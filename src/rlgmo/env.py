"""ポジション制御 MDP（GMO レバレッジ取引シミュレータ）。

MDP の定義
----------
- **意思決定周期**: 1 分足のクローズごと（上位足 5/15 分の情報は状態に含む）。
- **行動**: 目標ポジション比率 a ∈ {-1, -0.5, 0, +0.5, +1}（既定）。
  実効エクスポージャ = a × レバレッジ上限 × 有効証拠金 × ボラターゲット係数。
- **執行規約（ルックアヘッド防止）**: 時刻 t のクローズ情報で決めた注文は
  **t+1 のオープンで約定**する。約定価格はスプレッド + スリッページのぶん不利側。
- **損益**: t のクローズ → t+1 のオープン は旧ポジション、t+1 のオープン → クローズ は
  新ポジションで評価する（ギャップとイントラバーを分離）。
- **建玉サイズ**: 常に「有効証拠金 × 目標比率 × レバレッジ上限」に再調整する（定率）。
  含み損が出れば自動的に建玉が縮む＝実質的なリスク一定運用になり、ロスカットは
  基本的に「1 バーで大きくギャップした場合」にのみ発生する。微小なズレでの
  無駄な発注は `rebalance_tolerance` で抑える。
- **コスト**: 取引コスト（片道 = スプレッド/2 + スリッページ）+ 建玉管理料
  （0.04%/日・JST 06:00 課金）。GMO レバレッジの取引手数料は 0。
- **終了条件**: 証拠金維持率がロスカット水準を割る / 日次損失上限に到達 /
  エピソード長に到達。

報酬
----
既定は「対数エクイティ変化（コスト込み）をボラで正規化したもの」。
これは対数効用 = Kelly 的な資金成長率の最大化に対応し、スケール不変で学習が安定する。
オプションで Moody & Saffell の Differential Sharpe Ratio (DSR) も選べる。
追加ペナルティ（回転率・ドローダウン）は「実コストで足りないぶんの抑制」として使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .costs import CostConfig, carry_flags, carry_rate_per_bar, effective_trade_rate

ACCOUNT_FEATURES = 6
BARS_PER_YEAR = 365 * 24 * 60


@dataclass
class RewardConfig:
    """報酬の設計パラメータ。"""

    kind: str = "logret"           # "logret" | "dsr"
    vol_norm: bool = True          # 1 バーあたり目標ボラで正規化してスケールを揃える
    target_vol_ann: float = 0.20   # 正規化に使う年率ボラ（= 目標リスク水準）
    turnover_penalty: float = 2.0e-4   # |Δポジション| への追加ペナルティ（実コストとは別）
    dd_penalty: float = 0.5        # ドローダウン増分へのペナルティ
    scale: float = 1.0
    dsr_eta: float = 1 / 1440      # DSR の EMA 係数（≒ 1 日）
    bankrupt_penalty: float = 1.0  # ロスカット到達時の一括ペナルティ


@dataclass
class EnvConfig:
    """環境の設定。"""

    leverage_cap: float = 2.0                 # 個人向けレバレッジ上限（GMO は 2 倍）
    actions: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    episode_len: int = 1440                   # 1 エピソード = 1 日ぶん
    initial_equity: float = 1_000_000.0
    maintenance_margin: float = 0.75          # 証拠金維持率のロスカット水準
    daily_loss_limit: float = 0.02            # 当日 -2% で打ち切り（リスクレイヤの模擬）
    vol_target: bool = True                   # ボラターゲットでサイズを調整
    vol_target_ann: float = 0.20
    vol_scale_cap: float = 1.0                # ボラターゲットによる増幅の上限
    rebalance_tolerance: float = 0.02         # 建玉評価額のズレがこの割合未満なら発注しない
    randomize_costs: bool = True              # ドメインランダマイゼーション（学習時のみ）
    cost_jitter: float = 0.5                  # 学習時にコストを ±50% 揺らす
    reward: RewardConfig = field(default_factory=RewardConfig)
    cost: CostConfig = field(default_factory=CostConfig)


class TradingEnv:
    """1 分足のポジション制御環境（Gym 互換の最小 API）。

    Example:
        >>> env = TradingEnv(features, meta, EnvConfig())
        >>> obs, info = env.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env.step(2)
    """

    def __init__(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
        cfg: EnvConfig | None = None,
        training: bool = True,
    ) -> None:
        if not features.index.equals(meta.index):
            raise ValueError("features と meta の index が一致していません")
        self.cfg = cfg or EnvConfig()
        self.training = training
        self.feature_names = list(features.columns)
        self.index = pd.DatetimeIndex(features.index)

        self._feats = np.ascontiguousarray(features.to_numpy(dtype=np.float32))
        self._open = meta["open"].to_numpy(dtype=np.float64)
        self._close = meta["close"].to_numpy(dtype=np.float64)
        self._vol_1m = np.nan_to_num(meta["vol_1m"].to_numpy(dtype=np.float64), nan=1e-4)
        self._vol_ratio = np.nan_to_num(meta["vol_ratio"].to_numpy(dtype=np.float64), nan=1.0)
        self._carry = carry_flags(self.index, self.cfg.cost.carry_hour_jst)
        self._n = len(self.index)

        self.n_actions = len(self.cfg.actions)
        self.observation_dim = self._feats.shape[1] + ACCOUNT_FEATURES
        self._rng = np.random.default_rng()
        self._target_vol_bar = self.cfg.reward.target_vol_ann / np.sqrt(BARS_PER_YEAR)
        # コストカリキュラム用の倍率（学習序盤だけコストを軽くする。詳細は agents/ppo.py）
        self.cost_scale = 1.0
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(
        self, seed: int | None = None, start: int | None = None, episode_len: int | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """エピソードを初期化する。

        Args:
            seed: 乱数シード。
            start: 開始バー（None なら学習時はランダム、評価時は先頭）。
            episode_len: エピソード長（None なら設定値、評価時は残り全部）。
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        length = episode_len or (self.cfg.episode_len if self.training else self._n - 2)
        max_start = max(1, self._n - length - 2)
        if start is None:
            start = int(self._rng.integers(0, max_start)) if self.training else 0
        self._t = int(np.clip(start, 0, max_start))
        self._end = min(self._t + length, self._n - 2)

        self.equity = self.cfg.initial_equity
        self._peak = self.equity
        self._notional = 0.0
        self._pos = 0.0
        self._entry_price = self._close[self._t]
        self._bars_held = 0
        self._day_start_equity = self.equity
        self._day_id = self._jst_day(self._t)
        self._dsr_a = 0.0
        self._dsr_b = 1e-8
        # ドメインランダマイゼーション: エピソードごとにコスト水準を揺らす
        jitter = 1.0
        if self.training and self.cfg.randomize_costs:
            jitter = float(np.exp(self._rng.uniform(-1, 1) * np.log1p(self.cfg.cost_jitter)))
        self._cost_jitter = jitter
        return self._obs(), {"t": self._t, "equity": self.equity}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """1 バー進める。

        Args:
            action: `cfg.actions` のインデックス。

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        t = self._t
        cfg = self.cfg
        target = float(cfg.actions[int(action)]) * self._size_scale(t)
        prev_pos, prev_notional = self._pos, self._notional

        # --- 執行: t+1 のオープンで約定
        fill_open = self._open[t + 1]
        desired_notional = target * cfg.leverage_cap * self.equity
        marked_prev = prev_notional * (fill_open / self._close[t])
        # 建玉は毎バー「有効証拠金 × 目標比率」に再調整される（= 実質的なボラ調整）。
        # ただし僅かなズレで発注し続けるとスプレッドを無駄に払うため、許容幅を設ける。
        if abs(desired_notional - marked_prev) < cfg.rebalance_tolerance * cfg.leverage_cap * self.equity:
            target_notional, traded = marked_prev, 0.0
            target = marked_prev / max(cfg.leverage_cap * self.equity, 1e-9)
        else:
            target_notional = desired_notional
            traded = desired_notional - marked_prev
        rate = float(effective_trade_rate(cfg.cost, self._vol_ratio[t + 1])) * self._cost_jitter * self.cost_scale
        trade_cost = abs(traded) * rate

        # --- 損益: ギャップ（旧ポジション）+ イントラバー（新ポジション）
        r_gap = fill_open / self._close[t] - 1.0
        r_intra = self._close[t + 1] / fill_open - 1.0
        pnl = prev_notional * r_gap + target_notional * r_intra

        # --- 建玉管理料（JST 06:00 課金）
        carry_cost = (
            abs(target_notional) * carry_rate_per_bar(cfg.cost, 1, bool(self._carry[t + 1])) * self.cost_scale
        )

        prev_equity = self.equity
        self.equity = max(prev_equity + pnl - trade_cost - carry_cost, 1e-9)
        self._notional = target_notional * (1.0 + r_intra)
        self._pos = target
        if abs(target) < 1e-9:
            self._bars_held = 0
            self._entry_price = self._close[t + 1]
        elif np.sign(target) != np.sign(prev_pos) or abs(prev_pos) < 1e-9:
            self._bars_held = 1
            self._entry_price = fill_open
        else:
            self._bars_held += 1

        # --- 報酬
        log_ret = float(np.log(self.equity / prev_equity))
        self._peak = max(self._peak, self.equity)
        dd_increment = max(0.0, (self._peak - self.equity) / self._peak - (self._peak - prev_equity) / self._peak)
        reward = self._reward(log_ret, abs(target - prev_pos), dd_increment)

        # --- 終了判定
        self._t = t + 1
        terminated = False
        margin_ratio = self._margin_ratio()
        if margin_ratio < cfg.maintenance_margin:  # ロスカット
            self._force_flat(rate)
            reward -= cfg.reward.bankrupt_penalty
            terminated = True
        if self._jst_day(self._t) != self._day_id:  # 日次リセット
            self._day_id = self._jst_day(self._t)
            self._day_start_equity = self.equity
        elif self.equity < self._day_start_equity * (1 - cfg.daily_loss_limit):  # 日次損失上限
            self._force_flat(rate)
            terminated = True
        truncated = self._t >= self._end

        info = {
            "t": self._t,
            "ts": self.index[self._t],
            "equity": self.equity,
            "position": self._pos,
            "trade_cost": trade_cost,
            "carry_cost": carry_cost,
            "pnl": pnl,
            "log_ret": log_ret,
            "turnover": abs(target - prev_pos),
            "margin_ratio": margin_ratio,
            "price": self._close[self._t],
        }
        return self._obs(), float(reward), terminated, truncated, info

    # -------------------------------------------------------------- internals
    def _size_scale(self, t: int) -> float:
        """ボラターゲットによるサイズ調整係数。"""
        if not self.cfg.vol_target:
            return 1.0
        target_bar = self.cfg.vol_target_ann / np.sqrt(BARS_PER_YEAR)
        vol = max(self._vol_1m[t], 1e-6)
        return float(np.clip(target_bar / (vol * self.cfg.leverage_cap), 0.05, self.cfg.vol_scale_cap))

    def _margin_ratio(self) -> float:
        """証拠金維持率 = 有効証拠金 / 必要証拠金（必要証拠金 = 建玉評価額 / レバレッジ上限）。"""
        required = abs(self._notional) / self.cfg.leverage_cap
        return float("inf") if required < 1e-9 else self.equity / required

    def _force_flat(self, rate: float) -> None:
        self.equity = max(self.equity - abs(self._notional) * rate, 1e-9)
        self._notional = 0.0
        self._pos = 0.0
        self._bars_held = 0

    def _reward(self, log_ret: float, turnover: float, dd_increment: float) -> float:
        cfg = self.cfg.reward
        if cfg.kind == "dsr":
            base = self._differential_sharpe(log_ret)
        else:
            base = log_ret / self._target_vol_bar if cfg.vol_norm else log_ret
        penalty = cfg.turnover_penalty * turnover / (self._target_vol_bar if cfg.vol_norm else 1.0)
        return cfg.scale * (base - penalty - cfg.dd_penalty * dd_increment / self._target_vol_bar)

    def _differential_sharpe(self, r: float) -> float:
        """Moody & Saffell (1998) の Differential Sharpe Ratio。"""
        eta, a, b = self.cfg.reward.dsr_eta, self._dsr_a, self._dsr_b
        da, db = r - a, r * r - b
        denom = (b - a * a) ** 1.5
        dsr = 0.0 if denom < 1e-12 else (b * da - 0.5 * a * db) / denom
        self._dsr_a, self._dsr_b = a + eta * da, b + eta * db
        return float(np.clip(dsr * eta, -10, 10))

    def _jst_day(self, t: int) -> int:
        return int((self.index[t].value // 60_000_000_000 + 9 * 60) // 1440)

    def _obs(self) -> np.ndarray:
        t = self._t
        price = self._close[t]
        vol = max(self._vol_1m[t], 1e-6)
        unreal = 0.0
        if abs(self._pos) > 1e-9:
            unreal = np.sign(self._pos) * np.log(price / self._entry_price) / (vol * np.sqrt(max(self._bars_held, 1)))
        account = np.array(
            [
                self._pos,
                abs(self._notional) / max(self.equity * self.cfg.leverage_cap, 1e-9),
                np.clip(unreal, -5, 5),
                np.log1p(self._bars_held) / np.log(1440),
                np.clip((self._peak - self.equity) / self._peak * 10, 0, 5),
                np.clip((self.equity / self._day_start_equity - 1) / self.cfg.daily_loss_limit, -3, 3),
            ],
            dtype=np.float32,
        )
        return np.concatenate([self._feats[t], account])


class SyncVectorEnv:
    """複数の `TradingEnv` を同期的にまとめて進める簡易ベクトル環境。

    RL の分散を下げるにはバッチで異なる時期を同時に経験させるのが効く。
    終了したエピソードは自動的に別のランダム区間から再開する（auto-reset）。
    """

    def __init__(self, envs: list[TradingEnv]) -> None:
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_dim = envs[0].observation_dim
        self.n_actions = envs[0].n_actions

    def set_cost_scale(self, scale: float) -> None:
        """全環境のコスト倍率を設定する（コストカリキュラム用）。"""
        for env in self.envs:
            env.cost_scale = scale

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = [env.reset(seed=None if seed is None else seed + i)[0] for i, env in enumerate(self.envs)]
        return np.stack(obs)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(obs_list), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos
