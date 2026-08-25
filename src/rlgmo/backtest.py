"""バックテスト実行とベースライン方策。

RL の成績は**必ずベースラインと並べて**評価する。1 分足では、コスト後で
Buy & Hold や単純モメンタムに勝てない学習結果が普通に出る。勝てないなら採用しない。
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .env import TradingEnv

Policy = Callable[[np.ndarray], int]


class Agentish(Protocol):
    def probs(self, obs: np.ndarray) -> np.ndarray: ...


def run_policy(env: TradingEnv, policy: Policy, start: int = 0, episode_len: int | None = None) -> pd.DataFrame:
    """方策を区間全体に適用し、バーごとの記録を返す。

    Args:
        env: 評価用の環境（`training=False` を推奨）。
        policy: 観測 → 行動インデックス。
        start: 開始バー。
        episode_len: バー数（None なら最後まで）。

    Returns:
        columns=[equity, position, pnl, trade_cost, carry_cost, price, reward] の DataFrame。
    """
    obs, _ = env.reset(start=start, episode_len=episode_len)
    rows = []
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "ts": info["ts"], "equity": info["equity"], "position": info["position"],
                "pnl": info["pnl"], "trade_cost": info["trade_cost"], "carry_cost": info["carry_cost"],
                "price": info["price"], "reward": reward, "margin_ratio": info["margin_ratio"],
            }
        )
        if terminated or truncated:
            break
    return pd.DataFrame(rows).set_index("ts")


def ensemble_policy(
    agents: list[Agentish],
    action_values: tuple[float, ...],
    confidence: float = 0.0,
    deterministic: bool = True,
) -> Policy:
    """複数シードのエージェントを平均する方策を作る。

    各エージェントの行動確率を平均し、**期待ポジション** Σ p_i·a_i を計算して
    最も近い行動に丸める。単一シードの RL は分散が大きく再現しないため、
    実運用では 5〜10 シードのアンサンブルを基本とする。

    Args:
        agents: `probs(obs)` を持つエージェント列。
        action_values: 各行動に対応するポジション比率（`EnvConfig.actions`）。
        confidence: 期待ポジションの絶対値がこの閾値未満ならフラットにする
            （確信度が低い局面で無駄な回転を避ける）。
        deterministic: False ならアンサンブル分布からサンプルする。

    Returns:
        観測 → 行動インデックス の関数。
    """
    values = np.asarray(action_values, dtype=float)
    flat_idx = int(np.argmin(np.abs(values)))
    rng = np.random.default_rng(0)

    def policy(obs: np.ndarray) -> int:
        probs = np.mean([a.probs(obs)[0] for a in agents], axis=0)
        if not deterministic:
            return int(rng.choice(len(values), p=probs / probs.sum()))
        expected = float(probs @ values)
        if abs(expected) < confidence:
            return flat_idx
        return int(np.argmin(np.abs(values - expected)))

    return policy


# ----------------------------------------------------------------- ベースライン
def flat_policy(action_values: tuple[float, ...]) -> Policy:
    """常にノーポジション。コストだけを見るための下限ベンチマーク。"""
    idx = int(np.argmin(np.abs(np.asarray(action_values))))
    return lambda obs: idx


def long_policy(action_values: tuple[float, ...]) -> Policy:
    """常にフルロング（Buy & Hold 相当。レバレッジとボラターゲットは環境側で適用）。"""
    idx = int(np.argmax(np.asarray(action_values)))
    return lambda obs: idx


def momentum_policy(env: TradingEnv, feature: str = "ret_10_15m", threshold: float = 0.5) -> Policy:
    """15 分足モメンタムの符号で建てる単純ベースライン。"""
    col = env.feature_names.index(feature)
    values = np.asarray(env.cfg.actions)
    long_idx, short_idx = int(np.argmax(values)), int(np.argmin(values))
    flat_idx = int(np.argmin(np.abs(values)))

    def policy(obs: np.ndarray) -> int:
        signal = obs[col]
        if signal > threshold:
            return long_idx
        if signal < -threshold:
            return short_idx
        return flat_idx

    return policy


def trend_policy(
    env: TradingEnv,
    lookback_bars: int,
    long_only: bool = True,
    gain: float = 1.5,
    vol_window: int = 30,
) -> Policy:
    """時系列モメンタム（トレンドフォロー）のルールベース方策。

    実データ検証で見つかった唯一の優位性（`docs/real_data_findings.md` 5 節、
    `scripts/signal_survey.py`）をそのまま方策にしたもの。RL の比較対象であり、
    現時点では**これ自体が最有力の戦略**である。

        シグナル = 過去 lookback_bars のログリターン / (1バーのボラ × √lookback)
        目標ポジション = clip(シグナル / gain, -1, 1)   （long_only なら下限 0）

    価格系列は環境が保持しているものをそのまま使うため、特徴量セットに依存しない。
    サイズ調整（ボラターゲット）とコストは環境側が担当する。

    Args:
        env: 対象の環境（価格系列と現在位置を参照する）。
        lookback_bars: モメンタムのルックバック（環境のバー単位）。
        long_only: True ならショートしない。実測ではショート側が負けている。
        gain: シグナルを目標ポジションに変換する際の除数。小さいほど強気に建てる。
        vol_window: ボラ推定の窓（バー数）。

    Returns:
        観測 → 行動インデックス の関数。
    """
    close = env._close
    log_close = np.log(close)
    logret = np.diff(log_close, prepend=log_close[0])
    vol = pd.Series(logret).rolling(vol_window, min_periods=vol_window // 2).std().to_numpy()
    ret = np.full_like(log_close, np.nan)
    ret[lookback_bars:] = log_close[lookback_bars:] - log_close[:-lookback_bars]
    with np.errstate(invalid="ignore", divide="ignore"):
        signal = ret / (vol * np.sqrt(lookback_bars))
    signal = np.nan_to_num(np.clip(signal / gain, -1.0, 1.0), nan=0.0)
    if long_only:
        signal = np.clip(signal, 0.0, 1.0)
    values = np.asarray(env.cfg.actions, dtype=float)

    def policy(obs: np.ndarray) -> int:
        return int(np.argmin(np.abs(values - signal[env._t])))

    return policy


def random_policy(n_actions: int, seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(n_actions))


def delayed_policy(policy: Policy, delay_bars: int, flat_action: int) -> Policy:
    """`delay_bars` バー前の観測で意思決定する方策に変換する（執行遅延のストレス試験）。

    レイテンシ・障害・約定待ちで判断が遅れる状況を、環境を変えずに再現する。
    遅延に弱い方策は「直近 1 分の情報にだけ乗っている」ことを意味し、実運用で崩れる。
    """
    from collections import deque

    buffer: deque = deque(maxlen=delay_bars + 1)

    def wrapped(obs: np.ndarray) -> int:
        buffer.append(obs)
        if len(buffer) <= delay_bars:
            return flat_action
        return policy(buffer[0])

    return wrapped
