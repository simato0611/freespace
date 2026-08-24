"""PPO エージェント（離散行動）。

**なぜ PPO か**

- 環境が高速なシミュレータなので、オンポリシーのサンプル効率の悪さは問題にならない。
- 報酬が小さくノイジーな金融環境では、クリッピングによる保守的な更新と
  アドバンテージ正規化が学習を安定させる（DQN 系は Q 値のスケール推定が不安定になりがち）。
- 方策が確率分布として出るため、**確信度が低い局面では自然にフラット寄り**になり、
  アンサンブル平均も取りやすい。

代替案: 経験再利用を効かせたいなら QR-DQN / SAC-discrete、
本番ログから学ぶ段階では CQL / IQL といったオフライン RL を検討する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..env import SyncVectorEnv
from .networks import ActorCritic


@dataclass
class PPOConfig:
    """PPO のハイパーパラメータ。"""

    lr: float = 3e-4
    n_steps: int = 512            # 環境ごとのロールアウト長
    batch_size: int = 1024
    epochs: int = 4               # ノイジーな環境では多く回しすぎない
    gamma: float = 0.999          # 1 分足で実効ホライズン ≒ 1000 分
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01        # エントロピー係数の初期値（下の目標エントロピー制御で自動調整）
    ent_coef_final: float = 0.001
    ent_coef_max: float = 0.1     # 自動調整の上限
    ent_target_start: float = 0.60  # 目標エントロピー（ln(行動数) に対する比）: 序盤は探索的
    ent_target_final: float = 0.05  # 終盤は決定的に
    adaptive_entropy: bool = True   # False なら単純な線形減衰
    explore_eps_start: float = 0.10  # 一様分布を混ぜる割合（探索の下限）
    explore_eps_final: float = 0.01
    cost_curriculum_start: float = 0.25  # 学習序盤のコスト倍率（0 に近いほど探索しやすい）
    cost_curriculum_frac: float = 0.3    # 学習全体のこの割合をかけて実コストまで戻す
    bc_steps: int = 0                    # 行動クローニングの事前学習ステップ数（0 で無効）
    bc_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03       # 超えたらそのエポックで打ち切り
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    shared_trunk: bool = False    # 方策と価値で胴体を共有しない（価値損失の干渉を防ぐ）
    reward_scale: float = -1.0    # 収集した報酬に掛ける係数。負なら (1 - gamma) を使う
    device: str = "cpu"
    seed: int = 0


class PPOAgent:
    """PPO 学習器。

    Example:
        >>> agent = PPOAgent(obs_dim, n_actions, PPOConfig())
        >>> agent.learn(vec_env, total_steps=200_000)
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: PPOConfig | None = None) -> None:
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        self.net = ActorCritic(
            obs_dim, n_actions, self.cfg.hidden, self.cfg.dropout, self.cfg.shared_trunk
        ).to(self.device)
        self._ent_coef = self.cfg.ent_coef
        self._reward_scale = (1 - self.cfg.gamma) if self.cfg.reward_scale < 0 else self.cfg.reward_scale
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.obs_dim, self.n_actions = obs_dim, n_actions

    # ------------------------------------------------------------------ 学習
    def learn(
        self,
        vec_env: SyncVectorEnv,
        total_steps: int,
        callback=None,
        callback_every: int = 20_000,
        log_every: int = 10,
    ) -> dict:
        """ロールアウト収集と方策更新を繰り返す。

        Args:
            vec_env: ベクトル環境。
            total_steps: 総環境ステップ数（= n_envs × ロールアウト回数 × n_steps）。
            callback: `callback(agent, step) -> dict | None` 形式の評価コールバック。
                返り値に "score" があれば最良モデルの保存に使う。
            callback_every: コールバックを呼ぶ環境ステップ間隔。
            log_every: ログ出力の更新回数間隔。

        Returns:
            学習履歴の dict。
        """
        cfg = self.cfg
        n_envs = vec_env.num_envs
        obs = torch.as_tensor(vec_env.reset(seed=cfg.seed), dtype=torch.float32, device=self.device)
        rollout_size = cfg.n_steps * n_envs
        n_updates = max(1, total_steps // rollout_size)
        history = {"step": [], "policy_loss": [], "value_loss": [], "entropy": [], "reward": [], "eval": []}
        best_score, best_state, next_cb = -np.inf, None, callback_every

        for update in range(n_updates):
            frac = update / max(n_updates - 1, 1)
            # コストカリキュラム: 実コストをいきなり課すと「何もしない」局所解に張り付くため、
            # 序盤はコストを軽くして方向性のシグナルを学ばせ、徐々に実コストへ戻す。
            if cfg.cost_curriculum_start < 1.0:
                ramp = min(1.0, frac / max(cfg.cost_curriculum_frac, 1e-9))
                vec_env.set_cost_scale(cfg.cost_curriculum_start + (1 - cfg.cost_curriculum_start) * ramp)
            for group in self.opt.param_groups:  # 線形減衰
                group["lr"] = cfg.lr * (1 - 0.9 * frac)
            # 目標エントロピー: 序盤は探索的、終盤は決定的。金融環境では放っておくと
            # 「常にフラット」（コストを払わない解）に早期収束してエントロピーが 0 に潰れる。
            ent_target = np.log(self.n_actions) * (
                cfg.ent_target_start + (cfg.ent_target_final - cfg.ent_target_start) * frac
            )

            eps = cfg.explore_eps_start + (cfg.explore_eps_final - cfg.explore_eps_start) * frac
            buf, obs = self._collect(vec_env, obs, eps)
            stats = self._update(buf, ent_target, eps)
            step = (update + 1) * rollout_size

            history["step"].append(step)
            history["reward"].append(float(buf["rewards"].mean()) / self._reward_scale)
            for key in ("policy_loss", "value_loss", "entropy"):
                history[key].append(stats[key])
            if update % log_every == 0:
                print(
                    f"[ppo] step={step:>9,} reward/bar={history['reward'][-1]:+.4f} "
                    f"pi={stats['policy_loss']:+.4f} vf={stats['value_loss']:.4f} "
                    f"H={stats['entropy']:.3f}/{ent_target:.2f} c_ent={self._ent_coef:.4f} "
                    f"eps={eps:.3f} kl={stats['kl']:.4f}"
                )
            if callback is not None and step >= next_cb:
                next_cb += callback_every
                result = callback(self, step) or {}
                history["eval"].append({"step": step, **result})
                score = result.get("score", -np.inf)
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                print(f"[eval] step={step:>9,} " + " ".join(f"{k}={v:.3f}" for k, v in result.items() if isinstance(v, float)))

        if best_state is not None:  # 検証成績が最良の重みを採用（早期終了に相当）
            self.net.load_state_dict(best_state)
        history["best_score"] = best_score
        return history

    # -------------------------------------------------------------- 内部処理
    def _collect(self, vec_env: SyncVectorEnv, obs: torch.Tensor, eps: float = 0.0) -> tuple[dict, torch.Tensor]:
        cfg, n_envs = self.cfg, vec_env.num_envs
        obs_buf = torch.zeros(cfg.n_steps, n_envs, self.obs_dim, device=self.device)
        act_buf = torch.zeros(cfg.n_steps, n_envs, dtype=torch.long, device=self.device)
        logp_buf = torch.zeros(cfg.n_steps, n_envs, device=self.device)
        val_buf = torch.zeros(cfg.n_steps, n_envs, device=self.device)
        rew_buf = torch.zeros(cfg.n_steps, n_envs, device=self.device)
        done_buf = torch.zeros(cfg.n_steps, n_envs, device=self.device)

        self.net.eval()
        for t in range(cfg.n_steps):
            action, logp, value = self.net.act(obs, eps=eps)
            next_obs, reward, done, _ = vec_env.step(action.cpu().numpy())
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action, logp, value
            # 報酬を (1-γ) 倍しておく。アドバンテージは正規化されるので方策勾配は不変だが、
            # 価値のターゲットが O(1) に収まり、価値損失が暴れなくなる。
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device) * self._reward_scale
            done_buf[t] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, last_value = self.net(obs)

        # GAE(λ)
        adv = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(n_envs, device=self.device)
        for t in reversed(range(cfg.n_steps)):
            next_value = last_value if t == cfg.n_steps - 1 else val_buf[t + 1]
            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        buf = {
            "obs": obs_buf.reshape(-1, self.obs_dim),
            "actions": act_buf.reshape(-1),
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "returns": (adv + val_buf).reshape(-1),
            "rewards": rew_buf,
        }
        return buf, obs

    def _update(self, buf: dict, ent_target: float, eps: float = 0.0) -> dict:
        """1 ロールアウトぶんの方策・価値更新。

        `ent_target` は目標エントロピー。実測が下回ればエントロピー係数を上げ、
        上回れば下げる（乗算制御）。これが無いと、コストを避ける自明解に張り付いて
        探索が止まる。
        """
        cfg = self.cfg
        self.net.train()
        n = buf["obs"].shape[0]
        idx = np.arange(n)
        adv = (buf["adv"] - buf["adv"].mean()) / (buf["adv"].std() + 1e-8)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_batches = 0
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                batch = torch.as_tensor(idx[start : start + cfg.batch_size], device=self.device)
                logp, entropy, value = self.net.evaluate(buf["obs"][batch], buf["actions"][batch], eps=eps)
                ratio = torch.exp(logp - buf["logp"][batch])
                a = adv[batch]
                policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * a).mean()
                value_loss = nn.functional.mse_loss(value, buf["returns"][batch])
                loss = policy_loss + cfg.vf_coef * value_loss - self._ent_coef * entropy.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()  # 近似 KL
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += kl
                n_batches += 1
            if stats["kl"] / max(n_batches, 1) > cfg.target_kl:  # 更新しすぎを防ぐ
                break
        for key in stats:
            stats[key] /= max(n_batches, 1)
        if cfg.adaptive_entropy:  # 目標エントロピーへの乗算制御
            factor = 1.05 if stats["entropy"] < ent_target else 1 / 1.05
            self._ent_coef = float(np.clip(self._ent_coef * factor, cfg.ent_coef_final, cfg.ent_coef_max))
        stats["ent_coef"] = self._ent_coef
        stats["ent_target"] = ent_target
        return stats


    # ------------------------------------------------------- 行動クローニング
    def pretrain(self, vec_env: SyncVectorEnv, teacher, steps: int = 20_000, batch_size: int = 512) -> float:
        """簡単なルールベース方策を教師にした事前学習（行動クローニング）。

        コストのある取引環境では、無作為な初期方策は「何もしない（フラット）」という
        報酬 0 の局所解に落ちやすい。モメンタム等の粗い教師を数万ステップ模倣させて
        から PPO を始めると、探索が「取引する側」から始まるため、この罠を避けられる。
        教師の性能自体は重要ではない（PPO が上書きする）。

        Args:
            vec_env: 観測を集めるための環境。
            teacher: 観測 → 行動インデックス の関数。
            steps: 収集・学習するサンプル数。
            batch_size: ミニバッチサイズ。

        Returns:
            最終の教師一致率（0〜1）。
        """
        if steps <= 0:
            return float("nan")
        obs_list, act_list = [], []
        obs = vec_env.reset(seed=self.cfg.seed)
        while len(obs_list) * vec_env.num_envs < steps:
            actions = np.array([teacher(o) for o in obs])
            obs_list.append(obs.copy())
            act_list.append(actions)
            obs, _, _, _ = vec_env.step(actions)
        x = torch.as_tensor(np.concatenate(obs_list), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.concatenate(act_list), dtype=torch.long, device=self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.bc_lr)
        self.net.train()
        accuracy = 0.0
        for epoch in range(5):
            perm = torch.randperm(len(x), device=self.device)
            for start in range(0, len(x), batch_size):
                batch = perm[start : start + batch_size]
                logits, _ = self.net(x[batch])
                loss = nn.functional.cross_entropy(logits, y[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                opt.step()
            with torch.no_grad():
                self.net.eval()
                accuracy = float((self.net(x)[0].argmax(-1) == y).float().mean())
                self.net.train()
        print(f"[bc] 事前学習 完了: samples={len(x):,} 教師一致率={accuracy:.2%}")
        return accuracy

    # ------------------------------------------------------------------ 推論
    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.net.eval()
        tensor = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        action, _, _ = self.net.act(tensor, deterministic=deterministic)
        return action.cpu().numpy()

    @torch.no_grad()
    def probs(self, obs: np.ndarray) -> np.ndarray:
        """行動確率（アンサンブル平均や信頼度フィルタに使う）。"""
        self.net.eval()
        tensor = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        return self.net.action_probs(tensor).cpu().numpy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state_dict": self.net.state_dict(),
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PPOAgent":
        blob = torch.load(path, map_location=device, weights_only=False)
        cfg: PPOConfig = blob["cfg"]
        cfg.device = device
        agent = cls(blob["obs_dim"], blob["n_actions"], cfg)
        agent.net.load_state_dict(blob["state_dict"])
        return agent
