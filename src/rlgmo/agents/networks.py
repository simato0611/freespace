"""方策・価値関数のネットワーク。

金融時系列は S/N 比が低くサンプルあたりの情報量が小さいため、**小さく正則化の効いた**
ネットワークを使う。既定は 2 層 MLP + LayerNorm（幅 256/128）。
LSTM/GRU は表現力が上がる一方で過学習しやすいので、まず MLP で基準を作ってから
検討すること（状態にすでにマルチタイムフレームの履歴要約が入っている）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """離散行動（目標ポジション）の Actor-Critic。

    方策と価値で**胴体を共有しない**。取引環境では価値のスケールが大きく
    （1 バーの報酬が小さくても、割引和は桁が上がる）、共有すると価値損失の勾配が
    方策側の表現を壊す。パラメータ数は数万規模なので分けても十分に軽い。

    Args:
        obs_dim: 観測次元。
        n_actions: 行動数。
        hidden: 隠れ層の幅。
        dropout: 学習時のドロップアウト率（過学習抑制）。
        shared_trunk: True なら胴体を共有する（メモリ優先の場合のみ）。
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        shared_trunk: bool = False,
    ):
        super().__init__()

        def make_trunk() -> nn.Sequential:
            layers: list[nn.Module] = []
            last = obs_dim
            for width in hidden:
                layers += [orthogonal_init(nn.Linear(last, width)), nn.LayerNorm(width), nn.Tanh(),
                           nn.Dropout(dropout)]
                last = width
            return nn.Sequential(*layers)

        self.shared_trunk = shared_trunk
        self.trunk = make_trunk()
        self.value_trunk = self.trunk if shared_trunk else make_trunk()
        last = hidden[-1]
        self.policy_head = orthogonal_init(nn.Linear(last, n_actions), gain=0.01)
        self.value_head = orthogonal_init(nn.Linear(last, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy_head(self.trunk(obs))
        value = self.value_head(self.value_trunk(obs)).squeeze(-1)
        return logits, value

    def _dist(self, logits: torch.Tensor, eps: float) -> torch.distributions.Categorical:
        """一様分布と混ぜた行動分布を返す。

        `eps` は探索の下限。純粋な softmax 方策は、取引環境では「常にフラット」の
        局所解に落ちた時点でエントロピーが 0 に潰れ、エントロピーボーナスの勾配も
        消えて二度と抜け出せなくなる。一様分布を ε だけ混ぜておくと、
        行動確率に下限（ε/|A|）が保証され、勾配が消えない。
        """
        probs = torch.softmax(logits, dim=-1)
        if eps > 0:
            probs = probs * (1 - eps) + eps / probs.shape[-1]
        return torch.distributions.Categorical(probs=probs)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False, eps: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動をサンプル（または貪欲選択）する。

        Returns:
            (action, log_prob, value)
        """
        logits, value = self(obs)
        dist = self._dist(logits, eps)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    @torch.no_grad()
    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self(obs)
        return torch.softmax(logits, dim=-1)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, eps: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """収集時と同じ探索率 `eps` のもとで log 確率・エントロピー・価値を返す。"""
        logits, value = self(obs)
        dist = self._dist(logits, eps)
        return dist.log_prob(actions), dist.entropy(), value
