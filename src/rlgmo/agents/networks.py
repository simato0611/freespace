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

    Args:
        obs_dim: 観測次元。
        n_actions: 行動数。
        hidden: 隠れ層の幅。
        dropout: 学習時のドロップアウト率（過学習抑制）。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden:
            layers += [orthogonal_init(nn.Linear(last, width)), nn.LayerNorm(width), nn.Tanh(), nn.Dropout(dropout)]
            last = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = orthogonal_init(nn.Linear(last, n_actions), gain=0.01)
        self.value_head = orthogonal_init(nn.Linear(last, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動をサンプル（または貪欲選択）する。

        Returns:
            (action, log_prob, value)
        """
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    @torch.no_grad()
    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self(obs)
        return torch.softmax(logits, dim=-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
