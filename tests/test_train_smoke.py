"""学習パイプラインの疎通テスト（短時間で回るサイズ）。"""

import dataclasses

import numpy as np

from rlgmo.agents.ppo import PPOAgent, PPOConfig
from rlgmo.backtest import ensemble_policy, flat_policy, momentum_policy, run_policy
from rlgmo.data.synthetic import make_synthetic_ohlcv
from rlgmo.env import EnvConfig, SyncVectorEnv, TradingEnv
from rlgmo.features import FeatureConfig, build_features


def build(training=True, n=60 * 24 * 12):
    ohlcv = make_synthetic_ohlcv(n, seed=31)
    features, meta = build_features(ohlcv, FeatureConfig(scale_window=1440))
    cfg = EnvConfig(episode_len=240)
    if not training:
        cfg = dataclasses.replace(cfg, randomize_costs=False)
    return features, meta, cfg


def test_ppo_learns_without_errors_and_stays_finite():
    features, meta, cfg = build()
    vec = SyncVectorEnv([TradingEnv(features, meta, cfg, training=True) for _ in range(2)])
    agent = PPOAgent(vec.observation_dim, vec.n_actions, PPOConfig(n_steps=64, batch_size=128, epochs=2, seed=0))
    history = agent.learn(vec, total_steps=2048, log_every=1000)
    assert len(history["step"]) >= 1
    assert all(np.isfinite(v) for v in history["reward"])
    probs = agent.probs(np.zeros((1, vec.observation_dim), dtype=np.float32))
    assert np.isfinite(probs).all() and abs(probs.sum() - 1.0) < 1e-5


def test_behavior_cloning_imitates_teacher():
    features, meta, cfg = build()
    envs = [TradingEnv(features, meta, cfg, training=True) for _ in range(2)]
    vec = SyncVectorEnv(envs)
    agent = PPOAgent(vec.observation_dim, vec.n_actions, PPOConfig(seed=0, bc_lr=3e-3))
    accuracy = agent.pretrain(vec, momentum_policy(envs[0]), steps=2000, batch_size=256)
    assert accuracy > 0.8  # 粗い教師でも十分に模倣できる


def test_cost_curriculum_scales_costs():
    features, meta, cfg = build()
    vec = SyncVectorEnv([TradingEnv(features, meta, cfg, training=True) for _ in range(2)])
    vec.set_cost_scale(0.25)
    assert all(env.cost_scale == 0.25 for env in vec.envs)


def test_exploration_floor_keeps_entropy_positive():
    """一様分布の混合により、行動確率に下限が入る（エントロピー崩壊の予防）。"""
    features, meta, cfg = build()
    env = TradingEnv(features, meta, cfg, training=True)
    agent = PPOAgent(env.observation_dim, env.n_actions, PPOConfig(seed=0))
    import torch

    logits = torch.tensor([[50.0, -50.0, -50.0, -50.0, -50.0]])  # 極端に確信した方策
    dist = agent.net._dist(logits, eps=0.1)
    assert float(dist.probs.min()) >= 0.1 / 5 - 1e-6
    assert float(dist.entropy()) > 0.05


def test_saved_agent_roundtrips(tmp_path):
    features, meta, cfg = build()
    env = TradingEnv(features, meta, cfg, training=True)
    agent = PPOAgent(env.observation_dim, env.n_actions, PPOConfig(seed=0))
    path = agent.save(tmp_path / "agent.pt")
    loaded = PPOAgent.load(path)
    obs = np.zeros((3, env.observation_dim), dtype=np.float32)
    assert np.allclose(agent.probs(obs), loaded.probs(obs))


def test_ensemble_policy_falls_back_to_flat_below_confidence():
    features, meta, cfg = build(training=False)
    env = TradingEnv(features, meta, cfg, training=False)

    class Uniform:
        def probs(self, obs):
            return np.full((1, len(cfg.actions)), 1 / len(cfg.actions))

    policy = ensemble_policy([Uniform()], cfg.actions, confidence=0.15)
    flat = flat_policy(cfg.actions)
    assert policy(np.zeros(env.observation_dim)) == flat(np.zeros(env.observation_dim))

    record = run_policy(env, policy, start=0, episode_len=200)
    assert record["position"].abs().max() == 0.0
    assert record["equity"].iloc[-1] == cfg.initial_equity  # フラットならコストも発生しない


def test_trend_policy_follows_price_direction():
    """トレンドフォロー方策は、上昇局面でロング、下降局面でフラット（long_only）になる。"""
    import pandas as pd

    from rlgmo.backtest import trend_policy
    from rlgmo.env import TradingEnv

    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    up = np.linspace(1e7, 1.4e7, n // 2)
    down = np.linspace(1.4e7, 1.0e7, n - n // 2)
    prices = np.concatenate([up, down]) * (1 + 0.001 * np.sin(np.arange(n)))
    meta = pd.DataFrame({"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices,
                         "volume": np.ones(n), "vol_1m": np.full(n, 5e-3), "vol_ratio": np.ones(n)}, index=idx)
    features = pd.DataFrame(np.zeros((n, 2), dtype=np.float32), columns=["a", "b"], index=idx)
    env = TradingEnv(features, meta, EnvConfig(vol_target=False, randomize_costs=False), training=False)
    policy = trend_policy(env, lookback_bars=48, long_only=True)
    values = np.asarray(env.cfg.actions)

    env.reset(start=0)
    positions = []
    while True:
        action = policy(None)
        positions.append(values[action])
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    positions = np.array(positions)
    assert positions.min() >= 0.0                      # long_only
    assert positions[100:190].mean() > 0.5             # 上昇局面ではロング
    assert positions[260:380].mean() < 0.2             # 下降局面では降りる
