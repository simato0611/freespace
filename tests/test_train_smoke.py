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
