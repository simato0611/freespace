"""環境の損益・コスト・リスク管理の会計検証。"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from rlgmo.costs import CostConfig
from rlgmo.data.synthetic import make_synthetic_ohlcv
from rlgmo.env import EnvConfig, RewardConfig, SyncVectorEnv, TradingEnv
from rlgmo.features import FeatureConfig, build_features

ZERO_COST = CostConfig(half_spread_bp=0.0, slippage_bp=0.0, taker_fee_bp=0.0, carry_mode="none")


def make_env_from_prices(prices, cfg: EnvConfig) -> TradingEnv:
    """任意の価格列から環境を作る（会計検証用。特徴量の中身は使わない）。"""
    n = len(prices)
    idx = pd.date_range("2026-03-01", periods=n, freq="1min", tz="UTC")
    meta = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": np.ones(n), "vol_1m": np.full(n, 1e-3), "vol_ratio": np.ones(n)},
        index=idx,
    )
    features = pd.DataFrame(np.zeros((n, 3), dtype=np.float32), columns=["f0", "f1", "f2"], index=idx)
    return TradingEnv(features, meta, cfg, training=False)


def base_cfg(**kwargs) -> EnvConfig:
    defaults = dict(leverage_cap=1.0, vol_target=False, randomize_costs=False, cost=ZERO_COST,
                    daily_loss_limit=1.0, reward=RewardConfig(turnover_penalty=0.0, dd_penalty=0.0))
    defaults.update(kwargs)
    return EnvConfig(**defaults)


def test_flat_position_keeps_equity_constant():
    env = make_env_from_prices(np.linspace(1e7, 1.2e7, 200), base_cfg())
    env.reset(start=0)
    flat = env.cfg.actions.index(0.0)
    for _ in range(150):
        _, _, terminated, truncated, info = env.step(flat)
        assert info["equity"] == pytest.approx(env.cfg.initial_equity)
        if terminated or truncated:
            break


def test_full_long_tracks_price_when_costs_are_zero():
    """コスト 0・レバレッジ 1 倍のフルロングは、価格の変化率をほぼ再現する。"""
    ohlcv = make_synthetic_ohlcv(60 * 24 * 2, seed=21)
    features, meta = build_features(ohlcv, FeatureConfig(scale_window=720))
    env = TradingEnv(features, meta, base_cfg(), training=False)
    env.reset(start=0)
    long_idx = int(np.argmax(env.cfg.actions))
    start_price = meta["close"].iloc[0]
    while True:
        _, _, terminated, truncated, info = env.step(long_idx)
        if terminated or truncated:
            break
    price_ratio = info["price"] / start_price
    equity_ratio = env.equity / env.cfg.initial_equity
    assert equity_ratio == pytest.approx(price_ratio, rel=0.01)


def test_turnover_cost_matches_spread_exactly():
    """定価格でロング⇄フラットを往復した損失は、支払ったスプレッドの合計に一致する。"""
    price = 1e7
    cost = CostConfig(half_spread_bp=3.0, slippage_bp=1.0, carry_mode="none", spread_vol_beta=0.0)
    env = make_env_from_prices(np.full(60, price), base_cfg(cost=cost))
    env.reset(start=0)
    long_idx, flat_idx = int(np.argmax(env.cfg.actions)), env.cfg.actions.index(0.0)
    total_cost = 0.0
    for i in range(40):
        _, _, _, truncated, info = env.step(long_idx if i % 2 == 0 else flat_idx)
        total_cost += info["trade_cost"]
        if truncated:
            break
    assert env.cfg.initial_equity - env.equity == pytest.approx(total_cost, rel=1e-9)
    assert total_cost > 0


def test_carry_fee_charged_once_per_day():
    """建玉管理料は 06:00 JST をまたぐバーで 1 回だけ、建玉評価額の 0.04% 課金される。"""
    n = 60 * 24 + 10
    idx = pd.date_range("2026-03-01 00:00", periods=n, freq="1min", tz="UTC")  # JST 09:00 開始
    prices = np.full(n, 1e7)
    meta = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices,
                         "volume": np.ones(n), "vol_1m": np.full(n, 1e-3), "vol_ratio": np.ones(n)}, index=idx)
    features = pd.DataFrame(np.zeros((n, 2), dtype=np.float32), columns=["a", "b"], index=idx)
    cfg = base_cfg(cost=CostConfig(half_spread_bp=0.0, slippage_bp=0.0, carry_mode="daily_0600",
                                   carry_rate_daily=0.0004))
    env = TradingEnv(features, meta, cfg, training=False)
    env.reset(start=0)
    long_idx = int(np.argmax(env.cfg.actions))
    charges = []
    while True:
        _, _, terminated, truncated, info = env.step(long_idx)
        if info["carry_cost"] > 0:
            charges.append(info["carry_cost"])
        if terminated or truncated:
            break
    assert len(charges) == 1
    assert charges[0] == pytest.approx(env.cfg.initial_equity * 0.0004, rel=1e-6)


def test_liquidation_on_large_adverse_move():
    """定率運用では建玉が自動で縮むため、ロスカットは 1 バーの急落（ギャップ）で起きる。

    レバレッジ 2 倍・フルロングで 1 分に -20% のフラッシュクラッシュが来ると、
    有効証拠金は 0.6 倍・建玉評価額は 0.8 倍 → 維持率 0.75 でロスカットに触れる。
    """
    prices = np.concatenate([np.full(5, 1e7), np.full(30, 0.78e7)])
    env = make_env_from_prices(prices, base_cfg(leverage_cap=2.0))
    env.reset(start=0)
    long_idx = int(np.argmax(env.cfg.actions))
    terminated = False
    for _ in range(len(prices) - 3):
        _, _, terminated, truncated, info = env.step(long_idx)
        if terminated or truncated:
            break
    assert terminated
    assert env._pos == 0.0
    assert 0.5 < env.equity / env.cfg.initial_equity < 0.7


def test_daily_loss_limit_flattens_for_the_day_without_ending_episode():
    """日次損失上限は「当日フラット・翌日再開」。エピソードは終了させない。

    終了させてしまうと、バックテストが最初の悪い日で打ち切られ、評価期間が
    実際より短くなる（実運用のリスクレイヤとも挙動が食い違う）。
    """
    prices = np.concatenate([np.full(5, 1e7), np.linspace(1e7, 0.95e7, 120)])
    cfg = dataclasses.replace(base_cfg(leverage_cap=1.0), daily_loss_limit=0.02)
    env = make_env_from_prices(prices, cfg)
    env.reset(start=0)
    long_idx = int(np.argmax(env.cfg.actions))
    blocked_seen = False
    steps = 0
    for _ in range(len(prices) - 3):
        _, _, terminated, truncated, info = env.step(long_idx)
        steps += 1
        blocked_seen |= info["blocked"]
        if info["blocked"]:
            assert info["position"] == 0.0        # 当日はフラットに固定される
        assert not terminated                      # ロスカットではないので終了しない
        if truncated:
            break
    assert blocked_seen
    assert steps > 60  # 最初の損失日で打ち切られていない


def test_blocked_state_clears_on_new_jst_day():
    n = 60 * 30
    idx = pd.date_range("2026-03-01 19:00", periods=n, freq="1min", tz="UTC")  # JST 04:00 開始
    prices = np.concatenate([np.full(60, 1e7), np.linspace(1e7, 0.9e7, 240), np.full(n - 300, 0.9e7)])
    meta = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices,
                         "volume": np.ones(n), "vol_1m": np.full(n, 1e-3), "vol_ratio": np.ones(n)}, index=idx)
    features = pd.DataFrame(np.zeros((n, 2), dtype=np.float32), columns=["a", "b"], index=idx)
    cfg = dataclasses.replace(base_cfg(leverage_cap=1.0), daily_loss_limit=0.02)
    env = TradingEnv(features, meta, cfg, training=False)
    env.reset(start=0)
    long_idx = int(np.argmax(env.cfg.actions))
    blocked, unblocked_after = False, False
    while True:
        _, _, terminated, truncated, info = env.step(long_idx)
        if info["blocked"]:
            blocked = True
        elif blocked:
            unblocked_after = True  # 日付が変わって再開した
        if terminated or truncated:
            break
    assert blocked and unblocked_after


def test_vector_env_autoresets():
    ohlcv = make_synthetic_ohlcv(60 * 24 * 4, seed=22)
    features, meta = build_features(ohlcv, FeatureConfig(scale_window=720))
    cfg = EnvConfig(episode_len=30)
    vec = SyncVectorEnv([TradingEnv(features, meta, cfg, training=True) for _ in range(3)])
    obs = vec.reset(seed=0)
    assert obs.shape == (3, vec.observation_dim)
    seen_done = False
    for _ in range(80):
        obs, rewards, dones, _ = vec.step(np.random.randint(0, vec.n_actions, size=3))
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        seen_done |= dones.any()
    assert seen_done  # エピソード終了後も自動で再開している


def test_observation_contains_account_state():
    ohlcv = make_synthetic_ohlcv(60 * 24 * 3, seed=23)
    features, meta = build_features(ohlcv, FeatureConfig(scale_window=720))
    env = TradingEnv(features, meta, base_cfg(), training=False)
    obs, _ = env.reset(start=0)
    assert obs.shape[0] == features.shape[1] + 6
    assert obs[-6] == 0.0  # 初期ポジションはフラット
    obs, *_ = env.step(int(np.argmax(env.cfg.actions)))
    assert obs[-6] > 0.0  # ロング後はポジションが観測に反映される


def test_action_repeat_advances_multiple_bars_without_extra_trades():
    """判断間隔 (action_repeat) を空けると、保持中のバーでは発注コストが発生しない。"""
    n = 200
    idx = pd.date_range("2026-03-01", periods=n, freq="1min", tz="UTC")
    prices = np.full(n, 1e7)
    meta = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices,
                         "volume": np.ones(n), "vol_1m": np.full(n, 1e-3), "vol_ratio": np.ones(n)}, index=idx)
    features = pd.DataFrame(np.zeros((n, 2), dtype=np.float32), columns=["a", "b"], index=idx)
    cost = CostConfig(half_spread_bp=3.0, slippage_bp=0.0, carry_mode="none", spread_vol_beta=0.0)
    cfg = dataclasses.replace(base_cfg(cost=cost), action_repeat=15)
    env = TradingEnv(features, meta, cfg, training=False)

    env.reset(start=0)
    t0 = env._t
    long_idx = int(np.argmax(env.cfg.actions))
    _, _, _, _, info = env.step(long_idx)
    assert env._t == t0 + 15                       # 1 ステップで 15 バー進む
    assert info["ts"] == idx[t0 + 15]

    # 建てた直後の 1 回だけコストが出る。同じ目標を出し続ければ追加コストは出ない。
    first_cost = cfg.initial_equity - env.equity
    assert first_cost > 0
    assert info["trade_cost"] == pytest.approx(first_cost, rel=1e-9)  # 最初のサブバーのコストが集計されている
    equity_after_entry = env.equity
    for _ in range(3):
        env.step(long_idx)
    assert env.equity == pytest.approx(equity_after_entry, rel=1e-9)


def test_action_repeat_still_charges_carry_while_holding():
    """保持中も建玉管理料は毎バー判定される（06:00 JST をまたげば課金）。"""
    n = 60 * 24 + 60
    idx = pd.date_range("2026-03-01 00:00", periods=n, freq="1min", tz="UTC")
    prices = np.full(n, 1e7)
    meta = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices,
                         "volume": np.ones(n), "vol_1m": np.full(n, 1e-3), "vol_ratio": np.ones(n)}, index=idx)
    features = pd.DataFrame(np.zeros((n, 2), dtype=np.float32), columns=["a", "b"], index=idx)
    cfg = dataclasses.replace(
        base_cfg(cost=CostConfig(half_spread_bp=0.0, slippage_bp=0.0, carry_mode="daily_0600",
                                 carry_rate_daily=0.0004)),
        action_repeat=37,  # 課金バーが判断間隔の途中に来る設定（集計漏れの検出）
    )
    env = TradingEnv(features, meta, cfg, training=False)
    env.reset(start=0)
    long_idx = int(np.argmax(env.cfg.actions))
    charged = 0.0
    while True:
        _, _, terminated, truncated, info = env.step(long_idx)
        charged += info["carry_cost"]
        if terminated or truncated:
            break
    # 1 日で 1 回、建玉評価額の 0.04%
    assert 0.0 < charged < cfg.initial_equity * 0.0005
    assert charged == pytest.approx(cfg.initial_equity * 0.0004, rel=0.02)
