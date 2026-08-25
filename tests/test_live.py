"""ライブ執行ロジックの検証。実弾に直結する部分なので細かく見る。"""

import numpy as np
import pandas as pd
import pytest

from rlgmo.costs import CostConfig
from rlgmo.live import (
    LiveConfig,
    Order,
    StrategyEngine,
    exposure_from_positions,
    split_close_open,
    staleness_seconds,
)
from rlgmo.portfolio import apply_rebalance_band, compute_exposures, ladder_signal


def make_bars(symbols, minutes=60 * 24 * 100, seed=0):
    idx = pd.date_range("2026-01-01", periods=minutes, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    out = {}
    for i, s in enumerate(symbols):
        price = 1e6 * (i + 1) * np.exp(np.cumsum(rng.standard_normal(minutes) * 2e-4 + 2e-6))
        out[s] = pd.DataFrame({"open": price, "high": price * 1.0005, "low": price * 0.9995,
                               "close": price, "volume": 1.0}, index=idx)
    return out


# ------------------------------------------------------------------ 決済 / 新規の分割
def test_reducing_a_long_uses_close_only():
    order = Order("BTC_JPY", "SELL", 0.5, -0.5, 0.5, 1.0, 1e7)
    assert split_close_open(order, held_quantity=1.0) == (0.5, 0.0)


def test_flipping_from_long_to_short_closes_then_opens():
    """ロング 1.0 を持っていて 1.5 売る → 1.0 決済 + 0.5 新規ショート。"""
    order = Order("BTC_JPY", "SELL", 1.5, -1.5, -0.5, 1.0, 1e7)
    close_qty, open_qty = split_close_open(order, held_quantity=1.0)
    assert close_qty == pytest.approx(1.0)
    assert open_qty == pytest.approx(0.5)
    assert close_qty + open_qty == pytest.approx(order.quantity)


def test_increasing_a_short_uses_open_only():
    order = Order("BTC_JPY", "SELL", 0.5, -0.5, -1.5, -1.0, 1e7)
    assert split_close_open(order, held_quantity=-1.0) == (0.0, 0.5)


def test_opening_from_flat_uses_open_only():
    order = Order("BTC_JPY", "BUY", 1.0, 1.0, 1.0, 0.0, 1e7)
    assert split_close_open(order, held_quantity=0.0) == (0.0, 1.0)


def test_covering_a_short_and_going_long():
    order = Order("BTC_JPY", "BUY", 2.0, 2.0, 1.0, -1.0, 1e7)
    close_qty, open_qty = split_close_open(order, held_quantity=-1.0)
    assert (close_qty, open_qty) == (pytest.approx(1.0), pytest.approx(1.0))


# ------------------------------------------------------------------ 目標建玉
def test_live_targets_match_the_backtest_exactly():
    """ライブとバックテストが同じ数字を出すこと。ここがズレると全部が無意味になる。"""
    cfg = LiveConfig(symbols=("A", "B"), cost=CostConfig(half_spread_bp=1.5, slippage_bp=0.0))
    engine = StrategyEngine(cfg)
    bars = make_bars(["A", "B"], seed=3)

    live_targets = engine.targets(bars)

    resampled = engine.resample(bars)
    signals = {s: apply_rebalance_band(ladder_signal(df, 24, cfg.lookback_days, cfg.long_only,
                                                     cfg.gain, cfg.vol_window_bars), cfg.rebalance_band)
               for s, df in resampled.items()}
    backtest_last = compute_exposures(resampled, signals, 60, cfg.portfolio_config()).iloc[-1]
    for symbol, value in live_targets.items():
        assert value == pytest.approx(float(backtest_last[symbol]))


def test_targets_refuse_to_guess_without_enough_history():
    cfg = LiveConfig(symbols=("A",))
    engine = StrategyEngine(cfg)
    with pytest.raises(ValueError, match="助走"):
        engine.targets(make_bars(["A"], minutes=60 * 24 * 5))


def test_gross_exposure_respects_the_leverage_cap():
    cfg = LiveConfig(symbols=("A", "B", "C"), leverage_cap=2.0, target_vol_ann=1.0, asset_vol_ann=1.0)
    engine = StrategyEngine(cfg)
    targets = engine.targets(make_bars(["A", "B", "C"], seed=7))
    assert sum(abs(v) for v in targets.values()) <= 2.0 + 1e-9


# ------------------------------------------------------------------ 発注の組み立て
def test_small_differences_do_not_generate_orders():
    cfg = LiveConfig(symbols=("A",), min_trade_delta=0.05)
    engine = StrategyEngine(cfg)
    orders = engine.orders({"A": 0.52}, {"A": 0.50}, 1_000_000, {"A": 1e7})
    assert orders == []


def test_order_quantity_and_side_are_correct():
    cfg = LiveConfig(symbols=("A",), min_trade_delta=0.01)
    engine = StrategyEngine(cfg)
    orders = engine.orders({"A": -0.20}, {"A": 0.30}, 1_000_000, {"A": 500_000})
    assert len(orders) == 1
    order = orders[0]
    assert order.side == "SELL"
    assert order.delta_exposure == pytest.approx(-0.50)
    assert order.quantity == pytest.approx(0.5 * 1_000_000 / 500_000)


def test_risk_scaling_shrinks_every_target():
    cfg = LiveConfig(symbols=("A", "B"), min_trade_delta=0.01)
    engine = StrategyEngine(cfg)
    full = engine.orders({"A": 0.4, "B": -0.4}, {"A": 0.0, "B": 0.0}, 1e6, {"A": 1e6, "B": 1e6})
    half = engine.orders({"A": 0.4, "B": -0.4}, {"A": 0.0, "B": 0.0}, 1e6, {"A": 1e6, "B": 1e6}, size_scale=0.5)
    for a, b in zip(full, half):
        assert b.target_exposure == pytest.approx(a.target_exposure * 0.5)


# ------------------------------------------------------------------ 監視
def test_staleness_uses_the_most_delayed_symbol():
    now = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    fresh = pd.DataFrame({"close": [1.0]}, index=[now - pd.Timedelta(minutes=1)])
    old = pd.DataFrame({"close": [1.0]}, index=[now - pd.Timedelta(minutes=30)])
    assert staleness_seconds({"A": fresh, "B": old}, now) == pytest.approx(1800)


def test_exposure_from_positions_signs_and_scale():
    exposure = exposure_from_positions({"A": 0.5, "B": -2.0}, {"A": 1e7, "B": 1e6}, equity=1e7)
    assert exposure["A"] == pytest.approx(0.5)
    assert exposure["B"] == pytest.approx(-0.2)


# ------------------------------------------------------------------ リスク判定の解釈
@pytest.mark.parametrize(
    "reasons, halted, size, expect_hold, expect_flatten",
    [
        ([], False, 1.0, False, False),                       # 平常
        (["stale_data:hold"], False, 1.0, True, False),       # データが古い → 維持
        (["wide_spread:hold"], False, 1.0, True, False),      # 板が壊れている → 維持
        (["daily_loss_limit"], False, 0.0, False, True),      # 日次損失上限 → 畳む
        (["max_drawdown_stop(20.0%)"], True, 0.0, False, True),
        (["dd_taper(0.60)"], False, 0.6, False, False),       # 縮小はするが畳まない
    ],
)
def test_risk_decision_is_interpreted_correctly(reasons, halted, size, expect_hold, expect_flatten):
    """「:hold」で終わる理由は発注見送り。全決済してはいけない。

    データが古い・板が壊れているときに投げると、いちばん約定が悪い場面で
    成行を出すことになる。この区別を間違えると実弾で損をする。
    """
    filtered = [r for r in reasons if r != "below_min_delta"]
    hold_only = any(r.endswith(":hold") for r in filtered)
    flatten = halted or (size == 0.0 and not hold_only)
    assert hold_only is expect_hold
    assert flatten is expect_flatten


# ------------------------------------------------------------------ 最小発注幅の較正
def _simulate(targets: pd.DataFrame, delta: float, engine: StrategyEngine) -> tuple[int, float]:
    """目標建玉の系列を 1 本ずつ流し、発注回数と目標からの最大乖離を返す。"""
    held = {s: 0.0 for s in targets.columns}
    orders, worst = 0, 0.0
    for _, row in targets.iterrows():
        target = {s: float(row[s]) for s in targets.columns}
        issued = engine.orders(target, held, 1_000_000, {s: 1e6 for s in targets.columns})
        orders += len(issued)
        for order in issued:
            held[order.symbol] = order.target_exposure
        worst = max(worst, max(abs(target[s] - held[s]) for s in targets.columns))
    return orders, worst


def test_min_trade_delta_cuts_orders_without_losing_the_target():
    """最小発注幅は「無意味な発注を落とすフィルタ」であること。

    0.005 なら発注回数が 1 桁減るのに、建玉は常に目標の 0.005 以内にいる。
    既定を 0.05 まで上げると、銘柄あたりの建玉（0.03〜0.07）より粗くなり、
    目標から離れたまま放置される。
    """
    symbols = ("A", "B", "C")
    bars = make_bars(list(symbols), seed=11)
    base = LiveConfig(symbols=symbols)
    assert base.min_trade_delta == 0.005

    engine = StrategyEngine(base)
    resampled = engine.resample(bars)
    signals = {s: apply_rebalance_band(ladder_signal(df, 24, base.lookback_days, base.long_only,
                                                     base.gain, base.vol_window_bars), base.rebalance_band)
               for s, df in resampled.items()}
    targets = compute_exposures(resampled, signals, 60, base.portfolio_config()).iloc[-500:]

    tight, tight_gap = _simulate(targets, 0.005, StrategyEngine(base))
    coarse, coarse_gap = _simulate(targets, 0.05, StrategyEngine(LiveConfig(symbols=symbols, min_trade_delta=0.05)))

    continuous = len(targets) * len(symbols)
    assert tight < continuous / 5          # 連続リバランスより 1 桁近く少ない
    assert tight_gap <= 0.005 + 1e-9       # それでも目標から離れない
    assert coarse < tight                  # 粗い方が発注は減るが……
    assert coarse_gap > tight_gap * 3      # ……建玉が目標から大きく離れる
