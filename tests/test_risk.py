"""リスクレイヤの検証（RL の出力に関わらず守られること）。"""

from rlgmo.risk import RiskLimits, RiskManager


def test_position_is_capped():
    rm = RiskManager(RiskLimits(max_position=0.5), equity=1_000_000)
    size, _ = rm.apply(1.0, 1_000_000, 0.5, 1.0, 5, 3.0)
    assert size == 0.5


def test_daily_loss_limit_flattens():
    rm = RiskManager(RiskLimits(daily_loss_limit=0.02), equity=1_000_000)
    size, info = rm.apply(1.0, 975_000, 0.5, 1.0, 5, 3.0, current=1.0)
    assert size == 0.0 and "daily_loss_limit" in info["reasons"]


def test_hard_stop_is_sticky():
    rm = RiskManager(RiskLimits(max_drawdown_stop=0.10), equity=1_000_000)
    rm.apply(1.0, 880_000, 0.5, 1.0, 5, 3.0)
    size, info = rm.apply(1.0, 1_000_000, 0.5, 1.0, 5, 3.0)  # 資金が戻っても自動再開しない
    assert size == 0.0 and info["halted"]


def test_stale_data_and_wide_spread_hold_position():
    rm = RiskManager(RiskLimits(max_data_staleness_sec=60, max_half_spread_bp=5), equity=1_000_000)
    hold, info = rm.apply(-1.0, 1_000_000, 0.5, 1.0, staleness_sec=300, margin_ratio=3.0, current=0.5)
    assert hold == 0.5 and "stale_data:hold" in info["reasons"]
    hold, info = rm.apply(-1.0, 1_000_000, 0.5, 20.0, staleness_sec=5, margin_ratio=3.0, current=0.5)
    assert hold == 0.5 and "wide_spread:hold" in info["reasons"]


def test_volatility_cap_shrinks_size():
    rm = RiskManager(RiskLimits(max_vol_ann=1.0), equity=1_000_000)
    size, _ = rm.apply(1.0, 1_000_000, vol_ann=2.0, half_spread_bp=1.0, staleness_sec=5, margin_ratio=3.0)
    assert size == 0.5


def test_min_trade_delta_avoids_micro_orders():
    rm = RiskManager(RiskLimits(min_trade_delta=0.2), equity=1_000_000)
    size, info = rm.apply(0.55, 1_000_000, 0.5, 1.0, 5, 3.0, current=0.5)
    assert size == 0.5 and "below_min_delta" in info["reasons"]


def test_consecutive_losses_trigger_cooldown():
    rm = RiskManager(RiskLimits(consecutive_loss_stop=3, cooldown_bars=2), equity=1_000_000)
    for _ in range(3):
        rm.on_trade_result(-100)
    size, info = rm.apply(1.0, 1_000_000, 0.5, 1.0, 5, 3.0)
    assert size == 0.0 and "cooldown" in info["reasons"]
