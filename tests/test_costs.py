"""コストモデルの検証。"""

import pandas as pd

from rlgmo.costs import CostConfig, carry_flags, carry_rate_per_bar, effective_trade_rate


def test_carry_flag_fires_once_per_day_at_0600_jst():
    idx = pd.date_range("2026-01-01", periods=60 * 24 * 5, freq="1min", tz="UTC")
    flags = carry_flags(idx, hour_jst=6)
    assert flags.sum() == 5  # 5 日ぶん = 5 回
    fired = pd.DatetimeIndex(idx[flags]).tz_convert("Asia/Tokyo")
    assert set(fired.hour) == {6} and set(fired.minute) == {0}


def test_carry_modes():
    cfg = CostConfig(carry_rate_daily=0.0004)
    assert carry_rate_per_bar(cfg, 1, True) == 0.0004
    assert carry_rate_per_bar(cfg, 1, False) == 0.0
    prorata = CostConfig(carry_mode="prorata", carry_rate_daily=0.0004)
    assert abs(carry_rate_per_bar(prorata, 1, False) * 1440 - 0.0004) < 1e-12
    assert carry_rate_per_bar(CostConfig(carry_mode="none"), 1, True) == 0.0


def test_spread_widens_with_volatility():
    cfg = CostConfig(half_spread_bp=2.0, slippage_bp=0.5, spread_vol_beta=1.0)
    calm = effective_trade_rate(cfg, 1.0)
    stormy = effective_trade_rate(cfg, 3.0)
    assert calm == (2.0 + 0.5) * 1e-4
    assert stormy > calm and abs(stormy - (2.0 * 3 + 0.5) * 1e-4) < 1e-12
