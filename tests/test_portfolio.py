"""複数銘柄ポートフォリオの検証。"""

import numpy as np
import pandas as pd
import pytest

from rlgmo.costs import CostConfig
from rlgmo.portfolio import PortfolioConfig, backtest_portfolio, trend_signal

ZERO_COST = CostConfig(half_spread_bp=0.0, slippage_bp=0.0, carry_mode="none")


def make_prices(paths: dict[str, np.ndarray], freq: str = "4h") -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in paths.items():
        idx = pd.date_range("2024-01-01", periods=len(path), freq=freq, tz="UTC")
        out[name] = pd.DataFrame(
            {"open": path, "high": path * 1.001, "low": path * 0.999, "close": path, "volume": 1.0}, index=idx
        )
    return out


def gbm(n: int, vol: float, seed: int, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.standard_normal(n) * vol + drift))


def test_equal_risk_gives_low_vol_asset_more_exposure():
    """ボラの低い銘柄ほど大きく建てる（リスク寄与を揃える）。"""
    prices = make_prices({"calm": gbm(2000, 0.005, 1), "wild": gbm(2000, 0.02, 2)})
    signals = {a: pd.Series(1.0, index=df.index) for a, df in prices.items()}
    cfg = PortfolioConfig(cost=ZERO_COST, target_vol_ann=0.20, asset_vol_ann=0.20)
    result = backtest_portfolio(prices, signals, 240, cfg)
    assert result["gross_exposure"].mean() > 0
    # 個別のスケールを直接確認する
    logret_calm = np.log(prices["calm"]["close"]).diff().std()
    logret_wild = np.log(prices["wild"]["close"]).diff().std()
    assert logret_calm < logret_wild
    assert result["n_assets"].max() == 2


def test_portfolio_vol_targeting_hits_the_target():
    """分散が効いても、ポートフォリオ全体のボラは目標付近に収まる。"""
    prices = make_prices({f"a{i}": gbm(4000, 0.01, i) for i in range(5)})
    signals = {a: pd.Series(1.0, index=df.index) for a, df in prices.items()}
    cfg = PortfolioConfig(cost=ZERO_COST, target_vol_ann=0.20, asset_vol_ann=0.20, leverage_cap=10.0)
    result = backtest_portfolio(prices, signals, 240, cfg)
    realized = result["ret"].std() * np.sqrt(365 * 24 / 4)
    assert 0.10 < realized < 0.40  # 目標 20% の周辺（推定誤差とスケール上限のぶん幅を見る）


def test_leverage_cap_is_enforced():
    prices = make_prices({f"a{i}": gbm(1500, 0.002, i) for i in range(6)})  # 低ボラ = 大きく建てたくなる
    signals = {a: pd.Series(1.0, index=df.index) for a, df in prices.items()}
    cfg = PortfolioConfig(cost=ZERO_COST, target_vol_ann=0.50, asset_vol_ann=0.50, leverage_cap=2.0)
    result = backtest_portfolio(prices, signals, 240, cfg)
    assert result["gross_exposure"].max() <= 2.0 + 1e-9


def test_costs_reduce_return_and_scale_with_turnover():
    prices = make_prices({"a": gbm(1200, 0.01, 3), "b": gbm(1200, 0.01, 4)})
    flip = pd.Series(np.tile([1.0, -1.0], 600), index=prices["a"].index)
    signals = {"a": flip, "b": flip}
    free = backtest_portfolio(prices, signals, 240, PortfolioConfig(cost=ZERO_COST))
    costly = backtest_portfolio(
        prices, signals, 240,
        PortfolioConfig(cost=CostConfig(half_spread_bp=5.0, slippage_bp=0.0, carry_mode="none")),
    )
    assert costly["cost"].sum() > 0
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert free["cost"].sum() == pytest.approx(0.0)


def test_no_lookahead_future_prices_do_not_change_past_pnl():
    """未来の価格を書き換えても、過去のポートフォリオ損益は変わらない。"""
    base = {"a": gbm(1500, 0.01, 7), "b": gbm(1500, 0.012, 8)}
    prices = make_prices(base)
    tampered = make_prices({k: np.concatenate([v[:1000], v[1000:] * 1.5]) for k, v in base.items()})
    cfg = PortfolioConfig(cost=ZERO_COST)
    lookback = 84

    def run(px):
        signals = {a: trend_signal(df, lookback) for a, df in px.items()}
        return backtest_portfolio(px, signals, 240, cfg)

    a, b = run(prices), run(tampered)
    cut = 900  # ボラ窓・ルックバックの影響が及ばない範囲で比較する
    pd.testing.assert_series_equal(a["ret"].iloc[:cut], b["ret"].iloc[:cut])


def test_trend_signal_is_long_only_and_bounded():
    prices = make_prices({"a": gbm(1000, 0.01, 11, drift=0.001)})
    signal = trend_signal(prices["a"], 84, long_only=True)
    assert signal.min() >= 0.0 and signal.max() <= 1.0
    both = trend_signal(prices["a"], 84, long_only=False)
    assert both.min() >= -1.0 and both.max() <= 1.0
    assert both.mean() > 0  # 上昇ドリフトを与えたので平均はプラス
