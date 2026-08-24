"""評価指標の検証。"""

import numpy as np
import pandas as pd

from rlgmo.metrics import BARS_PER_YEAR, deflated_sharpe, equity_metrics, summarize


def make_equity(returns: np.ndarray) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=len(returns), freq="1min", tz="UTC")
    return pd.Series(1e6 * np.exp(np.cumsum(returns)), index=idx)


def test_sharpe_matches_analytic_value():
    """平均・標準偏差が既知の決定論的な系列で年率換算が正しいことを確認する。

    （ランダム系列を使うと、1 年ぶんのデータでも Sharpe の推定誤差は ±1 程度になる。
    これは「バックテストの Sharpe をそのまま信じてはいけない」ことの直接的な理由でもある。）
    """
    per_bar_vol = 1e-4
    target_sharpe = 2.0
    mu = target_sharpe / np.sqrt(BARS_PER_YEAR) * per_bar_vol
    ret = mu + per_bar_vol * np.tile([1.0, -1.0], BARS_PER_YEAR // 2)
    metrics = equity_metrics(make_equity(ret))
    assert abs(metrics["sharpe"] - target_sharpe) < 0.01
    assert abs(metrics["ann_vol"] - per_bar_vol * np.sqrt(BARS_PER_YEAR)) < 0.01


def test_drawdown_and_turnover():
    ret = np.concatenate([np.full(100, 1e-4), np.full(100, -2e-4), np.full(100, 1e-4)])
    equity = make_equity(ret)
    pos = pd.Series(np.tile([1.0, 1.0, 0.0, 0.0], 75), index=equity.index)
    metrics = equity_metrics(equity, pos)
    assert metrics["max_drawdown"] < -0.015
    assert metrics["n_trades"] == 149  # 2 バーごとに出入り
    assert 0.4 < metrics["exposure"] < 0.6


def test_deflated_sharpe_penalizes_many_trials():
    kwargs = dict(sharpe_ann=1.5, n_bars=BARS_PER_YEAR, trial_sharpe_std_ann=0.8)
    single = deflated_sharpe(n_trials=1, **kwargs)
    many = deflated_sharpe(n_trials=500, **kwargs)
    assert 0.0 <= many < single <= 1.0


def test_summarize_includes_cost_drag():
    ret = np.random.default_rng(1).standard_normal(5000) * 1e-4
    equity = make_equity(ret)
    pos = pd.Series(np.sign(np.sin(np.arange(5000) / 30)), index=equity.index)
    costs = pd.Series(np.full(5000, 10.0), index=equity.index)
    out = summarize(equity, pos, costs, n_trials=10, trial_sharpe_std_ann=0.5)
    assert out["cost_drag_ann"] > 0
    assert 0.0 <= out["deflated_sharpe_p"] <= 1.0
