"""バックテスト評価指標。

1 分足のバーリターンから年率指標を作る。暗号資産は 24/365 で取引されるため
年間バー数 = 365 × 24 × 60 = 525,600 を用いる。

**Sharpe は「多数試した中の最良値」では過大評価される**ため、試行回数を考慮した
Deflated Sharpe Ratio (Bailey & Lopez de Prado) も併せて算出する。scipy への依存を
避けるため、正規分布の CDF / 逆関数は自前実装している。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BARS_PER_YEAR = 365 * 24 * 60


def equity_metrics(
    equity: pd.Series,
    positions: pd.Series | None = None,
    bars_per_year: int = BARS_PER_YEAR,
    trade_threshold: float = 0.05,
) -> dict:
    """エクイティカーブから主要指標を計算する。

    Args:
        equity: バーごとの有効証拠金（index は時刻）。
        positions: バーごとのポジション比率（回転率・稼働率の算出に使う）。
        bars_per_year: 年率換算に使う 1 年あたりのバー数。
        trade_threshold: 「1 回の売買」とみなすポジション変化の下限。ボラターゲットや
            建玉の値洗いで毎バー微小に変動するため、閾値なしでは取引回数が実態の
            数百倍になる。

    Returns:
        指標の dict。
    """
    equity = equity.astype(float)
    ret = np.log(equity).diff().dropna()
    n = len(ret)
    if n < 2:
        return {"n_bars": int(n), "sharpe": 0.0, "total_return": 0.0}

    # 完全にフラットな方策（std=0）でも全キーを返す。呼び出し側で KeyError を起こさない。
    mean, std = ret.mean(), ret.std(ddof=1)
    downside = ret[ret < 0].std(ddof=1)
    scale = np.sqrt(bars_per_year)
    sharpe = float(mean / std * scale) if std > 0 else 0.0
    sortino = float(mean / downside * scale) if downside and downside > 0 else 0.0
    peak = equity.cummax()
    max_dd = float((equity / peak - 1.0).min())
    years = n / bars_per_year
    cagr = float(equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    daily = ret.resample("1D").sum()

    out = {
        "n_bars": int(n),
        "days": round(n / 1440, 1),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": float(cagr),
        "ann_vol": float(std * scale),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "skew": float(ret.skew()) if std > 0 else 0.0,
        "excess_kurtosis": float(ret.kurtosis()) if std > 0 else 0.0,
        "hit_rate_daily": float((daily > 0).mean()),
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
    }
    if positions is not None:
        pos = positions.astype(float).reindex(equity.index).fillna(0.0)
        turn = pos.diff().abs().fillna(0.0)
        n_trades = int((turn > trade_threshold).sum())
        out.update(
            {
                "exposure": float(pos.abs().mean()),
                "turnover_per_day": float(turn.sum() / max(n / 1440, 1e-9)),
                "n_trades": n_trades,
                "avg_hold_bars": float(pos.abs().sum() / max(n_trades, 1)),
                "long_share": float((pos > 0).mean()),
                "short_share": float((pos < 0).mean()),
                "flat_share": float((pos == 0).mean()),
            }
        )
    return out


def deflated_sharpe(
    sharpe_ann: float,
    n_bars: int,
    n_trials: int,
    trial_sharpe_std_ann: float = 1.0,
    skew: float = 0.0,
    kurt: float = 3.0,
    bars_per_year: int = BARS_PER_YEAR,
) -> float:
    """Deflated Sharpe Ratio: 「真の Sharpe > 0」である確率（0〜1）。

    「N 通り試して最良を選んだ」という選択バイアスと、リターンの歪度・尖度を補正する。
    0.95 以上が採用の目安。`n_trials` は正直に数えること（ハイパラ探索・シード・
    特徴量セットの試行をすべて含む）。

    Args:
        sharpe_ann: 年率 Sharpe。
        n_bars: 観測バー数。
        n_trials: 試行総数。
        trial_sharpe_std_ann: 試行間の年率 Sharpe の標準偏差（ウォークフォワードの
            全試行から実測するのが望ましい）。
        skew: リターンの歪度。
        kurt: リターンの尖度（正規分布で 3）。
        bars_per_year: 年率換算のバー数。
    """
    if n_bars < 10 or n_trials < 1:
        return float("nan")
    gamma = 0.5772156649015329  # Euler-Mascheroni 定数
    sr = sharpe_ann / math.sqrt(bars_per_year)
    sigma = max(trial_sharpe_std_ann, 1e-9) / math.sqrt(bars_per_year)
    e_max = sigma * ((1 - gamma) * _norm_ppf(1 - 1 / n_trials) + gamma * _norm_ppf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return _norm_cdf((sr - e_max) * math.sqrt(n_bars - 1) / denom)


def summarize(
    equity: pd.Series,
    positions: pd.Series,
    costs: pd.Series | None = None,
    n_trials: int = 1,
    trial_sharpe_std_ann: float = 1.0,
) -> dict:
    """評価レポート用のまとめ（コスト内訳・DSR を含む）。"""
    out = equity_metrics(equity, positions)
    if costs is not None and len(costs):
        years = max(len(equity) / BARS_PER_YEAR, 1e-9)
        out["cost_drag_ann"] = float(costs.sum() / equity.iloc[0] / years)
    out["n_trials"] = n_trials
    out["deflated_sharpe_p"] = deflated_sharpe(
        out.get("sharpe", 0.0),
        out.get("n_bars", 0),
        n_trials,
        trial_sharpe_std_ann=trial_sharpe_std_ann,
        skew=out.get("skew", 0.0),
        kurt=out.get("excess_kurtosis", 0.0) + 3.0,
    )
    return out


def _norm_cdf(x: float) -> float:
    """標準正規分布の累積分布関数。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """標準正規分布の逆関数（Acklam の有理近似）。"""
    if not 0.0 < p < 1.0:
        return 0.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
