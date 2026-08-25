#!/usr/bin/env python3
"""複数銘柄の等リスク・ポートフォリオをバックテストする。

単一銘柄では当たり外れの振れが大きいトレンドフォローを、7 銘柄に等リスク配分して
分散させる。ルール（14 日モメンタム・ロングオンリー・4 時間判断）は BTC の探索で
確定したものを**変更せずに**使う。

    --period dev      開発期間のみ（探索・確認用）
    --period holdout  封印していたホールドアウト（一度だけ使う）
    --period all      全期間

Example:
    python scripts/portfolio_backtest.py --dir data/raw/perp --period dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.metrics import deflated_sharpe, equity_metrics  # noqa: E402
from rlgmo.portfolio import PortfolioConfig, backtest_portfolio, trend_signal  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def load(dir_path: Path, grid_hours: int) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(dir_path.glob("*_1h.parquet")):
        df = pd.read_parquet(path)
        resampled = df.resample(f"{grid_hours}h", label="right", closed="right").agg(AGG).dropna(subset=["close"])
        out[path.stem.split("_")[0]] = resampled
    return out


def slice_period(prices: dict[str, pd.DataFrame], start, end) -> dict[str, pd.DataFrame]:
    return {a: df.loc[start:end] for a, df in prices.items() if len(df.loc[start:end]) > 100}


def report(name: str, result: pd.DataFrame, grid_minutes: int, n_trials: int) -> dict:
    metrics = equity_metrics(result["equity"])
    gross = equity_metrics(result["gross_equity"])
    days = len(result) * grid_minutes / 1440
    row = {
        "Sharpe": metrics["sharpe"], "グロスSharpe": gross["sharpe"],
        "年率": metrics["cagr"], "リターン": metrics["total_return"],
        "最大DD": metrics["max_drawdown"], "年率ボラ": metrics["ann_vol"],
        "回転/日": float(result["turnover"].sum() / max(days, 1e-9)),
        "平均グロス建玉": float(result["gross_exposure"].mean()),
        "コスト/年": float(result["cost"].sum() / max(days / 365, 1e-9)),
        "日数": round(days),
        "DSR_p": deflated_sharpe(metrics["sharpe"], len(result), n_trials, 0.8,
                                 metrics.get("skew", 0.0), metrics.get("excess_kurtosis", 0.0) + 3,
                                 bars_per_year=365 * 24 * 60 / grid_minutes),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="data/raw/perp")
    parser.add_argument("--grid", type=int, default=4, help="判断間隔（時間）")
    parser.add_argument("--lookback-days", type=float, default=14.0)
    parser.add_argument("--period", default="dev", choices=["dev", "holdout", "all"])
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--n-trials", type=int, default=81)
    parser.add_argument("--out", default="runs/portfolio")
    args = parser.parse_args()

    grid_minutes = args.grid * 60
    per_day = max(1, 24 // args.grid)
    lookback = int(args.lookback_days * per_day)
    holdout = pd.Timestamp(args.holdout_start, tz="UTC")
    warmup = pd.Timedelta(hours=args.grid * (lookback + 60))

    prices_all = load(Path(args.dir), args.grid)
    if args.period == "dev":
        window = (None, holdout)
    elif args.period == "holdout":
        window = (holdout - warmup, None)
    else:
        window = (None, None)
    prices = slice_period(prices_all, *window)
    print(f"[portfolio] {args.period} / 銘柄 {list(prices)} / 判断間隔 {args.grid} 時間 / "
          f"ルックバック {args.lookback_days} 日")

    cfg = PortfolioConfig(target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol,
                          cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5,
                                          carry_mode="daily_0600", spread_vol_beta=0.0))
    variants = {
        "trend_portfolio": {a: trend_signal(df, lookback, long_only=True) for a, df in prices.items()},
        "buyhold_portfolio": {a: pd.Series(1.0, index=df.index) for a, df in prices.items()},
        "trend_BTC_only": None,
    }
    rows, curves = {}, {}
    for name, signals in variants.items():
        if name == "trend_BTC_only":
            if "BTC" not in prices:
                continue
            sub = {"BTC": prices["BTC"]}
            signals = {"BTC": trend_signal(prices["BTC"], lookback, long_only=True)}
        else:
            sub = prices
        result = backtest_portfolio(sub, signals, grid_minutes, cfg)
        if args.period == "holdout":
            result = result.loc[holdout:]
            result["equity"] = 1e6 * (1 + result["ret"]).cumprod()
            result["gross_equity"] = 1e6 * (1 + result["gross_pnl"]).cumprod()
        rows[name] = report(name, result, grid_minutes, args.n_trials)
        curves[name] = result

    table = pd.DataFrame(rows).T
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"portfolio_{args.period}.csv")
    for name, curve in curves.items():
        curve.to_csv(out_dir / f"curve_{args.period}_{name}.csv")

    show = table.copy()
    for col in ("年率", "リターン", "最大DD", "年率ボラ", "コスト/年", "平均グロス建玉"):
        show[col] = pd.to_numeric(show[col]).mul(100).round(1).astype(str) + "%"
    for col in ("Sharpe", "グロスSharpe", "回転/日", "DSR_p", "日数"):
        show[col] = pd.to_numeric(show[col]).round(2)
    print("\n" + show.to_string())

    # 年別
    curve = curves["trend_portfolio"]
    yearly = curve.groupby(curve.index.year).apply(
        lambda x: pd.Series({
            "Sharpe": x["ret"].mean() / x["ret"].std() * np.sqrt(365 * 24 * 60 / grid_minutes) if x["ret"].std() > 0 else 0.0,
            "リターン": (1 + x["ret"]).prod() - 1,
            "平均グロス建玉": x["gross_exposure"].mean()}))
    yearly["Sharpe"] = yearly["Sharpe"].round(2)
    for col in ("リターン", "平均グロス建玉"):
        yearly[col] = (yearly[col] * 100).round(1).astype(str) + "%"
    print("\n=== trend_portfolio 年別 ===")
    print(yearly.to_string())
    print(f"\n出力: {out_dir}/portfolio_{args.period}.csv")


if __name__ == "__main__":
    main()
