#!/usr/bin/env python3
"""同一ルールを複数銘柄に適用して、優位性が銘柄固有でないかを確認する。

**チューニングではない**: パラメータ（ルックバック・グリッド・コスト・ボラターゲット）は
BTC の探索で決めた値をそのまま固定して使う。ここで見たいのは
「その効果が BTC 固有の遺物なのか、暗号資産に共通する性質なのか」だけである。

データ形式は Huobi 公開データセット（id, open, high, low, close, vol, count, amount。
id は epoch 秒のバー開始時刻）を想定。

Example:
    python scripts/cross_asset_check.py --dir /path/to/alt --lookback-days 14 --grid 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.data.resample import resample_ohlcv  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from signal_survey import alpha_vs_benchmark, buy_hold, momentum, simulate  # noqa: E402


def load_huobi(asset_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(asset_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "id" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id").sort_values("id")
    close_time = pd.to_datetime(df["id"].astype("int64") + 60, unit="s", utc=True)
    volume = df["amount"] if "amount" in df.columns else df["vol"]
    return pd.DataFrame(
        {"open": df["open"].astype(float).to_numpy(), "high": df["high"].astype(float).to_numpy(),
         "low": df["low"].astype(float).to_numpy(), "close": df["close"].astype(float).to_numpy(),
         "volume": volume.astype(float).to_numpy()},
        index=pd.DatetimeIndex(close_time, name="close_time"),
    ).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="銘柄ごとのサブディレクトリを含むパス")
    parser.add_argument("--grid", type=int, default=240)
    parser.add_argument("--lookback-days", type=float, default=14.0)
    parser.add_argument("--half-spread-bp", type=float, default=2.0)
    parser.add_argument("--slippage-bp", type=float, default=0.5)
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--out", default="runs/analysis/cross_asset.csv")
    args = parser.parse_args()

    cost = CostConfig(half_spread_bp=args.half_spread_bp, slippage_bp=args.slippage_bp,
                      carry_mode="daily_0600", spread_vol_beta=0.0)
    per_day = max(1, 1440 // args.grid)
    lookback = int(args.lookback_days * per_day)
    rows = {}
    for asset_dir in sorted(Path(args.dir).iterdir()):
        if not asset_dir.is_dir():
            continue
        raw = load_huobi(asset_dir)
        if len(raw) < 60 * 24 * 90:
            print(f"[skip] {asset_dir.name}: データ不足 ({len(raw):,} バー)")
            continue
        df = resample_ohlcv(raw, args.grid)
        for label, long_only in (("trend_long", True), ("buy_hold", None)):
            if long_only is None:
                signal = buy_hold(df)
            else:
                signal = momentum(df, lookback).clip(lower=0)
            result = simulate(df, signal, args.grid, cost, args.target_vol)
            metrics = equity_metrics(1e6 * (1 + result["net"]).cumprod(), result["exposure"])
            key = (asset_dir.name.upper(), label)
            rows[key] = {
                "期間": f"{df.index[0]:%Y-%m}〜{df.index[-1]:%Y-%m}",
                "Sharpe": metrics["sharpe"], "リターン": metrics["total_return"],
                "最大DD": metrics["max_drawdown"], "年率ボラ": metrics["ann_vol"],
                "稼働率": float(result["exposure"].abs().mean()),
            }
        bench = simulate(df, buy_hold(df), args.grid, cost, args.target_vol)["net"]
        sig = momentum(df, lookback).clip(lower=0)
        rows[(asset_dir.name.upper(), "trend_long")].update(
            alpha_vs_benchmark(simulate(df, sig, args.grid, cost, args.target_vol)["net"], bench, args.grid))

    table = pd.DataFrame(rows).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["銘柄", "戦略"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)

    show = table.copy()
    for col in ("リターン", "最大DD", "年率ボラ", "稼働率", "α/年"):
        if col in show:
            show[col] = pd.to_numeric(show[col], errors="coerce").mul(100).round(1).astype(str) + "%"
    for col in ("Sharpe", "β", "情報比"):
        if col in show:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
    print("\n" + show.to_string())

    trend = table.xs("trend_long", level="戦略")["Sharpe"].astype(float)
    hold = table.xs("buy_hold", level="戦略")["Sharpe"].astype(float)
    print(f"\ntrend_long の Sharpe: 平均 {trend.mean():+.2f} / 中央値 {trend.median():+.2f} / "
          f"プラスの銘柄 {int((trend > 0).sum())}/{len(trend)}")
    print(f"buy_hold  の Sharpe: 平均 {hold.mean():+.2f} / プラスの銘柄 {int((hold > 0).sum())}/{len(hold)}")
    print(f"trend が buy&hold に勝った銘柄: {int((trend > hold).sum())}/{len(trend)}")
    print(f"出力: {args.out}")
    _ = np


if __name__ == "__main__":
    main()
