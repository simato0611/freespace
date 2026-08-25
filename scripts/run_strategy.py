#!/usr/bin/env python3
"""確定版の戦略を、指定したデータ・期間で実行する（正式な実装）。

**戦略 v2（現行の最有力案）**

    銘柄        レバレッジ取引できる全銘柄（GMO なら BTC/ETH/XRP/LTC/BCH）
    シグナル     トレンド・ラダー（5/14/30/60 日のモメンタムを平均）、**両建て**
    サイジング   銘柄ごとにボラターゲット（等リスク） → ポートフォリオ全体のボラを目標へ
                 → 総建玉にレバレッジ上限
    判断        1 時間ごと。ただし目標が 0.10 以上動いたときだけ建て直す（更新バンド）
    執行        指値優先。建玉管理料の回避オーバーレイは既定で無効
                （往復コストが 4bp を下回るときだけ得になるため）

開発期間での実績（`docs/strategy_search.md` 参照。ホールドアウトは未使用）:

    時代A 2017-10〜2020-05 / Huobi 6 銘柄  … Sharpe 1.64、最大DD −10.8%
    時代B 2020-01〜2024-12 / Perp 7 銘柄   … Sharpe 1.82、最大DD −12.4%、年率 35.3%
    （いずれも目標ボラ 15%、片道 1.5bp の想定）

Example:
    python scripts/run_strategy.py --data-dir data/raw/perp --end 2024-12-31
    python scripts/run_strategy.py --data-dir data/raw/perp --assets BTC,ETH,XRP --target-vol 0.15
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
from rlgmo.portfolio import (  # noqa: E402
    LADDER_DAYS,
    PortfolioConfig,
    apply_rebalance_band,
    backtest_portfolio,
    carry_avoidance_mask,
    ladder_signal,
)

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def load_prices(data_dir: Path, grid_hours: int, assets: list[str] | None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.parquet")):
        asset = path.stem.split("_")[0].upper()
        if assets and asset not in assets:
            continue
        df = pd.read_parquet(path)
        missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
        if missing:
            continue
        resampled = df.resample(f"{grid_hours}h", label="right", closed="right").agg(
            {k: v for k, v in AGG.items() if k in df.columns}
        )
        out[asset] = resampled.dropna(subset=["close"])
    return out


def build_signals(prices: dict[str, pd.DataFrame], grid_hours: int, band: float,
                  long_only: bool, carry_avoidance: bool) -> dict[str, pd.Series]:
    bars_per_day = max(1, 24 // grid_hours)
    signals = {}
    for asset, df in prices.items():
        signal = ladder_signal(df, bars_per_day, long_only=long_only)
        if band > 0:
            signal = apply_rebalance_band(signal, band)
        if carry_avoidance:
            signal = signal * carry_avoidance_mask(pd.DatetimeIndex(df.index))
        signals[asset] = signal
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/raw/perp")
    parser.add_argument("--assets", default=None, help="カンマ区切り（既定は全銘柄）")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--eval-from", default=None,
                        help="計測開始日。これ以前はシグナルの助走にのみ使い、成績には含めない")
    parser.add_argument("--warmup-days", type=int, default=180)
    parser.add_argument("--benchmark", action="store_true", help="同期間の買い持ちも表示する")
    parser.add_argument("--grid", type=int, default=1, help="判断間隔（時間）")
    parser.add_argument("--band", type=float, default=0.10, help="建玉更新バンド")
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--cost-bp", type=float, default=1.5, help="片道コスト (bp)")
    parser.add_argument("--carry-avoidance", action="store_true",
                        help="06:00 JST の課金バーだけフラットにする（片道 1.5bp 以下でのみ有効）")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--n-trials", type=int, default=115)
    parser.add_argument("--out", default="runs/strategy")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")] if args.assets else None
    prices = load_prices(Path(args.data_dir), args.grid, assets)
    start = args.start
    if args.eval_from:  # 助走ぶんだけ手前から読み込む（成績には含めない）
        start = str((pd.Timestamp(args.eval_from, tz="UTC") - pd.Timedelta(days=args.warmup_days)).date())
    if start or args.end:
        prices = {a: df.loc[start:args.end] for a, df in prices.items()}
    prices = {a: df for a, df in prices.items() if len(df) > 24 * 90 // args.grid}
    if not prices:
        raise SystemExit("データがありません")

    grid_minutes = args.grid * 60
    cfg = PortfolioConfig(
        target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol,
        cost=CostConfig(half_spread_bp=args.cost_bp, slippage_bp=0.0,
                        carry_mode="daily_0600", spread_vol_beta=0.0),
    )
    signals = build_signals(prices, args.grid, args.band, args.long_only, args.carry_avoidance)
    result = backtest_portfolio(prices, signals, grid_minutes, cfg)
    if args.eval_from:  # 助走を切り落とし、エクイティを 1 から取り直す
        result = result.loc[pd.Timestamp(args.eval_from, tz="UTC"):]
        result["equity"] = 1e6 * (1 + result["ret"]).cumprod()
        result["gross_equity"] = 1e6 * (1 + result["gross_pnl"]).cumprod()
    metrics = equity_metrics(result["equity"])
    days = len(result) * grid_minutes / 1440

    span = f"{result.index[0]:%Y-%m-%d}〜{result.index[-1]:%Y-%m-%d}"
    print(f"[戦略 v2] 銘柄 {sorted(prices)} / {span} ({days:.0f} 日)")
    print(f"          ラダー {LADDER_DAYS} 日 / {'ロングのみ' if args.long_only else '両建て'} / "
          f"{args.grid} 時間判断 / 更新バンド {args.band} / 目標ボラ {args.target_vol:.0%} / "
          f"片道 {args.cost_bp}bp{' / 06時課金回避' if args.carry_avoidance else ''}")
    print(f"\n  Sharpe          {metrics['sharpe']:+.2f}")
    print(f"  年率リターン      {metrics['cagr']:+.1%}")
    print(f"  年率ボラ         {metrics['ann_vol']:.1%}")
    print(f"  最大ドローダウン   {metrics['max_drawdown']:+.1%}")
    print(f"  Calmar          {metrics['calmar']:.2f}")
    print(f"  回転率          {result['turnover'].sum() / days:.2f} /日")
    print(f"  コスト          {result['cost'].sum() / (days / 365):.1%} /年")
    print(f"  平均グロス建玉    {result['gross_exposure'].mean():.1%}")
    print(f"  平均ネット建玉    {result['net_exposure'].mean():+.1%}")
    print(f"  Deflated Sharpe {deflated_sharpe(metrics['sharpe'], len(result), args.n_trials, 0.8, bars_per_year=365 * 24 * 60 / grid_minutes):.2f}")

    periods = 365 * 24 * 60 / grid_minutes
    yearly = result.groupby(result.index.year).apply(
        lambda x: pd.Series({
            "Sharpe": x["ret"].mean() / x["ret"].std() * np.sqrt(periods) if x["ret"].std() > 0 else 0.0,
            "リターン": (1 + x["ret"]).prod() - 1,
            "ネット建玉": x["net_exposure"].mean()}))
    show = yearly.copy()
    show["Sharpe"] = show["Sharpe"].round(2)
    for col in ("リターン", "ネット建玉"):
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    print("\n年別:")
    print(show.to_string())

    if args.benchmark:
        bench_signals = {a: pd.Series(1.0, index=df.index) for a, df in prices.items()}
        bench = backtest_portfolio(prices, bench_signals, grid_minutes, cfg)
        if args.eval_from:
            bench = bench.loc[pd.Timestamp(args.eval_from, tz="UTC"):]
            bench["equity"] = 1e6 * (1 + bench["ret"]).cumprod()
        bm = equity_metrics(bench["equity"])
        print(f"\n  [参考] 同期間の買い持ち: Sharpe {bm['sharpe']:+.2f} / "
              f"リターン {bm['total_return']:+.1%} / 最大DD {bm['max_drawdown']:+.1%}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "equity.csv")
    print(f"\n出力: {out_dir}/equity.csv")


if __name__ == "__main__":
    main()
