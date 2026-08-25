#!/usr/bin/env python3
"""シグナルの多様化: トレンドと相関の低い「スリーブ」を探す。

現行の戦略 v2 は全銘柄が同じトレンド則で動いているため、トレンドが効かない年に
弱点が集中する。そこで**トレンドと相関の低い別の収益源**を足せないかを調べる。

採用条件（先に固定する）:
    1. 単体で、時代A・時代B の**両方**で Sharpe > 0
    2. トレンド・スリーブとの相関が低い（|ρ| < 0.3 を目安）
    3. 混ぜたときに、両方の時代でブレンドの Sharpe が上がる

3 つすべてを満たさないスリーブは採用しない。片方の時代でしか効かないものは
その時代への当てはめとみなす。ホールドアウト（時代C）はここでも触らない。

Example:
    python scripts/sleeve_search.py --era-b data/raw/perp --era-a data/raw/alt2017
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.portfolio import (  # noqa: E402
    PortfolioConfig,
    apply_rebalance_band,
    backtest_portfolio,
    ladder_signal,
)

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# ------------------------------------------------------------------ スリーブ定義
def sleeve_trend(prices: dict[str, pd.DataFrame], per_day: int) -> dict[str, pd.Series]:
    """S1 トレンド・ラダー（現行の戦略 v2）。"""
    return {a: ladder_signal(df, per_day, long_only=False) for a, df in prices.items()}


def sleeve_short_reversal(prices: dict[str, pd.DataFrame], per_day: int, days: int = 2) -> dict[str, pd.Series]:
    """S2 短期リバーサル: 直近の急変を逆張りする。

    トレンド（数週間）とは逆の時間軸で働くため、原理的に相関が低い。
    """
    out = {}
    for asset, df in prices.items():
        logret = np.log(df["close"]).diff()
        window = int(days * per_day)
        move = (np.log(df["close"]) - np.log(df["close"].shift(window)))
        vol = logret.rolling(20 * per_day, min_periods=5 * per_day).std() * np.sqrt(window)
        out[asset] = (-(move / vol.replace(0.0, np.nan)) / 2.0).clip(-1, 1).fillna(0.0)
    return out


def _cross_sectional(values: pd.DataFrame, sign: float = 1.0) -> pd.DataFrame:
    """各時点で銘柄横断に順位づけし、平均ゼロ（市場中立）のウェイトにする。"""
    ranked = values.rank(axis=1, pct=True)
    centered = (ranked - 0.5) * 2.0
    return centered.sub(centered.mean(axis=1), axis=0) * sign


def sleeve_cs_momentum(prices: dict[str, pd.DataFrame], per_day: int, days: int = 14) -> dict[str, pd.Series]:
    """S3 クロスセクション・モメンタム: 相対的に強い銘柄を買い、弱い銘柄を売る。

    市場全体の方向には賭けない（合計ゼロ）ので、時系列トレンドとは別の収益源になりうる。
    """
    panel = pd.DataFrame({a: np.log(df["close"]).diff(int(days * per_day)) for a, df in prices.items()})
    weights = _cross_sectional(panel, +1.0)
    return {a: weights[a].reindex(prices[a].index).fillna(0.0) for a in prices}


def sleeve_cs_reversal(prices: dict[str, pd.DataFrame], per_day: int, days: int = 3) -> dict[str, pd.Series]:
    """S4 クロスセクション・リバーサル: 相対的に売られすぎた銘柄を買う。"""
    panel = pd.DataFrame({a: np.log(df["close"]).diff(int(days * per_day)) for a, df in prices.items()})
    weights = _cross_sectional(panel, -1.0)
    return {a: weights[a].reindex(prices[a].index).fillna(0.0) for a in prices}


def sleeve_breakout(prices: dict[str, pd.DataFrame], per_day: int, days: int = 20) -> dict[str, pd.Series]:
    """S5 ドンチャン・ブレイクアウト: 直近レンジの上抜け/下抜けに乗る。

    連続値のモメンタムとは建玉の入り方が違う（保ち合いでは何もしない）。
    """
    out = {}
    window = int(days * per_day)
    for asset, df in prices.items():
        hh = df["high"].rolling(window, min_periods=window // 2).max().shift(1)
        ll = df["low"].rolling(window, min_periods=window // 2).min().shift(1)
        raw = pd.Series(np.nan, index=df.index)
        raw[df["close"] > hh] = 1.0
        raw[df["close"] < ll] = -1.0
        out[asset] = raw.ffill().fillna(0.0)
    return out


def sleeve_weekend(prices: dict[str, pd.DataFrame], per_day: int) -> dict[str, pd.Series]:
    """S6 週末効果: 流動性の薄い週末を避け、平日だけ買い持ちする。

    暗号資産では週末に機関のフローが細ることが知られている。純粋な季節性なので
    トレンドとは無関係に効く（か、効かない）。
    """
    out = {}
    for asset, df in prices.items():
        weekday = pd.DatetimeIndex(df.index).tz_convert("UTC").dayofweek
        out[asset] = pd.Series(np.where(weekday < 5, 1.0, 0.0), index=df.index)
    return out


SLEEVES = {
    "S1 トレンド・ラダー": sleeve_trend,
    "S2 短期リバーサル": sleeve_short_reversal,
    "S3 CS モメンタム": sleeve_cs_momentum,
    "S4 CS リバーサル": sleeve_cs_reversal,
    "S5 ブレイクアウト": sleeve_breakout,
    "S6 週末回避": sleeve_weekend,
}


# ------------------------------------------------------------------ 実行
def load_prices(data_dir: Path, grid_hours: int, end: str | None) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        if "close" not in df.columns:
            continue
        resampled = df.resample(f"{grid_hours}h", label="right", closed="right").agg(
            {k: v for k, v in AGG.items() if k in df.columns}).dropna(subset=["close"])
        if end:
            resampled = resampled.loc[:end]
        if len(resampled) > 24 * 120 // grid_hours:
            out[path.stem.split("_")[0].upper()] = resampled
    return out


def run_sleeve(prices, signals, grid_minutes, cfg, band):
    if band > 0:
        signals = {a: apply_rebalance_band(s, band) for a, s in signals.items()}
    return backtest_portfolio(prices, signals, grid_minutes, cfg)


def blend(signal_sets: list[dict[str, pd.Series]], weights: list[float]) -> dict[str, pd.Series]:
    """スリーブのシグナルを重み付きで合成する（建玉レベルで相殺されるのでコストも正しく減る）。"""
    assets = signal_sets[0].keys()
    out = {}
    for asset in assets:
        total = None
        for signals, weight in zip(signal_sets, weights):
            part = signals[asset] * weight
            total = part if total is None else total.add(part, fill_value=0.0)
        out[asset] = total.clip(-1, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era-a", default="data/raw/alt2017")
    parser.add_argument("--era-b", default="data/raw/perp")
    parser.add_argument("--era-b-end", default="2024-12-31")
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--cost-bp", type=float, default=1.5)
    parser.add_argument("--out", default="runs/analysis")
    args = parser.parse_args()

    grid_minutes = args.grid * 60
    per_day = max(1, 24 // args.grid)
    cfg = PortfolioConfig(target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol,
                          cost=CostConfig(half_spread_bp=args.cost_bp, slippage_bp=0.0,
                                          carry_mode="daily_0600", spread_vol_beta=0.0))
    eras = {
        "時代A": load_prices(Path(args.era_a), args.grid, None),
        "時代B": load_prices(Path(args.era_b), args.grid, args.era_b_end),
    }

    all_returns, summary = {}, {}
    for era, prices in eras.items():
        print(f"\n===== {era}: {sorted(prices)} =====")
        returns = {}
        rows = {}
        for name, builder in SLEEVES.items():
            result = run_sleeve(prices, builder(prices, per_day), grid_minutes, cfg, args.band)
            metrics = equity_metrics(result["equity"])
            days = len(result) * grid_minutes / 1440
            rows[name] = {"Sharpe": metrics["sharpe"], "年率": metrics["cagr"],
                          "最大DD": metrics["max_drawdown"],
                          "回転/日": float(result["turnover"].sum() / max(days, 1e-9))}
            returns[name] = result["ret"]
        table = pd.DataFrame(rows).T
        show = table.copy()
        for col in ("年率", "最大DD"):
            show[col] = pd.to_numeric(show[col]).mul(100).round(1).astype(str) + "%"
        for col in ("Sharpe", "回転/日"):
            show[col] = pd.to_numeric(show[col]).round(2)
        print(show.to_string())

        daily = pd.DataFrame(returns).resample("1D").sum()
        corr = daily.corr()
        print("\nトレンドとの相関（日次）:")
        print(corr["S1 トレンド・ラダー"].round(2).to_string())
        all_returns[era] = daily
        summary[era] = table

    # --- 混ぜたときに良くなるか
    print("\n\n===== ブレンド（トレンド 1 : 候補 1 のリスク配分）=====")
    print(f"{'構成':34s} {'時代A':>7} {'時代B':>7} {'最小':>7} {'A最大DD':>8} {'B最大DD':>8}")
    base = {}
    for era, prices in eras.items():
        result = run_sleeve(prices, sleeve_trend(prices, per_day), grid_minutes, cfg, args.band)
        base[era] = equity_metrics(result["equity"])
    print(f"{'S1 単体（現行 v2）':34s} {base['時代A']['sharpe']:>7.2f} {base['時代B']['sharpe']:>7.2f} "
          f"{min(base['時代A']['sharpe'], base['時代B']['sharpe']):>7.2f} "
          f"{base['時代A']['max_drawdown']:>8.1%} {base['時代B']['max_drawdown']:>8.1%}")

    results = {}
    for name, builder in SLEEVES.items():
        if name.startswith("S1"):
            continue
        row = {}
        for era, prices in eras.items():
            signals = blend([sleeve_trend(prices, per_day), builder(prices, per_day)], [0.5, 0.5])
            result = run_sleeve(prices, signals, grid_minutes, cfg, args.band)
            row[era] = equity_metrics(result["equity"])
        results[name] = row
        print(f"{'S1 + ' + name:34s} {row['時代A']['sharpe']:>7.2f} {row['時代B']['sharpe']:>7.2f} "
              f"{min(row['時代A']['sharpe'], row['時代B']['sharpe']):>7.2f} "
              f"{row['時代A']['max_drawdown']:>8.1%} {row['時代B']['max_drawdown']:>8.1%}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for era, table in summary.items():
        table.to_csv(Path(args.out) / f"sleeves_{era}.csv")
        all_returns[era].corr().to_csv(Path(args.out) / f"sleeve_corr_{era}.csv")
    print(f"\n出力: {args.out}/sleeves_*.csv")


if __name__ == "__main__":
    main()
