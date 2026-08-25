#!/usr/bin/env python3
"""ポートフォリオ戦略の改良候補を、複数の「時代」で横並びに比較する。

**検証構造**（ホールドアウトを再消費しないための工夫）:

    時代A  2017-10〜2020-05  Huobi 6 銘柄（ETH/XRP/LTC/ETC/EOS/LINK）… 独立検証
    時代B  2020-01〜2024-12  Perp 7 銘柄 …………………………………………… 開発（改良はここで判断）
    時代C  2025-01〜2026-03  Perp 7 銘柄 …………………………………………… 封印（最後に一度だけ）

改良案は「時代B の年別で安定して効き、かつ時代A でも効く」ことを条件に採否を決める。
時代C は触らない。片方の時代でしか効かない改良は、その時代への当てはめとみなす。

改良候補（事前登録）:
    V0 base            14 日モメンタム単独（現行）
    V1 multi_lookback  5/14/30/60 日の平均（ルックバック選択のリスクを消す）
    V2 +dd_control     ポートフォリオのドローダウンに応じて建玉を縮める
    V3 +vol_regime     自身のボラが異常に高い銘柄を外す
    V4 +slow_filter    長期（100 日）トレンドが下向きの銘柄は建てない
    V5 long_short      ショートも建てる（BTC 単体では負けたが、分散すると別かもしれない）
    V6 maker_cost      指値執行を想定（片道 0.5bp）。コスト削減の効果量を測る
    V7 combo           時代Bで有効だった要素の組み合わせ

Example:
    python scripts/improve_portfolio.py --perp-dir data/raw/perp --huobi-dir /path/to/alt
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
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.portfolio import PortfolioConfig, backtest_portfolio, trend_signal  # noqa: E402
from cross_asset_check import load_huobi  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
LOOKBACK_DAYS = (5, 14, 30, 60)


# --------------------------------------------------------------------------- シグナル
def multi_lookback(df: pd.DataFrame, per_day: int, long_only: bool = True) -> pd.Series:
    """複数ルックバックの平均。単一の値を選ぶリスク（＝当てはめ）を消す。"""
    parts = [trend_signal(df, int(d * per_day), long_only=False) for d in LOOKBACK_DAYS]
    combined = pd.concat(parts, axis=1).mean(axis=1)
    return combined.clip(lower=0) if long_only else combined


def vol_regime_mask(df: pd.DataFrame, per_day: int, quantile: float = 0.9) -> pd.Series:
    """自身のボラが過去 1 年の上位 `quantile` を超える局面を外す。"""
    logret = np.log(df["close"]).diff()
    vol = logret.rolling(20 * per_day, min_periods=5 * per_day).std()
    threshold = vol.rolling(365 * per_day, min_periods=60 * per_day).quantile(quantile)
    return (vol <= threshold).astype(float).fillna(1.0)


def slow_trend_mask(df: pd.DataFrame, per_day: int, days: int = 100) -> pd.Series:
    """長期トレンドが上向きのときだけ建てる。"""
    log_close = np.log(df["close"])
    return (log_close > log_close.shift(int(days * per_day))).astype(float).fillna(0.0)


def apply_drawdown_control(result: pd.DataFrame, signals: dict, prices: dict, grid_minutes: int,
                           cfg: PortfolioConfig, threshold: float = 0.10, floor: float = 0.3) -> pd.DataFrame:
    """1 回目の結果からドローダウンを測り、それに応じて建玉を縮めて再計算する。

    ドローダウンは過去の実現値のみから計算し、**1 バー遅らせて**適用するので未来は使わない。
    """
    equity = result["equity"]
    drawdown = 1 - equity / equity.cummax()
    scale = (1 - (drawdown - threshold).clip(lower=0) / max(threshold, 1e-9)).clip(floor, 1.0).shift(1).fillna(1.0)
    scaled = {a: (s.reindex(scale.index).fillna(0.0) * scale).reindex(prices[a].index).fillna(0.0)
              for a, s in signals.items()}
    return backtest_portfolio(prices, scaled, grid_minutes, cfg)


# --------------------------------------------------------------------------- 評価
def evaluate(result: pd.DataFrame, grid_minutes: int) -> dict:
    metrics = equity_metrics(result["equity"])
    days = len(result) * grid_minutes / 1440
    return {
        "Sharpe": metrics["sharpe"], "年率": metrics["cagr"], "最大DD": metrics["max_drawdown"],
        "年率ボラ": metrics["ann_vol"], "Calmar": metrics["calmar"],
        "回転/日": float(result["turnover"].sum() / max(days, 1e-9)),
        "コスト/年": float(result["cost"].sum() / max(days / 365, 1e-9)),
    }


def yearly_sharpe(result: pd.DataFrame, grid_minutes: int) -> pd.Series:
    periods = 365 * 24 * 60 / grid_minutes
    return result.groupby(result.index.year)["ret"].apply(
        lambda x: x.mean() / x.std() * np.sqrt(periods) if x.std() > 0 else 0.0)


def build_variants(prices: dict[str, pd.DataFrame], per_day: int, base_lookback: int):
    """事前登録した改良候補のシグナル群を作る。"""
    base = {a: trend_signal(df, base_lookback, long_only=True) for a, df in prices.items()}
    multi = {a: multi_lookback(df, per_day, long_only=True) for a, df in prices.items()}
    multi_ls = {a: multi_lookback(df, per_day, long_only=False) for a, df in prices.items()}
    vol_mask = {a: vol_regime_mask(df, per_day) for a, df in prices.items()}
    slow_mask = {a: slow_trend_mask(df, per_day) for a, df in prices.items()}
    return {
        "V0 base(14d)": base,
        "V1 multi_lookback": multi,
        "V3 +vol_regime": {a: multi[a] * vol_mask[a] for a in prices},
        "V4 +slow_filter": {a: multi[a] * slow_mask[a] for a in prices},
        "V5 long_short": multi_ls,
        "V7 combo(multi+slow)": {a: multi[a] * slow_mask[a] for a in prices},
    }


def run_era(name: str, prices: dict[str, pd.DataFrame], grid_minutes: int, per_day: int,
            base_lookback: int, cfg: PortfolioConfig, maker_cfg: PortfolioConfig) -> pd.DataFrame:
    variants = build_variants(prices, per_day, base_lookback)
    rows, yearly = {}, {}
    for label, signals in variants.items():
        result = backtest_portfolio(prices, signals, grid_minutes, cfg)
        rows[label] = evaluate(result, grid_minutes)
        yearly[label] = yearly_sharpe(result, grid_minutes)
        if label == "V1 multi_lookback":
            dd_result = apply_drawdown_control(result, signals, prices, grid_minutes, cfg)
            rows["V2 +dd_control"] = evaluate(dd_result, grid_minutes)
            yearly["V2 +dd_control"] = yearly_sharpe(dd_result, grid_minutes)
            maker = backtest_portfolio(prices, signals, grid_minutes, maker_cfg)
            rows["V6 maker_cost(0.5bp)"] = evaluate(maker, grid_minutes)
            yearly["V6 maker_cost(0.5bp)"] = yearly_sharpe(maker, grid_minutes)
    table = pd.DataFrame(rows).T
    show = table.copy()
    for col in ("年率", "最大DD", "年率ボラ", "コスト/年"):
        show[col] = pd.to_numeric(show[col]).mul(100).round(1).astype(str) + "%"
    for col in ("Sharpe", "Calmar", "回転/日"):
        show[col] = pd.to_numeric(show[col]).round(2)
    print(f"\n===== {name} =====")
    print(show.to_string())
    print("\n年別 Sharpe:")
    print(pd.DataFrame(yearly).round(2).to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perp-dir", default="data/raw/perp")
    parser.add_argument("--huobi-dir", default=None, help="時代A（Huobi 6 銘柄）の展開先")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--out", default="runs/analysis")
    args = parser.parse_args()

    grid_minutes = args.grid * 60
    per_day = max(1, 24 // args.grid)
    base_lookback = int(14 * per_day)
    cost = CostConfig(half_spread_bp=2.0, slippage_bp=0.5, carry_mode="daily_0600", spread_vol_beta=0.0)
    maker = CostConfig(half_spread_bp=0.5, slippage_bp=0.0, carry_mode="daily_0600", spread_vol_beta=0.0)
    cfg = PortfolioConfig(target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol, cost=cost)
    maker_cfg = PortfolioConfig(target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol, cost=maker)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 時代B（開発）
    holdout = pd.Timestamp(args.holdout_start, tz="UTC")
    prices_b = {}
    for path in sorted(Path(args.perp_dir).glob("*_1h.parquet")):
        df = pd.read_parquet(path).resample(f"{args.grid}h", label="right", closed="right").agg(AGG)
        prices_b[path.stem.split("_")[0]] = df.dropna(subset=["close"]).loc[:holdout]
    table_b = run_era("時代B: Perp 7 銘柄 2020-01〜2024-12（開発）", prices_b, grid_minutes, per_day,
                      base_lookback, cfg, maker_cfg)
    table_b.to_csv(out_dir / "improve_era_b.csv")

    # --- 時代A（独立検証）
    if args.huobi_dir:
        prices_a = {}
        for asset_dir in sorted(Path(args.huobi_dir).iterdir()):
            if not asset_dir.is_dir():
                continue
            raw = load_huobi(asset_dir)
            if len(raw) < 60 * 24 * 120:
                continue
            prices_a[asset_dir.name.upper()] = raw.resample(
                f"{args.grid}h", label="right", closed="right").agg(AGG).dropna(subset=["close"])
        if prices_a:
            table_a = run_era("時代A: Huobi 6 銘柄 2017-10〜2020-05（独立検証）", prices_a, grid_minutes,
                              per_day, base_lookback, cfg, maker_cfg)
            table_a.to_csv(out_dir / "improve_era_a.csv")
            merged = pd.DataFrame({"時代A_Sharpe": table_a["Sharpe"], "時代B_Sharpe": table_b["Sharpe"]})
            merged["両時代の最小値"] = merged.min(axis=1)
            print("\n===== 両時代の突き合わせ（最小値が高い案が頑健）=====")
            print(merged.round(2).sort_values("両時代の最小値", ascending=False).to_string())
            merged.to_csv(out_dir / "improve_merged.csv")
    print(f"\n出力: {out_dir}/improve_*.csv")


if __name__ == "__main__":
    main()
