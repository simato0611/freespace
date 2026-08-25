#!/usr/bin/env python3
"""新しい情報源（ファンディング・ベーシス・建玉残高）の仮説探索。

`docs/real_data_findings.md` の結論は「OHLCV だけでは予測力が足りない」だった。
ここでは**別系統の情報**を足したときに予測力が増えるかを測る。

パーペチュアル先物のファンディングとベーシスは、レバレッジ需給の直接的な観測値である:

- ファンディングが大きく正 = ロングが過密（ロングがショートに支払っている）
- ベーシス（先物 − 現物）が大きく正 = 先物側にレバレッジ買いが溜まっている
- 建玉残高（OI）の急増 + 価格上昇 = 新規ロードの積み上がり（＝清算連鎖の燃料）

**プロトコル**: 開発期間のみで探索し、ホールドアウト（`--holdout-start` 以降）は触らない。
試行本数を数えて `docs/experiment_log.md` に記録する。

Example:
    python scripts/perp_survey.py --dir data/raw/perp --grid 4 --holdout-start 2025-01-01
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
from signal_survey import _zscore, alpha_vs_benchmark, buy_hold, momentum, simulate  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
       "funding_1h": "sum", "funding_spread": "mean", "basis": "last", "oi": "last"}


def load_asset(path: Path, grid_hours: int) -> pd.DataFrame:
    """1 時間足を `grid_hours` 時間足へ集約する（ファンディングは合計、ベーシス/OI は終値）。"""
    df = pd.read_parquet(path)
    cols = {k: v for k, v in AGG.items() if k in df.columns}
    out = df.resample(f"{grid_hours}h", label="right", closed="right").agg(cols)
    return out.dropna(subset=["close"])


# ------------------------------------------------------------------ シグナル定義
def build_signals(df: pd.DataFrame, per_day: int) -> dict[str, pd.Series]:
    """事前登録したシグナル一覧。方向は経済的な理屈で先に決める（当てはめてから決めない）。"""
    trend = momentum(df, 14 * per_day)                     # 既に確定している 14 日モメンタム
    trend_long = trend.clip(lower=0)
    window = 30 * per_day
    signals: dict[str, pd.Series] = {
        "trend_long（既存）": trend_long,
        "buy_hold": buy_hold(df),
    }
    if "funding_1h" in df:
        fz = _zscore(df["funding_1h"], window)
        # ファンディングが高い = ロング過密 → 逆張り（ショート寄り）が定説
        signals["funding_contrarian"] = (-fz / 2).clip(-1, 1)
        # 逆向き（対照群）: 高ファンディング = 強気の証拠、という仮説
        signals["funding_momentum"] = (fz / 2).clip(-1, 1)
        # キャリー: ファンディングがマイナス = ロングが受け取れる
        signals["funding_carry_long"] = (-fz / 2).clip(0, 1)
        # 既存トレンドに「過熱フィルタ」を掛ける
        signals["trend_x_funding_filter"] = trend_long * (fz < 1.5).astype(float)
    if "basis" in df:
        bz = _zscore(df["basis"], window)
        signals["basis_contrarian"] = (-bz / 2).clip(-1, 1)
        signals["trend_x_basis_filter"] = trend_long * (bz < 1.5).astype(float)
    if "oi" in df and df["oi"].notna().mean() > 0.5:
        oi_chg = np.log(df["oi"].replace(0, np.nan)).diff(per_day)
        oiz = _zscore(oi_chg, window)
        price_chg = np.sign(np.log(df["close"]).diff(per_day))
        # OI 増 + 価格上昇 = 新規ロングの積み上がり → 継続（トレンド確認）
        signals["oi_price_confirm"] = (oiz.clip(-2, 2) / 2 * price_chg).clip(-1, 1)
        # OI 急増そのものは清算リスク → 逆張り
        signals["oi_surge_contrarian"] = (-oiz / 2).clip(-1, 1)
        signals["trend_x_oi_filter"] = trend_long * (oiz < 1.5).astype(float)
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="data/raw/perp")
    parser.add_argument("--grid", type=int, default=4, help="判断間隔（時間）")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--half-spread-bp", type=float, default=2.0)
    parser.add_argument("--slippage-bp", type=float, default=0.5)
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--out", default="runs/analysis/perp_survey.csv")
    args = parser.parse_args()

    grid_minutes = args.grid * 60
    per_day = max(1, 24 // args.grid)
    cost = CostConfig(half_spread_bp=args.half_spread_bp, slippage_bp=args.slippage_bp,
                      carry_mode="daily_0600", spread_vol_beta=0.0)
    holdout = pd.Timestamp(args.holdout_start, tz="UTC")

    per_asset: dict[tuple[str, str], dict] = {}
    pooled: dict[str, list[pd.Series]] = {}
    for path in sorted(Path(args.dir).glob("*_1h.parquet")):
        asset = path.stem.split("_")[0]
        df = load_asset(path, args.grid)
        dev = df.loc[:holdout]
        if len(dev) < 90 * per_day:
            continue
        bench = simulate(dev, buy_hold(dev), grid_minutes, cost, args.target_vol)["net"]
        for name, signal in build_signals(dev, per_day).items():
            result = simulate(dev, signal, grid_minutes, cost, args.target_vol)
            metrics = equity_metrics(1e6 * (1 + result["net"]).cumprod(), result["exposure"])
            row = {"Sharpe": metrics["sharpe"], "年率": metrics["cagr"], "最大DD": metrics["max_drawdown"],
                   "回転/日": metrics["turnover_per_day"], "稼働率": float(result["exposure"].abs().mean())}
            row.update(alpha_vs_benchmark(result["net"], bench, grid_minutes))
            per_asset[(asset, name)] = row
            pooled.setdefault(name, []).append(result["net"].rename(asset))

    table = pd.DataFrame(per_asset).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=["銘柄", "シグナル"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)

    # 銘柄をまたいだ集計（等ウェイトで日次リターンを平均 = 分散投資したときの姿）
    summary = {}
    for name, series in pooled.items():
        combined = pd.concat(series, axis=1).mean(axis=1).dropna()
        equity = 1e6 * (1 + combined).cumprod()
        metrics = equity_metrics(equity)
        by_asset = table.xs(name, level="シグナル")["Sharpe"].astype(float)
        summary[name] = {
            "等ウェイトSharpe": metrics["sharpe"],
            "年率": metrics["cagr"],
            "最大DD": metrics["max_drawdown"],
            "銘柄別Sharpe平均": by_asset.mean(),
            "プラスの銘柄": f"{int((by_asset > 0).sum())}/{len(by_asset)}",
        }
    summary_table = pd.DataFrame(summary).T.sort_values("等ウェイトSharpe", ascending=False)
    show = summary_table.copy()
    for col in ("年率", "最大DD"):
        show[col] = pd.to_numeric(show[col]).mul(100).round(1).astype(str) + "%"
    for col in ("等ウェイトSharpe", "銘柄別Sharpe平均"):
        show[col] = pd.to_numeric(show[col]).round(2)
    print(f"[perp] 開発期間 〜 {args.holdout_start}（ホールドアウトは未使用） / 判断間隔 {args.grid} 時間")
    print("\n=== 銘柄等ウェイトでの成績（開発期間のみ）===")
    print(show.to_string())
    print(f"\n試行本数: {len(summary)}（`docs/experiment_log.md` に記録すること）")
    print(f"出力: {args.out}")


if __name__ == "__main__":
    main()
