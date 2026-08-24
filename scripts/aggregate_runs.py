#!/usr/bin/env python3
"""ウォークフォワードの実行結果を突き合わせて比較表にする。

各 fold のテスト区間の損益を**連結して 1 本のアウトオブサンプル曲線**にし、
そこから最終的な指標（Sharpe・最大 DD・コスト内訳・Deflated Sharpe）を出す。
fold ごとの Sharpe を単純平均するより、実際に運用した場合の姿に近い。

Example:
    python scripts/aggregate_runs.py runs/btc_real:1分判断 runs/btc_real_h60:60分判断 \
        runs/btc_real_h240:240分判断
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.metrics import deflated_sharpe, equity_metrics, infer_bars_per_year  # noqa: E402


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """fold ごとのテスト記録を連結し、(連結記録, fold 別レポート) を返す。"""
    frames = []
    for path in sorted(run_dir.glob("fold*_test.csv"), key=lambda p: int(p.stem.split("_")[0][4:])):
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
        df["fold"] = int(path.stem.split("_")[0][4:])
        frames.append(df)
    if not frames:
        raise SystemExit(f"テスト記録が見つかりません: {run_dir}/fold*_test.csv")
    record = pd.concat(frames).sort_index()
    report_path = run_dir / "walkforward_report.csv"
    report = pd.read_csv(report_path) if report_path.exists() else None
    return record, report


def chain_equity(record: pd.DataFrame) -> pd.Series:
    """fold をまたいで複利で連結したエクイティ曲線を作る。"""
    equity = []
    level = 1.0
    for _, group in record.groupby("fold", sort=True):
        rel = group["equity"] / group["equity"].iloc[0]
        equity.append(rel * level)
        level = float(rel.iloc[-1] * level)
    return pd.concat(equity)


def _annual_rate(record: pd.DataFrame, column: str, years: float) -> float:
    """バーごとの金額を「直前の有効証拠金に対する率」に直し、年率の平均寄与にする。

    損益の内訳（グロス / 取引コスト / 建玉管理料）は、単純合計でも複利連結でも
    比較しづらい（資金が減るほど同じ bp が小さい金額になる）。ここでは
    「年あたり何 % の寄与か」に揃える。グロス − コスト ≒ 純リターン（年率）になる。
    """
    total = 0.0
    for _, group in record.groupby("fold", sort=True):
        prev_equity = group["equity"].shift(1)
        prev_equity.iloc[0] = (group["equity"].iloc[0] - group["pnl"].iloc[0]
                               + group["trade_cost"].iloc[0] + group["carry_cost"].iloc[0])
        total += float((group[column] / prev_equity.replace(0.0, np.nan)).fillna(0.0).sum())
    return total / max(years, 1e-9)


def summarize_run(run_dir: Path, label: str, n_trials: int) -> dict:
    record, report = load_run(run_dir)
    equity = chain_equity(record)
    metrics = equity_metrics(equity, record["position"])
    bars_per_year = infer_bars_per_year(pd.DatetimeIndex(equity.index))
    days = len(record) / max(bars_per_year / 365.0, 1e-9)
    initial = 1.0
    trial_std = float(report["rl_sharpe"].std()) if report is not None and len(report) > 1 else 1.0

    out = {
        "戦略": label,
        "fold数": int(record["fold"].nunique()),
        "OOS日数": round(days),
        "純リターン": metrics["total_return"],
        "Sharpe": metrics["sharpe"],
        "最大DD": metrics["max_drawdown"],
        "年率ボラ": metrics["ann_vol"],
        "回転/日": metrics["turnover_per_day"],
        "平均|建玉|": metrics["exposure"],
        # グロス／コストは「1 バー前の有効証拠金に対する比率」を複利で連結して求める
        # （fold ごとの単純合計は、fold 内で資金が増減するぶんだけ歪む）。
        "グロス/年": _annual_rate(record, "pnl", days / 365.0),
        "取引コスト/年": -_annual_rate(record, "trade_cost", days / 365.0),
        "管理料/年": -_annual_rate(record, "carry_cost", days / 365.0),
        "DSR_p": deflated_sharpe(metrics["sharpe"], len(equity), n_trials, trial_std,
                                 metrics.get("skew", 0.0), metrics.get("excess_kurtosis", 0.0) + 3.0,
                                 bars_per_year=bars_per_year),
    }
    if report is not None:
        out["fold勝率"] = float((report["rl_sharpe"] > 0).mean())
        for name in ("flat", "long", "momentum"):
            col = f"{name}_sharpe"
            if col in report:
                out[f"vs_{name}"] = float((report["rl_sharpe"] > report[col]).mean())
    _ = initial
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="実行ディレクトリ（'path' または 'path:表示名'）")
    parser.add_argument("--n-trials", type=int, default=20, help="Deflated Sharpe 用の総試行回数")
    parser.add_argument("--out", default="runs/comparison.csv")
    args = parser.parse_args()

    rows = []
    for spec in args.runs:
        path, _, label = spec.partition(":")
        run_dir = Path(path)
        if not run_dir.exists():
            print(f"[skip] {run_dir} が見つかりません")
            continue
        rows.append(summarize_run(run_dir, label or run_dir.name, args.n_trials))

    if not rows:
        raise SystemExit("集計対象がありません")
    table = pd.DataFrame(rows).set_index("戦略")
    pct = ["純リターン", "最大DD", "年率ボラ", "グロス/年", "取引コスト/年", "管理料/年", "平均|建玉|"]
    shown = table.copy()
    for col in pct:
        if col in shown:
            shown[col] = (shown[col] * 100).round(2).astype(str) + "%"
    for col in ("Sharpe", "回転/日", "DSR_p", "fold勝率", "vs_flat", "vs_long", "vs_momentum"):
        if col in shown:
            shown[col] = shown[col].round(3)
    print("\n" + shown.to_string())
    table.to_csv(args.out)
    print(f"\n出力: {args.out}")
    print("\n※ 'vs_flat' は「常にノーポジより Sharpe が高かった fold の割合」。"
          "\n   ここが 0.5 を大きく下回る戦略は、単に取引しない方が良いということ。")


if __name__ == "__main__":
    main()
