#!/usr/bin/env python3
"""GMO の実データで、本番設定（configs/gmo_live.yaml）をそのまま検証する。

引き継ぎ先（Desktop 版 Claude Code）が最初に走らせるスクリプト。
**バックテスト用のパラメータを別途持たず、実運用と同じ設定ファイルを読む**ので、
「検証したものと動かすものが違う」という事故が起きない。

判定は docs/HANDOFF.md §3 ステップ4 のゲート表に従って自動で出す。
数字が想定より低くても**パラメータを触らないこと**（docs/HANDOFF.md §4）。

Example:
    # 1分足を置いてある場合（1時間足に自動で畳む）
    python scripts/gmo_validate.py --dir data/raw/gmo --config configs/gmo_live.yaml

    # 突き合わせ用に既存の海外データで動かして、比較の基準を作る
    python scripts/gmo_validate.py --dir data/raw/perp --symbols BTC ETH XRP BNB DOGE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.portfolio import (  # noqa: E402
    PortfolioConfig, apply_rebalance_band, backtest_portfolio, compute_exposures, ladder_signal,
)

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

# docs/HANDOFF.md §3 ステップ4 のゲート表（GO の下限, NO-GO の下限）
GATES = {
    "ホールドアウト Sharpe": (0.6, 0.2, "higher"),
    "最大DD": (0.15, 0.25, "lower"),
    "発注回数/日": (20.0, 40.0, "lower"),
    "月次勝率": (0.55, 0.45, "higher"),
    "BTCベータ(絶対値)": (0.3, 0.5, "lower"),
}


def load_prices(dir_path: Path, grid_hours: int, symbols: list[str] | None) -> dict[str, pd.DataFrame]:
    """*_1min / *_1h / 素の *.parquet のどれでも読み、共通グリッドに畳む。"""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(dir_path.glob("*.parquet")):
        name = path.stem.split("_")[0]
        if symbols and name not in symbols and path.stem not in symbols:
            continue
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        cols = {k: v for k, v in AGG.items() if k in df.columns}
        out[name] = df.resample(f"{grid_hours}h", label="right", closed="right").agg(cols).dropna(subset=["close"])
    if not out:
        raise SystemExit(f"{dir_path} に parquet が見つかりません")
    return out


def describe(ret: pd.Series, btc_ret: pd.Series, orders_per_day: float, label: str) -> dict:
    m = equity_metrics(1e6 * (1 + ret).cumprod())
    daily = pd.DataFrame({"s": ret, "b": btc_ret.reindex(ret.index).fillna(0.0)}).resample("1D").sum()
    beta = float(np.polyfit(daily["b"], daily["s"], 1)[0]) if daily["b"].std() > 0 else float("nan")
    monthly = ret.resample("ME").sum()
    return {
        "区間": label,
        "Sharpe": m["sharpe"],
        "年率": m["cagr"],
        "最大DD": m["max_drawdown"],
        "年率ボラ": m["ann_vol"],
        "BTC相関": float(daily["s"].corr(daily["b"])),
        "BTCベータ": beta,
        "月次勝率": float((monthly > 0).mean()),
        "月数": len(monthly),
        "発注/日": orders_per_day,
    }


def gate_and_report(rows: list[dict], holdout: dict) -> bool:
    checks = {
        "ホールドアウト Sharpe": holdout["Sharpe"],
        "最大DD": abs(holdout["最大DD"]),
        "発注回数/日": holdout["発注/日"],
        "月次勝率": holdout["月次勝率"],
        "BTCベータ(絶対値)": abs(holdout["BTCベータ"]),
    }
    print("\n=== 採用ゲート（docs/HANDOFF.md §3 ステップ4）===")
    all_go = True
    for name, (go, nogo, direction) in GATES.items():
        value = checks[name]
        if direction == "higher":
            verdict = "GO" if value >= go else ("NO-GO" if value < nogo else "要検討")
        else:
            verdict = "GO" if value <= go else ("NO-GO" if value > nogo else "要検討")
        all_go &= verdict == "GO"
        mark = {"GO": "○", "要検討": "△", "NO-GO": "✗"}[verdict]
        print(f"  {mark} {name:<22} {value:>7.2f}   （GO 基準 {'≥' if direction == 'higher' else '≤'} {go}）")
    return all_go


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="OHLCV parquet を置いたディレクトリ")
    parser.add_argument("--config", default="configs/gmo_live.yaml", help="本番設定（これがパラメータの唯一の正）")
    parser.add_argument("--symbols", nargs="*", default=None, help="銘柄を絞る（既定は設定ファイルの全銘柄）")
    parser.add_argument("--holdout-start", default="2025-01-01", help="封印期間の開始。安易に動かさないこと")
    parser.add_argument("--btc", default="BTC", help="ベータ計算の基準にする銘柄名")
    parser.add_argument("--out", default="runs/gmo_validate")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    s, c, r, e = raw.get("strategy", {}), raw.get("cost", {}), raw.get("risk", {}), raw.get("execution", {})
    grid_hours = int(s.get("grid_hours", 1))
    grid_minutes = grid_hours * 60
    bars_per_day = max(1, 24 // grid_hours)

    symbols = args.symbols or [x.split("_")[0] for x in raw["data"]["symbols"]]
    prices = load_prices(Path(args.dir), grid_hours, symbols)

    cfg = PortfolioConfig(
        target_vol_ann=float(e.get("target_vol_ann", 0.15)),
        asset_vol_ann=float(e.get("asset_vol_ann", 0.15)),
        max_weight=float(e.get("max_weight", 0.5)),
        leverage_cap=float(e.get("leverage_cap", 2.0)),
        cost=CostConfig(half_spread_bp=float(c.get("half_spread_bp", 1.5)),
                        slippage_bp=float(c.get("slippage_bp", 0.0)),
                        taker_fee_bp=float(c.get("taker_fee_bp", 0.0)),
                        carry_rate_daily=float(c.get("carry_rate_daily", 0.0004)),
                        carry_mode=str(c.get("carry_mode", "daily_0600")),
                        spread_vol_beta=0.0),
    )
    signals = {
        a: apply_rebalance_band(
            ladder_signal(df, bars_per_day, tuple(s.get("lookback_days", (5, 14, 30, 60))),
                          bool(s.get("long_only", False)), float(s.get("gain", 1.5)),
                          int(s.get("vol_window_bars", 30))),
            float(s.get("rebalance_band", 0.10)))
        for a, df in prices.items()
    }
    result = backtest_portfolio(prices, signals, grid_minutes, cfg)
    exposure = compute_exposures(prices, signals, grid_minutes, cfg)

    # 発注回数は最小発注幅を通した後の数で数える（実運用と同じ）
    delta = float(r.get("min_trade_delta", 0.005))
    values = exposure.to_numpy(dtype=float, copy=True)
    current = np.zeros(values.shape[1])
    for i in range(values.shape[0]):
        move = np.abs(values[i] - current) >= delta
        current = np.where(move, values[i], current)
        values[i] = current
    held = pd.DataFrame(values, index=exposure.index, columns=exposure.columns)
    issued = (held - held.shift(1)).abs() > 1e-12

    print(f"[検証] 設定 {args.config} / データ {args.dir}")
    print(f"       銘柄 {sorted(prices)} / 判断間隔 {grid_hours}h / 最小発注幅 {delta}")
    print(f"       期間 {result.index[0]:%Y-%m-%d} 〜 {result.index[-1]:%Y-%m-%d}（{len(result):,} 本）")

    if args.btc not in prices:
        raise SystemExit(f"ベータ計算の基準 {args.btc} がデータにありません（--btc で指定）")
    btc_ret = prices[args.btc]["close"].reindex(result.index).ffill().pct_change().fillna(0.0)

    ho = pd.Timestamp(args.holdout_start, tz="UTC")
    rows = []
    for label, mask in (("全期間", slice(None)), ("開発期間", result.index < ho), ("ホールドアウト", result.index >= ho)):
        part = result.loc[mask]
        if len(part) < bars_per_day * 30:
            print(f"[warn] {label} が短すぎるため省略（{len(part)} 本）")
            continue
        sub = issued.loc[part.index]
        rows.append(describe(part["ret"], btc_ret, sub.to_numpy().sum() / (len(sub) * grid_minutes / 1440), label))

    table = pd.DataFrame(rows).set_index("区間")
    show = table.copy()
    for col in ("年率", "最大DD", "年率ボラ", "月次勝率"):
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    for col in ("Sharpe", "BTC相関", "BTCベータ", "発注/日"):
        show[col] = show[col].round(2)
    print("\n" + show.to_string())

    holdout = next((row for row in rows if row["区間"] == "ホールドアウト"), None)
    if holdout is None:
        print("\n[warn] ホールドアウト区間のデータが足りないため、ゲート判定を省略しました")
    else:
        all_go = gate_and_report(rows, holdout)
        print(f"\n総合判定: {'GO — 次のステップ（実効スプレッド実測）へ' if all_go else '要検討 — docs/HANDOFF.md §4 を読んでから判断すること'}")
        print("※ 数字が想定より低くても、パラメータを触ってはいけない（後知恵の当てはめになる）")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "summary.csv")
    result[["ret", "equity", "gross_exposure", "net_exposure", "cost"]].to_csv(out / "curve.csv")
    held.to_csv(out / "exposure.csv")
    print(f"\n保存: {out}/summary.csv, curve.csv, exposure.csv")


if __name__ == "__main__":
    main()
