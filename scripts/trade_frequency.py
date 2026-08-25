#!/usr/bin/env python3
"""発注回数を実測し、最小発注幅（min_trade_delta）を較正する。

バックテストは毎バー目標建玉に張り替えるので、そのまま実運用すると
「7 銘柄 × 24 本 = 1 日 168 回」の発注指示になる。実際にはその大半が
建玉比 0.001 未満の微調整で、板を叩く意味がない。

そこで実行系は**最小発注幅**を持つ（`LiveConfig.min_trade_delta`）。
このスクリプトは幅を振って「発注回数」と「成績」の両方を測り、
連続リバランスと同じ成績を保てる最大の幅を選ぶための材料を出す。

Example:
    python scripts/trade_frequency.py --assets BTC ETH XRP BNB DOGE
    python scripts/trade_frequency.py --period holdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlgmo.costs import CostConfig, carry_flags, carry_rate_per_bar  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.portfolio import (  # noqa: E402
    PortfolioConfig, apply_rebalance_band, compute_exposures, ladder_signal,
)

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
DELTAS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)
FLIP_FLOOR = 0.01   # これ未満の建玉はゼロ扱い（方向転換の数え方）


def load(dir_path: Path, grid_hours: int, assets: list[str] | None) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(dir_path.glob("*_1h.parquet")):
        name = path.stem.split("_")[0]
        if assets and name not in assets:
            continue
        df = pd.read_parquet(path)
        out[name] = df.resample(f"{grid_hours}h", label="right", closed="right").agg(AGG).dropna(subset=["close"])
    return out


def gate(exposure: pd.DataFrame, delta: float) -> pd.DataFrame:
    """最小発注幅を適用する。差が幅未満の銘柄は建玉を据え置く（＝発注しない）。"""
    if delta <= 0.0:
        return exposure
    values = exposure.to_numpy(dtype=float, copy=True)
    current = np.zeros(values.shape[1])
    for i in range(values.shape[0]):
        move = np.abs(values[i] - current) >= delta
        current = np.where(move, values[i], current)
        values[i] = current
    return pd.DataFrame(values, index=exposure.index, columns=exposure.columns)


def evaluate(exposure: pd.DataFrame, gap: pd.DataFrame, intra: pd.DataFrame,
             grid_minutes: int, cfg: PortfolioConfig, delta: float) -> dict:
    """発注回数と成績を同時に測る。"""
    held = gate(exposure, delta)
    diff = (held - held.shift(1)).fillna(0.0)
    orders = (diff.abs() > 1e-12)                       # 銘柄ごとの「発注した」回数
    days = len(held) * grid_minutes / 1440

    one_way = (cfg.cost.half_spread_bp + cfg.cost.slippage_bp + cfg.cost.taker_fee_bp) * 1e-4
    flags = carry_flags(held.index, cfg.cost.carry_hour_jst)
    carry_rate = pd.Series(np.where(flags, carry_rate_per_bar(cfg.cost, grid_minutes, True),
                                    carry_rate_per_bar(cfg.cost, grid_minutes, False)), index=held.index)
    gross_pnl = (held.shift(1) * gap + held * intra).sum(axis=1)
    cost = diff.abs().sum(axis=1) * one_way + carry_rate * held.abs().sum(axis=1)
    ret = (gross_pnl - cost).fillna(0.0).iloc[:-1]
    metrics = equity_metrics(1e6 * (1 + ret).cumprod())

    # 売買方向の転換。建玉比 0.01 未満の往復はゼロ近傍の揺れなので数えない
    sign = np.sign(held.where(held.abs() > FLIP_FLOOR, 0.0).to_numpy())
    flips = int(((sign[1:] * sign[:-1]) < 0).sum())
    return {
        "最小発注幅": delta,
        "発注/日": orders.to_numpy().sum() / max(days, 1e-9),
        "方向転換/月": flips / max(days / 30.4, 1e-9),
        "Sharpe": metrics["sharpe"],
        "年率": metrics["cagr"],
        "最大DD": metrics["max_drawdown"],
        "回転/日": float(diff.abs().sum().sum() / max(days, 1e-9)),
        "_per_asset": (orders.sum() / max(days, 1e-9)).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="data/raw/perp")
    parser.add_argument("--assets", nargs="*", default=None, help="銘柄を絞る（既定は全部）")
    parser.add_argument("--grid", type=int, default=1, help="判断間隔（時間）")
    parser.add_argument("--period", default="all", choices=["dev", "holdout", "all"])
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--band", type=float, default=0.10)
    args = parser.parse_args()

    grid_minutes = args.grid * 60
    bars_per_day = max(1, 24 // args.grid)
    holdout = pd.Timestamp(args.holdout_start, tz="UTC")

    prices = load(Path(args.dir), args.grid, args.assets)
    cfg = PortfolioConfig(target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol,
                          cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5,
                                          carry_mode="daily_0600", spread_vol_beta=0.0))
    signals = {a: apply_rebalance_band(ladder_signal(df, bars_per_day, long_only=False), args.band)
               for a, df in prices.items()}
    exposure = compute_exposures(prices, signals, grid_minutes, cfg)
    gap, intra = exposure.attrs["gap"], exposure.attrs["intra"]
    if args.period == "dev":
        mask = exposure.index < holdout
    elif args.period == "holdout":
        mask = exposure.index >= holdout
    else:
        mask = slice(None)
    exposure, gap, intra = exposure.loc[mask], gap.loc[mask], intra.loc[mask]

    print(f"[頻度] {args.period} / 銘柄 {list(prices)} / 判断間隔 {args.grid} 時間 / {len(exposure)} 本")
    rows = [evaluate(exposure, gap, intra, grid_minutes, cfg, d) for d in DELTAS]
    table = pd.DataFrame(rows).drop(columns=["_per_asset"])
    for col in ("年率", "最大DD"):
        table[col] = (table[col] * 100).round(1).astype(str) + "%"
    for col in ("発注/日", "方向転換/月", "Sharpe", "回転/日"):
        table[col] = table[col].round(2)
    print("\n" + table.to_string(index=False))

    chosen = next(r for r in rows if r["最小発注幅"] == 0.005)
    print("\n=== 最小発注幅 0.005 の銘柄別 発注/日 ===")
    for asset, n in sorted(chosen["_per_asset"].items(), key=lambda kv: -kv[1]):
        print(f"  {asset:<5} {n:5.2f}")
    print(f"  合計  {sum(chosen['_per_asset'].values()):5.2f}")


if __name__ == "__main__":
    main()
