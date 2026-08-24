#!/usr/bin/env python3
"""板情報から実効スプレッドを実測し、コストモデルの `half_spread_bp` を較正する。

バックテストのコスト仮定が甘いと、1 分足戦略の成績は簡単に虚構になる。
運用前に必ず、自分が実際に建てるサイズで「板を食ったときの平均約定価格」を実測すること。

Example:
    # 60 分間、10 秒ごとに BTC_JPY の板を記録し、0.05 BTC を成行で食う場合のコストを推定
    python scripts/measure_spread.py --symbol BTC_JPY --minutes 60 --interval 10 --size 0.05
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://api.coin.z.com/public/v1"


def walk_book(levels: list[dict], size: float) -> float:
    """指定サイズを板で食ったときの平均約定価格を返す（数量が足りなければ最終価格）。"""
    remaining, cost = size, 0.0
    for level in levels:
        price, qty = float(level["price"]), float(level["size"])
        take = min(remaining, qty)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            break
    filled = size - max(remaining, 0.0)
    return cost / filled if filled > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC_JPY")
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--interval", type=float, default=10.0, help="サンプリング間隔（秒）")
    parser.add_argument("--size", type=float, default=0.01, help="想定発注数量")
    parser.add_argument("--out", default="data/raw/spread_samples.csv")
    args = parser.parse_args()

    rows = []
    deadline = time.time() + args.minutes * 60
    while time.time() < deadline:
        try:
            book = requests.get(f"{BASE}/orderbooks", params={"symbol": args.symbol}, timeout=10).json()["data"]
            bids, asks = book["bids"], book["asks"]
            best_bid, best_ask = float(bids[0]["price"]), float(asks[0]["price"])
            mid = (best_bid + best_ask) / 2
            buy_vwap, sell_vwap = walk_book(asks, args.size), walk_book(bids, args.size)
            rows.append(
                {
                    "ts": pd.Timestamp.utcnow(),
                    "mid": mid,
                    "top_half_spread_bp": (best_ask - best_bid) / 2 / mid * 1e4,
                    "eff_half_spread_bp": (buy_vwap - sell_vwap) / 2 / mid * 1e4,  # サイズ込みの実効片道コスト
                }
            )
        except Exception as err:  # noqa: BLE001
            print(f"[warn] {err}")
        time.sleep(args.interval)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("サンプルを取得できませんでした")
    df.to_csv(args.out, index=False)
    q = df["eff_half_spread_bp"].quantile([0.5, 0.9, 0.99])
    print(f"サンプル数: {len(df)}  → {args.out}")
    print(f"実効片道スプレッド (bp)  中央値 {q[0.5]:.2f} / 90% {q[0.9]:.2f} / 99% {q[0.99]:.2f}")
    print(f"最良気配のみ (bp)        中央値 {df['top_half_spread_bp'].median():.2f}")
    print(f"\nconfigs/*.yaml の env.cost.half_spread_bp には保守的に {np.ceil(q[0.9]):.0f} 前後を設定することを推奨。")


if __name__ == "__main__":
    main()
