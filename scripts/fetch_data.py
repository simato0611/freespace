#!/usr/bin/env python3
"""GMO コイン Public API から 1 分足を取得して保存する。

Example:
    python scripts/fetch_data.py --symbol BTC_JPY --start 2023-01-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlgmo.data.gmo_klines import FetchConfig, fetch_klines_range, save_parquet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC_JPY", help="レバレッジ銘柄（例 BTC_JPY, ETH_JPY）")
    parser.add_argument("--interval", default="1min", choices=["1min", "5min", "10min", "15min", "30min"])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sleep", type=float, default=0.5, help="リクエスト間隔（秒）")
    args = parser.parse_args()

    cfg = FetchConfig(symbol=args.symbol, interval=args.interval, sleep_sec=args.sleep)
    df = fetch_klines_range(args.start, args.end, cfg)
    if df.empty:
        raise SystemExit("データを取得できませんでした（期間・銘柄・ネットワークを確認してください）")

    out = args.out or f"data/raw/{args.symbol}_{args.interval}.parquet"
    path = save_parquet(df, out)
    gaps = df.index.to_series().diff().dt.total_seconds().div(60).fillna(1)
    print(f"保存: {path}  bars={len(df):,}  {df.index[0]} 〜 {df.index[-1]}")
    print(f"欠損バー: {int((gaps > 1).sum()):,} 箇所（最大 {gaps.max():.0f} 分の空白）")


if __name__ == "__main__":
    main()
