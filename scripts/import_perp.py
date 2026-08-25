#!/usr/bin/env python3
"""パーペチュアル先物データセット（複数銘柄・ファンディング・OI・ベーシス）を取り込む。

データ元:
    https://github.com/PietroC21/Crypto-PerpetualFutures
    git clone --depth 1 https://github.com/PietroC21/Crypto-PerpetualFutures.git

7 銘柄（BTC/ETH/SOL/XRP/BNB/DOGE/AVAX）の 1 時間足で、2020-01 〜 2026-03 を収録。
`docs/real_data_findings.md` の結論「OHLCV だけでは足りない。板やファンディングのような
別系統の情報が要る」に対応する、**新しい情報源**である。

取り込む列:
    open/high/low/close/volume  現物（binance spot）の OHLCV。GMO のレバレッジ取引に最も近い価格系列
    funding_1h                  各取引所のファンディングレート（1 時間換算）の平均
    funding_spread              取引所間のばらつき（標準偏差）
    basis                       (perp VWAP − spot VWAP) / spot VWAP。先物の需給プレミアム
    oi                          建玉残高（利用可能な取引所の合計）

Example:
    python scripts/import_perp.py --src /path/to/Crypto-PerpetualFutures --out data/raw/perp
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ASSETS = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "AVAX")


def load_asset(src: Path, asset: str) -> pd.DataFrame:
    """1 銘柄ぶんの価格・ファンディング・OI・ベーシスを 1 枚の DataFrame にまとめる。"""
    price_files = sorted(glob.glob(str(src / "data" / asset / f"binance_{asset}_*.parquet")))
    master_file = src / "data" / asset / f"{asset}_master.parquet"
    if not price_files or not master_file.exists():
        return pd.DataFrame()

    px = pd.read_parquet(price_files[0]).set_index("timestamp").sort_index()
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")

    out = pd.DataFrame(index=px.index)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = px[f"{col}_spot_binance"].astype(float)
    # ベーシス: 先物が現物からどれだけ乖離しているか（先物の需給プレミアム）
    out["basis"] = (px["vwap_perp_binance"] - px["vwap_spot_binance"]) / px["vwap_spot_binance"]

    master = pd.read_parquet(master_file)
    master.index = pd.DatetimeIndex(master.index)
    if master.index.tz is None:
        master.index = master.index.tz_localize("UTC")
    funding_cols = [c for c in master.columns if c.startswith("funding_rate_1h")]
    oi_cols = [c for c in master.columns if c.startswith("open_interest")]
    funding = master[funding_cols].astype(float)
    out["funding_1h"] = funding.mean(axis=1).reindex(out.index)
    out["funding_spread"] = funding.std(axis=1).reindex(out.index)
    if oi_cols:
        out["oi"] = master[oi_cols].astype(float).sum(axis=1, min_count=1).reindex(out.index)

    out = out.dropna(subset=["close"])
    out.index.name = "close_time"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="クローンした Crypto-PerpetualFutures のパス")
    parser.add_argument("--out", default="data/raw/perp")
    args = parser.parse_args()

    src, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'銘柄':>6} {'バー数':>8} {'期間':>22} {'funding欠損':>11} {'OI欠損':>8} {'年率ボラ':>8}")
    for asset in ASSETS:
        df = load_asset(src, asset)
        if df.empty:
            print(f"{asset:>6}  データなし")
            continue
        df.to_parquet(out_dir / f"{asset}_1h.parquet")
        vol = np.log(df["close"]).diff().std() * np.sqrt(365 * 24)
        oi_missing = df["oi"].isna().mean() if "oi" in df else 1.0
        print(f"{asset:>6} {len(df):>8,} {df.index[0]:%Y-%m}〜{df.index[-1]:%Y-%m} "
              f"{df['funding_1h'].isna().mean():>11.1%} {oi_missing:>8.1%} {vol:>8.1%}")
    print(f"\n出力: {out_dir}/")


if __name__ == "__main__":
    main()
