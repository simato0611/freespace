#!/usr/bin/env python3
"""公開データセット（Bitstamp BTC/USD 1 分足）を本リポジトリの OHLCV 形式へ変換する。

GMO の Public API に到達できない環境で、**実際の市場データ**を使って戦略とパイプラインを
検証するための入口。データ元:

    https://github.com/ff137/bitstamp-btcusd-minute-data
    （2012-01-01 以降の 1 分足。欠損・重複なし。日次で更新）

    git clone --depth 1 https://github.com/ff137/bitstamp-btcusd-minute-data.git

**GMO の BTC_JPY との差**（結果を読むときに必ず考慮すること）:

- 取引所・通貨建てが違う（Bitstamp BTC/USD ↔ GMO BTC_JPY レバレッジ）。BTC_JPY は
  裁定により BTC/USD × USD/JPY にほぼ追随するが、JPY レッグぶんの差（年率 10% 程度の
  為替ボラ）と、GMO 板固有のマイクロ構造の差が残る。分足のダイナミクス（ボラのクラスタ性・
  厚い裾・ほぼゼロの自己相関・日中季節性）は実データそのものなので、
  **「学習機構とコスト仮定の検証」には十分**だが、**GMO での実運用成績の保証ではない**。
- コストモデル（スプレッド・建玉管理料）は GMO の設定のまま使う。ここが検証の本体である。
- 最終判断は必ず GMO 自身の 1 分足（`scripts/fetch_data.py`）で取り直して行うこと。

Example:
    python scripts/import_bitstamp.py --src /path/to/bitstamp-btcusd-minute-data \
        --start 2024-01-01 --end 2026-08-20 --out data/raw/BTCUSD_bitstamp_1min.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rlgmo.data.gmo_klines import save_parquet  # noqa: E402


def load_repo(src: Path) -> pd.DataFrame:
    """バルク履歴と日次更新を結合し、クローズ時刻 index の OHLCV を返す。"""
    frames = []
    for path in sorted((src / "data").rglob("*.csv*")):
        frames.append(pd.read_csv(path))
        print(f"  読み込み: {path.name} ({len(frames[-1]):,} 行)")
    if not frames:
        raise SystemExit(f"CSV が見つかりません: {src}/data")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    # timestamp はバーのオープン時刻（epoch 秒）。本リポジトリの規約に合わせてクローズ時刻にする。
    close_time = pd.to_datetime(df["timestamp"].astype("int64") + 60, unit="s", utc=True)
    out = pd.DataFrame(
        {c: df[c].astype(float).to_numpy() for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex(close_time, name="close_time"),
    )
    return out.sort_index()


def report_quality(df: pd.DataFrame) -> None:
    gaps = df.index.to_series().diff().dt.total_seconds().div(60)
    zero_vol = (df["volume"] == 0).mean()
    ret = (df["close"] / df["close"].shift(1) - 1).dropna()
    print(f"  バー数        : {len(df):,}  ({df.index[0]} 〜 {df.index[-1]})")
    print(f"  欠損          : {int((gaps > 1).sum()):,} 箇所 (最大 {gaps.max():.0f} 分)")
    print(f"  出来高ゼロ足  : {zero_vol:.1%}")
    print(f"  年率ボラ      : {ret.std() * (365 * 24 * 60) ** 0.5:.1%}")
    print(f"  1 分リターンの尖度: {ret.kurtosis():.1f}（正規分布なら 0）")
    print(f"  ラグ1 自己相関: {ret.autocorr(1):+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="クローンした bitstamp-btcusd-minute-data のパス")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="data/raw/BTCUSD_bitstamp_1min.parquet")
    args = parser.parse_args()

    print("[import] 読み込み中...")
    df = load_repo(Path(args.src))
    print("[import] 全期間:")
    report_quality(df)

    sliced = df.loc[args.start : args.end] if args.end else df.loc[args.start :]
    print(f"[import] 切り出し ({args.start} 〜 {args.end or '最新'}):")
    report_quality(sliced)

    path = save_parquet(sliced, args.out)
    print(f"[import] 保存: {path}")


if __name__ == "__main__":
    main()
