#!/usr/bin/env python3
"""価格データの健全性を、独立したソースとの突き合わせで検証する。

**なぜ必要か**: 公開データセットには、価格水準もボラも正しいのに
**バーの時刻が実際の値動きと対応していない**ものがある。この種の破損は
「欠損なし・重複なし」のチェックを素通りし、統計量も正常に見えるため、
バックテストを走らせても気づけない。実際に本リポジトリでは、ある公開データの
バルク履歴がこれに該当し、短期の予測力の測定結果を無意味にしていた。

検出方法は単純で、**同じ資産の独立した 2 つのソースを突き合わせる**。
健全なら時間足リターンの相関は 0.99 以上になる。0.9 を下回るなら、
どちらかの時刻付けが壊れている。

    python scripts/verify_data.py --a data/raw/perp/BTC_1h.parquet \
        --b data/raw/BTCUSDT_huobi_1min.parquet --freq 1h

`--shift` は、片方が「バーの開始時刻」で、もう片方が「終了時刻」で索引されている
場合の補正（1 本ぶんずらすと相関が跳ね上がるなら、規約の違いであって破損ではない）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
GOOD, SUSPECT = 0.95, 0.80


def load(path: Path, freq: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    cols = {k: v for k, v in AGG.items() if k in df.columns}
    return df.resample(freq, label="right", closed="right").agg(cols).dropna(subset=["close"])


def compare(a: pd.DataFrame, b: pd.DataFrame, shifts=(-1, 0, 1)) -> dict:
    """時刻を前後にずらしながら、リターン相関が最大になる組み合わせを探す。"""
    best = {"shift": 0, "corr": -2.0, "n": 0}
    for shift in shifts:
        moved = np.log(b["close"]).copy()
        moved.index = moved.index + (moved.index[1] - moved.index[0]) * shift
        joined = pd.concat([np.log(a["close"]).rename("a"), moved.rename("b")], axis=1, sort=True).dropna()
        if len(joined) < 200:
            continue
        ret = joined.diff().dropna()
        c = float(ret["a"].corr(ret["b"]))
        if c > best["corr"]:
            level = (np.exp(joined["a"] - joined["b"]) - 1).median() * 1e4
            best = {"shift": shift, "corr": c, "n": len(ret), "level_bp": float(level),
                    "vol_a": float(ret["a"].std() * np.sqrt(365 * 24)), "vol_b": float(ret["b"].std() * np.sqrt(365 * 24))}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="検証したいデータ")
    parser.add_argument("--b", required=True, help="突き合わせる独立ソース")
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--by-year", action="store_true", help="年ごとに判定する（部分的な破損を見つける）")
    args = parser.parse_args()

    a, b = load(Path(args.a), args.freq), load(Path(args.b), args.freq)
    lo, hi = max(a.index[0], b.index[0]), min(a.index[-1], b.index[-1])
    if lo >= hi:
        raise SystemExit("2 つのデータに重なる期間がありません")
    a, b = a.loc[lo:hi], b.loc[lo:hi]
    print(f"重なる期間: {lo:%Y-%m-%d} 〜 {hi:%Y-%m-%d}（{args.freq} 足）")

    def verdict(c: float) -> str:
        return "○ 正常" if c >= GOOD else ("△ 要確認" if c >= SUSPECT else "✗ 破損の疑い")

    overall = compare(a, b)
    print(f"\n全期間: 相関 {overall['corr']:.4f}（{overall['shift']:+d} 本ずらし）  {verdict(overall['corr'])}")
    print(f"  価格差の中央値 {overall['level_bp']:+.1f}bp / 年率ボラ A {overall['vol_a']:.1%} · B {overall['vol_b']:.1%}")
    if overall["shift"] != 0:
        print(f"  ※ {overall['shift']:+d} 本ずらすと一致する = バーの時刻規約（開始/終了）の違い。破損ではない")

    if args.by_year:
        print(f"\n{'年':>6} {'相関':>8} {'ずらし':>7} {'判定':>12}")
        for year in range(lo.year, hi.year + 1):
            ya, yb = a.loc[str(year)], b.loc[str(year)]
            if len(ya) < 500 or len(yb) < 500:
                continue
            r = compare(ya, yb)
            print(f"{year:>6} {r['corr']:>8.4f} {r['shift']:>+7d} {verdict(r['corr']):>12}")

    if overall["corr"] < SUSPECT:
        print("\n価格水準とボラが一致していても、時刻の対応が壊れているデータは実在する。")
        print("この状態のデータで短期戦略を検証しても結果に意味は無い。ソースを差し替えること。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
