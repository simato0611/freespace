#!/usr/bin/env python3
"""資金調達率キャリー（いわゆる Funding Rate Arbitrage）を、この repo の作法で検証する。

**この環境では取引所 API が遮断されているためデータを取得できない。**
ネットワークが通る環境（手元の PC など）で `--fetch` を付けて実行すればデータを取得し、
そのまま検証まで走る。

## この戦略は「アービトラージ」ではない

現物ロング + 無期限先物ショートで価格変動を消し、資金調達率を受け取り続ける形は、
**リスクプレミアムを売るキャリー取引**である。docs/strategy_search.md 67 節で、
信用スプレッドを使った 99 年の検証を行った結果は次のとおり:

    Sharpe 0.23 / 最大DD −67.6% / 歪度 −0.98 / 尖度 19.3 / 月次勝率 59%
    大恐慌 −59.6%、70年代 −21.3%、ITバブル −10.6%、リーマン −15.7%（すべてマイナス）

**月の 59% で勝つのに Sharpe は 0.23。** 負けるときに大きく負けるためである。
資金調達率キャリーが同じ構造を持つかどうかを、実データで確かめるのがこのスクリプトの目的。

## 検証の作法（この repo 共通）

1. **2 つの独立した時代**に分けて評価する。既定は強気相場（〜2021）と弱気相場（2022〜）
2. **弱気相場を必ず含める。** 2022-2023 の BTC は 20〜25% の期間で資金調達率がマイナス
3. コストを甘くしない。現物手数料・先物手数料・スプレッド・借入金利を全部入れる
4. 歪度と最悪月を必ず見る。**Sharpe だけ見ると、この戦略の危険は見えない**

Example:
    # データ取得つき（ネットワークが通る環境で）
    python scripts/funding_carry.py --fetch --symbols BTCUSDT ETHUSDT --out data/raw/funding

    # 取得済みデータで検証だけ
    python scripts/funding_carry.py --dir data/raw/funding
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BINANCE = "https://fapi.binance.com/fapi/v1/fundingRate"
PERIODS_PER_YEAR = 365 * 3          # 8 時間ごとに 1 回

# 既定の時代分割。相場環境が正反対の 2 期間を選んである
ERAS = [("強気 2019-2021", "2019-01-01", "2021-12-31"),
        ("弱気 2022-2024", "2022-01-01", "2024-12-31")]


def fetch_funding(symbol: str, start: str, end: str, sleep_sec: float = 0.3) -> pd.DataFrame:
    """Binance の資金調達率履歴を取得する。1 回 1000 件までなので繰り返す。"""
    import requests

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows: list[dict] = []
    while start_ms < end_ms:
        res = requests.get(BINANCE, params={"symbol": symbol, "startTime": start_ms,
                                            "endTime": end_ms, "limit": 1000}, timeout=20)
        res.raise_for_status()
        chunk = res.json()
        if not chunk:
            break
        rows.extend(chunk)
        last = int(chunk[-1]["fundingTime"])
        if last <= start_ms:
            break
        start_ms = last + 1
        time.sleep(sleep_sec)
    if not rows:
        return pd.DataFrame(columns=["rate"], index=pd.DatetimeIndex([], name="time", tz="UTC"))
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    out = pd.DataFrame({"rate": df["fundingRate"].astype(float)}, index=idx)
    out.index.name = "time"
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_dir(path: Path) -> dict[str, pd.Series]:
    out = {}
    for f in sorted(path.glob("*.parquet")):
        df = pd.read_parquet(f)
        s = df["rate"]
        s.index = pd.DatetimeIndex(s.index)
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        out[f.stem] = s.sort_index()
    return out


def carry_returns(rate: pd.Series, spot_fee_bp: float, perp_fee_bp: float,
                  half_spread_bp: float, borrow_ann: float, rebalance_days: int) -> pd.Series:
    """デルタニュートラルのキャリー収益。

    受け取り: 資金調達率（ショート側が受け取る = rate が正なら収益）
    払う:     現物と先物の往復コスト（建てるときと畳むとき）、および現物購入資金の金利

    Args:
        rate: 8 時間ごとの資金調達率。
        spot_fee_bp / perp_fee_bp: 片道の手数料（bp）。
        half_spread_bp: 片道の実効スプレッド（bp）。
        borrow_ann: 現物購入に充てる資金の年率調達コスト。
        rebalance_days: 何日ごとに建て直すか（コストの発生頻度）。
    """
    income = rate.copy()                                     # ショート側が受け取る
    funding_per_year = PERIODS_PER_YEAR
    carry_cost = borrow_ann / funding_per_year               # 1 期間あたりの資金コスト

    # 建て直しのコストは、該当する期間にだけ計上する
    per_rebalance = 2 * (spot_fee_bp + perp_fee_bp + half_spread_bp) * 1e-4   # 現物+先物の往復
    periods_per_rebalance = max(1, int(rebalance_days * 3))
    trade_cost = pd.Series(0.0, index=rate.index)
    trade_cost.iloc[::periods_per_rebalance] = per_rebalance

    return (income - carry_cost - trade_cost).dropna()


def stats(r: pd.Series) -> dict:
    if len(r) < 30 or r.std() == 0:
        return {}
    eq = (1 + r).cumprod()
    ann = r.mean() * PERIODS_PER_YEAR
    vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
    monthly = r.resample("ME").sum()
    return {
        "Sharpe": ann / vol if vol > 0 else np.nan,
        "年率": ann,
        "年率ボラ": vol,
        "最大DD": float((eq / eq.cummax() - 1).min()),
        "歪度": float(r.skew()),
        "尖度": float(r.kurtosis()),
        "月次勝率": float((monthly > 0).mean()),
        "最悪の月": float(monthly.min()),
        "期間数": len(r),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="取引所からデータを取得する（要ネットワーク）")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", default="2019-09-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--dir", default="data/raw/funding")
    parser.add_argument("--out", default=None, help="--fetch の保存先（既定は --dir と同じ）")
    parser.add_argument("--spot-fee-bp", type=float, default=10.0, help="現物の片道手数料")
    parser.add_argument("--perp-fee-bp", type=float, default=5.0, help="先物の片道手数料")
    parser.add_argument("--half-spread-bp", type=float, default=2.0)
    parser.add_argument("--borrow-ann", type=float, default=0.03, help="現物購入資金の年率調達コスト")
    parser.add_argument("--rebalance-days", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.out or args.dir)
    if args.fetch:
        data_dir.mkdir(parents=True, exist_ok=True)
        for sym in args.symbols:
            print(f"[取得] {sym} {args.start} 〜 {args.end}")
            df = fetch_funding(sym, args.start, args.end)
            if df.empty:
                print(f"  取得できませんでした（ネットワーク・銘柄名を確認してください）")
                continue
            df.to_parquet(data_dir / f"{sym}.parquet")
            print(f"  {len(df):,} 件  {df.index[0]:%Y-%m-%d} 〜 {df.index[-1]:%Y-%m-%d}"
                  f"  平均 {df['rate'].mean()*1e4:.2f}bp/8h  マイナスの割合 {(df['rate']<0).mean()*100:.1f}%")

    series = load_dir(Path(args.dir))
    if not series:
        raise SystemExit(
            f"{args.dir} に資金調達率データがありません。\n"
            f"ネットワークが通る環境で --fetch を付けて実行してください:\n"
            f"    python scripts/funding_carry.py --fetch --symbols {' '.join(args.symbols)}")

    print(f"\nコスト前提: 現物 {args.spot_fee_bp}bp / 先物 {args.perp_fee_bp}bp / "
          f"スプレッド {args.half_spread_bp}bp / 資金 {args.borrow_ann*100:.1f}%年 / "
          f"{args.rebalance_days}日ごとに建て直し\n")

    for name, rate in series.items():
        r = carry_returns(rate, args.spot_fee_bp, args.perp_fee_bp,
                          args.half_spread_bp, args.borrow_ann, args.rebalance_days)
        print("=" * 72)
        print(f"{name}   {rate.index[0]:%Y-%m-%d} 〜 {rate.index[-1]:%Y-%m-%d}   "
              f"資金調達率がマイナスの割合 {(rate < 0).mean()*100:.1f}%")
        print("=" * 72)
        rows = []
        for label, lo, hi in [("全期間", None, None)] + ERAS:
            seg = r if lo is None else r.loc[lo:hi]
            s = stats(seg)
            if not s:
                continue
            rows.append({"区間": label, **s})
        if not rows:
            print("  データが足りません")
            continue
        t = pd.DataFrame(rows).set_index("区間")
        show = t.copy()
        for c in ("年率", "年率ボラ", "最大DD", "月次勝率", "最悪の月"):
            show[c] = (show[c] * 100).round(1).astype(str) + "%"
        for c in ("Sharpe", "歪度", "尖度"):
            show[c] = show[c].round(2)
        print(show.to_string())

        full = t.loc["全期間"]
        print(f"\n判定:")
        print(f"  {'✗' if full['歪度'] < -0.5 else '○'} 歪度 {full['歪度']:+.2f}"
              f"（−0.5 を下回るなら『たまに大きく負ける』形。キャリーの典型）")
        print(f"  {'✗' if len(rows) > 2 and t.loc[ERAS[1][0], 'Sharpe'] < 0 else '○'} "
              f"弱気相場での成績（時代を分けて符号が反転していないか）")
        print(f"  参考: 信用キャリー 99 年は Sharpe 0.23 / 歪度 −0.98 / 危機で全部マイナス"
              f"（docs/strategy_search.md 67 節）\n")


if __name__ == "__main__":
    main()
