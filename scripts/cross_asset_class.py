#!/usr/bin/env python3
"""第十三次探索: 非暗号資産で、パラメータ無変更のまま戦略を検証する。

プロトコルは docs/strategy_search.md 57 節で**実行前に固定**した。結果は 58 節。

データは `arch` パッケージがローカル同梱している実データ（ネットワーク不要）。
S&P500 / NASDAQ 日足 1999-2018、WTI 原油 日足 1986-2019。
**269 試行のどれにも使われていない、資産クラスも年代も独立したデータ**である。

**注意（実装上の罠）**: S&P500 / NASDAQ の `Open` は無調整、`Adj Close` は配当調整済である。
両者を混ぜて `Open(t+1)/AdjClose(t)` のようなリターンを作ると、配当ぶんの系統的な
バイアスが入る（実測で年 +4% ほどずれた）。ここでは **`Adj Close` だけ**を使い、
シグナルは t 日の終値で確定、t→t+1 の終値リターンを取る（1 日ラグ）。

Example:
    python scripts/cross_asset_class.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlgmo.portfolio import LADDER_DAYS, apply_rebalance_band, trend_signal  # noqa: E402

GAIN, VOL_WINDOW, BAND = 1.5, 30, 0.10      # 暗号資産版と同一。動かさない
TARGET_VOL, LEV_CAP = 0.15, 2.0
PPY = 252


def load(name: str) -> pd.Series:
    import arch.data.nasdaq, arch.data.sp500, arch.data.wti  # noqa: F401
    if name == "SP500":
        return arch.data.sp500.load()["Adj Close"].dropna()
    if name == "NASDAQ":
        return arch.data.nasdaq.load()["Adj Close"].dropna()
    return arch.data.wti.load()["DCOILWTICO"].dropna()


def backtest(px: pd.Series, signal: pd.Series, half_spread_bp: float = 1.0) -> pd.Series:
    fwd = px.pct_change().shift(-1)                       # t→t+1（1 日ラグ）
    vol = np.log(px).diff().rolling(20, min_periods=5).std() * np.sqrt(PPY)
    exposure = (signal * (TARGET_VOL / vol.replace(0.0, np.nan)).clip(0.0, LEV_CAP)).clip(-2, 2).fillna(0.0)
    turnover = (exposure - exposure.shift(1)).abs().fillna(0.0)
    return (exposure * fwd - turnover * half_spread_bp * 1e-4).dropna()


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(PPY)) if x.std() > 0 else float("nan")


def sig_trend(px: pd.Series, days: int) -> pd.Series:
    return apply_rebalance_band(
        trend_signal(pd.DataFrame({"close": px}), days, False, GAIN, VOL_WINDOW), BAND)


def sig_ladder(px: pd.Series, horizons=LADDER_DAYS) -> pd.Series:
    parts = [trend_signal(pd.DataFrame({"close": px}), d, False, GAIN, VOL_WINDOW) for d in horizons]
    return apply_rebalance_band(pd.concat(parts, axis=1).mean(axis=1), BAND)


def main() -> None:
    markets = ["SP500", "NASDAQ", "WTI"]
    print("パラメータは暗号資産版と完全に同一（gain 1.5・ボラ窓 30・バンド 0.10）\n")

    print("=== ホライズン別（暗号資産のラダーは 5/14/30/60 日）===")
    rows = []
    for name in markets:
        px = load(name)
        row = {"市場": name, "年数": round(len(px) / PPY, 1)}
        for d in LADDER_DAYS:
            row[f"{d}日"] = round(sharpe(backtest(px, sig_trend(px, d))), 2)
        row["ラダー"] = round(sharpe(backtest(px, sig_ladder(px))), 2)
        row["買い持ち"] = round(sharpe(px.pct_change().dropna()), 2)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== より長いルックバック ===")
    rows = []
    for name in markets:
        px = load(name)
        row = {"市場": name}
        for d in (90, 120, 180, 250):
            row[f"{d}日"] = round(sharpe(backtest(px, sig_trend(px, d))), 2)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 実装の健全性確認: 古典的な 200 日移動平均（ロングのみ）===")
    for name in markets:
        px = load(name)
        s = (px > px.rolling(200).mean()).astype(float)
        print(f"  {name:<8} 200日MA {sharpe(backtest(px, s)):+.2f}   買い持ち {sharpe(px.pct_change().dropna()):+.2f}")

    print("\n=== 46 節の損益分岐（0.4 最低 / 0.7 採用候補）との突き合わせ ===")
    print("各市場で最良のルックバックを使っても足りるか（有利に見積もっても、の意味）")
    for name in markets:
        px = load(name)
        best = max((sharpe(backtest(px, sig_trend(px, d))), d) for d in (5, 14, 30, 60, 90, 120, 180, 250))
        verdict = "採用候補" if best[0] > 0.7 else ("最低ライン通過" if best[0] > 0.4 else "**不足**")
        print(f"  {name:<8} 最良 {best[0]:+.2f}（{best[1]}日）→ {verdict}")


if __name__ == "__main__":
    main()
