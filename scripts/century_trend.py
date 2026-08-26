#!/usr/bin/env python3
"""第十四次探索: 92 年のデータでトレンド追随の原理そのものを検証する。

プロトコルは docs/strategy_search.md 62 節で**実行前に固定**した。結果は 63 節。

暗号資産の 5.7 年で Sharpe 1.0 が出ても、それが 5.7 年ぶんの幸運なのか普遍的な性質なのかは
区別できない。`arch` 同梱の Fama-French 月次（1926-2018、92 年）と社債利回り（1919-2018、
100 年）はその区別に使える。大恐慌・戦時・70 年代インフレ・リーマンを含む。

**暗号資産戦略のパラメータは一切変更しない。**原理の確認であって改良ではない。

Example:
    python scripts/century_trend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LOOKBACKS = (1, 3, 6, 12)          # 月。時系列モメンタム研究の標準
VOL_WINDOW = 36                    # 月。ボラ正規化の窓
TARGET_VOL = 0.15
ERAS = [("1926-1956", "1926", "1956"), ("1957-1987", "1957", "1987"), ("1988-2018", "1988", "2018")]


def load_factors() -> pd.DataFrame:
    """Fama-French 月次。index が YYYYMM をナノ秒として誤解釈しているので復元する。"""
    import arch.data.frenchdata
    df = arch.data.frenchdata.load()
    ym = df.index.astype("int64").astype(str).str[-6:]          # 192607 のような文字列に戻す
    df = df.copy()
    df.index = pd.to_datetime(ym, format="%Y%m")
    return df / 100.0                                            # % → 小数


def load_credit() -> pd.DataFrame:
    """社債利回り月次から、概算の総リターンを作る。

    債券リターン ≈ 利回り/12 − デュレーション × 利回り変化。
    長期社債なのでデュレーションは 10 年で近似する（厳密ではないが符号と桁は保てる）。
    """
    import arch.data.default
    y = arch.data.default.load() / 100.0
    dur = 10.0
    out = {}
    for col in y.columns:
        out[col] = (y[col].shift(1) / 12.0 - dur * y[col].diff()).dropna()
    return pd.DataFrame(out)


def tsmom(returns: pd.Series, lookback: int) -> pd.Series:
    """時系列モメンタム。過去 N か月の累積リターンの符号方向へ、ボラ調整して建てる。"""
    cum = returns.rolling(lookback).sum()
    vol = returns.rolling(VOL_WINDOW, min_periods=12).std() * np.sqrt(12)
    signal = np.sign(cum)
    size = (TARGET_VOL / vol.replace(0.0, np.nan)).clip(0.0, 3.0)
    return (signal * size).shift(1).fillna(0.0)                  # t 月末に判断、t+1 月に保有


def evaluate(returns: pd.Series, position: pd.Series, cost_bp: float = 10.0) -> pd.Series:
    turnover = (position - position.shift(1)).abs().fillna(0.0)
    return (position * returns - turnover * cost_bp * 1e-4).dropna()


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(12)) if len(x) > 12 and x.std() > 0 else float("nan")


def by_era(returns: pd.Series, position: pd.Series, cost_bp: float = 10.0) -> dict:
    r = evaluate(returns, position, cost_bp)
    out = {"全期間": round(sharpe(r), 2)}
    for label, lo, hi in ERAS:
        seg = r.loc[lo:hi]
        out[label] = round(sharpe(seg), 2) if len(seg) > 24 else np.nan
    return out


def main() -> None:
    ff = load_factors()
    credit = load_credit()
    print(f"Fama-French: {ff.index[0]:%Y-%m} 〜 {ff.index[-1]:%Y-%m}（{len(ff)} か月 = {len(ff)/12:.0f} 年）")
    print(f"社債       : {credit.index[0]:%Y-%m} 〜 {credit.index[-1]:%Y-%m}（{len(credit)} か月）\n")

    series = {
        "Mkt-RF（株式市場）": ff["Mkt-RF"],
        "SMB（小型株）": ff["SMB"],
        "HML（バリュー）": ff["HML"],
        "AAA社債": credit["AAA"],
        "BAA社債": credit["BAA"],
    }

    for name, ret in series.items():
        print(f"=== {name} ===")
        rows = []
        for lb in LOOKBACKS:
            row = {"ルックバック": f"{lb}か月"}
            row.update(by_era(ret, tsmom(ret, lb)))
            rows.append(row)
        # 参考: 買い持ち（ボラ調整）
        vol = ret.rolling(VOL_WINDOW, min_periods=12).std() * np.sqrt(12)
        hold = (TARGET_VOL / vol.replace(0.0, np.nan)).clip(0.0, 3.0).shift(1).fillna(0.0)
        row = {"ルックバック": "買い持ち"}
        row.update(by_era(ret, hold))
        rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    print("=== 判定: 3 時代すべてで Sharpe > 0 か（ルックバック 12 か月）===")
    for name, ret in series.items():
        d = by_era(ret, tsmom(ret, 12))
        eras = [d[e[0]] for e in ERAS]
        ok = all(pd.notna(v) and v > 0 for v in eras)
        print(f"  {name:<18} {eras}  → {'○ 全時代プラス' if ok else '✗ 符号が反転する時代あり'}")


if __name__ == "__main__":
    main()
