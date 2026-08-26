#!/usr/bin/env python3
"""第十二次探索: 出来高と足内構造に情報があるか（結論: 無い）。

結論は docs/strategy_search.md 53 節。volume と足内構造は手元 OHLCV の最後の未使用データ
だったが、OBV 型はトレンドとの相関 0.5〜0.6（モメンタムのノイズ版）、終値位置は時代間で
不整合、条件づけは 12 通り試して両時代同符号の改善がゼロだった。

Example:
    python scripts/volume_features.py
"""

import sys
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
import sleeve_lab as L
from rlgmo.costs import CostConfig
from rlgmo.portfolio import PortfolioConfig, apply_rebalance_band, ladder_signal
GM, BPD = 60, 24
cfg = PortfolioConfig(target_vol_ann=0.15, asset_vol_ann=0.15,
    cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5, carry_mode="daily_0600", spread_vol_beta=0.0))

def era_data(era):
    prices = L.load_era(era); index = L.align(prices)
    gap, intra = L.returns_frames(prices, index)
    if era == "B":
        index = index[index < L.HOLDOUT]
        prices = {a: df.loc[df.index < L.HOLDOUT] for a,df in prices.items()}
        gap, intra = gap.loc[index], intra.loc[index]
    return prices, index, gap, intra

def norm(s, w=30*24):
    """ローリング標準化して [-1,1] に収める（トレンドシグナルと同じ土俵に乗せる）"""
    m = s.rolling(w, min_periods=w//4).mean(); sd = s.rolling(w, min_periods=w//4).std()
    return ((s-m)/sd.replace(0,np.nan)/1.5).clip(-1,1).fillna(0.0)

def f_obv(df, days):
    r = np.log(df["close"]).diff()
    signed = np.sign(r) * df["volume"]
    return norm(signed.rolling(int(days*BPD), min_periods=int(days*BPD)//2).sum())

def f_close_loc(df, days):
    rng = (df["high"]-df["low"]).replace(0,np.nan)
    loc = ((df["close"]-df["low"])/rng - 0.5)
    return norm(loc.rolling(int(days*BPD), min_periods=int(days*BPD)//2).mean())

def c_volume(df, days):
    """出来高の増加度 → 0〜1 の乗数"""
    v = df["volume"]
    ratio = v.rolling(int(days*BPD)).mean() / v.rolling(int(days*BPD*6)).mean().replace(0,np.nan)
    return ratio.clip(0.3, 2.0).fillna(1.0)

def c_illiq(df, days):
    """Amihud 型の非流動性 → 0〜1 の乗数"""
    r = np.log(df["close"]).diff().abs()
    il = (r / df["volume"].replace(0,np.nan)).rolling(int(days*BPD)).mean()
    med = il.rolling(int(days*BPD*6), min_periods=int(days*BPD)).median()
    return (il/med.replace(0,np.nan)).clip(0.3,2.0).fillna(1.0)

def c_range(df, days):
    rng = (df["high"]-df["low"])/df["close"]
    ratio = rng.rolling(int(days*BPD)).mean()/rng.rolling(int(days*BPD*6)).mean().replace(0,np.nan)
    return ratio.clip(0.3,2.0).fillna(1.0)

def run_sig(prices, index, gap, intra, sig):
    e = L.vol_target(L.asset_exposures(prices, sig, index, cfg, GM), gap, intra, cfg, GM)
    return L.evaluate(e, gap, intra, cfg, GM)

for era in ["A","B"]:
    prices, index, gap, intra = era_data(era)
    base_sig = {a: apply_rebalance_band(ladder_signal(df,BPD,(5,14,30,60),False,1.5,30), 0.10) for a,df in prices.items()}
    r0, m0 = run_sig(prices,index,gap,intra, base_sig)
    print("="*76); print(f"{L.ERAS[era]['label']}   v2 基準 Sharpe {m0['sharpe']:.3f}"); print("="*76)

    print("--- H13a/H14a 単体シグナル（16 節の条件で判定）---")
    rows=[]
    for name, fn in [("H13a OBV型", f_obv), ("H14a 終値位置", f_close_loc)]:
        for days in (3, 7, 14):
            sig = {a: apply_rebalance_band(fn(df, days), 0.10) for a,df in prices.items()}
            r,m = run_sig(prices,index,gap,intra, sig)
            d = pd.DataFrame({"a":r0,"b":r}).resample("1D").sum()
            rows.append(dict(シグナル=f"{name} {days}日", Sharpe=round(m["sharpe"],2),
                             v2相関=round(d["a"].corr(d["b"]),2), 回転=round(m["turnover_per_day"],2)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- H13b/H13c/H14b トレンドへの条件づけ（乗数）---")
    rows=[]
    for name, fn in [("H13b 出来高増加", c_volume), ("H13c 非流動性", c_illiq), ("H14b レンジ拡大", c_range)]:
        for days in (3, 7):
            for sign in (+1, -1):
                sig={}
                for a,df in prices.items():
                    mult = fn(df, days)
                    mult = mult if sign>0 else (1.0/mult.replace(0,np.nan)).clip(0.3,2.0).fillna(1.0)
                    s = ladder_signal(df,BPD,(5,14,30,60),False,1.5,30) * mult
                    sig[a] = apply_rebalance_band(s.clip(-1,1), 0.10)
                r,m = run_sig(prices,index,gap,intra, sig)
                rows.append(dict(条件=f"{name} {days}日 {'順' if sign>0 else '逆'}",
                                 Sharpe=round(m["sharpe"],2), 差=round(m["sharpe"]-m0["sharpe"],3),
                                 回転=round(m["turnover_per_day"],2)))
    t=pd.DataFrame(rows); print(t.to_string(index=False))
    print()
