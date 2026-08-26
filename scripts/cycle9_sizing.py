#!/usr/bin/env python3
"""第九次探索: 「1 本のベットの使い方」を変えられるか（開発期間のみ）。

プロトコルと採用条件は docs/strategy_search.md 40 節で**実行前に固定**してある。
結果は 41 節。H6（ホライズン一致度）だけが保留、他は不採用。

- H6 ホライズン一致度を確信度としてサイジングに使う
- H8 確信度が低い局面はフラットにする
- H10 BTC のトレンドを他銘柄に当てる（リードラグ）

Example:
    python scripts/cycle9_sizing.py
"""

import sys
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
import sleeve_lab as L
from rlgmo.costs import CostConfig
from rlgmo.portfolio import PortfolioConfig, LADDER_DAYS, apply_rebalance_band, trend_signal, ladder_signal
from rlgmo.metrics import equity_metrics

GM, BPD = 60, 24
def mkcfg(tv=0.15):
    return PortfolioConfig(target_vol_ann=tv, asset_vol_ann=tv,
        cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5, carry_mode="daily_0600", spread_vol_beta=0.0))

def era_data(era):
    prices = L.load_era(era); index = L.align(prices)
    gap, intra = L.returns_frames(prices, index)
    if era == "B":
        index = index[index < L.HOLDOUT]
        prices = {a: df.loc[df.index < L.HOLDOUT] for a,df in prices.items()}
        gap, intra = gap.loc[index], intra.loc[index]
    return prices, index, gap, intra

def run(prices, index, gap, intra, sig, cfg=None):
    cfg = cfg or mkcfg()
    e = L.vol_target(L.asset_exposures(prices, sig, index, cfg, GM), gap, intra, cfg, GM)
    return L.evaluate(e, gap, intra, cfg, GM)

def horizon_parts(df, gain=1.5, vw=30):
    return pd.concat([trend_signal(df, max(2,int(d*BPD)), False, gain, vw) for d in LADDER_DAYS], axis=1)

def sig_base(prices, rb=0.10):
    return {a: apply_rebalance_band(ladder_signal(df, BPD, LADDER_DAYS, False, 1.5, 30), rb) for a,df in prices.items()}

def sig_agree(prices, power, rb=0.10):
    """H6: 4本の一致度で確信度を作る。agreement = |mean| / (mean|.|) ∈ [0,1]"""
    out={}
    for a,df in prices.items():
        p = horizon_parts(df)
        m = p.mean(axis=1)
        agree = (m.abs() / p.abs().mean(axis=1).replace(0,np.nan)).clip(0,1).fillna(0.0)
        out[a] = apply_rebalance_band(((m/1.5).clip(-1,1) * agree**power).fillna(0.0), rb)
    return out

def sig_threshold(prices, thr, rb=0.10):
    """H8: |シグナル| が閾値未満ならフラット"""
    out={}
    for a,df in prices.items():
        s = ladder_signal(df, BPD, LADDER_DAYS, False, 1.5, 30)
        out[a] = apply_rebalance_band(s.where(s.abs() >= thr, 0.0), rb)
    return out

def sig_leadlag(prices, index, weight, rb=0.10):
    """H10: 自分のトレンドと BTC のトレンドを混ぜる。weight=1 なら完全に BTC 依存"""
    if "BTC" not in prices: return None
    btc = ladder_signal(prices["BTC"], BPD, LADDER_DAYS, False, 1.5, 30).reindex(index)
    out={}
    for a,df in prices.items():
        own = ladder_signal(df, BPD, LADDER_DAYS, False, 1.5, 30).reindex(index)
        out[a] = apply_rebalance_band(((1-weight)*own + weight*btc).fillna(0.0), rb)
    return out

print("="*78)
print("H6 ホライズン一致度による確信度サイジング（power=0 が現行 v2）")
print("="*78)
res={}
for era in ["A","B"]:
    prices, index, gap, intra = era_data(era)
    _, m0 = run(prices,index,gap,intra, sig_base(prices))
    row={"power=0(v2)": (m0["sharpe"], m0["turnover_per_day"])}
    for pw in (0.5, 1.0, 2.0):
        _, m = run(prices,index,gap,intra, sig_agree(prices, pw))
        row[f"power={pw}"] = (m["sharpe"], m["turnover_per_day"])
    res[era]=row
print(pd.DataFrame({e:{k:round(v[0],3) for k,v in r.items()} for e,r in res.items()}).to_string())
print("\n回転/日:")
print(pd.DataFrame({e:{k:round(v[1],2) for k,v in r.items()} for e,r in res.items()}).to_string())

print("\n"+"="*78)
print("H8 確信度の閾値（thr=0 が現行 v2 = 常時在場）")
print("="*78)
res={}
for era in ["A","B"]:
    prices, index, gap, intra = era_data(era)
    row={}
    for thr in (0.0, 0.10, 0.20, 0.30, 0.40):
        _, m = run(prices,index,gap,intra, sig_threshold(prices, thr))
        row[f"thr={thr:.2f}"] = (m["sharpe"], m["turnover_per_day"], m["gross_exposure"])
    res[era]=row
print(pd.DataFrame({e:{k:round(v[0],3) for k,v in r.items()} for e,r in res.items()}).to_string())
print("\n平均グロス建玉:")
print(pd.DataFrame({e:{k:round(v[2],3) for k,v in r.items()} for e,r in res.items()}).to_string())

print("\n"+"="*78)
print("H10 BTC リードラグ（weight=0 が現行 v2 = 自分のトレンドのみ）")
print("="*78)
res={}
for era in ["A","B"]:
    prices, index, gap, intra = era_data(era)
    row={}
    for w in (0.0, 0.15, 0.30, 0.50, 1.0):
        s = sig_leadlag(prices, index, w)
        _, m = run(prices,index,gap,intra, s)
        row[f"w={w:.2f}"] = (m["sharpe"], m["turnover_per_day"])
    res[era]=row
print(pd.DataFrame({e:{k:round(v[0],3) for k,v in r.items()} for e,r in res.items()}).to_string())
