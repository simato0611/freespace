#!/usr/bin/env python3
"""第十一次探索: パラメータ選択に当てはめが無かったかを全数検査する。

結論は docs/strategy_search.md 49 節。gain/vol_window/band の格子 64 構成を全部回すと
**全構成が両時代で勝ち**、現行値は上位 31%/41% の中位だった。つまりこの 3 つに関しては
当てはめの余地が実質的に無かった。格子の範囲（時代B 1.56〜1.91）は、ライブで期待すべき
水準の予測区間として使える。

Example:
    python scripts/param_grid_audit.py
"""

import sys, itertools
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
import sleeve_lab as L
from rlgmo.costs import CostConfig
from rlgmo.portfolio import PortfolioConfig, apply_rebalance_band, ladder_signal
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

def raw_exposure(prices, index, cfg, gain, vw, rb):
    sig = {a: apply_rebalance_band(ladder_signal(df, BPD, (5,14,30,60), False, gain, vw), rb)
           for a,df in prices.items()}
    return L.asset_exposures(prices, sig, index, cfg, GM)

GAINS  = (1.0, 1.5, 2.0, 2.5)
VWS    = (15, 30, 45, 60)
BANDS  = (0.05, 0.10, 0.15, 0.20)

for era in ["A","B"]:
    prices, index, gap, intra = era_data(era); cfg = mkcfg()
    print("="*74); print(f"{L.ERAS[era]['label']}"); print("="*74)

    # --- 格子上の全構成を個別に評価（診断: 選択の運はどれだけあるか）
    grid, raws = [], {}
    for g, v, b in itertools.product(GAINS, VWS, BANDS):
        e_raw = raw_exposure(prices, index, cfg, g, v, b)
        raws[(g,v,b)] = e_raw
        e = L.vol_target(e_raw, gap, intra, cfg, GM)
        r, m = L.evaluate(e, gap, intra, cfg, GM)
        grid.append(dict(gain=g, vw=v, band=b, Sharpe=m["sharpe"], 回転=m["turnover_per_day"]))
    G = pd.DataFrame(grid)
    cur = G[(G.gain==1.5)&(G.vw==30)&(G.band==0.10)]["Sharpe"].iloc[0]
    print(f"格子 {len(G)} 構成の Sharpe 分布:")
    print(f"  現行(1.5/30/0.10) {cur:.3f}  |  中央値 {G.Sharpe.median():.3f}  平均 {G.Sharpe.mean():.3f}")
    print(f"  最小 {G.Sharpe.min():.3f}  最大 {G.Sharpe.max():.3f}  標準偏差 {G.Sharpe.std():.3f}")
    print(f"  現行の順位: {int((G.Sharpe > cur).sum())+1} / {len(G)}  (上位 {100*(G.Sharpe>cur).mean():.0f}%)")

    # --- H11: 建玉レベルでアンサンブル平均
    ens_raw = sum(raws.values())/len(raws)
    e = L.vol_target(ens_raw, gap, intra, cfg, GM)
    r_ens, m_ens = L.evaluate(e, gap, intra, cfg, GM)
    print(f"\nH11 アンサンブル(64構成の建玉平均): Sharpe {m_ens['sharpe']:.3f}  回転 {m_ens['turnover_per_day']:.2f}")
    print(f"   現行との差 {m_ens['sharpe']-cur:+.3f}  /  格子平均との差 {m_ens['sharpe']-G.Sharpe.mean():+.3f}")

    # 格子の広さを変えた感度
    for label, sub in [("狭い格子(3^3=27)", list(itertools.product((1.0,1.5,2.0),(15,30,45),(0.05,0.10,0.15)))),
                       ("gainのみ固定1.5", list(itertools.product((1.5,),VWS,BANDS)))]:
        ens2 = sum(raws[k] for k in sub)/len(sub)
        _, m2 = L.evaluate(L.vol_target(ens2, gap,intra,cfg,GM), gap,intra,cfg,GM)
        print(f"   {label}: {m2['sharpe']:.3f}")

    # --- H12: バンド位相をずらしたサブポートフォリオ
    def band_offset(sig, width, start):
        v = np.array(sig.to_numpy(), dtype=float, copy=True); cur_=0.0
        for i,t in enumerate(v):
            if i < start: v[i]=0.0; continue
            if not np.isfinite(t): t=cur_
            if abs(t-cur_) > width: cur_=float(t)
            v[i]=cur_
        return pd.Series(v, index=sig.index)
    base_sig = {a: ladder_signal(df, BPD, (5,14,30,60), False, 1.5, 30) for a,df in prices.items()}
    subs=[]
    for off in (0, 6, 12, 18):
        sig = {a: band_offset(s, 0.10, off) for a,s in base_sig.items()}
        subs.append(L.asset_exposures(prices, sig, index, cfg, GM))
    e12 = L.vol_target(sum(subs)/len(subs), gap,intra,cfg,GM)
    _, m12 = L.evaluate(e12, gap,intra,cfg,GM)
    print(f"\nH12 バンド位相ずらし(4本平均): Sharpe {m12['sharpe']:.3f}  現行との差 {m12['sharpe']-cur:+.3f}")
    print()
