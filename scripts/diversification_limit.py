#!/usr/bin/env python3
"""第十次探索: 分散の効き方を較正し、商品を足す価値を定量化する。

結論は docs/strategy_search.md 43〜46 節。分散に効くのは価格の相関ではなく
**戦略リターンの相関**であり、暗号資産では 0.37〜0.44 で有効独立ベット数は 2.0 しかない。

ただし暗号資産の単独トレンド Sharpe 1.13 は異常に高く、弱い商品を等リスクで足すと
希薄化の損が分散の利得を上回る。損益分岐は単独 Sharpe 0.4〜0.8。

Example:
    python scripts/diversification_limit.py
"""

import sys, itertools
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
import sleeve_lab as L
from rlgmo.costs import CostConfig
from rlgmo.portfolio import PortfolioConfig, apply_rebalance_band, ladder_signal
from rlgmo.metrics import equity_metrics

GM, BPD = 60, 24
cfg = PortfolioConfig(target_vol_ann=0.15, asset_vol_ann=0.15,
    cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5, carry_mode="daily_0600", spread_vol_beta=0.0))

def single_asset_returns(era):
    """各銘柄で単独にトレンド戦略を回し、その戦略リターンを返す。
    分散に効くのは価格の相関ではなく**戦略リターンの相関**である。"""
    prices, index, gap, intra = None, None, None, None
    prices = L.load_era(era); index = L.align(prices)
    gap, intra = L.returns_frames(prices, index)
    if era == "B":
        index = index[index < L.HOLDOUT]
        prices = {a: df.loc[df.index < L.HOLDOUT] for a,df in prices.items()}
        gap, intra = gap.loc[index], intra.loc[index]
    out = {}
    for a, df in prices.items():
        sig = {a: apply_rebalance_band(ladder_signal(df, BPD, (5,14,30,60), False, 1.5, 30), 0.10)}
        e = L.vol_target(L.asset_exposures({a: df}, sig, index, cfg, GM), gap[[a]], intra[[a]], cfg, GM)
        r, m = L.evaluate(e, gap[[a]], intra[[a]], cfg, GM)
        out[a] = r
    return pd.DataFrame(out)

def sharpe(x, ppy=365):
    return x.mean()/x.std()*np.sqrt(ppy) if x.std() > 0 else np.nan

for era in ["A","B"]:
    R = single_asset_returns(era).resample("1D").sum()
    n = R.shape[1]
    s1 = np.array([sharpe(R[c]) for c in R.columns])
    C = R.corr()
    rho = (C.to_numpy().sum() - n) / (n*(n-1))
    print("="*72); print(f"{L.ERAS[era]['label']} / {n} 銘柄"); print("="*72)
    print(f"単独銘柄の戦略 Sharpe: 平均 {s1.mean():.2f}  範囲 {s1.min():.2f}〜{s1.max():.2f}")
    print(f"**戦略リターン**の平均相関 rho = {rho:.3f}")
    print(f"有効独立ベット数 N_eff = N/(1+(N-1)rho) = {n/(1+(n-1)*rho):.2f}")

    # 実測: 銘柄数ごとの等リスク合成 Sharpe（全組み合わせ）
    rows=[]
    for k in range(1, n+1):
        vals=[]
        combos = list(itertools.combinations(R.columns, k))
        if len(combos) > 60:
            rs = np.random.default_rng(0); combos=[combos[i] for i in rs.choice(len(combos),60,replace=False)]
        for c in combos:
            sub = R[list(c)]
            w = 1.0/sub.std().replace(0,np.nan)           # 等リスク
            vals.append(sharpe((sub*w).sum(axis=1)))
        pred = s1.mean()*np.sqrt(k/(1+(k-1)*rho))          # 理論値
        rows.append(dict(銘柄数=k, 実測=round(np.nanmean(vals),3), 理論=round(pred,3),
                         最小=round(np.nanmin(vals),2), 最大=round(np.nanmax(vals),2)))
    t = pd.DataFrame(rows)
    print("\n銘柄数 vs 合成 Sharpe（等リスク・全組み合わせ平均）")
    print(t.to_string(index=False))
    err = (t["実測"]-t["理論"]).abs().mean()
    print(f"理論式の当てはまり: 平均絶対誤差 {err:.3f}")

    if era=="B":
        print("\n" + "="*72)
        print("投影: 既存 5 銘柄に、相関 rho' の商品を K 本足したら Sharpe はどうなるか")
        print("="*72)
        print("（前提: 追加商品の単独 Sharpe は暗号資産と同じ {:.2f}。".format(s1.mean()))
        print(" 既存 5 本の内部相関は実測 {:.2f} のまま、追加分との相関を rho' とする）".format(rho))
        base_n = 5
        rows=[]
        for K in (0, 3, 5, 10, 15, 20):
            row={"追加本数": K}
            for rp in (0.0, 0.05, 0.10, 0.20):
                N = base_n + K
                # ブロック相関行列の有効ベット数
                Cm = np.full((N,N), 0.0)
                Cm[:base_n,:base_n] = rho; Cm[base_n:,base_n:] = rp
                Cm[:base_n,base_n:] = rp;  Cm[base_n:,:base_n] = rp
                np.fill_diagonal(Cm, 1.0)
                w = np.ones(N)/N
                neff = 1.0/ (w @ Cm @ w)
                row[f"rho'={rp:.2f}"] = round(s1.mean()*np.sqrt(neff), 2)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
    print()
