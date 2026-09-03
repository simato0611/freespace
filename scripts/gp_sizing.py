#!/usr/bin/env python3
"""第十六次探索: ガウス過程回帰は上乗せできるか（開発期間のみ）。

プロトコルと事前予想は docs/strategy_search.md 70 節で**実行前に固定**した。

- **H21** GP の事後平均を建玉に使う → RL と同じ轍と予想（3 節・30 節）
- **H22** GP の**事後分散**でサイジングを絞る → 本命。H6（ホライズン一致度）の原理的な版

GP は学習点数の 3 乗で計算量が増えるため、1 時間足をそのまま使えない。
直近 1 年から 800 点を無作為抽出して学習し、1 か月ごとに再学習する。
**特徴量はラダーと同じ 4 本に固定する**（情報を足すと何が効いたか分からなくなる）。

Example:
    python scripts/gp_sizing.py --era both
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sleeve_lab as L  # noqa: E402
from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.portfolio import (  # noqa: E402
    LADDER_DAYS, PortfolioConfig, apply_rebalance_band, ladder_signal, trend_signal,
)

GM, BPD = 60, 24
N_TRAIN = 800            # GP の学習点数。3 乗で効くのでここが上限
REFIT_DAYS = 30          # 再学習の間隔
LOOKBACK_DAYS = 365      # 学習に使う直近期間


def era_data(era: str):
    prices = L.load_era(era)
    index = L.align(prices)
    gap, intra = L.returns_frames(prices, index)
    if era == "B":
        index = index[index < L.HOLDOUT]
        prices = {a: df.loc[df.index < L.HOLDOUT] for a, df in prices.items()}
        gap, intra = gap.loc[index], intra.loc[index]
    return prices, index, gap, intra


def features(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """ラダーと同じ 4 本。新しい情報は足さない。"""
    cols = {f"m{d}": trend_signal(df, max(2, int(d * BPD)), False, 1.5, 30).reindex(index).fillna(0.0)
            for d in LADDER_DAYS}
    return pd.DataFrame(cols, index=index)


def gp_walk_forward(X: pd.DataFrame, y: pd.Series, seed: int = 0, fwd_bars: int = 1
                    ) -> tuple[pd.Series, pd.Series]:
    """未来を使わないウォークフォワードで GP を回し、事後平均と事後標準偏差を返す。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    rng = np.random.default_rng(seed)
    n = len(X)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    step = REFIT_DAYS * BPD
    warm = LOOKBACK_DAYS * BPD                     # 助走。ここまでは予測しない
    Xv, yv = X.to_numpy(), y.to_numpy()

    for start in range(warm, n, step):
        lo = max(0, start - LOOKBACK_DAYS * BPD)
        # 目的変数が先 fwd_bars 本を見るので、その分だけ学習に使える範囲を切る（未来漏れ防止）
        idx = np.arange(lo, max(lo, start - fwd_bars))
        ok = np.isfinite(yv[idx]) & np.isfinite(Xv[idx]).all(axis=1)
        idx = idx[ok]
        if len(idx) < 200:
            continue
        take = rng.choice(idx, size=min(N_TRAIN, len(idx)), replace=False)
        Xtr, ytr = Xv[take], yv[take]
        ys = ytr.std()
        if ys == 0 or not np.isfinite(ys):
            continue
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                  + WhiteKernel(1.0, (1e-3, 1e2)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=0, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(Xtr, ytr / ys)
        end = min(start + step, n)
        block = np.arange(start, end)
        good = np.isfinite(Xv[block]).all(axis=1)
        if good.sum() == 0:
            continue
        m, s = gp.predict(Xv[block][good], return_std=True)
        mu[block[good]] = m
        sd[block[good]] = s
    return pd.Series(mu, index=X.index), pd.Series(sd, index=X.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", default="both", choices=["A", "B", "both"])
    ap.add_argument("--fwd-days", type=float, default=5.0,
                    help="予測先の日数。**この戦略の平均保有は約37日なので、"
                         "次バー（1時間）を予測させるのは問題の立て方が誤り**")
    ap.add_argument("--out", default="runs/gp_sizing")
    args = ap.parse_args()

    cfg = PortfolioConfig(target_vol_ann=0.15, asset_vol_ann=0.15,
                          cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5,
                                          carry_mode="daily_0600", spread_vol_beta=0.0))
    eras = ["A", "B"] if args.era == "both" else [args.era]
    rows = []

    for era in eras:
        prices, index, gap, intra = era_data(era)
        print(f"\n{'='*74}\n{L.ERAS[era]['label']} / 銘柄 {sorted(prices)} / {len(index):,} 本\n{'='*74}")

        base_sig, gp_mean_sig, gp_conf_sig = {}, {}, {}
        for asset, df in prices.items():
            X = features(df, index)
            # 予測先は保有期間に合わせる。次バーを当てる問題ではない（70 節の設計欠陥を修正）
            px = df["close"].reindex(index)
            h = max(1, int(args.fwd_days * BPD))
            vol = np.log(px).diff().rolling(20 * BPD, min_periods=50).std()
            y = ((np.log(px).shift(-h) - np.log(px))
                 / (vol * np.sqrt(h)).replace(0.0, np.nan)).clip(-5, 5)
            mu, sd = gp_walk_forward(X, y, fwd_bars=h)

            ladder = ladder_signal(df, BPD, LADDER_DAYS, False, 1.5, 30).reindex(index)
            base_sig[asset] = apply_rebalance_band(ladder.fillna(0.0), 0.10)

            # H21: 事後平均をそのままシグナルに使う
            z = (mu / mu.rolling(30 * BPD, min_periods=100).std().replace(0, np.nan)).clip(-1, 1)
            gp_mean_sig[asset] = apply_rebalance_band(z.fillna(0.0), 0.10)

            # H22: ラダーはそのまま。事後分散で確信度を作って掛ける
            #      分散が小さい（自信がある）ほど大きく建てる
            rel = sd / sd.rolling(90 * BPD, min_periods=200).median().replace(0, np.nan)
            conf = (1.0 / rel.clip(0.5, 2.0)).clip(0.3, 1.5).fillna(1.0)
            gp_conf_sig[asset] = apply_rebalance_band((ladder * conf).clip(-1, 1).fillna(0.0), 0.10)

            print(f"  {asset:<6} GP 予測できた割合 {np.isfinite(mu).mean()*100:5.1f}%  "
                  f"事後sd 中央値 {np.nanmedian(sd):.3f}")

        for name, sig in (("v2 ラダー（現行）", base_sig),
                          ("H21 GP 事後平均", gp_mean_sig),
                          ("H22 GP 事後分散でサイジング", gp_conf_sig)):
            e = L.vol_target(L.asset_exposures(prices, sig, index, cfg, GM), gap, intra, cfg, GM)
            _, m = L.evaluate(e, gap, intra, cfg, GM)
            rows.append(dict(時代=era, 構成=name, Sharpe=m["sharpe"], 年率=m["cagr"],
                             最大DD=m["max_drawdown"], 回転=m["turnover_per_day"]))

        t = pd.DataFrame([r for r in rows if r["時代"] == era]).drop(columns=["時代"]).set_index("構成")
        show = t.copy()
        for c in ("年率", "最大DD"):
            show[c] = (show[c] * 100).round(1).astype(str) + "%"
        for c in ("Sharpe", "回転"):
            show[c] = show[c].round(3)
        print()
        print(show.to_string())
        base = t.loc["v2 ラダー（現行）", "Sharpe"]
        for name in ("H21 GP 事後平均", "H22 GP 事後分散でサイジング"):
            print(f"  {name}: v2 との差 {t.loc[name,'Sharpe']-base:+.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(f"\n保存: {out}/results.csv")


if __name__ == "__main__":
    main()
