#!/usr/bin/env python3
"""第十七次探索: 学習モデルの総当たり（開発期間のみ）。

プロトコルと事前予想は docs/strategy_search.md 74 節で**実行前に固定**した。

### 設計の要点

特徴量も増やすので「モデルが効いたのか特徴量が効いたのか」を切り分ける必要がある。
**F1（ラダーと同じ4本）と F2（拡張14本）の 2 列**を作り、Ridge を対照群に置く。

- GBM-F2 > Ridge-F2 なら → 非線形性が効いた
- Ridge-F2 > ルール なら → 特徴量が効いた（モデルではない）
- F1 でどのモデルもルールに勝てないなら → モデルは無意味

### 評価（第十六次で確立した方法）

主指標は**アウトオブサンプルの IC**。実装の巧拙に左右されないため。
**重複を除いてサンプリングして測る**（5日先なら5日おき）。重複したまま測ると
見かけの当てはまりが作れることを第十六次で確認した。
**学習時の当てはまりは一切報告しない。**

Example:
    python scripts/model_zoo.py --era both
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
FWD_DAYS = 5
TRAIN_DAYS = 730          # 学習に使う直近期間
REFIT_DAYS = 60           # 再学習の間隔
MAX_TRAIN = 3000          # 学習点数の上限（計算量の都合）


def era_data(era: str):
    prices = L.load_era(era)
    index = L.align(prices)
    gap, intra = L.returns_frames(prices, index)
    if era == "B":
        index = index[index < L.HOLDOUT]
        prices = {a: df.loc[df.index < L.HOLDOUT] for a, df in prices.items()}
        gap, intra = gap.loc[index], intra.loc[index]
    return prices, index, gap, intra


def zscore(s: pd.Series, w: int) -> pd.Series:
    return ((s - s.rolling(w, min_periods=w // 4).mean())
            / s.rolling(w, min_periods=w // 4).std().replace(0.0, np.nan))


def make_features(df: pd.DataFrame, index: pd.DatetimeIndex,
                  btc: pd.DataFrame | None, extended: bool) -> pd.DataFrame:
    """F1: ラダーと同じ4本 / F2: それに10本足した拡張版。"""
    cols = {f"m{d}": trend_signal(df, max(2, int(d * BPD)), False, 1.5, 30).reindex(index)
            for d in LADDER_DAYS}
    if not extended:
        return pd.DataFrame(cols, index=index).fillna(0.0)

    d = df.reindex(index)
    close, high, low, vol_ = d["close"], d["high"], d["low"], d["volume"]
    lr = np.log(close).diff()
    w_s, w_l = 5 * BPD, 60 * BPD
    cols["vol_ratio"] = (lr.rolling(w_s).std() / lr.rolling(w_l).std().replace(0.0, np.nan)).clip(0, 5)
    cols["vol_level"] = zscore(lr.rolling(w_s).std(), w_l)
    cols["volume_ratio"] = (vol_.rolling(w_s).mean() / vol_.rolling(w_l).mean().replace(0.0, np.nan)).clip(0, 5)
    cols["volume_z"] = zscore(vol_, w_l)
    rng = ((high - low) / close.replace(0.0, np.nan))
    cols["range_ratio"] = (rng.rolling(w_s).mean() / rng.rolling(w_l).mean().replace(0.0, np.nan)).clip(0, 5)
    cols["close_loc"] = (((close - low) / (high - low).replace(0.0, np.nan) - 0.5)
                         .rolling(w_s).mean())
    cols["skew"] = lr.rolling(30 * BPD, min_periods=100).skew()
    if btc is not None:
        cols["btc_trend"] = ladder_signal(btc, BPD, LADDER_DAYS, False, 1.5, 30).reindex(index)
    else:
        cols["btc_trend"] = pd.Series(0.0, index=index)
    jst = index.tz_convert("Asia/Tokyo")
    cols["hour_sin"] = pd.Series(np.sin(2 * np.pi * jst.hour / 24), index=index)
    cols["dow"] = pd.Series(jst.dayofweek / 6.0, index=index)
    return pd.DataFrame(cols, index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_models(seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Ridge（線形）": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=100)),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=50,
                                              random_state=seed, n_jobs=-1),
        "勾配ブースティング": HistGradientBoostingRegressor(max_depth=4, max_iter=150,
                                                   min_samples_leaf=50, learning_rate=0.05,
                                                   random_state=seed),
        "MLP": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=400,
                                          early_stopping=True, random_state=seed)),
    }


def walk_forward(X: pd.DataFrame, y: pd.Series, model_factory, fwd_bars: int,
                 seed: int = 0) -> pd.Series:
    """未来を使わないウォークフォワード。学習は重複ありでよいが、評価は別途重複を除く。"""
    n = len(X)
    pred = np.full(n, np.nan)
    Xv, yv = X.to_numpy(), y.to_numpy()
    warm = TRAIN_DAYS * BPD
    step = REFIT_DAYS * BPD
    rng = np.random.default_rng(seed)

    for start in range(warm, n, step):
        lo = max(0, start - TRAIN_DAYS * BPD)
        idx = np.arange(lo, max(lo, start - fwd_bars))          # 未来漏れ防止
        ok = np.isfinite(yv[idx]) & np.isfinite(Xv[idx]).all(axis=1)
        idx = idx[ok]
        if len(idx) < 300:
            continue
        if len(idx) > MAX_TRAIN:
            idx = rng.choice(idx, size=MAX_TRAIN, replace=False)
        m = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m.fit(Xv[idx], yv[idx])
            except Exception:
                continue
            end = min(start + step, n)
            block = np.arange(start, end)
            good = np.isfinite(Xv[block]).all(axis=1)
            if good.sum() == 0:
                continue
            pred[block[good]] = m.predict(Xv[block][good])
    return pd.Series(pred, index=X.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", default="both", choices=["A", "B", "both"])
    ap.add_argument("--out", default="runs/model_zoo")
    args = ap.parse_args()

    cfg = PortfolioConfig(target_vol_ann=0.15, asset_vol_ann=0.15,
                          cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5,
                                          carry_mode="daily_0600", spread_vol_beta=0.0))
    h = FWD_DAYS * BPD
    eras = ["A", "B"] if args.era == "both" else [args.era]
    all_rows = []

    for era in eras:
        prices, index, gap, intra = era_data(era)
        btc = prices.get("BTC")
        print(f"\n{'='*78}\n{L.ERAS[era]['label']} / 銘柄 {sorted(prices)} / {len(index):,} 本"
              f" / 予測先 {FWD_DAYS} 日\n{'='*78}")

        targets, feats = {}, {}
        for asset, df in prices.items():
            px = df["close"].reindex(index)
            vol = np.log(px).diff().rolling(20 * BPD, min_periods=50).std()
            targets[asset] = ((np.log(px).shift(-h) - np.log(px))
                              / (vol * np.sqrt(h)).replace(0.0, np.nan)).clip(-5, 5)
            feats[asset] = {"F1": make_features(df, index, btc, extended=False),
                            "F2": make_features(df, index, btc, extended=True)}

        # --- 基準: ルール（ラダー）
        rule_sig = {a: apply_rebalance_band(
            ladder_signal(df, BPD, LADDER_DAYS, False, 1.5, 30).reindex(index).fillna(0.0), 0.10)
            for a, df in prices.items()}
        e = L.vol_target(L.asset_exposures(prices, rule_sig, index, cfg, GM), gap, intra, cfg, GM)
        _, mrule = L.evaluate(e, gap, intra, cfg, GM)
        rule_ic = np.mean([
            _ic(pd.Series(ladder_signal(prices[a], BPD, LADDER_DAYS, False, 1.5, 30).reindex(index)),
                targets[a], h) for a in prices])
        all_rows.append(dict(時代=era, 特徴量="F1", モデル="ルール（現行 v2）",
                             IC=rule_ic, Sharpe=mrule["sharpe"], 回転=mrule["turnover_per_day"]))
        print(f"\n基準: ルール（ラダー）  IC {rule_ic:+.3f}  Sharpe {mrule['sharpe']:.3f}\n")

        models = build_models()
        for fset in ("F1", "F2"):
            print(f"--- 特徴量 {fset}（{feats[list(prices)[0]][fset].shape[1]} 本）---")
            for mname, _ in models.items():
                sigs, ics = {}, []
                for asset, df in prices.items():
                    X, y = feats[asset][fset], targets[asset]
                    pred = walk_forward(X, y, lambda mn=mname: build_models()[mn], h)
                    ics.append(_ic(pred, y, h))
                    z = (pred / pred.rolling(60 * BPD, min_periods=200).std().replace(0, np.nan)).clip(-1, 1)
                    sigs[asset] = apply_rebalance_band(z.fillna(0.0), 0.10)
                e = L.vol_target(L.asset_exposures(prices, sigs, index, cfg, GM), gap, intra, cfg, GM)
                _, m = L.evaluate(e, gap, intra, cfg, GM)
                ic = float(np.nanmean(ics))
                all_rows.append(dict(時代=era, 特徴量=fset, モデル=mname, IC=ic,
                                     Sharpe=m["sharpe"], 回転=m["turnover_per_day"]))
                print(f"  {mname:<18} IC {ic:+.3f}   Sharpe {m['sharpe']:+.3f}   "
                      f"回転 {m['turnover_per_day']:.2f}")

    df = pd.DataFrame(all_rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)

    print(f"\n{'='*78}\n両時代のまとめ（IC / Sharpe）\n{'='*78}")
    piv = df.pivot_table(index=["特徴量", "モデル"], columns="時代", values=["IC", "Sharpe"])
    print(piv.round(3).to_string())
    print(f"\n保存: {out}/results.csv")


def _ic(pred: pd.Series, y: pd.Series, h: int) -> float:
    """重複を除いてサンプリングした IC。重複したまま測ると見かけの当てはまりが作れる。"""
    m = np.isfinite(pred) & np.isfinite(y)
    idx = np.where(m)[0][::h]
    if len(idx) < 30:
        return float("nan")
    a, b = pred.to_numpy()[idx], y.to_numpy()[idx]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    main()
