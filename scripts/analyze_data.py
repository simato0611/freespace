#!/usr/bin/env python3
"""実データの「情報量」を測る事前分析。

RL を回す前に（そして結果を解釈する前に）確認すべきこと:

1. **実測ボラから損益分岐グロス Sharpe を再計算する** — 設計書 2 節の表を、
   仮定値ではなく手元のデータで引き直す。
2. **1 分足リターンの自己相関** — 単純なモメンタム/リバーサルがどれだけ残っているか。
3. **特徴量の予測力（Ridge 回帰の OOS R²）** — 61 次元の特徴量から、先の
   1/5/15/60 分リターンをどれだけ説明できるか。ウォークフォワードで学習・評価する。
   ここで OOS R² が実質ゼロなら、RL に何を期待しても同じ結論になる
   （RL は予測力を作り出さない。既にある予測力の使い方を最適化するだけ）。
4. **予測に基づく単純戦略のコスト前後 Sharpe** — 予測値をボラでスケールしてポジションに
   するだけの素朴な戦略で、コスト前後の成績を出す。RL の下限ベンチマークになる。

Example:
    python scripts/analyze_data.py --config configs/btc_real.yaml --stride 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.config import load_config  # noqa: E402
from rlgmo.metrics import BARS_PER_YEAR  # noqa: E402
from rlgmo.pipeline import prepare_data  # noqa: E402
from rlgmo.walkforward import make_folds  # noqa: E402

HORIZONS = (1, 5, 15, 60)


def breakeven_table(ann_vol: float, round_trip_bp: float, carry_bp_per_day: float) -> pd.DataFrame:
    """実測ボラから「損益分岐に必要なグロス年率 Sharpe」を保有時間別に計算する。"""
    daily_bp = ann_vol / np.sqrt(365) * 1e4
    rows = []
    for hold_min in (5, 15, 60, 240, 720, 1440):
        trade_vol_bp = daily_bp * np.sqrt(hold_min / 1440)
        cost_bp = round_trip_bp + (carry_bp_per_day if hold_min >= 1440 else 0.0)
        n_per_year = 525_600 / hold_min
        rows.append(
            {
                "保有(分)": hold_min,
                "往復/日": round(1440 / hold_min, 1),
                "1トレードσ(bp)": round(trade_vol_bp, 1),
                "往復コスト(bp)": round(cost_bp, 1),
                "損益分岐グロスSharpe": round(cost_bp / trade_vol_bp * np.sqrt(n_per_year), 1),
            }
        )
    return pd.DataFrame(rows)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """切片付き Ridge の閉形式解。"""
    x1 = np.hstack([x, np.ones((len(x), 1))])
    a = x1.T @ x1 + alpha * np.eye(x1.shape[1])
    a[-1, -1] -= alpha  # 切片は正則化しない
    return np.linalg.solve(a, x1.T @ y)


def ridge_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))]) @ beta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/btc_real.yaml")
    parser.add_argument("--stride", type=int, default=3, help="回帰に使うサンプルの間引き間隔（重複ラベル対策）")
    parser.add_argument("--alpha", type=float, default=300.0, help="Ridge の正則化強度")
    parser.add_argument("--model", default="ridge", choices=["ridge", "gbdt"],
                        help="予測モデル。gbdt は非線形（scikit-learn の HistGradientBoosting）")
    parser.add_argument("--horizons", default="1,5,15,60", help="予測ホライズン（分・カンマ区切り）")
    parser.add_argument("--out", default="runs/analysis")
    args = parser.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(","))
    cfg = load_config(args.config)
    features, meta = prepare_data(cfg)
    close = meta["close"]
    logret = np.log(close).diff()
    ann_vol = float(logret.std() * np.sqrt(BARS_PER_YEAR))

    print("\n===== 1. 実測統計 =====")
    print(f"期間          : {features.index[0]} 〜 {features.index[-1]}  ({len(features):,} バー)")
    print(f"年率ボラ      : {ann_vol:.1%}")
    print(f"1分リターン尖度: {logret.kurtosis():.1f}（正規分布なら 0）")
    print("自己相関      : " + "  ".join(f"lag{k}={logret.autocorr(k):+.4f}" for k in (1, 2, 5, 15, 60)))
    hourly = logret.groupby(features.index.tz_convert("Asia/Tokyo").hour).std() * np.sqrt(BARS_PER_YEAR)
    print(f"日中ボラ(JST) : 最小 {hourly.min():.0%} ({hourly.idxmin()}時) / 最大 {hourly.max():.0%} ({hourly.idxmax()}時)")

    print("\n===== 2. 損益分岐グロス Sharpe（実測ボラで再計算） =====")
    cost = cfg.env.cost
    round_trip = 2 * (cost.half_spread_bp + cost.slippage_bp + cost.taker_fee_bp)
    table = breakeven_table(ann_vol, round_trip, cost.carry_rate_daily * 1e4)
    print(table.to_string(index=False))

    model_name = "Ridge（線形）" if args.model == "ridge" else "HistGradientBoosting（非線形）"
    print(f"\n===== 3. 特徴量の予測力（ウォークフォワード / {model_name}） =====")
    x_all = features.to_numpy(dtype=np.float64)
    folds = make_folds(features.index, cfg.walkforward)
    rows = []
    for horizon in horizons:
        fwd = (np.log(close).shift(-horizon) - np.log(close)).to_numpy()
        vol = logret.rolling(60).std().to_numpy() * np.sqrt(horizon)
        target = np.divide(fwd, vol, out=np.zeros_like(fwd), where=(vol > 0))  # ボラ正規化した先行リターン
        for fold in folds:
            tr = slice(fold.train.start, fold.train.stop - horizon)
            te = fold.test
            xtr, ytr = x_all[tr][:: args.stride], target[tr][:: args.stride]
            xte, yte = x_all[te][:: args.stride], target[te][:: args.stride]
            ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
            xtr, ytr, xte, yte = xtr[ok_tr], ytr[ok_tr], xte[ok_te], yte[ok_te]
            if len(xtr) < 1000 or len(xte) < 1000:
                continue
            if args.model == "ridge":
                pred = ridge_predict(xte, ridge_fit(xtr, ytr, args.alpha))
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor

                model = HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.05, max_depth=4,
                    l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                    random_state=0,
                )
                model.fit(xtr, ytr)
                pred = model.predict(xte)
            ss_res = float(((yte - pred) ** 2).sum())
            ss_tot = float(((yte - yte.mean()) ** 2).sum())
            ic = float(np.corrcoef(pred, yte)[0, 1])
            rows.append({"horizon": horizon, "fold": fold.idx, "oos_r2": 1 - ss_res / ss_tot, "ic": ic})
    pred_df = pd.DataFrame(rows)
    summary = pred_df.groupby("horizon").agg(
        oos_r2_mean=("oos_r2", "mean"), oos_r2_min=("oos_r2", "min"),
        ic_mean=("ic", "mean"), ic_std=("ic", "std"), folds=("fold", "count"))
    print(summary.round(5).to_string())
    print("\n※ IC（予測と実現の相関）から期待できる年率 Sharpe の上限（コスト無視・完全サイジング）:")
    for horizon in horizons:
        sub = pred_df[pred_df.horizon == horizon]
        if sub.empty:
            continue
        ic = sub["ic"].mean()
        n_per_year = 525_600 / horizon
        print(f"  {horizon:>3}分: IC={ic:+.4f} → 上限 Sharpe ≈ {ic * np.sqrt(n_per_year):+.1f}"
              f"  (損益分岐 {table.set_index('保有(分)')['損益分岐グロスSharpe'].get(horizon, float('nan'))})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / f"predictability_{args.model}.csv", index=False)
    table.to_csv(out_dir / "breakeven.csv", index=False)
    print(f"\n出力: {out_dir}/predictability_{args.model}.csv, {out_dir}/breakeven.csv")


if __name__ == "__main__":
    main()
