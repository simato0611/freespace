#!/usr/bin/env python3
"""封印していたホールドアウト期間で、候補戦略を**一度だけ**評価する。

探索（`signal_survey.py` / ウォークフォワード）で候補を決めたあと、最後に一回だけ走らせる。
ここで良い結果が出なかったからといって、戻って設計をいじってはいけない。それをやると
ホールドアウトはもうホールドアウトではなくなる。

評価する候補:
    buy_hold        ボラターゲット付きの買い持ち（ベンチマーク）
    trend_long      ロングオンリー時系列モメンタム（探索で選ばれたルール）
    trend_ls        ロング・ショート両建ての同ルール（比較用）
    rl_ensemble     ウォークフォワードで学習した PPO アンサンブル（--models 指定時）

Example:
    python scripts/final_holdout.py --config configs/btc_trend.yaml \
        --holdout-start 2025-07-01 --lookback-days 14 --models "runs/btc_trend_final/*.pt" \
        --n-trials 70
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.agents.ppo import PPOAgent  # noqa: E402
from rlgmo.backtest import ensemble_policy, flat_policy, long_policy, run_policy, trend_policy  # noqa: E402
from rlgmo.config import load_config  # noqa: E402
from rlgmo.data.gmo_klines import load_ohlcv  # noqa: E402
from rlgmo.features import build_features  # noqa: E402
from rlgmo.metrics import deflated_sharpe, infer_bars_per_year, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/btc_trend.yaml")
    parser.add_argument("--holdout-start", default="2025-07-01")
    parser.add_argument("--holdout-end", default=None)
    parser.add_argument("--lookback-days", type=float, default=14.0)
    parser.add_argument("--models", default=None, help="RL モデルの glob（省略時は RL を評価しない）")
    parser.add_argument("--n-trials", type=int, default=70,
                        help="探索で試した総本数。Deflated Sharpe の割り引きに使う（正直に数える）")
    parser.add_argument("--out", default="runs/holdout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # ホールドアウトを含めて読み込む（config の end はここでは無視する）
    ohlcv = load_ohlcv(cfg.data.path).loc[cfg.data.start :]
    if args.holdout_end:
        ohlcv = ohlcv.loc[: args.holdout_end]
    features, meta = build_features(ohlcv, cfg.features)

    start = pd.Timestamp(args.holdout_start, tz="UTC")
    warmup = int(args.lookback_days * 1440 / cfg.features.base_minutes) + 60  # シグナルの助走
    positions = features.index.searchsorted(start)
    lo = max(0, positions - warmup)
    features, meta = features.iloc[lo:], meta.iloc[lo:]
    eval_from = int(features.index.searchsorted(start))

    env_cfg = dataclasses.replace(cfg.env, randomize_costs=False)
    bar_minutes = cfg.features.base_minutes
    lookback_bars = int(args.lookback_days * 1440 / bar_minutes)
    print(f"[holdout] {features.index[eval_from]} 〜 {features.index[-1]} "
          f"({len(features) - eval_from:,} バー / {bar_minutes} 分足 / 助走 {warmup} バー)")

    from rlgmo.env import TradingEnv

    def make() -> TradingEnv:
        return TradingEnv(features, meta, env_cfg, training=False)

    candidates = {
        "buy_hold": lambda env: long_policy(cfg.env.actions),
        "flat": lambda env: flat_policy(cfg.env.actions),
        "trend_long": lambda env: trend_policy(env, lookback_bars, long_only=True),
        "trend_ls": lambda env: trend_policy(env, lookback_bars, long_only=False),
    }
    if args.models:
        paths = sorted(glob.glob(args.models))
        if not paths:
            raise SystemExit(f"モデルが見つかりません: {args.models}")
        agents = [PPOAgent.load(p) for p in paths]
        print(f"[holdout] RL アンサンブル: {len(agents)} モデル")
        candidates["rl_ensemble"] = lambda env: ensemble_policy(agents, cfg.env.actions, cfg.train.confidence)

    rows, records = {}, {}
    for name, factory in candidates.items():
        env = make()
        record = run_policy(env, factory(env), start=eval_from)
        metrics = summarize(record["equity"], record["position"],
                            record["trade_cost"] + record["carry_cost"], n_trials=args.n_trials)
        bars_per_year = infer_bars_per_year(pd.DatetimeIndex(record.index))
        rows[name] = {
            "リターン": metrics["total_return"], "Sharpe": metrics["sharpe"],
            "最大DD": metrics["max_drawdown"], "年率ボラ": metrics["ann_vol"],
            "回転/日": metrics["turnover_per_day"], "平均|建玉|": metrics["exposure"],
            "コスト/年": metrics.get("cost_drag_ann", float("nan")),
            "日数": metrics["days"],
        }
        records[name] = record

    bench = np.log(records["buy_hold"]["equity"]).diff().dropna()
    for name, record in records.items():
        ret = np.log(record["equity"]).diff().dropna()
        aligned = pd.concat([ret.rename("s"), bench.rename("b")], axis=1).dropna()
        beta = float(aligned["s"].cov(aligned["b"]) / aligned["b"].var()) if aligned["b"].var() > 0 else float("nan")
        residual = aligned["s"] - beta * aligned["b"]
        periods = infer_bars_per_year(pd.DatetimeIndex(record.index))
        rows[name]["β"] = beta
        rows[name]["情報比"] = float(residual.mean() / residual.std() * np.sqrt(periods)) if residual.std() > 0 else 0.0
        rows[name]["DSR_p"] = deflated_sharpe(
            rows[name]["Sharpe"], len(ret), args.n_trials, trial_sharpe_std_ann=0.8, bars_per_year=periods)

    table = pd.DataFrame(rows).T
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "holdout_metrics.csv")
    for name, record in records.items():
        record.to_csv(out_dir / f"holdout_{name}.csv")

    show = table.copy()
    for col in ("リターン", "最大DD", "年率ボラ", "平均|建玉|", "コスト/年"):
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    for col in ("Sharpe", "回転/日", "β", "情報比", "DSR_p", "日数"):
        show[col] = show[col].round(2)
    print("\n" + show.to_string())
    print(f"\n試行回数 {args.n_trials} で割り引いた Deflated Sharpe を併記している。"
          "\n0.95 以上でなければ「多重検定を考慮すると有意とは言えない」。")
    print(json.dumps({"holdout_start": args.holdout_start, "n_trials": args.n_trials}, ensure_ascii=False))
    print(f"出力: {out_dir}/holdout_metrics.csv")


if __name__ == "__main__":
    main()
