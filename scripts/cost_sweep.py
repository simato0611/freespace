#!/usr/bin/env python3
"""コスト感度分析: 学習済みモデルを、スプレッド仮定を変えて再評価する。

「今の戦略は、コストがいくらなら成立するのか」を直接測る。GMO レバレッジは
**取引手数料が無料**なので、成行（テイカー）をやめて指値（メイカー）で建てられれば
実効スプレッドは大きく下がる。その効果を定量化するためのもの。

Example:
    python scripts/cost_sweep.py --config configs/btc_real_h240.yaml \
        --run-dir runs/btc_real_h240 --spreads 2.5,1.0,0.5,0.25,0.0
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.agents.ppo import PPOAgent  # noqa: E402
from rlgmo.backtest import ensemble_policy  # noqa: E402
from rlgmo.config import load_config  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.pipeline import evaluate_policy, make_env, prepare_data  # noqa: E402
from rlgmo.walkforward import make_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--spreads", default="2.5,1.0,0.5,0.25,0.0",
                        help="片道コスト(bp)の候補（スプレッド/2 + スリッページの合計）")
    parser.add_argument("--out", default="runs/cost_sweep.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features, meta = prepare_data(cfg)
    folds = make_folds(features.index, cfg.walkforward)
    rows = []
    for one_way_bp in [float(x) for x in args.spreads.split(",")]:
        cost = dataclasses.replace(cfg.env.cost, half_spread_bp=one_way_bp, slippage_bp=0.0)
        env_cfg = dataclasses.replace(cfg.env, cost=cost)
        equity_parts, records, level = [], [], 1.0
        for fold in folds:
            paths = sorted(glob.glob(str(Path(args.run_dir) / f"fold{fold.idx}_seed*.pt")))
            if not paths:
                continue
            agents = [PPOAgent.load(p) for p in paths]
            env = make_env(features, meta, fold.test, env_cfg, training=False)
            record, _ = evaluate_policy(env, ensemble_policy(agents, cfg.env.actions, cfg.train.confidence))
            rel = record["equity"] / record["equity"].iloc[0]
            equity_parts.append(rel * level)
            level = float(rel.iloc[-1] * level)
            records.append(record)
        if not equity_parts:
            raise SystemExit(f"モデルが見つかりません: {args.run_dir}")
        equity = pd.concat(equity_parts)
        allrec = pd.concat(records)
        metrics = equity_metrics(equity, allrec["position"])
        base = allrec.groupby(allrec.index.date)["equity"].first().iloc[0]
        rows.append({
            "片道コスト(bp)": one_way_bp,
            "純リターン": metrics["total_return"],
            "Sharpe": metrics["sharpe"],
            "最大DD": metrics["max_drawdown"],
            "回転/日": metrics["turnover_per_day"],
            "取引コスト": -float(allrec["trade_cost"].sum() / base / len(folds)),
            "管理料": -float(allrec["carry_cost"].sum() / base / len(folds)),
        })
        print(f"  片道 {one_way_bp:>4.2f}bp → Sharpe {metrics['sharpe']:+6.2f}  "
              f"純 {metrics['total_return']:+7.2%}  最大DD {metrics['max_drawdown']:+6.2%}")

    table = pd.DataFrame(rows).set_index("片道コスト(bp)")
    table.to_csv(args.out)
    print("\n" + table.round(4).to_string())
    positive = table[table["Sharpe"] > 0]
    if len(positive):
        print(f"\n→ 片道 {positive.index.max():.2f}bp 以下なら Sharpe が正になる。")
    else:
        print("\n→ コストをゼロにしても Sharpe は正にならない（グロスに優位性が無い）。")
    print(f"出力: {args.out}")
    _ = np


if __name__ == "__main__":
    main()
