#!/usr/bin/env python3
"""学習済みエージェント（アンサンブル）を任意区間でバックテストする。

Example:
    python scripts/backtest.py --config configs/default.yaml \
        --models "runs/default/fold0_seed*.pt" --start 2026-05-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rlgmo.agents.ppo import PPOAgent  # noqa: E402
from rlgmo.backtest import ensemble_policy, flat_policy, long_policy, momentum_policy  # noqa: E402
from rlgmo.config import load_config  # noqa: E402
from rlgmo.pipeline import evaluate_policy, make_env, prepare_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", required=True, help="モデルの glob パターン")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="runs/backtest")
    parser.add_argument("--n-trials", type=int, default=1, help="Deflated Sharpe 用の総試行回数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features, meta = prepare_data(cfg)
    if args.start:
        features, meta = features.loc[args.start :], meta.loc[args.start :]
    if args.end:
        features, meta = features.loc[: args.end], meta.loc[: args.end]

    paths = sorted(glob.glob(args.models))
    if not paths:
        raise SystemExit(f"モデルが見つかりません: {args.models}")
    agents = [PPOAgent.load(p) for p in paths]
    print(f"[models] {len(agents)} 個のエージェントをアンサンブル: {[Path(p).name for p in paths]}")

    full = slice(0, len(features))
    env = make_env(features, meta, full, cfg.env, training=False)
    record, metrics = evaluate_policy(env, ensemble_policy(agents, cfg.env.actions, cfg.train.confidence),
                                      n_trials=args.n_trials)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record.to_csv(out_dir / "equity.csv")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    rows = {"RL(ensemble)": metrics}
    for name, policy in {
        "flat": flat_policy(cfg.env.actions),
        "long": long_policy(cfg.env.actions),
        "momentum": momentum_policy(env),
    }.items():
        _, base = evaluate_policy(make_env(features, meta, full, cfg.env, training=False), policy)
        rows[name] = base

    table = pd.DataFrame(rows).T[
        ["total_return", "sharpe", "max_drawdown", "ann_vol", "turnover_per_day", "exposure", "deflated_sharpe_p"]
    ]
    print("\n" + table.round(3).to_string())
    print(f"\n出力: {out_dir}/equity.csv, {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
