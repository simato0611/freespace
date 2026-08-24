#!/usr/bin/env python3
"""学習済みの fold 別モデルを、テスト区間で再評価する（再学習はしない）。

環境やコストモデルを修正したときに、学習をやり直さずに評価だけを取り直すためのもの。
出力は `train_walkforward.py` と同じ形式（`fold*_test.csv` と `walkforward_report.csv`）。

Example:
    python scripts/reevaluate_folds.py --config configs/btc_real_h60.yaml \
        --run-dir runs/btc_real_h60 --out runs/btc_real_h60_v2
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
from rlgmo.walkforward import make_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="学習済みモデル（fold*_seed*.pt）のあるディレクトリ")
    parser.add_argument("--out", default=None, help="出力先（既定は run-dir を上書き）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out or args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, meta = prepare_data(cfg)
    folds = make_folds(features.index, cfg.walkforward)
    rows = []
    for fold in folds:
        paths = sorted(glob.glob(str(run_dir / f"fold{fold.idx}_seed*.pt")))
        if not paths:
            continue
        agents = [PPOAgent.load(p) for p in paths]
        test_env = make_env(features, meta, fold.test, cfg.env, training=False)
        policy = ensemble_policy(agents, cfg.env.actions, confidence=cfg.train.confidence)
        record, metrics = evaluate_policy(test_env, policy, n_trials=len(paths) * len(folds))
        record.to_csv(out_dir / f"fold{fold.idx}_test.csv")

        row = {"fold": fold.idx, "test_start": fold.timestamps["test_start"],
               **{f"rl_{k}": v for k, v in metrics.items()}}
        for name, base in {
            "flat": flat_policy(cfg.env.actions),
            "long": long_policy(cfg.env.actions),
            "momentum": momentum_policy(test_env),
        }.items():
            _, bm = evaluate_policy(make_env(features, meta, fold.test, cfg.env, training=False), base)
            row[f"{name}_sharpe"] = bm.get("sharpe", 0.0)
            row[f"{name}_return"] = bm.get("total_return", 0.0)
        rows.append(row)
        print(f"fold {fold.idx}: RL sharpe={row['rl_sharpe']:+7.2f} ret={row['rl_total_return']:+7.2%} "
              f"日数={row['rl_days']:>5} 回転={row['rl_turnover_per_day']:.2f}/日 "
              f"| flat 0.00 long={row['long_sharpe']:+6.2f} mom={row['momentum_sharpe']:+6.2f}")

    if not rows:
        raise SystemExit(f"モデルが見つかりません: {run_dir}/fold*_seed*.pt")
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "walkforward_report.csv", index=False)
    print(f"\n出力: {out_dir}/walkforward_report.csv")
    print(json.dumps({"folds": len(report), "mean_sharpe": float(report["rl_sharpe"].mean()),
                      "win_rate": float((report["rl_sharpe"] > 0).mean()),
                      "vs_flat": float((report["rl_sharpe"] > 0).mean())}, indent=1))


if __name__ == "__main__":
    main()
