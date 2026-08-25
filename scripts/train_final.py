#!/usr/bin/env python3
"""ホールドアウト直前までのデータで、最終モデルを学習する。

ウォークフォワードの各 fold は「その時点で入手できたデータだけ」で学習しているが、
最後にホールドアウトへ出すモデルは**封印線の直前まで**のデータで学習しておきたい。
そのための 1 回だけの学習。検証区間（直近 `valid_days` 日）で最良の重みを選ぶ点は同じ。

Example:
    python scripts/train_final.py --config configs/btc_trend.yaml --out runs/btc_trend_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rlgmo.config import dump_config, load_config  # noqa: E402
from rlgmo.pipeline import prepare_data, train_fold  # noqa: E402
from rlgmo.walkforward import Fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.train.out_dir = args.out
    features, meta = prepare_data(cfg)
    index = pd.DatetimeIndex(features.index)

    wf = cfg.walkforward
    valid_start = index[-1] - pd.Timedelta(days=wf.valid_days)
    train_end = valid_start - pd.Timedelta(hours=wf.embargo_hours)
    train_start = train_end - pd.Timedelta(days=wf.train_days)
    fold = Fold(
        idx=0,
        train=slice(int(index.searchsorted(train_start)), int(index.searchsorted(train_end))),
        valid=slice(int(index.searchsorted(valid_start)), len(index)),
        test=slice(len(index), len(index)),
        timestamps={"train_start": train_start, "train_end": train_end,
                    "valid_start": valid_start, "valid_end": index[-1],
                    "test_start": index[-1], "test_end": index[-1]},
    )
    print(f"[final] 学習 {train_start:%Y-%m-%d}〜{train_end:%Y-%m-%d} "
          f"({fold.train.stop - fold.train.start:,} バー) / "
          f"検証 {valid_start:%Y-%m-%d}〜{index[-1]:%Y-%m-%d} ({fold.valid.stop - fold.valid.start:,} バー)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.snapshot.yaml")
    for seed in cfg.train.seeds:
        print(f"--- seed {seed} ---")
        agent, history = train_fold(features, meta, fold, cfg, seed)
        path = agent.save(out_dir / f"final_seed{seed}.pt")
        print(f"保存: {path}  検証ベストスコア {history['best_score']:+.3f}")


if __name__ == "__main__":
    main()
