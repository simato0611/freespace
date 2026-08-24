#!/usr/bin/env python3
"""ウォークフォワードで PPO エージェントを学習・評価する。

Example:
    # 合成データで配線確認（数分）
    python scripts/train_walkforward.py --config configs/smoke.yaml
    # 実データで本番学習
    python scripts/train_walkforward.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlgmo.config import dump_config, load_config  # noqa: E402
from rlgmo.pipeline import run_walkforward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-folds", type=int, default=None, help="先頭からこの数の fold だけ実行")
    parser.add_argument("--out", default=None, help="出力ディレクトリ（設定を上書き）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg.train.out_dir = args.out
    dump_config(cfg, Path(cfg.train.out_dir) / "config.snapshot.yaml")  # 再現性のため設定を保存
    run_walkforward(cfg, max_folds=args.max_folds)


if __name__ == "__main__":
    main()
