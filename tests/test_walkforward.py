"""ウォークフォワード分割の健全性（重なり・embargo）。"""

import pandas as pd

from rlgmo.walkforward import WalkForwardConfig, make_folds


def test_folds_are_ordered_and_non_overlapping():
    idx = pd.date_range("2026-01-01", periods=60 * 24 * 200, freq="1min", tz="UTC")
    cfg = WalkForwardConfig(train_days=60, valid_days=10, test_days=10, step_days=10, embargo_hours=24)
    folds = make_folds(idx, cfg)
    assert len(folds) > 3
    for fold in folds:
        assert fold.train.stop <= fold.valid.start
        assert fold.valid.stop <= fold.test.start
        # embargo のぶんギャップが空いていること（1 分足なので 24h = 1440 バー）
        assert fold.valid.start - fold.train.stop >= 1440
        assert fold.test.start - fold.valid.stop >= 1440
        assert fold.test.stop <= len(idx)


def test_test_windows_move_forward_without_overlap():
    idx = pd.date_range("2026-01-01", periods=60 * 24 * 200, freq="1min", tz="UTC")
    folds = make_folds(idx, WalkForwardConfig(train_days=60, valid_days=10, test_days=10, step_days=10))
    starts = [f.test.start for f in folds]
    assert starts == sorted(starts)
    for a, b in zip(folds, folds[1:]):
        assert b.test.start >= a.test.start + 1  # 前進している


def test_returns_empty_when_data_too_short():
    idx = pd.date_range("2026-01-01", periods=60 * 24 * 10, freq="1min", tz="UTC")
    assert make_folds(idx, WalkForwardConfig()) == []
