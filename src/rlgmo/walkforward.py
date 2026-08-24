"""ウォークフォワード分割（purge & embargo 付き）。

単純なランダム分割やシャッフルは時系列では致命的に楽観的な結果を生む。ここでは

    [ 学習 90 日 ][embargo][ 検証 15 日 ][embargo][ テスト 15 日 ]  → 15 日ずつ前進

というローリング分割を作る。embargo（既定 1 日）は、特徴量のローリング窓が
境界をまたいで情報を運ぶのを防ぐためのギャップである（López de Prado の purging）。

**テスト区間は最終評価に一度だけ使う**。ハイパーパラメータ・特徴量・報酬の選択は
すべて検証区間の成績だけで行うこと。テスト成績を見て設計を直した瞬間に、
その成績は「インサンプル」になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 90
    valid_days: int = 15
    test_days: int = 15
    step_days: int = 15
    embargo_hours: int = 24


@dataclass(frozen=True)
class Fold:
    """1 つのウォークフォワード分割（すべて index の位置スライス）。"""

    idx: int
    train: slice
    valid: slice
    test: slice
    timestamps: dict[str, pd.Timestamp]

    def describe(self) -> str:
        ts = self.timestamps
        return (
            f"fold {self.idx}: train {ts['train_start']:%Y-%m-%d}〜{ts['train_end']:%Y-%m-%d} "
            f"valid 〜{ts['valid_end']:%Y-%m-%d} test 〜{ts['test_end']:%Y-%m-%d}"
        )


def make_folds(index: pd.DatetimeIndex, cfg: WalkForwardConfig | None = None) -> list[Fold]:
    """時刻 index からウォークフォワード分割のリストを作る。

    Args:
        index: 1 分足のクローズ時刻（昇順・tz-aware）。
        cfg: 分割設定。

    Returns:
        Fold のリスト（データが足りなければ空）。
    """
    cfg = cfg or WalkForwardConfig()
    index = pd.DatetimeIndex(index)
    day = pd.Timedelta(days=1)
    embargo = pd.Timedelta(hours=cfg.embargo_hours)
    start, end = index[0], index[-1]

    folds: list[Fold] = []
    cursor = start
    i = 0
    while True:
        train_start = cursor
        train_end = train_start + cfg.train_days * day
        valid_start = train_end + embargo
        valid_end = valid_start + cfg.valid_days * day
        test_start = valid_end + embargo
        test_end = test_start + cfg.test_days * day
        if test_end > end:
            break
        folds.append(
            Fold(
                idx=i,
                train=_slice(index, train_start, train_end),
                valid=_slice(index, valid_start, valid_end),
                test=_slice(index, test_start, test_end),
                timestamps={
                    "train_start": train_start, "train_end": train_end,
                    "valid_start": valid_start, "valid_end": valid_end,
                    "test_start": test_start, "test_end": test_end,
                },
            )
        )
        cursor = cursor + cfg.step_days * day
        i += 1
    return folds


def _slice(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> slice:
    left = int(index.searchsorted(start, side="left"))
    right = int(index.searchsorted(end, side="left"))
    return slice(left, right)
