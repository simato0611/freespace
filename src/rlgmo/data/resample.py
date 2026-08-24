"""1 分足から上位足（5 分足・15 分足）へのリサンプルとアライン。

**ルックアヘッド防止の要点**

- リサンプルは `label="right", closed="right"` を使い、バーの index を
  「そのバーのクローズ時刻」にする。
- 1 分足グリッドへ戻すときは `merge_asof`（direction="backward"）で
  「その時刻までに **完成している** 上位足」だけを貼り付ける。
  ffill/reindex を素朴に使うと、まだ確定していない 15 分足の終値が
  1 分足の途中に漏れ込む（= 未来情報のリーク）。
"""

from __future__ import annotations

import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """1 分足 OHLCV を `minutes` 分足へ集約する（index はクローズ時刻）。"""
    out = df.resample(f"{minutes}min", label="right", closed="right").agg(AGG)
    return out.dropna(subset=["close"])


def align_to_base(base_index: pd.DatetimeIndex, higher: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """上位足の特徴量を 1 分足グリッドに「完成済みのものだけ」貼り付ける。

    Args:
        base_index: 1 分足のクローズ時刻 index。
        higher: 上位足のクローズ時刻を index に持つ DataFrame。
        suffix: 列名に付ける接尾辞（例 "_15m"）。

    Returns:
        base_index に揃えた DataFrame。
    """
    left = pd.DataFrame(index=base_index).reset_index().rename(columns={base_index.name or "index": "ts"})
    right = higher.reset_index()
    right = right.rename(columns={right.columns[0]: "ts"})
    merged = pd.merge_asof(
        left.sort_values("ts"),
        right.sort_values("ts"),
        on="ts",
        direction="backward",
        allow_exact_matches=True,  # クローズ時刻ちょうどのバーは確定済みなので使ってよい
    )
    merged = merged.set_index("ts")
    merged.index.name = base_index.name
    merged.columns = [f"{c}{suffix}" for c in merged.columns]
    return merged
