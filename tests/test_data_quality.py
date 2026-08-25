"""データ健全性チェックの検証。

価格水準もボラも正しいのに**バーの時刻が実際の値動きと対応していない**データは実在し、
欠損・重複チェックを素通りする。そういうデータを検出できることを確かめる。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_data import compare  # noqa: E402


def make_series(seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.004))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": price, "high": price * 1.001, "low": price * 0.999,
                         "close": price, "volume": 1.0}, index=idx)


def test_identical_sources_are_reported_healthy():
    a = make_series(1)
    b = a.copy()
    b["close"] = b["close"] * (1 + np.random.default_rng(9).standard_normal(len(b)) * 1e-5)  # 取引所差の微差
    result = compare(a, b)
    assert result["corr"] > 0.95
    assert result["shift"] == 0


def test_open_time_vs_close_time_convention_is_detected_as_a_shift():
    """規約違い（開始時刻 / 終了時刻）は 1 本ずらせば一致する。破損ではない。"""
    a = make_series(2)
    b = a.copy()
    b.index = b.index - pd.Timedelta(hours=1)
    result = compare(a, b)
    assert result["corr"] > 0.99
    assert result["shift"] == 1


def test_scrambled_timestamps_are_flagged():
    """日ごとにバーの順序を入れ替えたデータ。価格水準もボラも保たれるが相関は消える。"""
    a = make_series(3)
    ret = np.log(a["close"]).diff().fillna(0.0).to_numpy()
    rng = np.random.default_rng(4)
    scrambled = ret.copy()
    for start in range(0, len(scrambled) - 24, 24):  # 1 日ぶんの中で並べ替える
        block = scrambled[start:start + 24].copy()
        rng.shuffle(block)
        scrambled[start:start + 24] = block
    price = 100 * np.exp(np.cumsum(scrambled))
    b = a.copy()
    for col in ("open", "high", "low", "close"):
        b[col] = price
    result = compare(a, b)
    assert result["corr"] < 0.8            # 検出できる
    assert abs(result["vol_a"] - result["vol_b"]) < 0.05   # ボラは一致したまま = 統計量では気づけない
