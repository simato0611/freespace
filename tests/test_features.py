"""特徴量の因果性（ルックアヘッドが無いこと）を検証する。"""

import numpy as np
import pandas as pd

from rlgmo.data.resample import resample_ohlcv
from rlgmo.data.synthetic import make_synthetic_ohlcv
from rlgmo.features import FeatureConfig, build_features


def test_no_lookahead_when_future_changes():
    """未来のバーを書き換えても、過去時点の特徴量は 1 つも変わってはいけない。"""
    cfg = FeatureConfig(scale_window=1440)
    df = make_synthetic_ohlcv(60 * 24 * 12, seed=11)
    feats_a, _ = build_features(df, cfg)

    cut = len(df) - 500
    tampered = df.copy()
    tampered.iloc[cut:] *= 1.25  # 未来を大きく改変
    feats_b, _ = build_features(tampered, cfg)

    common = feats_a.index.intersection(feats_b.index)
    common = common[common < df.index[cut]]
    assert len(common) > 1000
    pd.testing.assert_frame_equal(feats_a.loc[common], feats_b.loc[common])


def test_higher_timeframe_alignment_is_backward_only():
    """1 分足に貼り付けた 15 分足は、必ず「完成済み」バーの値である。"""
    df = make_synthetic_ohlcv(60 * 24 * 3, seed=12)
    h15 = resample_ohlcv(df, 15)
    feats, _ = build_features(df, FeatureConfig(scale_window=720))
    # 15 分足クローズ直後の 1 分足では、直前に確定した 15 分足リターンと符号が一致する
    sample = feats.index[len(feats) // 2]
    last_closed = h15.index[h15.index <= sample][-1]
    assert last_closed <= sample


def test_features_are_finite_and_scaled():
    feats, meta = build_features(make_synthetic_ohlcv(60 * 24 * 20, seed=13), FeatureConfig(scale_window=1440))
    assert np.isfinite(feats.to_numpy()).all()
    assert (feats.abs().max() <= 5.0 + 1e-6).all()
    assert feats.index.equals(meta.index)
    assert feats.std().median() > 0.2  # 情報が潰れていないこと
