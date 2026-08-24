"""オフライン検証用の合成 1 分足生成器。

GMO の Public API に到達できない環境（CI・隔離ネットワーク）でも、パイプライン全体
（特徴量 → 環境 → 学習 → バックテスト）を通しで検証できるようにするためのもの。
**戦略の性能評価には使わない**。実データでのウォークフォワード結果のみを採用すること。

生成モデル:
    - 2 レジーム（トレンド / レンジ）のマルコフ切替ドリフト
    - GARCH(1,1) 的なボラティリティ + 日中ボラ季節性（JST 深夜は薄く、欧米時間で厚い）
    - まれなジャンプ（フラッシュクラッシュ再現）
    - 1 分を N サブステップに分割して high/low を整合的に構成
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    n_minutes: int = 60 * 24 * 30,
    start: str = "2026-01-01",
    s0: float = 15_000_000.0,
    seed: int = 0,
    substeps: int = 6,
    trend_strength: float = 0.35,
    jump_prob: float = 1.5e-5,
) -> pd.DataFrame:
    """合成 1 分足 OHLCV を生成する。

    Args:
        n_minutes: 生成する分数。
        start: 開始時刻（UTC）。
        s0: 初期価格（BTC_JPY を想定）。
        seed: 乱数シード。
        substeps: 1 分あたりのサブステップ数（high/low の生成に使う）。
        trend_strength: トレンドレジームでのドリフトの強さ（ボラ単位）。
        jump_prob: 1 サブステップあたりのジャンプ発生確率。

    Returns:
        columns=[open, high, low, close, volume]、index=クローズ時刻(UTC) の DataFrame。
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC") + pd.Timedelta(minutes=1)
    jst_hour = idx.tz_convert("Asia/Tokyo").hour.to_numpy()

    # 日中ボラ季節性: JST 2〜7 時（欧米深夜）は薄く、16〜24 時（欧州・NY）は厚い
    seasonal = 0.65 + 0.55 * np.sin((jst_hour - 3) / 24 * 2 * np.pi) ** 2

    # レジーム切替（平均滞在 ~6 時間）
    switch = rng.random(n_minutes) < 1 / 360
    regime = np.cumsum(switch) % 2

    base_vol = 8.3e-4  # 1 分あたり ~8bp → 年率 60% 程度（525,600 分/年）
    vol = np.empty(n_minutes)
    vol[0] = base_vol
    shock = rng.standard_normal(n_minutes)
    for t in range(1, n_minutes):  # GARCH(1,1) 風
        var = 0.03 * base_vol**2 + 0.90 * vol[t - 1] ** 2 + 0.07 * (vol[t - 1] * shock[t - 1]) ** 2
        vol[t] = np.sqrt(var)
    vol = vol * seasonal

    sub_vol = np.repeat(vol, substeps) / np.sqrt(substeps)
    sub_drift = np.repeat(np.where(regime == 1, trend_strength, -0.02) * vol, substeps) / substeps
    sign = np.repeat(rng.choice([-1.0, 1.0], size=n_minutes // 360 + 2), 360 * substeps)[: n_minutes * substeps]
    eps = rng.standard_normal(n_minutes * substeps)
    jumps = (rng.random(n_minutes * substeps) < jump_prob) * rng.standard_normal(n_minutes * substeps) * 0.02

    log_ret = sub_drift * sign + sub_vol * eps + jumps
    log_px = np.log(s0) + np.cumsum(log_ret)
    px = np.exp(log_px).reshape(n_minutes, substeps)

    open_ = np.empty(n_minutes)
    open_[0] = s0
    open_[1:] = px[:-1, -1]
    close = px[:, -1]
    high = np.maximum(px.max(axis=1), open_)
    low = np.minimum(px.min(axis=1), open_)
    volume = np.exp(rng.standard_normal(n_minutes) * 0.6) * (1 + 12 * np.abs(close / open_ - 1) / base_vol) * seasonal

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(idx, name="close_time"),
    )
