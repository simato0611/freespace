"""マルチタイムフレーム特徴量（1 分 / 5 分 / 15 分足）。

設計方針
--------
1. **因果性**: 時刻 t の特徴量は t のクローズまでに確定した情報のみを使う。
   上位足は `resample.align_to_base` で「完成済みバー」だけを貼り付ける。
2. **スケール不変**: 価格水準・ボラ水準が変わっても分布が動かないよう、リターンは
   EWMA ボラで、価格乖離は ATR で正規化する。生の価格・出来高は入れない
   （BTC_JPY は数年で桁が変わるため、学習期間の水準に過学習する）。
3. **正規化のリーク防止**: 標準化は学習データ全体の平均/分散ではなく、
   **因果的なローリング中央値・IQR** で行う。テスト期間の統計量が学習に漏れない。
4. **少数精鋭**: 1 分足の S/N 比は極めて低い。似た情報を持つ指標を大量に足すより、
   「トレンド / 平均回帰 / ボラ / 出来高（オーダーフロー代理） / 時間 / 建玉状態」の
   各軸から代表を選ぶ。既定で約 60 次元。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.resample import align_to_base, resample_ohlcv

# 構成上すでに [-1, 1] 近傍に収まっており、再標準化しない列の接頭辞
BOUNDED_PREFIX = ("ret_", "rsi_", "donch_", "body_", "effr_", "ofi_", "sin_", "cos_", "sess_", "carry_")


@dataclass
class FeatureConfig:
    """特徴量生成の設定。

    `base_minutes` は特徴量を作る基準グリッド（分）。1 なら 1 分足そのまま。
    `timeframes` は**基準グリッドの倍数**で指定する。例:

        base_minutes=1,  timeframes=(1, 5, 15)   → 1分/5分/15分足（短期戦略）
        base_minutes=60, timeframes=(1, 4, 24)   → 1時間/4時間/日足（多日スケール戦略）

    実データ検証（`docs/real_data_findings.md`）で分かったとおり、1〜15 分足だけの
    特徴量は最長でも 5 時間ぶんの情報しか持たず、コストを超える予測力が無い。
    多日スケールの時系列モメンタムには優位性があるため、基準グリッドを粗くして
    そちらを見られるようにしている。
    """

    base_minutes: int = 1
    timeframes: tuple[int, ...] = (1, 5, 15)
    ret_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    vol_span: int = 60          # 各タイムフレームでの EWMA ボラのスパン（バー数）
    rsi_period: int = 14
    atr_period: int = 14
    donchian: int = 48
    effr_period: int = 20
    ofi_period: int = 20        # オーダーフロー代理（符号付き出来高）の窓
    vwap_period: int = 60       # 1 分足での VWAP 乖離の窓
    scale_window: int = 60 * 24 * 7   # 因果ロバスト標準化の窓（1 分足バー数 = 7 日）
    clip: float = 5.0
    carry_hour_jst: int = 6
    session_hours_jst: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"tokyo": (8, 15), "europe": (16, 23), "us": (22, 6)}
    )


# --------------------------------------------------------------------------------------
# 基本指標
# --------------------------------------------------------------------------------------
def ewma_vol(ret: pd.Series, span: int) -> pd.Series:
    """EWMA 実現ボラティリティ（1 バーあたり）。"""
    return ret.ewm(span=span, min_periods=span // 2).std().replace(0.0, np.nan)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman の効率比: 正味変化 / 経路長。1 に近いほど素直なトレンド。"""
    net = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(period).sum()
    return (net / path.replace(0.0, np.nan)).clip(0, 1)


def signed_volume_imbalance(df: pd.DataFrame, period: int) -> pd.Series:
    """符号付き出来高の不均衡（板情報を使わないオーダーフロー代理）。

    ローソク内の終値位置で買い/売り出来高を按分する（Lee-Ready 的近似）。
    """
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    buy_frac = ((df["close"] - df["low"]) / rng).fillna(0.5).clip(0, 1)
    signed = df["volume"] * (2 * buy_frac - 1)
    return signed.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0.0, np.nan)


def causal_robust_scale(df: pd.DataFrame, window: int, clip: float) -> pd.DataFrame:
    """因果的なローリング中央値・IQR による標準化（未来情報を使わない）。

    平均/標準偏差ではなく中央値/IQR を使うのは、暗号資産のジャンプで統計量が
    壊れるのを防ぐため。IQR が 0 の期間（値動きが完全に止まった場合）は NaN → 0。
    """
    if df.empty:
        return df
    med = df.rolling(window, min_periods=window // 8).median()
    q75 = df.rolling(window, min_periods=window // 8).quantile(0.75)
    q25 = df.rolling(window, min_periods=window // 8).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    return ((df - med) / (iqr * 0.7413)).clip(-clip, clip)  # 0.7413 = 正規分布での IQR→σ 換算


# --------------------------------------------------------------------------------------
# タイムフレームごとのブロック
# --------------------------------------------------------------------------------------
def timeframe_block(df: pd.DataFrame, cfg: FeatureConfig, tf: int) -> pd.DataFrame:
    """1 つのタイムフレームぶんの特徴量を作る（index はそのタイムフレームのクローズ時刻）。

    Args:
        df: そのタイムフレームの OHLCV。
        cfg: 設定。
        tf: 基準グリッドの倍数（`minutes_per_bar = tf * cfg.base_minutes`）。
    """
    minutes_per_bar = tf * cfg.base_minutes
    # 「平常時の水準」を測る窓: 日中足なら 1 日ぶん、日足以上なら 30 本を下限にする
    baseline = max(30, 1440 // max(minutes_per_bar, 1))
    close = df["close"]
    logret = np.log(close).diff()
    vol = ewma_vol(logret, cfg.vol_span)
    out = pd.DataFrame(index=df.index)

    # --- トレンド / モメンタム（ボラ正規化リターン）
    for lag in cfg.ret_lags:
        out[f"ret_{lag}"] = (np.log(close) - np.log(close.shift(lag))) / (vol * np.sqrt(lag))

    # --- 平均回帰 / 位置
    a = atr(df, cfg.atr_period)
    for span in (12, 48):
        out[f"emadev_{span}"] = (close - close.ewm(span=span, min_periods=span).mean()) / a
    hh = df["high"].rolling(cfg.donchian).max()
    ll = df["low"].rolling(cfg.donchian).min()
    out["donch_pos"] = ((close - ll) / (hh - ll).replace(0.0, np.nan) - 0.5) * 2
    out["rsi_14"] = (rsi(close, cfg.rsi_period) - 50) / 50

    # --- ボラティリティ
    out["atr_rel"] = a / close                              # 価格に対する ATR（水準）
    out["volratio"] = np.log(vol / vol.rolling(baseline, min_periods=max(10, baseline // 4)).mean())
    park = np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))
    out["park_ratio"] = np.log(np.sqrt(park.rolling(12).mean()) / vol.replace(0.0, np.nan))

    # --- 出来高 / オーダーフロー代理
    out["ofi"] = signed_volume_imbalance(df, cfg.ofi_period)
    out["logvol"] = np.log1p(df["volume"]) - np.log1p(df["volume"]).rolling(
        baseline, min_periods=max(10, baseline // 4)).mean()

    # --- ローソク形状 / トレンド品質
    out["body_ratio"] = (close - df["open"]) / (df["high"] - df["low"]).replace(0.0, np.nan)
    out["effr"] = efficiency_ratio(close, cfg.effr_period) * 2 - 1

    return out.replace([np.inf, -np.inf], np.nan)


def time_features(index: pd.DatetimeIndex, cfg: FeatureConfig) -> pd.DataFrame:
    """時間帯・曜日・建玉管理料カットオフまでの残り時間。"""
    jst = index.tz_convert("Asia/Tokyo")
    minute_of_day = jst.hour * 60 + jst.minute
    out = pd.DataFrame(index=index)
    out["sin_tod"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["cos_tod"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["sin_dow"] = np.sin(2 * np.pi * jst.dayofweek / 7)
    out["cos_dow"] = np.cos(2 * np.pi * jst.dayofweek / 7)
    # 建玉管理料 (06:00 JST) までの残り時間 → 日跨ぎコストを織り込ませる
    mins_to_carry = (cfg.carry_hour_jst * 60 - minute_of_day) % 1440
    out["carry_sin"] = np.sin(2 * np.pi * mins_to_carry / 1440)
    out["carry_cos"] = np.cos(2 * np.pi * mins_to_carry / 1440)
    for name, (start, end) in cfg.session_hours_jst.items():
        hour = jst.hour.to_numpy()
        active = (hour >= start) & (hour < end) if start < end else (hour >= start) | (hour < end)
        out[f"sess_{name}"] = active.astype(float)
    return out


def build_features(ohlcv_1m: pd.DataFrame, cfg: FeatureConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1 分足 OHLCV からマルチタイムフレーム特徴量と市場メタ情報を作る。

    Args:
        ohlcv_1m: index=クローズ時刻(UTC)、columns=[open, high, low, close, volume]。
        cfg: 特徴量設定。

    Returns:
        (features, meta):
            features: 因果標準化済みの特徴量（NaN 行は除去済み）。
            meta: 環境が使う列（open/high/low/close/volume/vol_ratio/vol_1m）。features と同じ index。
    """
    cfg = cfg or FeatureConfig()
    base = ohlcv_1m if cfg.base_minutes == 1 else resample_ohlcv(ohlcv_1m, cfg.base_minutes)
    base_index = pd.DatetimeIndex(base.index)
    blocks = []
    for tf in cfg.timeframes:
        minutes = tf * cfg.base_minutes
        src = base if tf == 1 else resample_ohlcv(base, minutes)
        block = timeframe_block(src, cfg, tf)
        suffix = f"_{minutes}m" if minutes < 1440 else f"_{minutes // 1440}d"
        if tf == 1:
            block.columns = [f"{c}{suffix}" for c in block.columns]
            blocks.append(block)
        else:
            blocks.append(align_to_base(base_index, block, suffix))

    # 基準グリッドでの VWAP 乖離（執行タイミングの手掛かり）
    typical = (base["high"] + base["low"] + base["close"]) / 3
    vwap = (typical * base["volume"]).rolling(cfg.vwap_period).sum() / base["volume"].rolling(
        cfg.vwap_period
    ).sum().replace(0.0, np.nan)
    a1 = atr(base, cfg.atr_period)
    extra = pd.DataFrame({"vwapdev": (base["close"] - vwap) / a1}, index=base_index)

    feats = pd.concat(blocks + [extra], axis=1).replace([np.inf, -np.inf], np.nan)

    to_scale = [c for c in feats.columns if not c.startswith(BOUNDED_PREFIX)]
    scaled = feats.copy()
    scaled[to_scale] = causal_robust_scale(feats[to_scale], cfg.scale_window, cfg.clip)
    scaled[[c for c in feats.columns if c not in to_scale]] = feats[
        [c for c in feats.columns if c not in to_scale]
    ].clip(-cfg.clip, cfg.clip)

    scaled = pd.concat([scaled, time_features(base_index, cfg)], axis=1)

    # メタ情報: 執行価格とコストモデル用（すべて基準グリッド）
    base_ret = np.log(base["close"]).diff()
    base_vol = ewma_vol(base_ret, cfg.vol_span)
    long_window = max(60, 60 * 24 * 7 // max(cfg.base_minutes, 1))
    meta = base.copy()
    meta["vol_1m"] = base_vol          # 「1 バーあたりの実現ボラ」（列名は互換性のため据え置き）
    meta["vol_ratio"] = base_vol / base_vol.rolling(long_window, min_periods=long_window // 8).median()

    valid = scaled.dropna().index.intersection(meta.dropna().index)
    return scaled.loc[valid].astype(np.float32), meta.loc[valid]
