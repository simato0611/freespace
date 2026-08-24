"""取引コストモデル（GMO コイン レバレッジ取引）。

GMO コインのレバレッジ取引は **取引手数料が無料** である一方、実質コストは

    1. スプレッド（取引所板の bid/ask 差 + 気配の薄さ）
    2. スリッページ（成行執行時の不利約定・レイテンシ）
    3. 建玉管理料（レバレッジ手数料）: 建玉評価額に対して **0.04% / 日**
       日本時間 06:00 時点で保有している建玉に課金される

の 3 つで構成される。1 分足でのポジション制御では (1)(2) が回転率を、
(3) が「日をまたぐ保有」を直接罰する。したがって 06:00 JST までの残り時間は
状態変数として与える価値がある（`features.py` の `mins_to_carry` を参照）。

注意: 料率・課金時刻は変更されうるため、必ず公式の手数料ページで最新値を確認し
`configs/*.yaml` の値を更新すること。ここでは全てパラメータ化している。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BP = 1e-4


@dataclass(frozen=True)
class CostConfig:
    """コストパラメータ（すべて建玉評価額に対する率）。

    Attributes:
        half_spread_bp: 片道スプレッドの半値 (bp)。板の mid から実際の約定価格までの距離。
            BTC_JPY のレバレッジ取引所板では平常時 1〜3bp 程度だが、実測を推奨
            (`scripts/measure_spread.py`)。
        slippage_bp: 成行/IOC 執行の平均的な追加コスト (bp)。約定サイズ・板厚に依存。
        taker_fee_bp: 取引手数料 (bp)。GMO レバレッジは 0。現物や他所へ移植する際に使う。
        carry_rate_daily: 建玉管理料の日率。0.0004 = 0.04%/日。
        carry_mode: "daily_0600"（06:00 JST 時点の建玉に課金・実態に近い）、
            "prorata"（バーごとに日率を按分・報酬が滑らかになる）、"none"。
        carry_hour_jst: 課金時刻（JST の時）。
        spread_vol_beta: 実現ボラティリティ比に対するスプレッド拡大係数。
            実効スプレッド = half_spread_bp * (1 + beta * max(0, vol_ratio - 1))。
            ストレス時にスプレッドが広がる現実（= 荒れ相場ほど回転が高くつく）を再現する。
    """

    half_spread_bp: float = 2.0
    slippage_bp: float = 0.5
    taker_fee_bp: float = 0.0
    carry_rate_daily: float = 0.0004
    carry_mode: str = "daily_0600"
    carry_hour_jst: int = 6
    spread_vol_beta: float = 1.0

    @property
    def round_trip_rate(self) -> float:
        """エクスポージャ 1 単位を動かすときに支払う率（片道）。"""
        return (self.half_spread_bp + self.slippage_bp + self.taker_fee_bp) * BP


def effective_trade_rate(cfg: CostConfig, vol_ratio: np.ndarray | float = 1.0) -> np.ndarray | float:
    """ボラティリティ比に応じて拡大した片道コスト率を返す。

    Args:
        cfg: コスト設定。
        vol_ratio: 直近ボラ / 平常ボラ の比。1.0 で平常時。

    Returns:
        エクスポージャ変化 1 単位あたりのコスト率（float または配列）。
    """
    widen = 1.0 + cfg.spread_vol_beta * np.maximum(0.0, np.asarray(vol_ratio, dtype=float) - 1.0)
    spread = cfg.half_spread_bp * widen
    return (spread + cfg.slippage_bp + cfg.taker_fee_bp) * BP


def carry_flags(index: pd.DatetimeIndex, hour_jst: int = 6) -> np.ndarray:
    """各バーが「建玉管理料の課金時刻」をまたぐかどうかのフラグ配列を作る。

    バー i の期間 (t_{i-1}, t_i] が JST の `hour_jst` 時ちょうどを含むとき True。

    Args:
        index: バーのクローズ時刻（tz-aware。naive の場合は UTC とみなす）。
        hour_jst: 課金時刻（JST）。

    Returns:
        shape (len(index),) の bool 配列。
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    jst = idx.tz_convert("Asia/Tokyo")
    # 「前バーのクローズ時点の JST 日付境界カウンタ」が変化したか、で判定する。
    # hour_jst だけシフトすると、hour_jst をまたぐ瞬間に日付が変わる。
    shifted = pd.DatetimeIndex(jst - pd.Timedelta(hours=hour_jst))
    day_id = np.asarray(shifted.normalize().astype("int64"))
    flags = np.zeros(len(idx), dtype=bool)
    flags[1:] = day_id[1:] != day_id[:-1]
    return flags


def carry_rate_per_bar(cfg: CostConfig, bar_minutes: int, crosses_charge_time: bool) -> float:
    """1 バーぶんの建玉管理料率（エクスポージャ 1 単位あたり）。

    Args:
        cfg: コスト設定。
        bar_minutes: バーの長さ（分）。
        crosses_charge_time: そのバーが課金時刻をまたぐか。

    Returns:
        コスト率。
    """
    if cfg.carry_mode == "none":
        return 0.0
    if cfg.carry_mode == "prorata":
        return cfg.carry_rate_daily * bar_minutes / (24 * 60)
    if cfg.carry_mode == "daily_0600":
        return cfg.carry_rate_daily if crosses_charge_time else 0.0
    raise ValueError(f"unknown carry_mode: {cfg.carry_mode}")
