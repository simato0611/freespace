"""ライブ執行のロジック（ネットワークに依存しない部分）。

**設計の要点: 2 つの速度で回す**

    リスク監視ループ  … 既定 60 秒ごと。データ鮮度・スプレッド・証拠金維持率・
                        ドローダウン・日次損失を見る。異常なら即座に縮小/停止する
    リバランスループ  … 既定 1 時間ごと。シグナルを再計算して目標建玉を出す

シグナルは 5〜60 日のトレンドなので、再計算を細かくしても見えるのはノイズだけである
（実測: 5 分〜8 時間で Sharpe 1.64〜2.15、非単調で推定誤差の範囲。細かいほど
ドローダウンはむしろ悪化した）。一方で**リスク事象は分単位で起きる**ため、監視は速くする。

サイジングは `portfolio.compute_exposures` を**バックテストと共有**している。
ここを二重に実装すると必ずズレる（設計書 9.1 節）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostConfig
from .portfolio import PortfolioConfig, apply_rebalance_band, compute_exposures, ladder_signal

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


@dataclass
class LiveConfig:
    """ライブ執行の設定（`configs/gmo_live.yaml` に対応）。"""

    symbols: tuple[str, ...] = ("BTC_JPY", "ETH_JPY", "XRP_JPY", "LTC_JPY", "BCH_JPY")
    lookback_days: tuple[float, ...] = (5, 14, 30, 60)
    long_only: bool = False
    gain: float = 1.5
    vol_window_bars: int = 30
    grid_hours: int = 1                # リバランス間隔
    risk_interval_sec: int = 60        # リスク監視の間隔
    rebalance_band: float = 0.10
    asset_vol_ann: float = 0.15
    target_vol_ann: float = 0.15
    max_weight: float = 0.5
    leverage_cap: float = 2.0
    min_trade_delta: float = 0.005     # これ未満の建玉変更は発注しない。
                                       # 大きすぎると建玉が目標から離れ、小さすぎると
                                       # 無意味な微調整を毎時間出すことになる（実測で 0.005 が最適）
    cost: CostConfig = field(default_factory=lambda: CostConfig(half_spread_bp=1.5, slippage_bp=0.0))

    def portfolio_config(self) -> PortfolioConfig:
        return PortfolioConfig(
            target_vol_ann=self.target_vol_ann, asset_vol_ann=self.asset_vol_ann,
            leverage_cap=self.leverage_cap, max_weight=self.max_weight, cost=self.cost,
        )


@dataclass
class Order:
    """1 銘柄ぶんの発注指示。"""

    symbol: str
    side: str          # "BUY" | "SELL"
    quantity: float    # 銘柄の数量（BTC なら BTC 建て）
    delta_exposure: float
    target_exposure: float
    current_exposure: float
    price: float

    def describe(self) -> str:
        return (f"{self.symbol} {self.side} {self.quantity:.4f} "
                f"(建玉 {self.current_exposure:+.3f} → {self.target_exposure:+.3f})")


class StrategyEngine:
    """1 分足のバッファから目標建玉を計算し、発注指示に変換する。"""

    def __init__(self, cfg: LiveConfig) -> None:
        self.cfg = cfg

    def resample(self, bars_1min: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """1 分足を判断グリッドへ集約する（バックテストと同じ規約）。"""
        out = {}
        for symbol, df in bars_1min.items():
            r = df.resample(f"{self.cfg.grid_hours}h", label="right", closed="right").agg(
                {k: v for k, v in AGG.items() if k in df.columns}).dropna(subset=["close"])
            if len(r) > 0:
                out[symbol] = r
        return out

    def signals(self, bars: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """トレンド・ラダー + 建玉更新バンド。バックテストと同一の計算。"""
        per_day = max(1, 24 // self.cfg.grid_hours)
        return {
            symbol: apply_rebalance_band(
                ladder_signal(df, per_day, self.cfg.lookback_days, self.cfg.long_only,
                              self.cfg.gain, self.cfg.vol_window_bars),
                self.cfg.rebalance_band)
            for symbol, df in bars.items()
        }

    def targets(self, bars_1min: dict[str, pd.DataFrame]) -> dict[str, float]:
        """各銘柄の目標建玉（有効証拠金に対する倍率、符号つき）を返す。

        Raises:
            ValueError: 助走が足りず目標を計算できないとき。
        """
        bars = self.resample(bars_1min)
        if not bars:
            raise ValueError("価格バーがありません")
        need = int(max(self.cfg.lookback_days) * 24 / self.cfg.grid_hours) + self.cfg.vol_window_bars
        short = {s: len(df) for s, df in bars.items() if len(df) < need}
        if short:
            raise ValueError(f"助走が不足しています（必要 {need} 本）: {short}")
        exposures = compute_exposures(bars, self.signals(bars), self.cfg.grid_hours * 60,
                                      self.cfg.portfolio_config())
        last = exposures.iloc[-1]
        return {s: float(last.get(s, 0.0)) for s in bars}

    def orders(
        self,
        targets: dict[str, float],
        current_exposure: dict[str, float],
        equity: float,
        prices: dict[str, float],
        size_scale: float = 1.0,
    ) -> list[Order]:
        """目標と現状の差から発注指示を作る。

        Args:
            targets: 目標建玉（有効証拠金比）。
            current_exposure: 現在の建玉（有効証拠金比、符号つき）。
            equity: 有効証拠金。
            prices: 銘柄ごとの現在値。
            size_scale: リスクレイヤが返した縮小係数（0〜1）。

        Returns:
            発注が必要な銘柄ぶんの `Order`。差が小さいものは含めない。
        """
        orders = []
        for symbol, target in targets.items():
            scaled = target * size_scale
            current = current_exposure.get(symbol, 0.0)
            delta = scaled - current
            if abs(delta) < self.cfg.min_trade_delta:
                continue
            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue
            quantity = abs(delta) * equity / price
            orders.append(Order(symbol=symbol, side="BUY" if delta > 0 else "SELL",
                                quantity=quantity, delta_exposure=delta, target_exposure=scaled,
                                current_exposure=current, price=price))
        return orders


def split_close_open(order: Order, held_quantity: float) -> tuple[float, float]:
    """発注量を「決済ぶん」と「新規ぶん」に分ける。

    GMO のレバレッジ取引は、建玉を減らす/反対に返すときは決済注文
    （`/v1/closeBulkOrder`）、増やすときは新規注文（`/v1/order`）と API が分かれている。
    素朴に反対売買を出すと**両建てが積み上がる**ため、この分割が必要になる。

    Args:
        order: 発注指示。
        held_quantity: 現在の建玉数量（ロングが正、ショートが負）。

    Returns:
        (決済する数量, 新規で建てる数量)。合計は `order.quantity` に等しい。

    Example:
        >>> # ロング 1.0 を持っていて 1.5 売る → 1.0 を決済し、0.5 で新規ショート
        >>> split_close_open(Order("BTC_JPY", "SELL", 1.5, 0, 0, 0, 0), 1.0)
        (1.0, 0.5)
    """
    reducing = held_quantity != 0 and (order.side == "SELL") == (held_quantity > 0)
    close_qty = min(order.quantity, abs(held_quantity)) if reducing else 0.0
    return close_qty, order.quantity - close_qty


def exposure_from_positions(positions: dict[str, float], prices: dict[str, float], equity: float) -> dict[str, float]:
    """建玉数量（符号つき）を、有効証拠金に対する倍率へ換算する。"""
    if equity <= 0:
        return {s: 0.0 for s in positions}
    return {s: positions.get(s, 0.0) * prices.get(s, 0.0) / equity for s in positions}


def staleness_seconds(bars_1min: dict[str, pd.DataFrame], now: pd.Timestamp | None = None) -> float:
    """最も古い銘柄の、最新バーからの経過秒数。"""
    now = now or pd.Timestamp.now(tz="UTC")
    lags = [(now - pd.DatetimeIndex(df.index)[-1]).total_seconds() for df in bars_1min.values() if len(df)]
    return max(lags) if lags else float("inf")


def realized_vol_ann(bars_1min: dict[str, pd.DataFrame], window_bars: int = 1440) -> float:
    """ポートフォリオの代理として、銘柄平均の年率ボラを返す（リスク監視用）。"""
    vols = []
    for df in bars_1min.values():
        ret = np.log(df["close"]).diff().tail(window_bars)
        if ret.notna().sum() > 30:
            vols.append(float(ret.std() * np.sqrt(365 * 24 * 60)))
    return float(np.mean(vols)) if vols else 0.0
