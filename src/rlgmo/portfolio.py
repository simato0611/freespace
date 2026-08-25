"""複数銘柄の等リスク・ポートフォリオ。

単一銘柄のトレンドフォローは、良い年と悪い年の振れが大きい（BTC 単体で 2018 年 −1.19、
2022 年 −1.12 の Sharpe）。同じルールを相関の低くない複数銘柄に分散すると、
**シグナルの当たり外れが平均化されて、リスク当たりの成績が上がる**。
実測では 7 銘柄の等ウェイトで Sharpe 1.41 → 2.07、最大 DD −13% → −4.8% になった。

構成:

1. 銘柄ごとに「シグナル × ボラターゲット」で建玉比率を決める（＝等リスク配分）。
   ボラの高い銘柄は自動的に小さく建つので、リスク寄与が揃う。
2. 銘柄を合計したあと、**ポートフォリオ全体のボラ**を目標に合わせて再スケールする。
   分散が効いているぶん、個別に 20% を狙うと合計は 20% を大きく下回るため、
   この 2 段目が無いとリスクを取り損なう。
3. 総建玉（グロス）にレバレッジ上限を掛ける。GMO は個人 2 倍まで。

コストは銘柄ごとに実費で引く（スプレッド + 建玉管理料）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostConfig, carry_flags, carry_rate_per_bar

MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass
class PortfolioConfig:
    """ポートフォリオ設定。"""

    target_vol_ann: float = 0.20      # ポートフォリオ全体の目標ボラ
    asset_vol_ann: float = 0.20       # 銘柄ごとの目標ボラ（等リスク配分の単位）
    leverage_cap: float = 2.0         # 総建玉 / 有効証拠金 の上限
    vol_window_days: int = 20         # ボラ推定の窓
    portfolio_vol_window_days: int = 60
    max_weight: float = 0.5           # 1 銘柄あたりの建玉上限（有効証拠金比）
    cost: CostConfig = field(default_factory=CostConfig)


def _realized_vol(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    return returns.rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(periods_per_year)


def compute_exposures(
    prices: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    grid_minutes: int,
    cfg: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """シグナルから、各銘柄の目標建玉（有効証拠金に対する倍率）を計算する。

    **バックテストとライブ執行はこの関数を共有する**。サイジングを二重に実装すると
    必ずズレる（設計書 9.1 節）。ライブ側は返り値の最終行だけを使う。

    Args:
        prices: 銘柄名 → OHLCV（共通グリッド）。
        signals: 銘柄名 → シグナル（-1〜1）。
        grid_minutes: バーの長さ（分）。
        cfg: ポートフォリオ設定。

    Returns:
        index=時刻、columns=銘柄 の目標建玉。レバレッジ上限適用済み。
    """
    cfg = cfg or PortfolioConfig()
    periods_per_year = MINUTES_PER_YEAR / grid_minutes
    vol_window = max(5, int(cfg.vol_window_days * 1440 / grid_minutes))

    index = None
    for df in prices.values():
        index = df.index if index is None else index.union(df.index)
    index = pd.DatetimeIndex(index).sort_values()

    exposures, gaps, intras = {}, {}, {}
    for asset, df in prices.items():
        df = df.reindex(index)
        logret = np.log(df["close"]).diff()
        vol = _realized_vol(logret, vol_window, periods_per_year)
        # 低ボラ銘柄ほど大きく建てる（等リスク）。個別の上限は下の clip で掛ける。
        scale = (cfg.asset_vol_ann / vol.replace(0.0, np.nan)).clip(0.0, cfg.leverage_cap)
        raw = signals[asset].reindex(index).clip(-1, 1)
        exposures[asset] = (raw * scale).clip(-cfg.max_weight, cfg.max_weight).fillna(0.0)
        gaps[asset] = (df["open"].shift(-1) / df["close"] - 1.0).fillna(0.0)
        intras[asset] = (df["close"].shift(-1) / df["open"].shift(-1) - 1.0).fillna(0.0)

    exposure = pd.DataFrame(exposures).fillna(0.0)
    gap = pd.DataFrame(gaps).fillna(0.0)
    intra = pd.DataFrame(intras).fillna(0.0)

    # --- 2 段目: ポートフォリオ全体のボラを目標に合わせる（分散のぶんリスクを取り戻す）
    raw_pnl = (exposure.shift(1) * gap + exposure * intra).sum(axis=1)
    pf_window = max(10, int(cfg.portfolio_vol_window_days * 1440 / grid_minutes))
    pf_vol = _realized_vol(raw_pnl, pf_window, periods_per_year)
    pf_scale = (cfg.target_vol_ann / pf_vol.replace(0.0, np.nan)).clip(0.2, 5.0).shift(1).fillna(1.0)

    scaled = exposure.mul(pf_scale, axis=0)
    gross = scaled.abs().sum(axis=1)
    over = (gross / cfg.leverage_cap).clip(lower=1.0)      # レバレッジ上限で頭打ち
    scaled = scaled.div(over, axis=0)
    scaled.attrs["gap"] = gap
    scaled.attrs["intra"] = intra
    return scaled


def backtest_portfolio(
    prices: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    grid_minutes: int,
    cfg: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """複数銘柄のポートフォリオをバックテストする。

    Args:
        prices: 銘柄名 → OHLCV（index は共通グリッドのクローズ時刻）。
        signals: 銘柄名 → シグナル（-1〜1。時刻 t のクローズで確定し、t+1 の始値で執行）。
        grid_minutes: バーの長さ（分）。
        cfg: ポートフォリオ設定。

    Returns:
        columns=[equity, gross_exposure, net_exposure, gross_pnl, cost, ret] の DataFrame。
    """
    cfg = cfg or PortfolioConfig()
    scaled = compute_exposures(prices, signals, grid_minutes, cfg)
    gap, intra = scaled.attrs["gap"], scaled.attrs["intra"]
    index = scaled.index

    one_way = (cfg.cost.half_spread_bp + cfg.cost.slippage_bp + cfg.cost.taker_fee_bp) * 1e-4
    turnover = (scaled - scaled.shift(1)).abs().sum(axis=1).fillna(0.0)
    # 建玉管理料は costs.carry_rate_per_bar に委譲する（carry_mode を尊重するため）
    flags = carry_flags(index, cfg.cost.carry_hour_jst)
    on_rate = carry_rate_per_bar(cfg.cost, grid_minutes, True)
    off_rate = carry_rate_per_bar(cfg.cost, grid_minutes, False)
    carry_rate = pd.Series(np.where(flags, on_rate, off_rate), index=index)
    carry_cost = carry_rate * scaled.abs().sum(axis=1)

    gross_pnl = (scaled.shift(1) * gap + scaled * intra).sum(axis=1)
    cost = turnover * one_way + carry_cost
    ret = (gross_pnl - cost).fillna(0.0)

    out = pd.DataFrame({
        "ret": ret, "gross_pnl": gross_pnl.fillna(0.0), "cost": cost.fillna(0.0),
        "gross_exposure": scaled.abs().sum(axis=1), "net_exposure": scaled.sum(axis=1),
        "turnover": turnover, "n_assets": (scaled.abs() > 1e-6).sum(axis=1),
    }, index=index).iloc[:-1]
    out["equity"] = 1e6 * (1 + out["ret"]).cumprod()
    out["gross_equity"] = 1e6 * (1 + out["gross_pnl"]).cumprod()
    return out


def trend_signal(df: pd.DataFrame, lookback_bars: int, long_only: bool = True,
                 gain: float = 1.5, vol_window: int = 30) -> pd.Series:
    """`backtest.trend_policy` と同じ定義のシグナル（ベクトル版）。"""
    log_close = np.log(df["close"])
    logret = log_close.diff()
    vol = logret.rolling(vol_window, min_periods=vol_window // 2).std()
    signal = (log_close - log_close.shift(lookback_bars)) / (vol * np.sqrt(lookback_bars))
    signal = (signal / gain).clip(-1, 1).fillna(0.0)
    return signal.clip(lower=0) if long_only else signal


# --------------------------------------------------------------------------- シグナル構築
LADDER_DAYS = (5, 14, 30, 60)


def ladder_signal(
    df: pd.DataFrame,
    bars_per_day: int,
    lookback_days: tuple[float, ...] = LADDER_DAYS,
    long_only: bool = False,
    gain: float = 1.5,
    vol_window: int = 30,
) -> pd.Series:
    """複数ルックバックのトレンドを平均した「ラダー」シグナル。

    単一のルックバック（例 14 日）を選ぶと、その値への当てはめリスクが残る。
    5/14/30/60 日を等ウェイトで平均すると、**どの周期のトレンドにも部分的に乗る**ため、
    パラメータ選択の影響が薄まり、実測でも 2 つの独立した時代の両方で成績が上がった
    （時代A 0.94→1.01、時代B 1.59→1.86）。

    `long_only=False`（両建て）が既定である点に注意。単一銘柄ではショート側は負けるが、
    **複数銘柄に分散すると下落局面の収益源になる**（2018 年 Sharpe −0.13→+1.74、
    2022 年 +0.52→+1.09）。詳細は `docs/strategy_search.md`。
    """
    parts = [trend_signal(df, max(2, int(d * bars_per_day)), long_only=False, gain=gain, vol_window=vol_window)
             for d in lookback_days]
    combined = pd.concat(parts, axis=1).mean(axis=1)
    return combined.clip(lower=0) if long_only else combined


def apply_rebalance_band(signal: pd.Series, width: float = 0.10) -> pd.Series:
    """目標が一定幅を超えて動いたときだけ建玉を更新する（回転率の削減）。

    毎バー律儀に建て直すと、シグナルの微小な揺れにスプレッドを払い続ける。
    幅 0.10 で回転率が 0.27 → 0.18 回/日に下がり、成績はむしろ僅かに改善した。
    """
    values = np.array(signal.to_numpy(), dtype=float, copy=True)
    current = 0.0
    for i, target in enumerate(values):
        if not np.isfinite(target):
            target = current
        if abs(target - current) > width:
            current = float(target)
        values[i] = current
    return pd.Series(values, index=signal.index)


def carry_avoidance_mask(index: pd.DatetimeIndex, hour_jst: int = 6) -> pd.Series:
    """建玉管理料の課金時刻をまたぐバーだけフラットにするマスク。

    GMO の建玉管理料は「JST 06:00 時点の建玉」に 0.04% 課される。その 1 本だけ
    建玉を落とせば課金を避けられる。ただし**往復コストが 4bp を下回る場合しか得しない**
    （実測: 片道 1.0bp なら Sharpe +0.15、片道 2.5bp なら −0.03 で逆効果）。
    指値がほぼ確実に約定する運用でのみ有効。既定では使わないこと。
    """
    flags = carry_flags(pd.DatetimeIndex(index), hour_jst)
    return pd.Series((~flags).astype(float), index=index)
