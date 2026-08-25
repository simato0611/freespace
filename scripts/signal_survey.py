#!/usr/bin/env python3
"""仮説探索: 単純なシグナルを、実コスト込みで横並びに測る。

**なぜ RL より先にこれをやるのか**: 強化学習は「既にある予測力の使い方」を最適化する
道具であって、予測力を作り出す道具ではない。素朴なシグナルで優位性の影も見えない
時間軸・情報セットでは、RL を何時間回しても結果は変わらない（`docs/real_data_findings.md`）。
だからまず、**安価に測れるシグナルを大量に並べて、勝ち目のある領域を特定する**。

プロトコル（探索を始める前に固定する）:

- **ホールドアウト期間は触らない**。既定では `--holdout-start` 以降を完全に除外して探索する。
- ここで測るのは開発期間のみ。良い結果が出ても、それは「選ばれた最良」であり
  選択バイアスが乗っている。最終判断は Deflated Sharpe（試行回数で割り引く）で行う。
- 試したシグナルは**全部** `docs/experiment_log.md` に記録する（捨てたものも含む）。

Example:
    python scripts/signal_survey.py --data data/raw/BTCUSD_bitstamp_1min_full.parquet \
        --grid 1440 --holdout-start 2025-07-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.costs import CostConfig, carry_flags  # noqa: E402
from rlgmo.data.gmo_klines import load_ohlcv  # noqa: E402
from rlgmo.data.resample import resample_ohlcv  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402

MINUTES_PER_YEAR = 365 * 24 * 60


# ----------------------------------------------------------------------------- シグナル
def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-3, 3)


def momentum(df: pd.DataFrame, lookback: int, vol_window: int = 60) -> pd.Series:
    """時系列モメンタム: 過去 `lookback` バーのリターンをボラで正規化。"""
    logret = np.log(df["close"]).diff()
    ret = np.log(df["close"]) - np.log(df["close"].shift(lookback))
    vol = logret.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(lookback)
    return (ret / vol.replace(0.0, np.nan)).clip(-3, 3) / 1.5


def reversal(df: pd.DataFrame, lookback: int) -> pd.Series:
    return -momentum(df, lookback)


def ema_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """EMA クロス（乖離をボラで正規化した連続値）。"""
    close = df["close"]
    spread = close.ewm(span=fast, min_periods=fast).mean() - close.ewm(span=slow, min_periods=slow).mean()
    vol = np.log(close).diff().rolling(slow, min_periods=slow // 2).std() * close
    return (spread / vol.replace(0.0, np.nan)).clip(-3, 3) / 2.0


def donchian(df: pd.DataFrame, lookback: int) -> pd.Series:
    """ドンチャン・ブレイクアウト: 上抜けでロング、下抜けでショート、それ以外は維持。"""
    hh = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    ll = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    raw = pd.Series(np.nan, index=df.index)
    raw[df["close"] > hh] = 1.0
    raw[df["close"] < ll] = -1.0
    return raw.ffill().fillna(0.0)


def vol_filtered_long(df: pd.DataFrame, vol_window: int = 60) -> pd.Series:
    """低ボラ局面だけロング（ボラ・リスクプレミアムの素朴な実装）。"""
    vol = np.log(df["close"]).diff().rolling(20, min_periods=10).std()
    median = vol.rolling(vol_window * 4, min_periods=vol_window).median()
    return (vol < median).astype(float)


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def build_signals(df: pd.DataFrame, grid_minutes: int) -> dict[str, pd.Series]:
    """事前登録したシグナル一覧（探索の試行回数はここの本数で数える）。"""
    per_day = max(1, 1440 // grid_minutes)
    lb = {"1d": per_day, "3d": 3 * per_day, "7d": 7 * per_day, "14d": 14 * per_day,
          "30d": 30 * per_day, "90d": 90 * per_day}
    signals: dict[str, pd.Series] = {"buy_hold": buy_hold(df)}
    for name, k in lb.items():
        if k >= 2:
            signals[f"mom_{name}"] = momentum(df, k)
    for name in ("1d", "3d"):
        if lb[name] >= 2:
            signals[f"rev_{name}"] = reversal(df, lb[name])
    for fast, slow in (("1d", "7d"), ("3d", "30d"), ("7d", "90d")):
        if lb[fast] >= 2 and lb[slow] >= 2:
            signals[f"ema_{fast}_{slow}"] = ema_cross(df, lb[fast], lb[slow])
    for name in ("7d", "30d", "90d"):
        if lb[name] >= 2:
            signals[f"donchian_{name}"] = donchian(df, lb[name])
    signals["vol_filtered_long"] = vol_filtered_long(df)
    return signals


# ----------------------------------------------------------------------------- 検証
def simulate(
    df: pd.DataFrame,
    signal: pd.Series,
    grid_minutes: int,
    cost: CostConfig,
    target_vol_ann: float = 0.20,
    leverage_cap: float = 2.0,
    long_only: bool = False,
) -> pd.DataFrame:
    """シグナル → ボラターゲット → 次バー始値で執行 → コスト控除、をベクトル化して回す。

    ルックアヘッド防止: 時刻 t のクローズで決めた建玉は t+1 の始値で約定する。
    """
    logret = np.log(df["close"]).diff()
    vol_window = max(10, 1440 // grid_minutes * 20)  # 直近 20 日相当
    realized = logret.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(
        MINUTES_PER_YEAR / grid_minutes
    )
    scale = (target_vol_ann / realized.replace(0.0, np.nan)).clip(0.0, leverage_cap)
    raw = signal.clip(0, 1) if long_only else signal.clip(-1, 1)
    exposure = (raw * scale).fillna(0.0)                # 有効証拠金に対する建玉倍率

    open_, close = df["open"], df["close"]
    r_gap = (open_.shift(-1) / close - 1.0).fillna(0.0)          # t クローズ → t+1 始値
    r_intra = (close.shift(-1) / open_.shift(-1) - 1.0).fillna(0.0)  # t+1 始値 → クローズ
    exposure_prev = exposure.shift(1).fillna(0.0)

    one_way = (cost.half_spread_bp + cost.slippage_bp + cost.taker_fee_bp) * 1e-4
    turnover = (exposure - exposure_prev).abs()
    carry = pd.Series(carry_flags(pd.DatetimeIndex(df.index), cost.carry_hour_jst), index=df.index)
    carry_rate = carry.astype(float) * cost.carry_rate_daily * exposure.abs()

    gross = exposure_prev * r_gap + exposure * r_intra
    net = gross - turnover * one_way - carry_rate
    out = pd.DataFrame(
        {"exposure": exposure, "gross": gross, "net": net,
         "cost": turnover * one_way + carry_rate, "turnover": turnover},
        index=df.index,
    ).iloc[:-1]
    out["equity"] = 1e6 * (1 + out["net"]).cumprod()
    out["gross_equity"] = 1e6 * (1 + out["gross"]).cumprod()
    return out


def alpha_vs_benchmark(net: pd.Series, benchmark: pd.Series, grid_minutes: int) -> dict:
    """ベンチマーク（ボラターゲット済み Buy & Hold）に対する α・β・情報比を測る。

    暗号資産の上昇相場では、どんな買い持ち系の戦略でも Sharpe が高く出る。
    「本当に優位性があるのか、単に BTC ロングのベータを言い換えているだけか」を
    切り分けるため、ベンチマークで説明できない残差リターンだけを評価する。
    """
    aligned = pd.concat([net.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(aligned) < 30 or aligned["b"].std() == 0:
        return {"α/年": float("nan"), "β": float("nan"), "情報比": float("nan")}
    beta = float(aligned["s"].cov(aligned["b"]) / aligned["b"].var())
    residual = aligned["s"] - beta * aligned["b"]
    periods = MINUTES_PER_YEAR / grid_minutes
    ir = float(residual.mean() / residual.std() * np.sqrt(periods)) if residual.std() > 0 else 0.0
    return {"α/年": float(residual.mean() * periods), "β": beta, "情報比": ir}


def evaluate(result: pd.DataFrame, grid_minutes: int) -> dict:
    metrics = equity_metrics(result["equity"], result["exposure"])
    gross = equity_metrics(result["gross_equity"], result["exposure"])
    days = len(result) * grid_minutes / 1440
    return {
        "Sharpe": metrics["sharpe"],
        "グロスSharpe": gross["sharpe"],
        "年率リターン": metrics["cagr"],
        "最大DD": metrics["max_drawdown"],
        "年率ボラ": metrics["ann_vol"],
        "回転/日": metrics["turnover_per_day"],
        "コスト/年": float(result["cost"].sum() / max(days / 365, 1e-9)),
        "平均|建玉|": metrics["exposure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/raw/BTCUSD_bitstamp_1min_full.parquet")
    parser.add_argument("--grid", type=int, default=1440, help="判断間隔（分）")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--holdout-start", default="2025-07-01",
                        help="この日以降は探索に使わない（封印するホールドアウト）")
    parser.add_argument("--half-spread-bp", type=float, default=2.0)
    parser.add_argument("--slippage-bp", type=float, default=0.5)
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--out", default="runs/analysis/signal_survey.csv")
    args = parser.parse_args()

    ohlcv = load_ohlcv(args.data).loc[args.start :]
    df = resample_ohlcv(ohlcv, args.grid) if args.grid > 1 else ohlcv
    dev = df.loc[: pd.Timestamp(args.holdout_start, tz="UTC")]
    print(f"[survey] 判断間隔 {args.grid} 分 / 開発期間 {dev.index[0]:%Y-%m-%d} 〜 {dev.index[-1]:%Y-%m-%d} "
          f"({len(dev):,} バー) / ホールドアウト {args.holdout_start} 以降は未使用")

    cost = CostConfig(half_spread_bp=args.half_spread_bp, slippage_bp=args.slippage_bp,
                      carry_mode="daily_0600", spread_vol_beta=0.0)
    signals = build_signals(dev, args.grid)
    benchmark = simulate(dev, buy_hold(dev), args.grid, cost, args.target_vol)["net"]
    rows = {}
    yearly = {}
    for name, signal in signals.items():
        result = simulate(dev, signal, args.grid, cost, args.target_vol)
        rows[name] = evaluate(result, args.grid)
        rows[name].update(alpha_vs_benchmark(result["net"], benchmark, args.grid))
        by_year = result.groupby(result.index.year)["net"].apply(
            lambda x: x.mean() / x.std() * np.sqrt(MINUTES_PER_YEAR / args.grid) if x.std() > 0 else 0.0)
        yearly[name] = by_year
        rows[name]["年別Sharpe>0"] = float((by_year > 0).mean())

    table = pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)
    pd.DataFrame(yearly).round(2).to_csv(str(args.out).replace(".csv", "_yearly.csv"))

    show = table.copy()
    for col in ("年率リターン", "最大DD", "年率ボラ", "コスト/年", "平均|建玉|", "年別Sharpe>0", "α/年"):
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    for col in ("Sharpe", "グロスSharpe", "回転/日", "β", "情報比"):
        show[col] = show[col].round(2)
    print("\n" + show.to_string())
    print(f"\n試行本数: {len(signals)}（Deflated Sharpe の n_trials に加算すること）")
    print(f"出力: {args.out}")
    print("\n年別 Sharpe:")
    print(pd.DataFrame(yearly).round(1).to_string())


if __name__ == "__main__":
    main()
