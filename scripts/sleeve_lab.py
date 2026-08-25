#!/usr/bin/env python3
"""第八次探索: 同じ収益源から独立ベットを増やせるかを検証する（開発期間のみ）。

プロトコルと採用条件は docs/strategy_search.md 35 節で**実行前に固定**してある。
ここは実行するだけの場所であり、結果を見てから条件を緩めてはいけない。

- H1 ホライズン別スリーブ: ラダーは 4 本を平均して 1 つの数にしている。
  平均せず各ホライズンを個別にリスク配分すれば、有効ベット数が増えるか
- H2 残差トレンド: 市場ファクターを抜いた残差にトレンド則を当てる
- H3 分散度の条件づけ: 銘柄間の分散が大きい局面ほどトレンドが効くか

Example:
    python scripts/sleeve_lab.py --era both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlgmo.costs import CostConfig, carry_flags, carry_rate_per_bar  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.portfolio import (  # noqa: E402
    LADDER_DAYS, PortfolioConfig, apply_rebalance_band, ladder_signal, trend_signal,
)

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
MINUTES_PER_YEAR = 365 * 24 * 60
HOLDOUT = pd.Timestamp("2025-01-01", tz="UTC")

ERAS = {
    "A": dict(dirs=["data/handoff/prices/alt2017_1h", "data/handoff/prices/verify_sources"],
              end="2020-05-31", label="時代A 2017-10〜2020-05"),
    "B": dict(dirs=["data/handoff/prices/perp_1h"],
              end="2024-12-31", label="時代B 2020-01〜2024-12"),
}


def load_era(era: str, grid_hours: int = 1) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for d in ERAS[era]["dirs"]:
        for path in sorted(Path(d).glob("*.parquet")):
            name = path.stem.split("_")[0].replace("BTCUSDT", "BTC")
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.resample(f"{grid_hours}h", label="right", closed="right").agg(
                {k: v for k, v in AGG.items() if k in df.columns}).dropna(subset=["close"])
            out[name] = df.loc[: ERAS[era]["end"]]
    return out


# --------------------------------------------------------------- 共通の器
def align(prices: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    index = None
    for df in prices.values():
        index = df.index if index is None else index.union(df.index)
    return pd.DatetimeIndex(index).sort_values()


def realized_vol(returns: pd.Series, window: int, ppy: float) -> pd.Series:
    return returns.rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(ppy)


def asset_exposures(prices: dict[str, pd.DataFrame], signals: dict[str, pd.Series],
                    index: pd.DatetimeIndex, cfg: PortfolioConfig, grid_minutes: int) -> pd.DataFrame:
    """1 段目だけ（銘柄ごとの等リスク配分）。全体のボラ目標は別途かける。"""
    ppy = MINUTES_PER_YEAR / grid_minutes
    window = max(5, int(cfg.vol_window_days * 1440 / grid_minutes))
    out = {}
    for asset, df in prices.items():
        d = df.reindex(index)
        vol = realized_vol(np.log(d["close"]).diff(), window, ppy)
        scale = (cfg.asset_vol_ann / vol.replace(0.0, np.nan)).clip(0.0, cfg.leverage_cap)
        raw = signals[asset].reindex(index).clip(-1, 1)
        out[asset] = (raw * scale).clip(-cfg.max_weight, cfg.max_weight).fillna(0.0)
    return pd.DataFrame(out).fillna(0.0)


def returns_frames(prices: dict[str, pd.DataFrame], index: pd.DatetimeIndex):
    gaps, intras = {}, {}
    for asset, df in prices.items():
        d = df.reindex(index)
        gaps[asset] = (d["open"].shift(-1) / d["close"] - 1.0).fillna(0.0)
        intras[asset] = (d["close"].shift(-1) / d["open"].shift(-1) - 1.0).fillna(0.0)
    return pd.DataFrame(gaps).fillna(0.0), pd.DataFrame(intras).fillna(0.0)


def vol_target(exposure: pd.DataFrame, gap: pd.DataFrame, intra: pd.DataFrame,
               cfg: PortfolioConfig, grid_minutes: int) -> pd.DataFrame:
    """2 段目（全体のボラ目標）とレバレッジ上限。ラダーもスリーブ版も同じ器を通す。"""
    ppy = MINUTES_PER_YEAR / grid_minutes
    window = max(10, int(cfg.portfolio_vol_window_days * 1440 / grid_minutes))
    raw = (exposure.shift(1) * gap + exposure * intra).sum(axis=1)
    pf_vol = realized_vol(raw, window, ppy)
    scale = (cfg.target_vol_ann / pf_vol.replace(0.0, np.nan)).clip(0.2, 5.0).shift(1).fillna(1.0)
    scaled = exposure.mul(scale, axis=0)
    over = (scaled.abs().sum(axis=1) / cfg.leverage_cap).clip(lower=1.0)
    return scaled.div(over, axis=0)


def evaluate(exposure: pd.DataFrame, gap: pd.DataFrame, intra: pd.DataFrame,
             cfg: PortfolioConfig, grid_minutes: int) -> tuple[pd.Series, dict]:
    one_way = (cfg.cost.half_spread_bp + cfg.cost.slippage_bp + cfg.cost.taker_fee_bp) * 1e-4
    turnover = (exposure - exposure.shift(1)).abs().sum(axis=1).fillna(0.0)
    flags = carry_flags(exposure.index, cfg.cost.carry_hour_jst)
    carry = pd.Series(np.where(flags, carry_rate_per_bar(cfg.cost, grid_minutes, True),
                               carry_rate_per_bar(cfg.cost, grid_minutes, False)), index=exposure.index)
    gross = (exposure.shift(1) * gap + exposure * intra).sum(axis=1)
    ret = (gross - turnover * one_way - carry * exposure.abs().sum(axis=1)).fillna(0.0).iloc[:-1]
    m = equity_metrics(1e6 * (1 + ret).cumprod())
    days = len(ret) * grid_minutes / 1440
    m["turnover_per_day"] = float(turnover.iloc[:-1].sum() / max(days, 1e-9))
    m["gross_exposure"] = float(exposure.abs().sum(axis=1).mean())
    return ret, m


def band(sig: pd.Series, width: float) -> pd.Series:
    return apply_rebalance_band(sig, width)


# --------------------------------------------------------------- 構成
def build_ladder(prices, index, cfg, grid_minutes, bpd, gain, vw, rb) -> pd.DataFrame:
    """現行 v2: 4 本を平均して 1 つのシグナルにする。"""
    sig = {a: band(ladder_signal(df, bpd, LADDER_DAYS, False, gain, vw), rb) for a, df in prices.items()}
    return asset_exposures(prices, sig, index, cfg, grid_minutes)


def build_sleeves(prices, index, cfg, grid_minutes, bpd, gain, vw, rb,
                  gap, intra, horizons=LADDER_DAYS) -> pd.DataFrame:
    """H1: ホライズンごとに個別へリスク配分してから合成する。"""
    parts = []
    for L in horizons:
        sig = {a: band(trend_signal(df, max(2, int(L * bpd)), False, gain, vw), rb) for a, df in prices.items()}
        e = asset_exposures(prices, sig, index, cfg, grid_minutes)
        parts.append(vol_target(e, gap, intra, cfg, grid_minutes))   # 各スリーブを個別にボラ目標へ
    return sum(parts) / len(parts)


def market_residual_prices(prices, index) -> dict[str, pd.DataFrame]:
    """各銘柄から市場ファクター（等ウェイト）を抜いた合成価格を作る。"""
    logret = pd.DataFrame({a: np.log(df["close"].reindex(index)).diff() for a, df in prices.items()})
    market = logret.mean(axis=1)
    resid = logret.sub(market, axis=0).fillna(0.0)
    out = {}
    for a in prices:
        px = np.exp(resid[a].cumsum())
        out[a] = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=index)
    return out


def build_residual(prices, index, cfg, grid_minutes, bpd, gain, vw, rb, neutral: bool) -> pd.DataFrame:
    """H2: 残差にトレンド則を当てる。neutral=True なら建玉を横断的にゼロ和にする。"""
    res_px = market_residual_prices(prices, index)
    sig = {a: band(ladder_signal(df, bpd, LADDER_DAYS, False, gain, vw), rb) for a, df in res_px.items()}
    if neutral:
        s = pd.DataFrame(sig)
        s = s.sub(s.mean(axis=1), axis=0)          # 市場方向のベットを消す
        sig = {a: s[a] for a in s.columns}
    return asset_exposures(prices, sig, index, cfg, grid_minutes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", default="both", choices=["A", "B", "both"])
    parser.add_argument("--gain", type=float, default=1.5)
    parser.add_argument("--vol-window", type=int, default=30)
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--out", default="runs/sleeve_lab")
    args = parser.parse_args()

    grid_minutes, bpd = 60, 24
    cfg = PortfolioConfig(target_vol_ann=args.target_vol, asset_vol_ann=args.target_vol,
                          cost=CostConfig(half_spread_bp=2.0, slippage_bp=0.5,
                                          carry_mode="daily_0600", spread_vol_beta=0.0))
    eras = ["A", "B"] if args.era == "both" else [args.era]
    rows, curves = [], {}

    for era in eras:
        prices = load_era(era)
        index = align(prices)
        gap, intra = returns_frames(prices, index)
        if era == "B":                                  # 開発期間だけ。封印は触らない
            keep = index < HOLDOUT
            index = index[keep]
            prices = {a: df.loc[df.index < HOLDOUT] for a, df in prices.items()}
            gap, intra = gap.loc[index], intra.loc[index]
        print(f"\n{'='*72}\n{ERAS[era]['label']} / 銘柄 {sorted(prices)} / {len(index):,} 本\n{'='*72}")

        variants = {
            "v2 ラダー（現行）": build_ladder(prices, index, cfg, grid_minutes, bpd, args.gain, args.vol_window, args.band),
            "H1 ホライズン別スリーブ": build_sleeves(prices, index, cfg, grid_minutes, bpd, args.gain, args.vol_window, args.band, gap, intra),
            "H2 残差トレンド（方向あり）": build_residual(prices, index, cfg, grid_minutes, bpd, args.gain, args.vol_window, args.band, neutral=False),
            "H2 残差トレンド（中立化）": build_residual(prices, index, cfg, grid_minutes, bpd, args.gain, args.vol_window, args.band, neutral=True),
        }
        for name, raw_exp in variants.items():
            exp = vol_target(raw_exp, gap, intra, cfg, grid_minutes)
            ret, m = evaluate(exp, gap, intra, cfg, grid_minutes)
            curves[(era, name)] = ret
            rows.append(dict(時代=era, 構成=name, Sharpe=m["sharpe"], 年率=m["cagr"],
                             最大DD=m["max_drawdown"], 年率ボラ=m["ann_vol"],
                             回転=m["turnover_per_day"], グロス=m["gross_exposure"]))

        t = pd.DataFrame([r for r in rows if r["時代"] == era]).drop(columns=["時代"]).set_index("構成")
        show = t.copy()
        for c in ("年率", "最大DD", "年率ボラ"):
            show[c] = (show[c] * 100).round(1).astype(str) + "%"
        for c in ("Sharpe", "回転", "グロス"):
            show[c] = show[c].round(2)
        print(show.to_string())

        base = curves[(era, "v2 ラダー（現行）")]
        print("\n--- v2 との日次相関 ---")
        for name in variants:
            if name == "v2 ラダー（現行）":
                continue
            d = pd.DataFrame({"a": base, "b": curves[(era, name)]}).resample("1D").sum()
            print(f"  {name:<24} {d['a'].corr(d['b']):+.2f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "variants.csv", index=False)
    print(f"\n保存: {out}/variants.csv")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- 混合と局面分析
def blend_test(base_ret: pd.Series, sleeve_ret: pd.Series, weights=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
               grid_minutes: int = 60) -> pd.DataFrame:
    """リスク・パリティで混ぜる（各スリーブを自身のボラで割ってから配分）。

    18 節の基準に合わせる: **重みを変えても改善が続くか**を見る。
    特定の重みでだけ良くなるのは、分散効果ではなくノイズの当たりである。
    """
    ppy = MINUTES_PER_YEAR / grid_minutes
    a = base_ret / (base_ret.std() * np.sqrt(ppy))
    b = sleeve_ret.reindex(base_ret.index).fillna(0.0)
    b = b / (b.std() * np.sqrt(ppy))
    rows = []
    for w in weights:
        mix = (1 - w) * a + w * b
        mix = mix / (mix.std() * np.sqrt(ppy)) * 0.15        # 目標ボラを揃えて比較する
        m = equity_metrics(1e6 * (1 + mix).cumprod())
        rows.append(dict(重み=w, Sharpe=m["sharpe"], 最大DD=m["max_drawdown"]))
    return pd.DataFrame(rows)


def dispersion_study(prices: dict, index: pd.DatetimeIndex, base_ret: pd.Series,
                     grid_minutes: int = 60) -> pd.DataFrame:
    """H3: 銘柄間の分散が大きい局面ほどトレンドが効くか。

    分散度 = 直近 30 日の銘柄別リターンの横断標準偏差。3 分位で成績を割る。
    """
    logret = pd.DataFrame({a: np.log(df["close"].reindex(index)).diff() for a, df in prices.items()})
    disp = logret.std(axis=1).rolling(30 * 24, min_periods=200).mean()
    daily = pd.DataFrame({"ret": base_ret, "disp": disp.reindex(base_ret.index)}).resample("1D").agg(
        {"ret": "sum", "disp": "last"}).dropna()
    daily["tercile"] = pd.qcut(daily["disp"].shift(1), 3, labels=["低", "中", "高"])
    ppy = 365
    rows = []
    for name, grp in daily.groupby("tercile", observed=True):
        rows.append(dict(分散度=name, 日数=len(grp), 平均日次=grp["ret"].mean() * 1e4,
                         Sharpe=grp["ret"].mean() / grp["ret"].std() * np.sqrt(ppy) if grp["ret"].std() > 0 else np.nan))
    return pd.DataFrame(rows)
