#!/usr/bin/env python3
"""GMO コイン レバレッジ取引の実行ループ（戦略 v2・5 銘柄）。

**2 つの速度で回す**

    リスク監視  既定 60 秒ごと … データ鮮度・スプレッド・証拠金維持率・ドローダウン・
                                日次損失。異常なら即座に縮小/全決済する
    リバランス  既定 1 時間ごと … シグナルを再計算し、目標建玉との差だけを発注する

シグナルは 5〜60 日のトレンドなので、再計算を細かくしても見えるのはノイズだけ
（実測で 5 分〜8 時間の間に有意差なし、細かいほどドローダウンは悪化）。
一方でリスク事象は分単位で起きるため、監視だけを速くしている。

**GMO 固有の扱い**

- 建玉を**増やす**ときは新規注文（`/v1/order`）、**減らす/反対に返す**ときは決済注文
  （`/v1/closeBulkOrder`）。素朴に反対売買を出すと両建てが積み上がる。
- 建玉管理料は JST 06:00 時点の建玉に 0.04%。数日保有の戦略なので織り込み済み。
- 取引手数料は無料。実質コストはスプレッドと建玉管理料。

既定はドライラン（発注しない）。`--live` を付けたときだけ実際に発注する。

Example:
    python scripts/live_gmo.py --config configs/gmo_live.yaml            # ドライラン
    python scripts/live_gmo.py --config configs/gmo_live.yaml --live     # 実発注
    python scripts/live_gmo.py --config configs/gmo_live.yaml --once     # 1 回だけ実行して終了
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.data.gmo_klines import FetchConfig, fetch_klines_day  # noqa: E402
from rlgmo.live import (  # noqa: E402
    LiveConfig,
    Order,
    StrategyEngine,
    exposure_from_positions,
    realized_vol_ann,
    split_close_open,
    staleness_seconds,
)
from rlgmo.risk import RiskLimits, RiskManager  # noqa: E402

PUBLIC = "https://api.coin.z.com/public"
PRIVATE = "https://api.coin.z.com/private"


# --------------------------------------------------------------------------- API
def private_request(method: str, path: str, body: dict | None = None) -> dict:
    """GMO Private API への署名付きリクエスト。"""
    key, secret = os.environ.get("GMO_API_KEY"), os.environ.get("GMO_API_SECRET")
    if not key or not secret:
        raise RuntimeError("GMO_API_KEY / GMO_API_SECRET が設定されていません")
    timestamp = str(int(time.time() * 1000))
    payload = json.dumps(body) if body and method == "POST" else ""
    sign = hmac.new(secret.encode(), (timestamp + method + path + payload).encode(), hashlib.sha256).hexdigest()
    headers = {"API-KEY": key, "API-TIMESTAMP": timestamp, "API-SIGN": sign, "Content-Type": "application/json"}
    url = PRIVATE + path
    res = (requests.post(url, headers=headers, data=payload, timeout=15) if method == "POST"
           else requests.get(url, headers=headers, params=body, timeout=15))
    res.raise_for_status()
    out = res.json()
    if out.get("status") != 0:
        raise RuntimeError(f"GMO private API error: {out}")
    return out


def public_json(path: str, params: dict | None = None) -> dict:
    res = requests.get(PUBLIC + path, params=params, timeout=15)
    res.raise_for_status()
    out = res.json()
    if out.get("status") != 0:
        raise RuntimeError(f"GMO public API error: {out}")
    return out


def fetch_tickers(symbols: tuple[str, ...]) -> dict[str, dict]:
    """全銘柄の最良気配を 1 回のリクエストで取る。"""
    data = public_json("/v1/ticker").get("data", [])
    out = {}
    for row in data:
        if row.get("symbol") in symbols:
            bid, ask = float(row["bid"]), float(row["ask"])
            mid = (bid + ask) / 2
            out[row["symbol"]] = {"bid": bid, "ask": ask, "mid": mid, "last": float(row["last"]),
                                 "half_spread_bp": (ask - bid) / 2 / mid * 1e4 if mid else float("inf")}
    return out


def fetch_buffer(symbols: tuple[str, ...], days: int) -> dict[str, pd.DataFrame]:
    """各銘柄の 1 分足を、必要な助走ぶんだけ取得する。"""
    out = {}
    today = datetime.now(timezone.utc)
    for symbol in symbols:
        frames = []
        for back in range(days, -1, -1):
            date = (today - pd.Timedelta(days=back)).strftime("%Y%m%d")
            try:
                frames.append(fetch_klines_day(date, FetchConfig(symbol=symbol, interval="1min")))
            except RuntimeError as err:
                print(f"[warn] {symbol} {date}: {err}")
            time.sleep(0.35)      # Public API のレート制限に配慮
        if frames:
            df = pd.concat(frames)
            out[symbol] = df[~df.index.duplicated(keep="last")].sort_index()
    return out


def account_state(symbols: tuple[str, ...]) -> tuple[float, float, dict[str, float]]:
    """有効証拠金・証拠金維持率・銘柄ごとの正味建玉数量を取る。"""
    margin = private_request("GET", "/v1/account/margin").get("data", {})
    equity = float(margin.get("actualProfitLoss", 0.0))
    ratio = float(margin.get("marginRatio", 0.0) or 0.0) / 100.0 or float("inf")
    positions = {}
    for symbol in symbols:
        data = private_request("GET", "/v1/positionSummary", {"symbol": symbol}).get("data", {})
        net = 0.0
        for row in data.get("list", []) or []:
            qty = float(row["sumOrderQuantity"])
            net += qty if row["side"] == "BUY" else -qty
        positions[symbol] = net
    return equity, ratio, positions


def place(order: Order, held_qty: float, dry_run: bool) -> dict:
    """建玉を増やすなら新規、減らす/返すなら決済。GMO は API が分かれている。"""
    close_qty, open_qty = split_close_open(order, held_qty)
    actions = []
    if close_qty > 1e-9:
        actions.append(("決済", "/v1/closeBulkOrder",
                        {"symbol": order.symbol, "side": order.side, "executionType": "MARKET",
                         "size": f"{close_qty:.4f}"}))
    if open_qty > 1e-9:
        actions.append(("新規", "/v1/order",
                        {"symbol": order.symbol, "side": order.side, "executionType": "MARKET",
                         "size": f"{open_qty:.4f}"}))
    results = []
    for label, path, body in actions:
        if dry_run:
            print(f"  [ドライラン] {label}: {body}")
            results.append({"dry_run": True, "kind": label, **body})
        else:
            results.append({"kind": label, **private_request("POST", path, body)})
    return {"orders": results}


# --------------------------------------------------------------------------- ループ
def load_config(path: Path) -> tuple[LiveConfig, RiskLimits, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    s, c, r = raw.get("strategy", {}), raw.get("cost", {}), raw.get("risk", {})
    live = LiveConfig(
        symbols=tuple(raw["data"]["symbols"]),
        lookback_days=tuple(s.get("lookback_days", (5, 14, 30, 60))),
        long_only=bool(s.get("long_only", False)),
        gain=float(s.get("gain", 1.5)),
        vol_window_bars=int(s.get("vol_window_bars", 30)),
        grid_hours=int(s.get("grid_hours", 1)),
        risk_interval_sec=int(raw.get("execution", {}).get("risk_interval_sec", 60)),
        rebalance_band=float(s.get("rebalance_band", 0.10)),
        asset_vol_ann=float(s.get("asset_vol_ann", 0.15)),
        target_vol_ann=float(s.get("target_vol_ann", 0.15)),
        max_weight=float(s.get("max_weight", 0.5)),
        leverage_cap=float(s.get("leverage_cap", 2.0)),
        min_trade_delta=float(r.get("min_trade_delta", 0.005)),
        cost=CostConfig(half_spread_bp=float(c.get("half_spread_bp", 1.5)),
                        slippage_bp=float(c.get("slippage_bp", 0.0)),
                        carry_rate_daily=float(c.get("carry_rate_daily", 0.0004)),
                        carry_mode=str(c.get("carry_mode", "daily_0600"))),
    )
    limits = RiskLimits(
        max_position=float(r.get("max_position", 1.0)),
        daily_loss_limit=float(r.get("daily_loss_limit", 0.05)),
        max_drawdown_stop=float(r.get("max_drawdown_stop", 0.20)),
        drawdown_taper=float(r.get("drawdown_taper", 0.12)),
        max_vol_ann=float(r.get("max_vol_ann", 2.0)),
        max_half_spread_bp=float(r.get("max_half_spread_bp", 8.0)),
        max_data_staleness_sec=int(r.get("max_data_staleness_sec", 300)),
        min_margin_ratio=float(r.get("min_margin_ratio", 1.5)),
        min_trade_delta=float(r.get("min_trade_delta", 0.005)),
    )
    return live, limits, raw


def log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/gmo_live.yaml")
    parser.add_argument("--live", action="store_true", help="実際に発注する（既定はドライラン）")
    parser.add_argument("--once", action="store_true", help="1 回だけ実行して終了する")
    parser.add_argument("--equity", type=float, default=None, help="ドライラン時の想定資金（JPY）")
    parser.add_argument("--log", default="runs/live/decisions.csv")
    args = parser.parse_args()

    cfg, limits, raw = load_config(Path(args.config))
    engine = StrategyEngine(cfg)
    warmup_days = int(max(cfg.lookback_days)) + 10
    print(f"[live] 銘柄 {list(cfg.symbols)} / リバランス {cfg.grid_hours} 時間ごと / "
          f"リスク監視 {cfg.risk_interval_sec} 秒ごと / {'実発注' if args.live else 'ドライラン'}")
    print(f"[live] 助走 {warmup_days} 日ぶんの 1 分足を取得します…")

    bars = fetch_buffer(cfg.symbols, warmup_days)
    if not bars:
        raise SystemExit("価格データを取得できませんでした（ネットワーク・銘柄名を確認）")

    equity = args.equity or 1_000_000.0
    positions = {s: 0.0 for s in cfg.symbols}
    if args.live:
        equity, _, positions = account_state(cfg.symbols)
    risk = RiskManager(limits, equity)
    last_rebalance = None
    log_path = Path(args.log)

    while True:
        try:
            now = pd.Timestamp.now(tz="UTC")
            tickers = fetch_tickers(cfg.symbols)
            prices = {s: t["mid"] for s, t in tickers.items()}
            spread = max((t["half_spread_bp"] for t in tickers.values()), default=float("inf"))

            if args.live:
                equity, margin_ratio, positions = account_state(cfg.symbols)
            else:
                margin_ratio = float("inf")
            current = exposure_from_positions(positions, prices, equity)
            gross = sum(abs(v) for v in current.values())

            stale = staleness_seconds(bars, now)
            vol_ann = realized_vol_ann(bars)
            # リスクレイヤは「全建玉に掛ける縮小係数」を出す。current=1.0 は
            # 「いまフルスケールで建てている」という意味で渡す。
            size, decision = risk.apply(1.0, equity, vol_ann, spread, stale, margin_ratio, current=1.0)
            reasons = [r for r in decision.get("reasons", []) if r != "below_min_delta"]
            # 「:hold」で終わる理由（データが古い・板が壊れている）は**発注を見送る**という意味。
            # ここで全決済すると、いちばん約定が悪い場面で投げることになる。
            hold_only = any(r.endswith(":hold") for r in reasons)
            flatten = decision.get("halted") or (size == 0.0 and not hold_only)

            due = (not hold_only and (last_rebalance is None or
                   (now - last_rebalance) >= pd.Timedelta(hours=cfg.grid_hours)))
            if hold_only:
                print(f"[{now:%H:%M:%S}] 発注を見送り: {','.join(reasons)}（建玉は維持）")
            if due:
                # --- リバランス: 最新の足を継ぎ足してから目標を計算する
                fresh = fetch_buffer(cfg.symbols, 1)
                for symbol, df in fresh.items():
                    merged = pd.concat([bars.get(symbol, df.iloc[:0]), df])
                    bars[symbol] = merged[~merged.index.duplicated(keep="last")].sort_index()
                stale = staleness_seconds(bars, now)

                targets = engine.targets(bars)
                orders = engine.orders(targets, current, equity, prices, size_scale=size)
                print(f"\n[{now:%Y-%m-%d %H:%M} UTC] 資金 {equity:,.0f} / グロス建玉 {gross:.1%} / "
                      f"スプレッド {spread:.1f}bp / 鮮度 {stale:.0f}s / リスク係数 {size:.2f}"
                      + (f" / {','.join(reasons)}" if reasons else ""))
                for symbol, target in targets.items():
                    print(f"  {symbol:9s} 目標 {target * size:+.3f}  現在 {current.get(symbol, 0.0):+.3f}")
                if not orders:
                    print("  発注なし（差が最小発注幅 未満）")
                for order in orders:
                    print("  " + order.describe())
                    place(order, positions.get(order.symbol, 0.0), dry_run=not args.live)
                log_row(log_path, {"ts": now, "equity": equity, "gross": gross, "spread_bp": spread,
                                   "stale_s": stale, "vol_ann": vol_ann, "size_scale": size,
                                   "reasons": ";".join(reasons),
                                   **{f"target_{s}": v * size for s, v in targets.items()},
                                   **{f"current_{s}": v for s, v in current.items()}})
                last_rebalance = now
            elif flatten:
                # --- リスク監視で停止条件に触れたら、リバランスを待たずに畳む
                print(f"[{now:%H:%M:%S}] リスク停止: {','.join(reasons)} → 建玉を落とします")
                for symbol, exposure in current.items():
                    if abs(exposure) > cfg.min_trade_delta:
                        qty = abs(exposure) * equity / prices.get(symbol, 1.0)
                        order = Order(symbol=symbol, side="SELL" if exposure > 0 else "BUY", quantity=qty,
                                      delta_exposure=-exposure, target_exposure=0.0,
                                      current_exposure=exposure, price=prices.get(symbol, 0.0))
                        place(order, positions.get(symbol, 0.0), dry_run=not args.live)
                log_row(log_path, {"ts": now, "equity": equity, "gross": gross, "spread_bp": spread,
                                   "stale_s": stale, "size_scale": 0.0, "reasons": ";".join(reasons)})

            if args.once:
                break
            time.sleep(cfg.risk_interval_sec)
        except KeyboardInterrupt:
            print("\n停止しました")
            break
        except Exception as err:  # noqa: BLE001 - 監視ループは止めない
            print(f"[error] {err}")
            if args.once:
                raise
            time.sleep(cfg.risk_interval_sec)


if __name__ == "__main__":
    main()
