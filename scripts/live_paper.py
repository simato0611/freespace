#!/usr/bin/env python3
"""ライブ推論ループ（既定はドライラン / ペーパートレード）。

毎分のバークローズ後に

    1. 最新の 1 分足を取得（Public API）
    2. 直近バッファから特徴量を再計算（学習時と完全に同じコード）
    3. アンサンブル方策で目標ポジションを推論
    4. リスクレイヤ（`rlgmo.risk`）でクリップ
    5. 現在ポジションとの差分だけを発注

を行う。`--live` を付けない限り**発注はせず、意思決定ログのみを出力**する。

必要な環境変数（--live 時のみ）:
    GMO_API_KEY, GMO_API_SECRET

注意:
    - 本番投入前に、必ず同じコードで数週間のペーパートレードを行い、
      バックテストの想定コスト・約定価格と実測を突き合わせること。
    - 発注ロジックはユーザーの口座・数量制約に依存する。`place_order` は
      最小限の雛形であり、数量丸め・最小発注単位・エラー処理は各自で確認すること。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from rlgmo.agents.ppo import PPOAgent  # noqa: E402
from rlgmo.config import load_config  # noqa: E402
from rlgmo.data.gmo_klines import FetchConfig, fetch_klines_day  # noqa: E402
from rlgmo.features import build_features  # noqa: E402
from rlgmo.risk import RiskLimits, RiskManager  # noqa: E402

PUBLIC = "https://api.coin.z.com/public"
PRIVATE = "https://api.coin.z.com/private"


def private_request(method: str, path: str, body: dict | None = None) -> dict:
    """GMO Private API への署名付きリクエスト（--live 時のみ使用）。"""
    key, secret = os.environ.get("GMO_API_KEY"), os.environ.get("GMO_API_SECRET")
    if not key or not secret:
        raise RuntimeError("GMO_API_KEY / GMO_API_SECRET が設定されていません")
    timestamp = str(int(time.time() * 1000))
    payload = json.dumps(body) if body else ""
    sign = hmac.new(secret.encode(), (timestamp + method + path + payload).encode(), hashlib.sha256).hexdigest()
    headers = {"API-KEY": key, "API-TIMESTAMP": timestamp, "API-SIGN": sign, "Content-Type": "application/json"}
    url = PRIVATE + path
    res = requests.post(url, headers=headers, data=payload, timeout=15) if method == "POST" else requests.get(
        url, headers=headers, params=body, timeout=15
    )
    res.raise_for_status()
    out = res.json()
    if out.get("status") != 0:
        raise RuntimeError(f"GMO private API error: {out}")
    return out


def current_position(symbol: str) -> float:
    """建玉サマリから現在の正味数量を取得する（ロング +, ショート -）。"""
    data = private_request("GET", "/v1/positionSummary", {"symbol": symbol}).get("data", {})
    net = 0.0
    for row in data.get("list", []) or []:
        qty = float(row["sumOrderQuantity"])
        net += qty if row["side"] == "BUY" else -qty
    return net


def place_order(symbol: str, side: str, size: float, dry_run: bool = True) -> dict:
    """成行で差分を発注する（dry_run ではログのみ）。"""
    body = {"symbol": symbol, "side": side, "executionType": "MARKET", "size": f"{size:.4f}"}
    if dry_run:
        print(f"[DRY-RUN] order: {body}")
        return {"dry_run": True, **body}
    return private_request("POST", "/v1/order", body)


def latest_spread_bp(symbol: str) -> float:
    book = requests.get(f"{PUBLIC}/v1/orderbooks", params={"symbol": symbol}, timeout=10).json()["data"]
    bid, ask = float(book["bids"][0]["price"]), float(book["asks"][0]["price"])
    return (ask - bid) / 2 / ((ask + bid) / 2) * 1e4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", required=True, help="モデルの glob パターン")
    parser.add_argument("--live", action="store_true", help="実際に発注する（既定はドライラン）")
    parser.add_argument("--equity", type=float, default=1_000_000.0, help="運用資金（JPY）")
    parser.add_argument("--max-notional", type=float, default=500_000.0, help="建玉評価額の上限（JPY）")
    parser.add_argument("--log", default="runs/live/decisions.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = cfg.data.symbol
    agents = [PPOAgent.load(p) for p in sorted(glob.glob(args.models))]
    if not agents:
        raise SystemExit(f"モデルが見つかりません: {args.models}")
    values = np.asarray(cfg.env.actions, dtype=float)
    risk = RiskManager(RiskLimits(), equity=args.equity)
    fetch_cfg = FetchConfig(symbol=symbol, interval="1min")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    print(f"[live] symbol={symbol} models={len(agents)} live={args.live}")

    position = 0.0  # ポジション比率（-1〜1）
    while True:
        now = datetime.now(timezone.utc)
        time.sleep(max(0.0, 62 - now.second - now.microsecond / 1e6))  # バークローズ + 2 秒待つ
        try:
            today = datetime.now(timezone.utc)
            frames = [fetch_klines_day((today - pd.Timedelta(days=d)).strftime("%Y%m%d"), fetch_cfg) for d in (1, 0)]
            ohlcv = pd.concat(frames)
            ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()
            features, meta = build_features(ohlcv, cfg.features)
            if features.empty:
                print("[warn] 特徴量が空（ウォームアップ不足）")
                continue

            staleness = (pd.Timestamp.utcnow().tz_localize("UTC") - features.index[-1]).total_seconds()
            obs_market = features.iloc[-1].to_numpy(dtype=np.float32)
            account = np.array([position, abs(position), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs = np.concatenate([obs_market, account])

            probs = np.mean([a.probs(obs)[0] for a in agents], axis=0)
            target = float(probs @ values)
            if abs(target) < cfg.train.confidence:
                target = 0.0

            vol_ann = float(meta["vol_1m"].iloc[-1] * np.sqrt(365 * 24 * 60))
            spread_bp = latest_spread_bp(symbol)
            equity = args.equity
            margin_ratio = float("inf")
            if args.live:
                equity = float(private_request("GET", "/v1/account/margin")["data"]["actualProfitLoss"])
                margin_ratio = float(private_request("GET", "/v1/account/margin")["data"].get("marginRatio", 999)) / 100

            size, decision = risk.apply(target, equity, vol_ann, spread_bp, staleness, margin_ratio, position)
            delta = size - position
            price = float(meta["close"].iloc[-1])
            qty = abs(delta) * min(args.max_notional, equity * cfg.env.leverage_cap) / price

            row = {
                "ts": features.index[-1], "price": price, "target": target, "size": size, "delta": delta,
                "qty": round(qty, 4), "spread_bp": round(spread_bp, 2), "vol_ann": round(vol_ann, 3),
                "staleness_s": round(staleness, 1), "reasons": ";".join(decision.get("reasons", [])),
            }
            print(json.dumps(row, default=str, ensure_ascii=False))
            pd.DataFrame([row]).to_csv(args.log, mode="a", header=not Path(args.log).exists(), index=False)

            if abs(delta) > 1e-6 and qty > 0:
                place_order(symbol, "BUY" if delta > 0 else "SELL", qty, dry_run=not args.live)
                position = size
        except KeyboardInterrupt:
            print("停止しました")
            break
        except Exception as err:  # noqa: BLE001 - ループは止めず、次のバーで再試行
            print(f"[error] {err}")


if __name__ == "__main__":
    main()
