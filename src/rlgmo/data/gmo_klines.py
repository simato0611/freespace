"""GMO コイン Public API から K 線（ローソク足）を取得する。

エンドポイント:
    GET https://api.coin.z.com/public/v1/klines
        ?symbol=BTC_JPY&interval=1min&date=YYYYMMDD

- `symbol` にレバレッジ取引銘柄（`BTC_JPY`, `ETH_JPY`, `XRP_JPY` など、末尾が `_JPY`）
  を指定する。現物銘柄は `BTC`, `ETH` のように `_JPY` が付かない。
- 分足 (`1min`〜`30min`) は `date=YYYYMMDD`、時間足以上は `date=YYYY` を渡す。
- Public API はレート制限があるため、リクエスト間隔を空けて連続取得する
  （既定 0.5 秒 + 429/5xx 時の指数バックオフ）。
- 返却される `openTime` はミリ秒 epoch。ここでは **バーのクローズ時刻を index** に持つ
  DataFrame に正規化する（`close_time = open_time + interval`）。バックテスト側で
  「そのバーの情報が使えるのはクローズ後」という規約を守るためである。

日付境界の解釈（UTC/JST）は API 仕様に依存するため、複数日ぶんを取得して
`openTime` で重複排除・ソートすることで吸収している。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.coin.z.com/public"
MINUTE_INTERVALS = {"1min": 1, "5min": 5, "10min": 10, "15min": 15, "30min": 30}
LEVERAGE_SYMBOLS = ("BTC_JPY", "ETH_JPY", "XRP_JPY", "LTC_JPY", "BCH_JPY")


@dataclass
class FetchConfig:
    symbol: str = "BTC_JPY"
    interval: str = "1min"
    sleep_sec: float = 0.5
    max_retries: int = 5
    timeout_sec: float = 15.0


def _request_json(url: str, params: dict, cfg: FetchConfig) -> dict:
    """指数バックオフ付きの GET。"""
    delay = cfg.sleep_sec
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            res = requests.get(url, params=params, timeout=cfg.timeout_sec)
            if res.status_code == 429 or res.status_code >= 500:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")
            res.raise_for_status()
            payload = res.json()
            if payload.get("status") != 0:
                raise RuntimeError(f"GMO API error: {payload}")
            return payload
        except Exception as err:  # noqa: BLE001 - リトライ対象を広く取る
            last_err = err
            time.sleep(delay * (2**attempt))
    raise RuntimeError(f"failed after {cfg.max_retries} retries: {last_err}")


def fetch_klines_day(date: str, cfg: FetchConfig) -> pd.DataFrame:
    """1 日ぶんの K 線を取得する。

    Args:
        date: 分足なら "YYYYMMDD"、時間足以上なら "YYYY"。
        cfg: 取得設定。

    Returns:
        columns=[open, high, low, close, volume]、index=close_time(UTC) の DataFrame。
    """
    payload = _request_json(
        f"{BASE_URL}/v1/klines",
        {"symbol": cfg.symbol, "interval": cfg.interval, "date": date},
        cfg,
    )
    rows = payload.get("data") or []
    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows)
    open_time = pd.to_datetime(df["openTime"].astype("int64"), unit="ms", utc=True)
    step = pd.Timedelta(minutes=MINUTE_INTERVALS.get(cfg.interval, 1))
    out = pd.DataFrame(
        {
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
        },
        index=pd.DatetimeIndex(open_time + step, name="close_time"),
    )
    return out.sort_index()


def fetch_klines_range(start: str, end: str, cfg: FetchConfig) -> pd.DataFrame:
    """期間 [start, end]（"YYYY-MM-DD"）の K 線を日次で連続取得する。

    Note:
        GMO の K 線は 2021-04-16 以降が提供対象。取得できない日は空としてスキップする。
    """
    days = pd.date_range(start, end, freq="D")
    frames = []
    for day in days:
        try:
            frames.append(fetch_klines_day(day.strftime("%Y%m%d"), cfg))
        except RuntimeError as err:
            print(f"[warn] {day.date()} skipped: {err}")
        time.sleep(cfg.sleep_sec)
    if not frames:
        return _empty_frame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except (ImportError, ValueError):  # pyarrow 未導入の環境では CSV にフォールバック
        path = path.with_suffix(".csv.gz")
        df.to_csv(path)
    return path


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """parquet / csv(.gz) のどちらでも読める OHLCV ローダ。"""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index = pd.DatetimeIndex(df.index, name="close_time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="close_time", tz="UTC"),
    )
