# 引き継ぎデータ一式 — rlgmo 戦略 v2

作成日 2026-08-25 / 対応ブランチ `claude/rl-crypto-trading-strategy-hb5qqb`

リポジトリは価格データを `.gitignore` で除外している（`data/raw/`）ため、
clone だけでは**検証の基準になるデータが手に入らない**。このバンドルがその不足分である。

---

## 0. 最初にやること

このデータは **2 通りの経路で届く**。どちらか手元にある方を使えばよい。

**A. リポジトリに同梱（既定）** — `git clone` しただけで `data/handoff/` に入っている。

```bash
python data/handoff/verify_bundle.py --repo .
```

**B. tar.gz で受け取った場合** — リポジトリの隣に展開する。

```bash
tar xzf rlgmo_handoff_data.tar.gz
python handoff_data/verify_bundle.py --repo ./freespace
```

**`verify_bundle.py` が通らないうちは先へ進まないこと。** 基準が再現しない状態で
GMO のデータを取っても、食い違いの原因がデータなのか環境なのか切り分けられなくなる。

期待される出力:

```
○ ファイル検証 OK（16 件）
○ 全期間       実測 1.577 / 基準 1.577（差 0.000）
○ 開発期間     実測 1.696 / 基準 1.696（差 0.000）
○ ホールドアウト 実測 1.059 / 基準 1.059（差 0.000）
```

確認できたら、既存のスクリプトが期待するパスに配置する（経路 A・B とも同じ）:

```bash
mkdir -p data/raw
cp -r data/handoff/prices/perp_1h        data/raw/perp
cp -r data/handoff/prices/alt2017_1h     data/raw/alt2017_1h
cp -r data/handoff/prices/verify_sources data/raw/verify_sources
```

配置しなくても `--dir data/handoff/prices/perp_1h` を渡せば動く。

---

## 1. 中身

### `prices/perp_1h/` — 主検証データ（7銘柄・1時間足）★これが本体

**戦略 v2 の検証は全てこのデータで行った。**ホールドアウト Sharpe 1.01 も、
コスト耐性も、銘柄組み合わせの総当たりも、根拠は全てここにある。
出所は Binance の無期限先物（USDT 建て）。

| 銘柄 | 本数 | 期間 |
|---|---:|---|
| BTC | 54,113 | 2020-01-01 〜 2026-03-05 |
| ETH | 54,113 | 2020-01-01 〜 2026-03-05 |
| XRP | 54,113 | 2020-01-01 〜 2026-03-05 |
| BNB | 54,113 | 2020-01-01 〜 2026-03-05 |
| DOGE | 54,113 | 2020-01-01 〜 2026-03-05 |
| SOL | 48,767 | 2020-08-11 〜 2026-03-05 |
| AVAX | 47,759 | 2020-09-22 〜 2026-03-05 |

**GMO 5 銘柄の代役**は BTC / ETH / XRP / BNB / DOGE。GMO の LTC_JPY / BCH_JPY に
対応する銘柄はここに無く、**その 2 銘柄は一度も検証できていない**（引き継ぎの主要な穴）。

うち **2025-01-01 以降が封印ホールドアウト**。既に 1 回使った。
`docs/HANDOFF.md` §4 のとおり、ここでパラメータを調整してはいけない。

### `prices/alt2017_1h/` — 第2の時代（6銘柄・1時間足）

2017-10 〜 2020-05 の Huobi データ。perp と**期間が重ならない独立した時代**であり、
「片方の時代だけで良く見える」当てはめを検出するために使った。
ラダー化やリバランスバンドの採否は、両方の時代で改善したことを条件に決めている。

EOS / ETC / ETH / LINK / LTC / XRP。1分足（165MB）から1時間足に畳んで同梱した。

### `prices/verify_sources/BTCUSDT_huobi_1h.parquet` — データ検証用の独立ソース

`scripts/verify_data.py` で突き合わせるための、素性の分かっている BTC データ。
**このファイルが Bitstamp バルク履歴の破損を検出した現物**である
（Huobi と Binance の相関 0.9986 に対し、Bitstamp は約 0）。

期間は 2017-10-26 〜 2020-05-31。**GMO データ（2021-04-16 以降）とは重ならない**ので、
GMO の検証相手には `prices/perp_1h/BTC_1h.parquet` を使うこと。

### `baseline/` — 突き合わせの基準

| ファイル | 中身 |
|---|---|
| `baseline_summary.csv` | 本番設定での区間別成績（下表） |
| `baseline_curve_daily.csv` | 日次のリターン・エクイティ・建玉・コスト（2,256日） |

```
区間          Sharpe     年率     最大DD    年率ボラ   BTC相関  BTCベータ  月次勝率  発注/日
全期間          1.577   30.03%   -14.10%   16.65%   -0.053   -0.015    61.3%   10.92
開発期間         1.696   32.77%   -14.10%   16.71%    0.002    0.000    61.7%   10.65
ホールドアウト      1.059   18.98%    -8.49%   16.41%   -0.422   -0.151    60.0%   12.08
```

再現コマンド:

```bash
python scripts/gmo_validate.py --dir data/raw/perp --symbols BTC ETH XRP BNB DOGE
```

**GMO 実データの結果がこれと大きく食い違ったら、戦略ではなくデータを疑うのが先。**

---

## 2. 同梱していないもの

| データ | 理由 | 入手方法 |
|---|---|---|
| **GMO の実データ** | **一度も取得できていない。これを取るのが引き継ぎの目的** | `scripts/fetch_data.py`（`docs/HANDOFF.md` §3 ステップ2） |
| 1分足の生データ（計 350MB） | 1時間足で判断する戦略なので不要。スプレッド実測には板情報を使う | `scripts/import_perp.py` / `import_bitstamp.py` |
| Bitstamp バルク履歴 | **破損している**（下記） | 再入手しないこと |
| 学習済み RL モデル | RL は単純ルールに負けたので使わない | `scripts/train_final.py` |

### ⚠️ Bitstamp バルク履歴について

`data/raw/BTCUSD_bitstamp_1min_full.parquet`（2012 〜 2025-01-07）は**壊れている**。
価格水準は他ソースと ±1bp で一致し、ボラも ±2% で一致するのに、
**バーの時刻が実際の値動きと対応していない**。欠損・重複チェックを素通りする種類の破損で、
短期の予測力の測定結果を丸ごと無意味にしていた。

**このバンドルには意図的に含めていない。再入手もしないこと。**
同じ罠を避けるため、新しい価格データを入れたら必ず `scripts/verify_data.py` を通すこと。

---

## 3. ファイル形式

- すべて Parquet（zstd 圧縮）、`columns = [open, high, low, close, volume]`
- index は **バーのクローズ時刻**（UTC, tz-aware）。
  「そのバーの情報が使えるのはクローズ後」という規約を守るため
- 読み込みは `pd.read_parquet(path)` でそのまま。`src/rlgmo/data/gmo_klines.load_ohlcv()` も使える

## 4. 完全性の確認

`SHA256SUMS` に全 16 ファイルのハッシュがある。`verify_bundle.py` が自動で照合する。
手動なら `sha256sum -c SHA256SUMS`。
