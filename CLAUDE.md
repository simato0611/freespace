# rlgmo — このリポジトリで作業するときの前提

> **⚠️ 最初に [`docs/HANDOFF.md`](docs/HANDOFF.md) を読むこと。**
> 引き継ぎ資料であり、次にやるべきことと、やってはいけないことが書いてある。

## これは何か

GMO コインのレバレッジ取引（5 銘柄）向けの、複数銘柄トレンドフォロー戦略。
設計・検証・実行系まで完成しており、**残っているのは GMO 実データでの確認だけ**。

- 戦略の中身: 5/14/30/60 日モメンタムの平均（ラダー）、両建て、等リスク配分、
  2 段階のボラターゲット（15%）、レバレッジ上限 2 倍、1 時間判断
- 実績: 封印ホールドアウト 15 か月で Sharpe 1.01 / 年率 18% / 最大 DD −8.7% / BTC ベータ −0.01
- **ライブ実績はゼロ。GMO の実データは一度も取得していない**（リモート環境から API に到達できなかった）

## 最重要のルール

このプロジェクトは**検証の作法を守ったことだけが価値**である。以下を破ると 5.7 年ぶんの検証が無意味になる。

1. **ホールドアウト（2025-01-01 以降）でパラメータを調整しない。** 既に 1 回使った
2. **成績が悪くてもパラメータを触らない。** 変えるなら、理由と変更前の数字を
   `docs/strategy_search.md` に記録してから変える
3. **新しい試行は必ず記録する。** 現在 245 試行。Deflated Sharpe はこの数で補正している
4. **バックテストとライブでサイジングを二重実装しない。**
   両者は `portfolio.compute_exposures()` を共有し、テストで固定している
5. **新しい価格データを入れたら必ず `scripts/verify_data.py` を通す。**
   過去に公開データの破損（価格は正常なのに時刻が値動きと対応していない）で検証結果が
   丸ごと無意味になった事故がある。欠損・重複チェックでは検出できない
6. **RL を再挑戦する前に `docs/strategy_search.md` の 3 節と 30 節を読む。**
   2 回試して 2 回とも単純ルールに負けている

## 開発

```bash
pip install -e .                              # RL も動かすなら '.[rl]'
python -m pytest tests -q                    # 73 件。変更前後で必ず通すこと
python data/handoff/verify_bundle.py --repo . # 基準データと基準の数字を確認
```

- パラメータの唯一の正は `configs/gmo_live.yaml`。コードにハードコードしない
- 実発注は `scripts/live_gmo.py --live` のときだけ。既定はドライラン
- API キーは環境変数 `GMO_API_KEY` / `GMO_API_SECRET` から読む。コードや設定に書かない
- コメントと docstring は日本語。既存のトーン（「なぜそうしたか」を書く）に合わせる

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `docs/HANDOFF.md` | **引き継ぎ資料。まずこれ** |
| `data/handoff/` | 検証の基準になる価格データ一式。`verify_bundle.py` で完全性を確認できる |
| `configs/gmo_live.yaml` | 本番設定。パラメータの唯一の正 |
| `src/rlgmo/portfolio.py` | 戦略の核。`compute_exposures()` をライブと共有 |
| `src/rlgmo/live.py` | ライブ執行ロジック。GMO の決済/新規の分割もここ |
| `scripts/gmo_validate.py` | 本番設定を流して GO/NO-GO まで出す検証スクリプト |
| `scripts/fetch_data.py` | GMO Public API からデータ取得 |
| `scripts/verify_data.py` | データ健全性の検証 |
| `docs/strategy_search.md` | 全 51 節・245 試行の記録。迷ったらここ |

## 既知の未対応

- `scripts/live_gmo.py:148,152` — 発注数量が全銘柄 `f"{qty:.4f}"` 固定。
  GMO の `/v1/symbols` が返す `sizeStep` / `minOrderSize` / `maxOrderSize` に合わせる必要がある
- `LTC_JPY` / `BCH_JPY` は一度も検証していない
- 実効スプレッド・発注数量上限・キャパシティは未測定
