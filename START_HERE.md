# rlgmo 引き継ぎパッケージ（完全版）

**このパッケージだけで完結する。**戦略のロジック・設定・テスト・検証の基準データ・
ドキュメント、および git 履歴の全てが入っている。外部リポジトリを参照しない。

前回配布した `rlgmo_handoff_data.tar.gz` は**価格データだけ**で、戦略コードが
入っていなかった。「データと主張された出力は検証できるが、戦略が主張どおりのものかは
検証できない」という指摘は正しい。これはその修正版である。

---

## 中身

| ファイル | 内容 |
|---|---|
| `rlgmo/`（このパッケージ） | リポジトリの作業ツリー一式。コード・設定・テスト・ドキュメント・基準データ。**これだけで検証まで完結する** |
| `rlgmo.gitbundle`（別送・16MB） | git 履歴の全て。clone して push まで戻せる。作業を GitHub に戻すなら使う |

アップロード上限の都合で 2 ファイルに分けている。作業ツリーだけで検証は完結するので、
履歴が要らなければ `rlgmo.gitbundle` は使わなくてよい。

## 使い方

**A. すぐ動かす**

```bash
cd rlgmo
pip install -e .
python -m pytest tests -q                     # 73 件
python data/handoff/verify_bundle.py --repo . # 基準データと基準の数字を確認
```

**B. git 履歴ごと復元する（別送の `rlgmo.gitbundle` を使う場合）**

```bash
git clone rlgmo.gitbundle rlgmo-repo
cd rlgmo-repo
git checkout claude/rl-crypto-trading-strategy-hb5qqb
git remote set-url origin https://github.com/simato0611/freespace.git
pip install -e .
python -m pytest tests -q
python data/handoff/verify_bundle.py --repo .
```

## 次に読むもの

1. **`rlgmo/docs/HANDOFF.md`** — 引き継ぎ資料の本体。手順・判定基準・禁止事項
2. `rlgmo/CLAUDE.md` — Claude Code が最初に読む前提
3. `rlgmo/configs/gmo_live.yaml` — パラメータの唯一の正
4. `rlgmo/src/rlgmo/portfolio.py` — 戦略の核

## 「戦略が主張どおりか」を自分で確かめる経路

主張を鵜呑みにする必要はない。以下は全てこのパッケージの中だけで完結する。

| 確かめたいこと | 方法 |
|---|---|
| シグナルの定義が文書どおりか | `src/rlgmo/portfolio.py` の `ladder_signal` / `compute_exposures` を読む。docs/HANDOFF.md §1 の式と対応する |
| ライブとバックテストが同じサイジングか | `tests/test_live.py::test_live_targets_match_the_backtest_exactly` が両者の一致を固定している |
| 主張された成績が出るか | `python scripts/gmo_validate.py --dir data/handoff/prices/perp_1h --symbols BTC ETH XRP BNB DOGE` |
| ホールドアウトが本当に分離されているか | `gmo_validate.py` の `--holdout-start`（既定 2025-01-01）で区間を切っている。開発期間 1.696 に対しホールドアウト 1.059 と落ちるのは、分離が効いている証拠 |
| データが壊れていないか | `python scripts/verify_data.py --a <検証したいデータ> --b data/handoff/prices/verify_sources/BTCUSDT_huobi_1h.parquet` |
| 試行回数の申告が正しいか | `docs/strategy_search.md` に全 73 節・337 試行が残っている |

**疑ってよい。むしろ疑ってほしい。**このプロジェクトで唯一守ってきたのは
「都合の悪い結果も残す」ことなので、記録と食い違う点が見つかったらそれが正しい。
