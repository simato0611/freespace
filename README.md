# rlgmo — 強化学習による仮想通貨取引戦略（GMO コイン レバレッジ / 1〜15 分足）

GMO コインのレバレッジ取引（`BTC_JPY` 等）を対象に、**1 分足ごとに目標ポジション比率を決める**
強化学習エージェント（PPO）の設計・学習・検証・運用の一式。

📄 **設計の本体は [`docs/strategy_design.md`](docs/strategy_design.md)**（MDP 定式化・コスト算術・検証プロトコル・採用ゲート・失敗モード）。

🔎 **戦略探索と判定は [`docs/strategy_search.md`](docs/strategy_search.md)。**

**最有力候補: 戦略 v2 — 複数銘柄・両建てのトレンド・ラダー**（`scripts/run_strategy.py`）

シグナルは 5/14/30/60 日モメンタムの平均、**両建て**、銘柄ごとにボラターゲット（等リスク）→
ポートフォリオ・ボラ 15% → レバレッジ上限 2 倍。1 時間判断・更新バンド 0.10・指値執行想定。

| 検証期間 | Sharpe | 年率 | 最大DD | Calmar |
|---|---:|---:|---:|---:|
| 時代A 2017-10〜2020-05 / Huobi 6 銘柄 | 1.64 | +31.7% | −10.8% | 2.92 |
| 時代B 2020-01〜2024-12 / Perp 7 銘柄 | 1.82 | +35.3% | −12.4% | 2.84 |
| **ホールドアウト 2025-01〜2026-03（封印・一度だけ）** | **+0.99** | **+17.7%** | **−7.6%** | **2.34** |
| （同期間の買い持ち） | −0.68 | −14.4% | −27.2% | — |

ホールドアウトでは**買い持ちが −14.4% の下落局面で +17.7%（年率）**、最大DD −7.6%。
コスト仮定を 2 倍（片道 2.5bp）にしても Sharpe +0.94、GMO 相当 3 銘柄でも +0.78。
**採用ゲート 6 つ中 5 つ通過**（落ちたのは Deflated Sharpe のみ＝試行 139 本では到達不能な基準）。
ただし 1.18 年では t 値 1.07 で、統計的な確証には期間が足りない。

改良前（ロングのみ・14 日単独）は 時代A 0.94 / 時代B 1.59・最大DD −27.5%、
ホールドアウト +0.61 だった。**最大の発見は「単一銘柄では損なショートが、
分散すると下落局面の収益源になる」**こと（2018 年 −0.13→**+1.74**、2022 年 +0.52→**+1.09**）。

収益源の多様化も試したが、**トレンド以外に機能する収益源は見つからなかった**
（短期リバーサル −1.55/−2.02、CS リバーサル −0.85/−3.40、ファンディング／ベーシス／OI も上乗せ無し）。

---

優位性が存在する時間軸を探した結果、**多日スケールのロングオンリー・トレンドフォロー**を発見:
開発期間（BTC 8.5 年）Sharpe **+1.44**、他 6 銘柄でもパラメータ無変更で **6/6 プラス**。
ただし**封印していたホールドアウト 14 ヶ月では Sharpe −0.75** で採用ゲートを満たさない
（この不調の大きさは 2018 年・2022 年と同程度で、分布の外ではない）。**現時点では実運用しない。**
なお同じ情報セットで学習した PPO は、この単純ルール（10 fold 平均 +0.61）に **+0.36** で負けている。

⚠️ **短期（1〜15 分足）の検証結果は [`docs/real_data_findings.md`](docs/real_data_findings.md)。**
BTC/USD 1 分足 153 万バー（2023-06〜2026-08）でウォークフォワード検証した結果:

| 判断間隔 | fold | OOS日数 | 純リターン | Sharpe | 回転/日 | グロス/年 | 取引コスト/年 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 分 | 2 | 159 | −49.98% | −12.01 | 7.34 | +2.2% | −157.2% |
| 60 分 | 9 | 758 | −74.25% | −5.18 | 2.68 | −3.3% | −58.9% |
| 240 分 | 9 | 765 | −34.39% | −1.38 | 1.08 | +9.6% | −23.9% |

**取引コストを完全にゼロにしても Sharpe は +0.11**（2.1 年 OOS・誤差 ±0.7）。
つまり損失の直接原因はコストだが、**コストを消しても儲からない**。
1 分足 OHLCV から作れる特徴量の予測力（IC 0.01〜0.02）は、損益分岐に必要な水準の
1/2〜1/30 しかない。実装の問題ではなく情報量の問題である。

---

## この設計の要点

1. **1 分足は「判断の頻度」であって「売買の頻度」ではない。**
   片道 2.5bp・往復 5bp のコストでは、15 分回転はグロス Sharpe 29、5 分回転は 87 を要求する
   （[コスト算術](docs/strategy_design.md#2-最重要-コストの算術この節が戦略の形を決める)）。
   本設計は 1/5/15 分足で判断し、**平均保有 2〜8 時間**に誘導する。

2. **コストは環境の中で実費として引く。** スプレッド・スリッページに加え、GMO 固有の
   **建玉管理料 0.04%/日（JST 06:00 課金）** を離散イベントとして再現し、
   「06:00 までの残り時間」を状態に入れて日跨ぎの判断を学習させる。

3. **ルックアヘッドを構造的に排除する。** 足のクローズ時刻を index にし、注文は必ず次バーのオープンで約定。
   上位足は `merge_asof(backward)` で完成済みバーのみ参照。標準化は 7 日ローリングの中央値・IQR（因果的）。
   これらは [`tests/test_features.py`](tests/test_features.py) で検証している。

4. **リスク管理は学習に任せない。** 日次損失上限・DD 停止・ボラ上限・スプレッド異常・データ鮮度・
   連敗ブレーキは決定論的なリスクレイヤ（[`src/rlgmo/risk.py`](src/rlgmo/risk.py)）で外付けする。

5. **単一シードのバックテストは信用しない。** 5〜10 シードのアンサンブル × ウォークフォワード
   （purge & embargo）× Deflated Sharpe で採否を判断する。

---

## セットアップ

```bash
pip install -r requirements.txt      # numpy / pandas / torch(CPU) / requests / pyyaml
python -m pytest tests/ -q           # 28 テスト（会計・因果性・リスク・指標）
```

## クイックスタート

```bash
# 0) 配線確認（合成データ・ネットワーク不要。性能評価には使わない）
python scripts/train_walkforward.py --config configs/smoke.yaml --max-folds 1

# 1) スプレッドの実測 → configs/default.yaml の half_spread_bp を較正
python scripts/measure_spread.py --symbol BTC_JPY --minutes 60 --size 0.01

# 2) 1 分足の取得（GMO Public API。2021-04-16 以降が対象）
python scripts/fetch_data.py --symbol BTC_JPY --start 2023-01-01 --end 2026-06-30

# 3) ウォークフォワード学習（fold ごとに 5 シード → アンサンブルでテスト）
python scripts/train_walkforward.py --config configs/default.yaml

# 4) 任意区間のバックテスト（ベースラインと並べて表示）
python scripts/backtest.py --config configs/default.yaml \
    --models "runs/default/fold0_seed*.pt" --start 2026-05-01 --end 2026-06-30

# 5) ペーパートレード（既定はドライラン。--live で実発注）
python scripts/live_paper.py --config configs/default.yaml --models "runs/default/fold0_seed*.pt"
```

## 実データでの再現手順

```bash
# 1) 公開データセット（Bitstamp BTC/USD 1分足、2012-01〜、日次更新、欠損なし）を取得
git clone --depth 1 https://github.com/ff137/bitstamp-btcusd-minute-data.git /tmp/btcdata
python scripts/import_bitstamp.py --src /tmp/btcdata --start 2023-06-01     --out data/raw/BTCUSD_bitstamp_1min.parquet

# 2) まず「そもそも予測力があるか」を測る（RL を回す前にこれをやる）
python scripts/analyze_data.py --config configs/btc_real.yaml --stride 5

# 3) ウォークフォワード学習（判断間隔 1分 / 60分 / 240分）
python scripts/train_walkforward.py --config configs/btc_real.yaml       # 判断間隔 1 分
python scripts/train_walkforward.py --config configs/btc_real_h60.yaml   # 判断間隔 60 分
python scripts/train_walkforward.py --config configs/btc_real_h240.yaml  # 判断間隔 240 分

# 4) 連結したアウトオブサンプル曲線で比較
python scripts/aggregate_runs.py runs/btc_real:1分判断 runs/btc_real_h60:60分判断     runs/btc_real_h240:240分判断
```

GMO 自身の 1 分足で同じことをする場合は `scripts/fetch_data.py` でデータを取り、
`configs/default.yaml` の `data.path` を差し替える（この実行環境からは GMO の API に
到達できなかったため、検証は公開データセットで行っている。詳細は
[`docs/real_data_findings.md`](docs/real_data_findings.md) の 1 節）。

## 動作確認済みの疎通例（合成データ・2 fold）

```
$ python scripts/train_walkforward.py --config configs/smoke.yaml --max-folds 2
=========== ウォークフォワード集計 ===========
fold 数           : 2
RL Sharpe 平均    : +77.62  (中央値 +77.62)
Sharpe > 0 の割合 : 100%
シードばらつき σ  : 7.86
ベースライン flat     : Sharpe 平均 +0.00
ベースライン long     : Sharpe 平均 -82.23
ベースライン momentum : Sharpe 平均 +75.97
```

**この数字は戦略の収益性を意味しない。** 合成データの生成モデルにトレンドを埋め込んで
あるため、モメンタム則ですら Sharpe 76 が出る。ここで確認しているのは
「学習が発散しないか / フラットに退行しないか / ベースラインと比較できているか」だけである。
実データでの評価は必ず [採用ゲート](docs/strategy_design.md#8-採用ゲートこの数字を満たさなければ本番に出さない)で行うこと。

## 計算コストの目安（CPU 4 コア）

| 処理 | 規模 | 所要時間 |
|---|---|---|
| 特徴量生成 | 2 年 = 約 105 万バー | 約 64 秒（2 回目以降は `data/cache/` から即時） |
| PPO 学習 | 20 万ステップ / 1 シード | 約 4 分 |
| 本番設定 | 150 万ステップ × 5 シード × 8 fold | **数十時間**。まず `--max-folds 1` で確認してから回す |

## リポジトリ構成

```
docs/strategy_design.md     設計書（まずここを読む）
docs/experiment_log.md      試行記録テンプレート（Deflated Sharpe の試行数を正直に数えるため）
configs/default.yaml        既定設定（コスト・報酬・PPO・ウォークフォワード）
configs/smoke.yaml          合成データでの配線確認用

src/rlgmo/
  data/gmo_klines.py        GMO Public API から K 線取得（クローズ時刻 index に正規化）
  data/resample.py          1分→5/15分 リサンプルと未来漏れのないアライン
  data/synthetic.py         オフライン検証用の合成 1 分足（レジーム切替 + GARCH + ジャンプ）
  features.py               マルチタイムフレーム特徴量（約 61 次元）+ 因果ロバスト標準化
  costs.py                  スプレッド/スリッページ/建玉管理料（06:00 JST 課金）
  env.py                    ポジション制御 MDP（Gym 互換）+ ベクトル環境
  agents/networks.py        Actor-Critic（MLP + LayerNorm + Dropout）
  agents/ppo.py             PPO（GAE・目標エントロピー制御・検証ベスト重みの採用）
  risk.py                   決定論的リスクレイヤ（RL の外側）
  walkforward.py            purge & embargo 付きウォークフォワード分割
  backtest.py               方策の実行とベースライン（flat / long / momentum）
  metrics.py                Sharpe・DD・回転率・Deflated Sharpe
  portfolio.py              複数銘柄の等リスク配分 + ポートフォリオ・ボラターゲット
  pipeline.py               データ→学習→検証→テストの一連の流れ

scripts/
  fetch_data.py             GMO Public API から 1 分足を取得
  import_bitstamp.py        公開データセット（BTC/USD 1分足）の取り込み
  signal_survey.py          仮説探索: 素朴なシグナルを実コスト込みで横並び比較
  analyze_data.py           予測力の直接測定（Ridge / GBDT の OOS R²・IC）
  cross_asset_check.py      同一ルールを他銘柄へ（パラメータ無変更で確認）
  train_walkforward.py      ウォークフォワード学習
  train_final.py            ホールドアウト直前までのデータで最終モデルを学習
  reevaluate_folds.py       再学習せずに評価だけ取り直す
  backtest.py / stress_test.py / cost_sweep.py / aggregate_runs.py
  final_holdout.py          封印期間で一度だけ評価する
  import_perp.py            7銘柄のファンディング/OI/ベーシス付きデータの取り込み
  perp_survey.py            新情報源（ファンディング/ベーシス/OI）の探索
  portfolio_backtest.py     複数銘柄・等リスク運用のバックテスト
  improve_portfolio.py      改良案を独立した 2 つの時代で突き合わせる
  run_strategy.py           確定版（戦略 v2）の正式な実装
  sleeve_search.py          トレンドと相関の低い収益源を探す（結果: 見つからず）
  measure_spread.py / live_paper.py
tests/                      28 テスト（環境の会計、特徴量の因果性、リスク、指標、分割）
```

## 実装上、特に注意した点

| 論点 | 実装 |
|---|---|
| ルックアヘッド | クローズ時刻 index、次バーオープン約定、`merge_asof(backward)`、因果標準化。未来を改変しても過去の特徴量が変わらないことをテストで検証 |
| 損益の会計 | 「前バー終値→次バー始値」を旧建玉、「始値→終値」を新建玉で評価。コスト 0・レバ 1 倍のフルロングが価格変化率と一致することをテストで確認 |
| 建玉管理料 | 06:00 JST をまたぐバーで 1 回だけ課金（`carry_mode: daily_0600`）。按分モードも選択可 |
| ロスカット | 証拠金維持率 = 有効証拠金 ÷ (建玉評価額 ÷ レバレッジ上限)。定率運用のため実質的にギャップ急落でのみ発生 |
| 過学習 | シードアンサンブル、コストのドメインランダマイゼーション（±50%）、小さいネットワーク、検証ベスト重みの採用 |
| 探索の崩壊 | 金融環境では「常にフラット」（報酬 0 の局所解）に早期収束しがち。一様分布の混合による探索下限 + 目標エントロピー制御 + コストカリキュラム + 行動クローニング暖機の 4 点で回避（[6.4 節](docs/strategy_design.md#64-常にフラットという罠この環境で最初にぶつかる壁)） |
| 回転率 | 実コスト + 報酬ペナルティ + 建玉の再調整許容幅 + 最小発注幅、の 4 段構え |

## 免責

本リポジトリは戦略設計・研究のための実装であり、投資助言ではない。レバレッジ取引は
元本を超える損失が生じうる。実運用の前に必ず、公式の手数料・仕様を確認し、
[採用ゲート](docs/strategy_design.md#8-採用ゲートこの数字を満たさなければ本番に出さない)を満たすまで
実資金を投入しないこと。バックテストの成績は将来の成績を保証しない。
