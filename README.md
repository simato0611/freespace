# rlgmo — 強化学習による仮想通貨取引戦略（GMO コイン レバレッジ / 1〜15 分足）

GMO コインのレバレッジ取引（`BTC_JPY` 等）を対象に、**1 分足ごとに目標ポジション比率を決める**
強化学習エージェント（PPO）の設計・学習・検証・運用の一式。

📄 **設計の本体は [`docs/strategy_design.md`](docs/strategy_design.md)**（MDP 定式化・コスト算術・検証プロトコル・採用ゲート・失敗モード）。

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
  pipeline.py               データ→学習→検証→テストの一連の流れ

scripts/                    fetch_data / measure_spread / train_walkforward / backtest / live_paper
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
| 探索の崩壊 | 金融環境では「常にフラット」に早期収束しがち。目標エントロピーへの乗算制御で探索を維持 |
| 回転率 | 実コスト + 報酬ペナルティ + 建玉の再調整許容幅 + 最小発注幅、の 4 段構え |

## 免責

本リポジトリは戦略設計・研究のための実装であり、投資助言ではない。レバレッジ取引は
元本を超える損失が生じうる。実運用の前に必ず、公式の手数料・仕様を確認し、
[採用ゲート](docs/strategy_design.md#8-採用ゲートこの数字を満たさなければ本番に出さない)を満たすまで
実資金を投入しないこと。バックテストの成績は将来の成績を保証しない。
