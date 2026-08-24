"""GMO コイン レバレッジ取引向け 強化学習トレーディング戦略。

サブモジュール:
    data        : GMO Public API からの K 線取得・リサンプル・合成データ
    features    : マルチタイムフレーム特徴量（1/5/15 分足）
    costs       : スプレッド・スリッページ・建玉管理料のコストモデル
    env         : ポジション制御 MDP（Gym 互換 API）
    agents.ppo  : PPO エージェント
    walkforward : ウォークフォワード分割
    metrics     : バックテスト評価指標
"""

__version__ = "0.1.0"
