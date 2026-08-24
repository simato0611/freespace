"""ウォークフォワード学習・評価のオーケストレーション。

    データ取得 → 特徴量 → 分割 → (fold ごとに) 学習 → 検証で早期終了 → テストで一度だけ評価

という流れを 1 本にまとめる。各 fold で複数シードを学習し、アンサンブルでテストする。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .agents.ppo import PPOAgent
from .backtest import ensemble_policy, flat_policy, long_policy, momentum_policy, run_policy
from .config import ExperimentConfig
from .data.gmo_klines import load_ohlcv
from .data.synthetic import make_synthetic_ohlcv
from .env import EnvConfig, SyncVectorEnv, TradingEnv
from .features import build_features
from .metrics import summarize
from .walkforward import Fold, make_folds


def prepare_data(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OHLCV を読み込み、特徴量とメタ情報を返す。"""
    if cfg.data.use_synthetic:
        ohlcv = make_synthetic_ohlcv(cfg.data.synthetic_minutes, seed=cfg.ppo.seed)
    else:
        ohlcv = load_ohlcv(cfg.data.path)
        ohlcv = ohlcv.loc[cfg.data.start : cfg.data.end]
    features, meta = build_features(ohlcv, cfg.features)
    print(f"[data] bars={len(features):,} 期間={features.index[0]} 〜 {features.index[-1]} 特徴量={features.shape[1]}")
    return features, meta


def make_env(
    features: pd.DataFrame, meta: pd.DataFrame, sl: slice, env_cfg: EnvConfig, training: bool
) -> TradingEnv:
    """スライスから環境を作る（評価時はコストのランダム化を無効化）。"""
    if not training:
        env_cfg = dataclasses.replace(env_cfg, randomize_costs=False)
    return TradingEnv(features.iloc[sl], meta.iloc[sl], env_cfg, training=training)


def evaluate_policy(env: TradingEnv, policy, n_trials: int = 1) -> tuple[pd.DataFrame, dict]:
    """方策を区間全体で走らせて指標を返す。"""
    record = run_policy(env, policy)
    metrics = summarize(
        record["equity"], record["position"], record["trade_cost"] + record["carry_cost"], n_trials=n_trials
    )
    return record, metrics


def train_fold(
    features: pd.DataFrame, meta: pd.DataFrame, fold: Fold, cfg: ExperimentConfig, seed: int
) -> tuple[PPOAgent, dict]:
    """1 fold・1 シードを学習する（検証成績が最良の重みを返す）。"""
    env_cfg = dataclasses.replace(cfg.env, episode_len=cfg.train.episode_len)
    train_envs = [make_env(features, meta, fold.train, env_cfg, training=True) for _ in range(cfg.train.n_envs)]
    vec = SyncVectorEnv(train_envs)
    valid_env = make_env(features, meta, fold.valid, env_cfg, training=False)

    ppo_cfg = dataclasses.replace(cfg.ppo, seed=seed)
    agent = PPOAgent(vec.observation_dim, vec.n_actions, ppo_cfg)

    def callback(agent: PPOAgent, step: int) -> dict:
        policy = ensemble_policy([agent], cfg.env.actions, confidence=cfg.train.confidence)
        _, metrics = evaluate_policy(valid_env, policy)
        # 選択基準: Sharpe から最大 DD のペナルティを引いた複合スコア（検証区間のみで決める）
        score = metrics.get("sharpe", 0.0) + 3.0 * min(metrics.get("max_drawdown", 0.0), 0.0)
        return {"score": float(score), "sharpe": float(metrics.get("sharpe", 0.0)),
                "max_dd": float(metrics.get("max_drawdown", 0.0)),
                "turnover": float(metrics.get("turnover_per_day", 0.0))}

    history = agent.learn(
        vec, total_steps=cfg.train.total_steps, callback=callback, callback_every=cfg.train.eval_every
    )
    return agent, history


def run_walkforward(cfg: ExperimentConfig, max_folds: int | None = None) -> pd.DataFrame:
    """ウォークフォワード全体を実行し、fold ごとのテスト成績表を返す。"""
    features, meta = prepare_data(cfg)
    folds = make_folds(features.index, cfg.walkforward)
    if not folds:
        raise ValueError("データ期間が短く fold を作れません。walkforward の日数設定を見直してください。")
    if max_folds:
        folds = folds[:max_folds]
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in folds:
        print(f"\n===== {fold.describe()} =====")
        agents, histories = [], []
        for seed in cfg.train.seeds:
            print(f"--- seed {seed} ---")
            agent, history = train_fold(features, meta, fold, cfg, seed)
            agent.save(out_dir / f"fold{fold.idx}_seed{seed}.pt")
            agents.append(agent)
            histories.append(history)

        n_trials = len(cfg.train.seeds) * len(folds)
        test_env = make_env(features, meta, fold.test, cfg.env, training=False)
        policy = ensemble_policy(agents, cfg.env.actions, confidence=cfg.train.confidence)
        record, metrics = evaluate_policy(test_env, policy, n_trials=n_trials)
        record.to_csv(out_dir / f"fold{fold.idx}_test.csv")

        baselines = {
            "flat": flat_policy(cfg.env.actions),
            "long": long_policy(cfg.env.actions),
            "momentum": momentum_policy(test_env),
        }
        base_metrics = {}
        for name, base_policy in baselines.items():
            _, bm = evaluate_policy(make_env(features, meta, fold.test, cfg.env, training=False), base_policy)
            base_metrics[name] = bm

        row = {"fold": fold.idx, "test_start": fold.timestamps["test_start"], **{f"rl_{k}": v for k, v in metrics.items()}}
        for name, bm in base_metrics.items():
            row[f"{name}_sharpe"] = bm.get("sharpe", 0.0)
            row[f"{name}_return"] = bm.get("total_return", 0.0)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else str(v)) for k, v in row.items()},
                         indent=1, ensure_ascii=False))

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "walkforward_report.csv", index=False)
    _print_summary(report)
    return report


def _print_summary(report: pd.DataFrame) -> None:
    sharpes = report["rl_sharpe"].to_numpy(dtype=float)
    print("\n=========== ウォークフォワード集計 ===========")
    print(f"fold 数           : {len(report)}")
    print(f"RL Sharpe 平均    : {np.nanmean(sharpes):+.2f}  (中央値 {np.nanmedian(sharpes):+.2f})")
    print(f"Sharpe > 0 の割合 : {np.mean(sharpes > 0):.0%}")
    print(f"シードばらつき σ  : {np.nanstd(sharpes):.2f}")
    for name in ("flat", "long", "momentum"):
        col = f"{name}_sharpe"
        if col in report:
            print(f"ベースライン {name:<9}: Sharpe 平均 {report[col].mean():+.2f}")
    print("※ テスト区間の成績で設計を変更したら、その成績はもうアウトオブサンプルではない。")
