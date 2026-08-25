#!/usr/bin/env python3
"""強化学習とルールベースのシグナルを、同じ土俵で比較する。

**公平な比較にするための設計**

- RL には戦略 v2 と同じ情報を与える（1 時間 / 6 時間 / 日足、最長 60 日のモメンタム）。
- **置き換えるのはシグナル層だけ**。サイジング（等リスク + ポートフォリオ・ボラターゲット）と
  ポートフォリオ構築は、ルール版とまったく同じ `backtest_portfolio` を通す。
- 学習は**全銘柄をまとめて 1 つの方策**に行わせる（銘柄ごとに別々の方策を作るより
  データ量が増え、過学習しにくい）。
- ホールドアウト（2025-01 以降）は使わない。開発期間のウォークフォワードだけで判定する。

以前この比較を行ったときのデータは、後から破損が判明したもの（`docs/strategy_search.md` 24 節）
だった。ここでは健全性を確認済みのデータで測り直している。

Example:
    python scripts/rl_vs_rule.py --data-dir data/raw/perp --end 2024-12-31 --folds 4
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from rlgmo.backtest import momentum_policy  # noqa: E402
from rlgmo.costs import CostConfig  # noqa: E402
from rlgmo.env import EnvConfig, SyncVectorEnv, TradingEnv  # noqa: E402
from rlgmo.features import FeatureConfig, build_features  # noqa: E402
from rlgmo.metrics import equity_metrics  # noqa: E402
from rlgmo.portfolio import PortfolioConfig, apply_rebalance_band, backtest_portfolio, ladder_signal  # noqa: E402

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def feature_config() -> FeatureConfig:
    """戦略 v2 と同じ時間スケールを覆う特徴量設定（日足 × ラグ 60 = 60 日）。"""
    return FeatureConfig(
        base_minutes=60,
        timeframes=(1, 6, 24),          # 1 時間 / 6 時間 / 日足
        ret_lags=(1, 2, 3, 5, 10, 20, 40, 60),
        vol_span=30, atr_period=14, donchian=48, effr_period=20, ofi_period=20, vwap_period=24,
        scale_window=24 * 90, clip=5.0,
    )


def load_assets(data_dir: Path, end: str | None) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        if "close" not in df.columns:
            continue
        df = df.resample("1h", label="right", closed="right").agg(
            {k: v for k, v in AGG.items() if k in df.columns}).dropna(subset=["close"])
        if end:
            df = df.loc[:end]
        if len(df) > 24 * 400:
            out[path.stem.split("_")[0].upper()] = df
    return out


def extract_signal(agent: PPOAgent, env: TradingEnv, actions: tuple[float, ...],
                   start: int, stop: int) -> pd.Series:
    """方策を走らせ、各バーで選んだ「目標ポジション比率」を取り出す。

    サイジングはポートフォリオ側で行うので、ここで欲しいのは方策の生の出力だけ。
    """
    obs, _ = env.reset(start=start, episode_len=stop - start - 2)
    values = np.asarray(actions, dtype=float)
    idx, out = [], []
    while True:
        probs = agent.probs(obs)[0]
        out.append(float(probs @ values))          # 期待ポジション（アンサンブル平均と同じ扱い）
        idx.append(env.index[env._t])
        obs, _, terminated, truncated, _ = env.step(int(np.argmax(probs)))
        if terminated or truncated:
            break
    return pd.Series(out, index=pd.DatetimeIndex(idx)).clip(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/raw/perp")
    parser.add_argument("--end", default="2024-12-31", help="ホールドアウトを避けるための打ち切り日")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-days", type=int, default=540)
    parser.add_argument("--test-days", type=int, default=240)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", default="runs/rl_vs_rule")
    args = parser.parse_args()

    prices = load_assets(Path(args.data_dir), args.end)
    fcfg = feature_config()
    print(f"[data] 銘柄 {sorted(prices)}")
    feats, metas = {}, {}
    for asset, df in prices.items():
        f, m = build_features(df, fcfg)
        feats[asset], metas[asset] = f, m
    index = sorted(set.intersection(*[set(f.index) for f in feats.values()]))
    index = pd.DatetimeIndex(index)
    print(f"[data] 共通バー {len(index):,} 本  {index[0]:%Y-%m-%d}〜{index[-1]:%Y-%m-%d}  特徴量 {feats[list(feats)[0]].shape[1]}")

    env_cfg = EnvConfig(leverage_cap=2.0, episode_len=24 * 30, vol_target=True, vol_target_ann=0.15,
                        rebalance_tolerance=0.05, randomize_costs=True, daily_loss_limit=0.05,
                        cost=CostConfig(half_spread_bp=1.5, slippage_bp=0.0,
                                        carry_mode="daily_0600", spread_vol_beta=0.0))
    eval_cfg = dataclasses.replace(env_cfg, randomize_costs=False)
    pcfg = PortfolioConfig(target_vol_ann=0.15, asset_vol_ann=0.15,
                           cost=CostConfig(half_spread_bp=1.5, slippage_bp=0.0,
                                           carry_mode="daily_0600", spread_vol_beta=0.0))

    step = args.test_days
    rows = []
    for fold in range(args.folds):
        t0 = index[0] + pd.Timedelta(days=fold * step)
        t1 = t0 + pd.Timedelta(days=args.train_days)
        t2 = t1 + pd.Timedelta(days=args.test_days)
        if t2 > index[-1]:
            break
        print(f"\n===== fold {fold}: 学習 {t0:%Y-%m-%d}〜{t1:%Y-%m-%d} / テスト 〜{t2:%Y-%m-%d} =====")

        # --- 全銘柄をまとめた環境で 1 つの方策を学習する
        train_envs = []
        for asset in feats:
            f = feats[asset].loc[t0:t1]
            m = metas[asset].loc[t0:t1]
            if len(f) > 24 * 120:
                train_envs.append(TradingEnv(f, m, env_cfg, training=True))
        vec = SyncVectorEnv(train_envs * max(1, 8 // len(train_envs)))
        agents = []
        for seed in range(args.seeds):
            agent = PPOAgent(vec.observation_dim, vec.n_actions,
                             PPOConfig(seed=seed, n_steps=256, batch_size=512, epochs=4, gamma=0.99,
                                       hidden=(128, 64), dropout=0.2, bc_steps=20_000,
                                       cost_curriculum_start=0.5, cost_curriculum_frac=0.2))
            teacher = momentum_policy(train_envs[0], "ret_20_1d", 0.5)
            agent.pretrain(vec, teacher, steps=20_000)
            agent.learn(vec, total_steps=args.steps, log_every=200)
            agents.append(agent)

        # --- テスト区間のシグナルを取り出し、ルールと同じサイジングに通す
        rl_signals, rule_signals, test_prices = {}, {}, {}
        for asset in feats:
            f = feats[asset].loc[t0:t2]
            m = metas[asset].loc[t0:t2]
            start = int(f.index.searchsorted(t1))
            if len(f) - start < 24 * 30:
                continue
            env = TradingEnv(f, m, eval_cfg, training=False)
            parts = [extract_signal(a, env, env_cfg.actions, start, len(f)) for a in agents]
            rl_signals[asset] = pd.concat(parts, axis=1).mean(axis=1)
            px = prices[asset].loc[t1:t2]
            test_prices[asset] = px
            rule_signals[asset] = apply_rebalance_band(ladder_signal(prices[asset].loc[t0:t2], 24, long_only=False), 0.10).loc[t1:t2]

        common = {a: test_prices[a] for a in rl_signals}
        res = {}
        for name, sig in (("RL", rl_signals), ("ルール(v2)", rule_signals)):
            aligned = {a: sig[a].reindex(common[a].index).ffill().fillna(0.0) for a in common}
            r = backtest_portfolio(common, aligned, 60, pcfg)
            mt = equity_metrics(r["equity"])
            res[name] = mt
            print(f"  {name:10s} Sharpe {mt['sharpe']:+.2f}  年率 {mt['cagr']:+7.1%}  最大DD {mt['max_drawdown']:+6.1%}  "
                  f"回転 {r['turnover'].sum()/(len(r)/24):.2f}/日")
        rows.append({"fold": fold, "test_start": t1, "rl": res["RL"]["sharpe"], "rule": res["ルール(v2)"]["sharpe"],
                     "rl_ret": res["RL"]["cagr"], "rule_ret": res["ルール(v2)"]["cagr"],
                     "rl_dd": res["RL"]["max_drawdown"], "rule_dd": res["ルール(v2)"]["max_drawdown"]})

    report = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    report.to_csv(Path(args.out) / "comparison.csv", index=False)
    print("\n=========== 集計 ===========")
    print(f"fold 数            : {len(report)}")
    print(f"RL 平均 Sharpe     : {report['rl'].mean():+.2f}  (中央値 {report['rl'].median():+.2f})")
    print(f"ルール 平均 Sharpe  : {report['rule'].mean():+.2f}  (中央値 {report['rule'].median():+.2f})")
    print(f"RL が勝った fold   : {int((report['rl'] > report['rule']).sum())}/{len(report)}")
    print(f"平均の差 (RL−ルール): {(report['rl'] - report['rule']).mean():+.2f}")
    print(f"出力: {args.out}/comparison.csv")


if __name__ == "__main__":
    main()
