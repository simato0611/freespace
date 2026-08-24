#!/usr/bin/env python3
"""採用前のストレステスト（設計書 7.5 節）。

学習済みアンサンブルを、想定が崩れた条件下で評価する。ここで壊れる方策は、
バックテストの数字がどれだけ良くても本番に出してはいけない。

シナリオ:
    base          想定どおり
    cost_x2       スプレッド・スリッページ 2 倍
    adverse_fill  常に不利側 +1bp で約定
    delay_1bar    判断が 1 バー遅れる（レイテンシ・障害）
    wide_spread   ボラ連動のスプレッド拡大を 3 倍に
    flash_crash   数分で -15% の急落を注入（ロスカット耐性）
    no_carry      建玉管理料なし（コストの内訳を見るための参考値）

Example:
    python scripts/stress_test.py --config configs/default.yaml \
        --models "runs/default/fold0_seed*.pt" --start 2026-05-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlgmo.agents.ppo import PPOAgent  # noqa: E402
from rlgmo.backtest import delayed_policy, ensemble_policy  # noqa: E402
from rlgmo.config import load_config  # noqa: E402
from rlgmo.env import TradingEnv  # noqa: E402
from rlgmo.features import build_features  # noqa: E402
from rlgmo.pipeline import evaluate_policy, prepare_data  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume"]


def inject_flash_crash(ohlcv: pd.DataFrame, depth: float = 0.15, bars: int = 5, at: float = 0.5) -> pd.DataFrame:
    """指定位置に数分間の急落と部分的な戻りを注入する。"""
    out = ohlcv.copy()
    start = int(len(out) * at)
    path = np.concatenate([np.linspace(0, -depth, bars), np.linspace(-depth, -depth * 0.6, bars * 3)])
    factor = np.ones(len(out))
    factor[start : start + len(path)] = 1 + path
    factor[start + len(path) :] = 1 - depth * 0.6
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    out["volume"] = out["volume"] * np.where(factor < 1, 5.0, 1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="runs/stress")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features, meta = prepare_data(cfg)
    if args.start:
        features, meta = features.loc[args.start :], meta.loc[args.start :]
    if args.end:
        features, meta = features.loc[: args.end], meta.loc[: args.end]

    paths = sorted(glob.glob(args.models))
    if not paths:
        raise SystemExit(f"モデルが見つかりません: {args.models}")
    agents = [PPOAgent.load(p) for p in paths]
    base_policy = ensemble_policy(agents, cfg.env.actions, cfg.train.confidence)
    flat_action = int(np.argmin(np.abs(np.asarray(cfg.env.actions))))
    base_env_cfg = dataclasses.replace(cfg.env, randomize_costs=False)

    def evaluate(env_cfg, feats, mt, policy):
        env = TradingEnv(feats, mt, env_cfg, training=False)
        _, metrics = evaluate_policy(env, policy)
        return metrics

    cost = base_env_cfg.cost
    scenarios = {
        "base": (base_env_cfg, features, meta, base_policy),
        "cost_x2": (
            dataclasses.replace(base_env_cfg, cost=dataclasses.replace(
                cost, half_spread_bp=cost.half_spread_bp * 2, slippage_bp=cost.slippage_bp * 2)),
            features, meta, base_policy),
        "adverse_fill": (
            dataclasses.replace(base_env_cfg, cost=dataclasses.replace(cost, slippage_bp=cost.slippage_bp + 1.0)),
            features, meta, base_policy),
        "delay_1bar": (base_env_cfg, features, meta, delayed_policy(base_policy, 1, flat_action)),
        "wide_spread": (
            dataclasses.replace(base_env_cfg, cost=dataclasses.replace(cost, spread_vol_beta=3.0)),
            features, meta, base_policy),
        "no_carry": (
            dataclasses.replace(base_env_cfg, cost=dataclasses.replace(cost, carry_mode="none")),
            features, meta, base_policy),
    }

    crashed = inject_flash_crash(meta[OHLCV])
    crash_features, crash_meta = build_features(crashed, cfg.features)
    scenarios["flash_crash"] = (base_env_cfg, crash_features, crash_meta, base_policy)

    rows = {}
    for name, (env_cfg, feats, mt, policy) in scenarios.items():
        rows[name] = evaluate(env_cfg, feats, mt, policy)
        print(f"[{name}] 完了")

    table = pd.DataFrame(rows).T[
        ["total_return", "sharpe", "max_drawdown", "ann_vol", "exposure", "turnover_per_day", "cost_drag_ann"]
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "stress_test.csv")
    print("\n" + table.round(3).to_string())

    base_sharpe = table.loc["base", "sharpe"]
    fragile = [n for n in ("cost_x2", "adverse_fill", "delay_1bar", "wide_spread")
               if table.loc[n, "sharpe"] < 0 <= base_sharpe]
    print("\n判定:")
    print(f"  base Sharpe = {base_sharpe:.2f}")
    if fragile:
        print(f"  ✗ 以下のシナリオで Sharpe が負になる: {', '.join(fragile)} → 採用しない")
    else:
        print("  ○ コスト・遅延のストレス下でも Sharpe は正")
    dd = table.loc["flash_crash", "max_drawdown"]
    print(f"  フラッシュクラッシュ時の最大 DD = {dd:.1%}"
          f" {'✗ 許容外' if dd < -0.25 else '○ 許容内'}")


if __name__ == "__main__":
    main()
