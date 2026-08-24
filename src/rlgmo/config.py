"""YAML 設定の読み込み（dataclass への再帰マッピング）。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .costs import CostConfig
from .env import EnvConfig, RewardConfig
from .features import FeatureConfig
from .agents.ppo import PPOConfig
from .walkforward import WalkForwardConfig

T = TypeVar("T")


@dataclass
class DataConfig:
    """データソース設定。"""

    symbol: str = "BTC_JPY"          # GMO レバレッジ銘柄
    interval: str = "1min"
    path: str = "data/raw/BTC_JPY_1min.parquet"
    start: str = "2023-01-01"
    end: str = "2026-06-30"
    use_synthetic: bool = False      # 実データが無い環境での配線確認用
    synthetic_minutes: int = 60 * 24 * 180


@dataclass
class TrainConfig:
    """学習ループ設定。"""

    total_steps: int = 1_500_000
    n_envs: int = 8
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)   # アンサンブル用シード
    eval_every: int = 100_000
    episode_len: int = 1440
    confidence: float = 0.15         # アンサンブル期待ポジションの発注閾値
    out_dir: str = "runs/default"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """YAML から `ExperimentConfig` を作る（未指定のキーは既定値）。"""
    if path is None:
        return ExperimentConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _from_dict(ExperimentConfig, raw)


def dump_config(cfg: Any, path: str | Path) -> Path:
    """設定を YAML に保存する（実験の再現性のため必ず保存する）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_dict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _from_dict(cls: type[T], data: dict) -> T:
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)  # type: ignore[arg-type]
        elif isinstance(value, list) and f.name in {"seeds", "actions", "timeframes", "ret_lags", "hidden"}:
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    # ネストした dataclass はアノテーションが文字列の場合があるので明示的に処理する
    nested = {"data": DataConfig, "features": FeatureConfig, "env": EnvConfig, "ppo": PPOConfig,
              "walkforward": WalkForwardConfig, "train": TrainConfig, "reward": RewardConfig, "cost": CostConfig}
    for name, sub_cls in nested.items():
        if name in data and isinstance(data[name], dict) and name in {f.name for f in dataclasses.fields(cls)}:  # type: ignore[arg-type]
            kwargs[name] = _from_dict(sub_cls, data[name])
    return cls(**kwargs)  # type: ignore[return-value]


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
