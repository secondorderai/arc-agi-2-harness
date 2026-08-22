from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "prism-ml/Bonsai-27B-mlx-1bit"
    api_key: str = "none"
    max_tokens: int = 1024
    thinking_budget_tokens: int = 512
    temperature: float = 0.7
    timeout_seconds: float = 300.0
    vision_enabled: bool = False
    structural_summary_enabled: bool = False


class MlxTTTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_path: str = ".runtime/models/qwen3-4b-arc-mlx-4bit"
    rank: int = Field(default=32, ge=1, le=256)
    scale: float = Field(default=10.0, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    epochs: int = Field(default=5, ge=1, le=100)
    min_epochs: int = Field(default=2, ge=1, le=100)
    early_stop_median_loss: float | None = Field(default=None, gt=0)
    early_stop_max_loss: float = Field(default=0.05, gt=0)
    train_token_layers: bool = False
    augmentation_count: int = Field(default=10, ge=1, le=10)
    inference_augmentations: int = Field(default=1, ge=1, le=10)
    scoring_augmentations: int = Field(default=0, ge=0, le=10)
    samples: int = Field(default=3, ge=0, le=32)
    include_greedy: bool = True
    temperature: float = Field(default=0.5, ge=0)
    top_p: float = Field(default=0.0, ge=0, le=1)
    dfs_min_probability: float | None = Field(default=None, gt=0, le=1)
    dfs_max_candidates: int = Field(default=32, ge=1, le=256)
    dfs_max_nodes: int = Field(default=2048, ge=1, le=100_000)
    max_tokens: int = Field(default=1024, ge=1, le=2048)
    max_grid_side: int = Field(default=30, ge=1, le=30)


class HybridWorldModelConfig(BaseModel):
    """Configuration for the object-world-model plus verified symbolic executor."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    symbolic_enabled: bool = True
    neural_enabled: bool = True
    neural_backend: Literal["mlx", "numpy"] = "mlx"
    checkpoint_path: str | None = None
    max_program_depth: int = Field(default=3, ge=1, le=8)
    max_search_nodes: int = Field(default=10_000, ge=1, le=1_000_000)
    max_symbolic_candidates: int = Field(default=16, ge=1, le=256)
    search_time_seconds: float = Field(default=30.0, gt=0, le=3_600)
    neural_top_k: int = Field(default=8, ge=1, le=128)
    require_demo_verification: bool = True


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_seconds: float = 41_400.0
    level_1_seconds: float = 5.0
    level_2_seconds: float = 240.0
    level_3_seconds: float = 600.0
    level_2_calls: int = 2
    level_3_calls: int = 3
    independent_calls: int = 2


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "local-mlx"
    model: ModelConfig = Field(default_factory=ModelConfig)
    mlx_ttt: MlxTTTConfig = Field(default_factory=MlxTTTConfig)
    hybrid: HybridWorldModelConfig = Field(default_factory=HybridWorldModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    skill_limit: int = 4
    seed: int = 42

    def sha256(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> SolverConfig:
    return SolverConfig.model_validate(yaml.safe_load(Path(path).read_text()))
