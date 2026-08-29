from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResponsesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth_mode: Literal["chatgpt_subscription", "api_key"] = "chatgpt_subscription"
    base_url: str = "https://api.openai.com/v1"
    model: Literal["gpt-5.6-luna", "gpt-5.6-terra"] = "gpt-5.6-luna"
    api_key_env: str = "OPENAI_API_KEY"
    codex_cli: str = "codex"
    subscription_daemon: bool = True
    reasoning_effort: Literal["xhigh"] = "xhigh"
    reasoning_context: Literal["all_turns"] = "all_turns"
    background: bool = True
    store: bool = True
    max_output_tokens: list[int] = Field(
        default_factory=lambda: [32_768, 65_536, 128_000], min_length=1
    )
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    poll_initial_seconds: float = Field(default=2.0, gt=0)
    poll_max_seconds: float = Field(default=60.0, gt=0)
    retry_initial_seconds: float = Field(default=5.0, gt=0)
    retry_max_seconds: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def valid_token_ladder(self) -> ResponsesConfig:
        if self.max_output_tokens != sorted(set(self.max_output_tokens)):
            raise ValueError("max_output_tokens must be unique and increasing")
        if self.max_output_tokens[-1] > 128_000:
            raise ValueError("V2 synthesis models support at most 128,000 output tokens")
        return self


class GuardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_source_chars: int = Field(default=20_000, ge=100, le=100_000)
    max_literal_scalars: int = Field(default=64, ge=4, le=10_000)
    wall_timeout_seconds: float = Field(default=3.0, gt=0)
    d4_transforms: bool = True
    color_permutations: int = Field(default=4, ge=0, le=20)
    require_leave_one_out: bool = True


class RankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    seed: int = 42
    epochs: int = Field(default=10, ge=1, le=1_000)
    alpha: float = Field(default=1e-4, gt=0)
    folds: int = Field(default=5, ge=2, le=20)
    direct_verified_target: int = Field(default=8, ge=1, le=100)


class V2ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "v2-luna-xhigh"
    openai: ResponsesConfig = Field(default_factory=ResponsesConfig)
    guards: GuardConfig = Field(default_factory=GuardConfig)
    ranker: RankerConfig = Field(default_factory=RankerConfig)
    no_progress_reset_rounds: int = Field(default=3, ge=1, le=100)
    retrieved_programs: int = Field(default=8, ge=0, le=100)
    # Worker count is operational and intentionally excluded from the frozen
    # experiment hash so an unfinished workspace can resume at a new concurrency.
    training_concurrency: int = Field(default=2, ge=1, le=16, exclude=True)
    max_refinement_rounds: int = Field(default=80, ge=1, le=10_000, exclude=True)
    seed: int = 42

    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def load_v2_config(path: str | Path) -> V2ExperimentConfig:
    return V2ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
