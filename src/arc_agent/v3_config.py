from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class V3TeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth_mode: Literal["chatgpt_subscription", "api_key"] = "chatgpt_subscription"
    base_url: str = "https://api.openai.com/v1"
    model: Literal["gpt-5.6-sol"] = "gpt-5.6-sol"
    api_key_env: str = "OPENAI_API_KEY"
    codex_cli: str = "codex"
    subscription_daemon: bool = True
    reasoning_effort: Literal["xhigh"] = "xhigh"
    reasoning_context: Literal["all_turns"] = "all_turns"
    background: bool = True
    store: bool = True
    max_output_tokens: list[int] = Field(
        default_factory=lambda: [4_096, 8_192, 16_384], min_length=1
    )
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    poll_initial_seconds: float = Field(default=2.0, gt=0)
    poll_max_seconds: float = Field(default=60.0, gt=0)
    retry_initial_seconds: float = Field(default=5.0, gt=0)
    retry_max_seconds: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def valid_token_ladder(self) -> V3TeacherConfig:
        if self.max_output_tokens != sorted(set(self.max_output_tokens)):
            raise ValueError("max_output_tokens must be unique and increasing")
        if self.max_output_tokens[-1] > 128_000:
            raise ValueError("teacher output token limits may not exceed 128,000")
        return self


class V3PilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: Literal[10] = 10
    family_pairs: Literal[2] = 2
    max_attempts_per_task: int = Field(default=12, ge=1, le=100)
    plateau_rounds: int = Field(default=3, ge=1, le=20)
    max_pipeline_steps: int = Field(default=16, ge=1, le=16)
    max_parameters: int = Field(default=20, ge=1, le=20)
    max_literal_scalars: int = Field(default=64, ge=1, le=1_000)


class V3SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieved_signatures: Literal[3] = 3
    mutations_per_signature: int = Field(default=32, ge=0, le=256)
    family_search_limit: int = Field(default=64, ge=1, le=1_024)
    global_search_limit: int = Field(default=128, ge=1, le=2_048)
    max_candidates_per_task: int = Field(default=256, ge=2, le=4_096)
    leave_one_out_weight: float = Field(default=25.0, ge=0)
    invariant_weight: float = Field(default=10.0, ge=0)
    complexity_weight: float = Field(default=1.0, ge=0)


class V3EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_active_seconds: float = Field(default=36_000.0, gt=0)
    hard_active_seconds: float = Field(default=43_200.0, gt=0)
    freeze_reserve_seconds: float = Field(default=7_200.0, ge=0)
    task_concurrency: int = Field(default=1, ge=1, le=16)

    @model_validator(mode="after")
    def valid_budget(self) -> V3EvaluationConfig:
        if self.target_active_seconds + self.freeze_reserve_seconds > self.hard_active_seconds:
            raise ValueError("target runtime plus freeze reserve exceeds the hard runtime")
        return self


class V3ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "v3-sol-xhigh-pilot"
    teacher: V3TeacherConfig = Field(default_factory=V3TeacherConfig)
    pilot: V3PilotConfig = Field(default_factory=V3PilotConfig)
    search: V3SearchConfig = Field(default_factory=V3SearchConfig)
    evaluation: V3EvaluationConfig = Field(default_factory=V3EvaluationConfig)
    seed: Literal[42] = 42

    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_v3_config(path: str | Path) -> V3ExperimentConfig:
    return V3ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
