"""Explicit Astra subscription teacher / pinned Nanbeige student; no API fallback."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from arc_agent.v4_config import MODEL_ID, MODEL_REVISION


class TeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auth_mode: Literal["chatgpt_subscription"] = "chatgpt_subscription"
    model: Literal["gpt-6-astra"] = "gpt-6-astra"
    reasoning_effort: Literal["xhigh"] = "xhigh"
    codex_cli: str = "codex"
    subscription_daemon: Literal[False] = False
    request_timeout_seconds: float = Field(default=60, gt=0, le=60)


class V5Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    student: Literal["Nanbeige/Nanbeige4.2-3B"] = MODEL_ID
    student_revision: Literal["3384e426066d1a49c3aea90a7190b81260a6533f"] = MODEL_REVISION
    # Reuse the existing login, never its old model calls, candidates, or scores.
    auth_workspace: Path = Path("runs/v3-pilot")
    data: Path = Path("data/ARC-AGI-2/data/training")
    split: Path = Path("runs/v4/split.json")
    max_tasks: int = Field(default=1, ge=1, le=3000)
    max_rounds: int = Field(default=4, ge=1, le=32)
    task_seconds: float = Field(default=1800, gt=0, le=3600)
    poll_seconds: float = Field(default=2, gt=0, le=30)
    # App Server does not enforce an output token cap. This is recorded as a hint only.
    output_token_hint: int = Field(default=8192, ge=1, le=128000)
    student_sequence_tokens: int = Field(default=8192, ge=1024, le=8192)


def load_config(path: Path) -> V5Config:
    return V5Config.model_validate(yaml.safe_load(path.read_text()))
