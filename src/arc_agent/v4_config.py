"""V4 never inherits a legacy model, API endpoint, adapter, or money budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
MODEL_REVISION = "3384e426066d1a49c3aea90a7190b81260a6533f"
RUNTIME_REVISION = "c6640a1c0cf7b38df342b67021a3900b04d092e7"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class V4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Literal["Nanbeige/Nanbeige4.2-3B"] = MODEL_ID
    model_revision: Literal["3384e426066d1a49c3aea90a7190b81260a6533f"] = MODEL_REVISION
    runtime_revision: Literal["c6640a1c0cf7b38df342b67021a3900b04d092e7"] = RUNTIME_REVISION
    manifest: Path = Path(".runtime/nanbeige/manifest.json")
    base_url: str = "http://127.0.0.1:8094"
    # Pinned checkpoint metadata ceiling, not a claim this context fits a local Mac.
    context_tokens: int = Field(default=4096, ge=1024, le=262144)
    # Preserve historical 2K defaults; 4K is an explicit, separately frozen protocol.
    max_new_tokens: int = Field(default=2048, ge=1, le=4096)
    max_reasoning_tokens: int = Field(default=1024, ge=0, le=1536)
    cache_type: Literal["f16", "q8_0"] = "q8_0"
    task_seconds: float = Field(default=300, gt=0, le=3600)
    timing_mode: Literal["fixed", "diagnostic"] = "fixed"
    deadline_schedule_seconds: tuple[float, ...] = ()
    diagnostic_request_seconds: float = Field(default=900, gt=0, le=900)
    diagnostic_stall_rounds: int = Field(default=5, ge=2, le=32)
    max_rounds: int = Field(default=8, ge=1, le=128)
    pilot_tasks: int = Field(default=20, ge=1, le=20)
    seed: int = 42
    temperature: float = Field(default=0.6, ge=0, le=2)
    history: Literal["retain", "strip"] = "retain"
    min_free_disk_gib: float = Field(default=10, ge=10)
    max_swap_growth_gib: float = Field(default=1, gt=0)
    # Explicit user-approved local opt-in; historical configurations keep stopping.
    warning_pressure_policy: Literal["stop", "record"] = "stop"
    # None preserves archived defaults; new sequential runs can disable the separate
    # host-RAM prompt cache without changing their durable reasoning/evidence state.
    prompt_cache_mib: Literal[0] | None = None

    @model_validator(mode="after")
    def local_only(self) -> V4Config:
        url = urlsplit(self.base_url)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "V4 requires a literal loopback HTTP endpoint; remote LLMs are forbidden"
            )
        if self.max_new_tokens >= self.context_tokens:
            raise ValueError("context must leave room for the prompt")
        if self.timing_mode == "fixed":
            if self.task_seconds > 300 or self.deadline_schedule_seconds or self.max_rounds > 32:
                raise ValueError("extended timing requires an explicitly diagnostic configuration")
        else:
            schedule = self.deadline_schedule_seconds
            if (
                not schedule
                or schedule[0] != self.task_seconds
                or any(not 0 < value <= 3600 for value in schedule)
                or any(a >= b for a, b in zip(schedule, schedule[1:], strict=False))
            ):
                raise ValueError(
                    "diagnostic deadlines must increase from task_seconds up to one hour"
                )
        return self


def load_v4_config(path: Path) -> V4Config:
    return V4Config.model_validate(yaml.safe_load(path.read_text()))


def verify_manifest(path: Path, config: V4Config) -> dict:
    manifest = json.loads(path.read_text())
    expected = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "runtime_revision": config.runtime_revision,
        "quantization": "Q4_K_M",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"incompatible Nanbeige lineage: {key}")
    if manifest.get("adapter") is not None:
        raise ValueError("Phase 0 accepts only the original Nanbeige checkpoint, not adapters")
    required = {"model", "server", "tokenizer_config", "model_config", "tokenizer_model"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("incomplete model manifest")
    for name, artifact in manifest["files"].items():
        target = Path(artifact["path"])
        if not target.is_file() or file_hash(target) != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
    model_config = json.loads(Path(manifest["files"]["model_config"]["path"]).read_text())
    if model_config.get("model_type") != "nanbeige" or model_config.get("vocab_size") != 166144:
        raise ValueError("not the full-vocabulary Nanbeige architecture")
    return manifest
