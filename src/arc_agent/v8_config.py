"""Frozen local student comparison settings; no provider model or paid-job fields."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class V8Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_config: Path = Path("configs/v8-nanbeige-local.yaml")
    data: Path = Path("data/ARC-AGI-2/data/training")
    split: Path = Path("runs/v4/split.json")
    seeds: tuple[Literal[42], Literal[43], Literal[44]] = (42, 43, 44)
    modes: tuple[Literal["direct"], Literal["symbolic"]] = ("direct", "symbolic")
    structured_style: Literal[
        "legacy",
        "compact_rectangular",
        "compact_identifiers",
        "reference_repair",
        "reference_repair_bounded_ws",
    ] = "legacy"
    witness_repair_policy: Literal["revise_symbolic", "repair_static_errors"] = "revise_symbolic"
    symbolic_cell_encoding: Literal["named_fields", "triples_v1"] = "named_fields"
    smoke_contexts: Literal["4k_then_8k", "8k_then_12k"] = "4k_then_8k"
    max_cycles: int = Field(default=2, ge=1, le=4)
    task_seconds: float = Field(default=900, gt=0, le=3600)
    global_seconds: float = Field(default=39600, gt=0, le=39600)
    cohort: Literal["frozen_development_20"] = "frozen_development_20"
    state_policy: Literal["explicit_symbolic_state_with_external_evidence"] = (
        "explicit_symbolic_state_with_external_evidence"
    )


def load_config(path):
    return V8Config.model_validate(yaml.safe_load(Path(path).read_text()))
