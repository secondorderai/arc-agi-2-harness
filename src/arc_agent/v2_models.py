from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from arc_agent.models import Grid


class SynthesizedProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = ""
    strategy_tags: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    python_source: str


class VerificationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: str
    expected: Grid | None = None
    actual: Grid | None = None
    error: str | None = None
    detail: str = ""


class InductionVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool = False
    static_safe: bool = False
    exact_cases: int = 0
    total_cases: int = 0
    mean_cell_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_balanced_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    source_tests_exact: int = 0
    source_tests_total: int = 0
    leave_one_out_exact: int = 0
    leave_one_out_total: int = 0
    transformed_exact: int = 0
    transformed_total: int = 0
    predictions: list[Grid] = Field(default_factory=list)
    failures: list[VerificationFailure] = Field(default_factory=list)
    score: float = 0.0


class ResponseUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        uncached = max(
            0,
            self.input_tokens - self.cached_input_tokens - self.cache_write_input_tokens,
        )
        return (
            uncached * 0.20 / 1_000_000
            + self.cached_input_tokens * 0.02 / 1_000_000
            + self.cache_write_input_tokens * 0.25 / 1_000_000
            + self.output_tokens * 1.20 / 1_000_000
        )


class ResponseSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_id: str
    status: Literal["queued", "in_progress", "completed", "failed", "cancelled", "incomplete"]
    body: dict[str, object]
    usage: ResponseUsage = Field(default_factory=ResponseUsage)


class BankProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    program_hash: str
    source_task_ids: list[str]
    hypothesis: str
    strategy_tags: list[str]
    invariants: list[str]
    python_source: str
    canonical_ast: str
    features: dict[str, float]
    verification: InductionVerification
    complexity: int


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    active_task_id: str | None = None
    completed_tasks: int = 0
    total_tasks: int = 0
    message: str = ""
