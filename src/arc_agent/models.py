from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Cell = Annotated[int, Field(ge=0, le=9)]
Grid = list[list[Cell]]


def grid_accuracies(predicted: Grid, expected: Grid) -> tuple[float, float]:
    """Return ordinary and expected-color-balanced cell accuracy."""
    if (
        not predicted
        or not expected
        or len(predicted) != len(expected)
        or len(predicted[0]) != len(expected[0])
        or any(len(row) != len(expected[0]) for row in predicted)
    ):
        return 0.0, 0.0
    cells = [
        (predicted[row][column], expected[row][column])
        for row in range(len(expected))
        for column in range(len(expected[0]))
    ]
    raw = sum(actual == wanted for actual, wanted in cells) / len(cells)
    colors = {wanted for _, wanted in cells}
    balanced = sum(
        sum(actual == wanted for actual, wanted in cells if wanted == color)
        / sum(wanted == color for _, wanted in cells)
        for color in colors
    ) / len(colors)
    return raw, balanced


def validate_grid(grid: Grid, *, label: str = "grid") -> Grid:
    if not grid or not grid[0]:
        raise ValueError(f"{label} must not be empty")
    width = len(grid[0])
    if not 1 <= len(grid) <= 30 or not 1 <= width <= 30:
        raise ValueError(f"{label} dimensions must be between 1 and 30")
    if any(len(row) != width for row in grid):
        raise ValueError(f"{label} must be rectangular")
    if any(
        not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9
        for row in grid
        for cell in row
    ):
        raise ValueError(f"{label} cells must be integers from 0 to 9")
    return grid


class ArcPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Grid
    output: Grid | None = None

    @model_validator(mode="after")
    def valid_grids(self) -> ArcPair:
        validate_grid(self.input, label="input")
        if self.output is not None:
            validate_grid(self.output, label="output")
        return self


class ArcTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    train: list[ArcPair]
    test: list[ArcPair]

    @model_validator(mode="after")
    def valid_task(self) -> ArcTask:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not self.train or not self.test:
            raise ValueError("a task needs at least one train and one test pair")
        if any(pair.output is None for pair in self.train):
            raise ValueError("all training pairs need outputs")
        return self


class Program(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    args: dict[str, Any] = Field(default_factory=dict)
    steps: list[Program] = Field(default_factory=list)

    def complexity(self) -> int:
        return 1 + sum(step.complexity() for step in self.steps) + len(self.args)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source: Literal[
        "deterministic",
        "llm_dsl",
        "llm_python",
        "direct",
        "mlx_ttt",
        "hybrid_symbolic",
        "hybrid_world_model",
        "fallback",
        "model_error",
    ]
    hypothesis: str = ""
    program: Program | None = None
    symbolic_program: dict[str, Any] | None = None
    python_source: str | None = None
    predictions: list[Grid] = Field(default_factory=list)
    verified: bool = False
    exact_train_pairs: int = 0
    train_pairs: int = 0
    train_cell_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    train_balanced_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    error: str | None = None
    score: float = 0.0


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_1: Grid
    attempt_2: Grid

    @model_validator(mode="after")
    def valid_attempts(self) -> Attempt:
        validate_grid(self.attempt_1, label="attempt_1")
        validate_grid(self.attempt_2, label="attempt_2")
        return self


Submission = dict[str, list[Attempt]]


class VerificationResult(BaseModel):
    exact_pairs: int
    total_pairs: int
    cell_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    balanced_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)

    @property
    def perfect(self) -> bool:
        return self.exact_pairs == self.total_pairs and not self.errors


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_seconds: float = 0.0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.latency_seconds += other.latency_seconds


class ModelResponse(BaseModel):
    level: int
    round_index: int
    content: str = ""
    reasoning: str = ""


class TaskRun(BaseModel):
    task_id: str
    initial_level: int
    final_level: int
    attempts: list[Attempt]
    candidates: list[Candidate]
    model_responses: list[ModelResponse] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0.0
    timed_out: bool = False


class RunSummary(BaseModel):
    run_id: str
    dataset_sha256: str
    config_sha256: str
    tasks: int
    test_outputs: int
    pass_at_2: float | None = None
    strict_task_accuracy: float | None = None
    elapsed_seconds: float
    usage: Usage
    level_counts: dict[int, int]
