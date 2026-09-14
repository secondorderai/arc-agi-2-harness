"""Astra local research evaluation. Development artifacts are NEVER SFT data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from arc_agent.models import ArcTask, validate_grid
from arc_agent.sandbox import ALLOWED_METHODS, SAFE_BUILTINS
from arc_agent.v2_codex import _response_id, _snapshot
from arc_agent.v2_config import GuardConfig
from arc_agent.v2_openai import TransientAPIError
from arc_agent.v2_sandbox import canonicalize_source
from arc_agent.v2_verifier import verification_feedback, verify_induction_program
from arc_agent.v4_config import content_hash
from arc_agent.v4_state import atomic_json, blind_task
from arc_agent.v5_config import TeacherConfig
from arc_agent.v5_teacher import AstraTeacher


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    teacher: TeacherConfig = Field(
        default_factory=lambda: TeacherConfig(
            codex_cli="/Applications/ChatGPT.app/Contents/Resources/codex"
        )
    )
    auth_workspace: Path = Path("runs/v3-pilot")
    data: Path = Path("data/ARC-AGI-2/data/training")
    split: Path = Path("runs/v4/split.json")
    cohort: Literal["frozen_v4_development_20"] = "frozen_v4_development_20"
    max_rounds: int = Field(default=4, ge=1, le=16)
    task_seconds: float = Field(default=1800, gt=0, le=3600)
    global_seconds: float = Field(default=39600, gt=0, le=39600)
    poll_seconds: float = Field(default=2, gt=0, le=30)
    output_token_hint: int = Field(default=16000, ge=1, le=128000)
    target: Literal[0.7] = 0.7


class SolverResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbolic_model: str = Field(min_length=1, max_length=8000)
    python_source: str | None = Field(max_length=20000)
    # A list of test grids, each represented by rows of unseparated color digits.
    direct_predictions: list[list[str]] | None
    alternative_predictions: list[list[str]] | None


def solver_schema():
    schema = SolverResponse.model_json_schema()
    schema["required"] = list(schema["properties"])
    return schema


SOLVER_INSTRUCTIONS = f"""Solve an ARC transformation from demonstrations and unlabelled inputs.
Return only the supplied JSON schema. Work autonomously from the supplied evidence; no questions.
Author a compact symbolic working model: grounded objects, roles, task-local definitions,
relations, competing rules, counterexamples and ordered actions. Keep uncertainty explicit.
This is an explicit solution artifact, not a request to disclose private chain of thought.
Derive the rule from ALL demonstrations; check shape, color roles and edge cases before answering.
Provide a general Python witness when feasible, and direct predictions for every test input.
For a credible alternative rule, also provide alternative_predictions; otherwise use null.
Predictions are lists of grids; each grid is a list of strings of color digits 0-9, no spaces.
Example syntax for two one-row test grids: [["012"],["330"]]. Do not copy this example as an answer.
The witness defines exactly def solve(train, grid): and returns a list-of-lists integer grid.
train is a list of (input_grid, output_grid) pairs. Derive task-dependent values from train/grid.
There is no fixed DSL restriction: use loops, conditionals, arithmetic, comprehensions, lists,
sets, dictionaries and top-level helper functions. The deterministic harness executes the text.
Python sandbox: no imports, classes, lambdas, exceptions, global/nonlocal, with, async,
dunder names/attributes, IO, eval, exec, reflection or third-party packages. Helper functions
must be defined at module scope. Allowed builtins: {", ".join(sorted(SAFE_BUILTINS))}.
Allowed methods: {", ".join(sorted(ALLOWED_METHODS))}. No other methods are available.
Do not embed whole observed grids or literal-grid lookup tables. Avoid long literal collections.
The witness is checked on demonstration inputs with and without each corresponding output in train.
Keep symbolic_model concise (roughly 500-1000 tokens); code must be complete, not pseudocode.
Do not call tools, inspect files, browse, use MCP, delegate, execute commands yourself, or use
remembered benchmark answers. Test labels are unavailable. Verifier feedback concerns demos only.
"""


class AstraSolver(AstraTeacher):
    def _thread_params(self, model):
        params = super()._thread_params(model)
        params["baseInstructions"] = SOLVER_INSTRUCTIONS
        params["developerInstructions"] = SOLVER_INSTRUCTIONS
        return params

    def _start_turn(self, *, thread_id, prompt, request_key, token_limit, model):
        self._ensure_ready(model)
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "effort": self.settings.reasoning_effort,
                "model": model,
                "clientUserMessageId": request_key,
                "outputSchema": solver_schema(),
            },
        )
        turn = result.get("turn")
        if not isinstance(turn, dict) or not turn.get("id"):
            raise TransientAPIError("solver turn/start returned no turn id")
        return _snapshot(
            response_id=_response_id(thread_id, str(turn["id"]), request_key),
            status="in_progress",
            thread_id=thread_id,
            turn_id=str(turn["id"]),
            token_limit=token_limit,
        )


def rows(grid):
    return ["".join(map(str, row)) for row in grid]


def initial_prompt(task: ArcTask) -> str:
    task = blind_task(task)
    return json.dumps(
        {
            "train": [{"input": rows(p.input), "output": rows(p.output)} for p in task.train],
            "test": [{"input": rows(p.input)} for p in task.test],
            "request": "Infer the shared transformation. Return a symbolic model, a general "
            "executable witness, and your best predictions. Test outputs are never provided.",
        },
        separators=(",", ":"),
    )


def decode_predictions(predictions, expected_count):
    if predictions is None:
        return None
    if len(predictions) != expected_count:
        raise ValueError("prediction count must equal the number of test inputs")
    grids = []
    for grid_rows in predictions:
        if any(not row or any(c not in "0123456789" for c in row) for row in grid_rows):
            raise ValueError("rows must contain only color digits")
        grid = [[int(c) for c in row] for row in grid_rows]
        grids.append(validate_grid(grid))
    return grids


def evaluate_candidate(response: SolverResponse, task: ArcTask, round_index: int) -> dict:
    """This boundary rejects test labels, even if its caller forgot to blind the task."""
    if any(p.output is not None for p in task.test):
        raise ValueError("verification requires a label-blind task")
    candidates, errors = [], []
    feedback, verified = "No executable witness supplied.", False
    for name in ("direct_predictions", "alternative_predictions"):
        try:
            predictions = decode_predictions(getattr(response, name), len(task.test))
            if predictions is not None:
                candidates.append(
                    {
                        "predictions": predictions,
                        "source": name,
                        "rank": [2, int(name == "direct_predictions"), round_index],
                        "ancestry": f"round:{round_index}",
                    }
                )
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
    verification = None
    if response.python_source:
        verification = verify_induction_program(
            response.python_source,
            task,
            include_test_labels=False,
            run_transformations=False,
            guards=GuardConfig(
                require_leave_one_out=True, d4_transforms=False, color_permutations=0
            ),
        )
        verified = verification.accepted
        feedback = verification_feedback(verification)
        if verification.predictions:
            base_failures = [f for f in verification.failures if f.case.startswith("demo_full_")]
            base_fit = verification.static_safe and not base_failures
            _, complexity = canonicalize_source(response.python_source)
            candidates.append(
                {
                    "predictions": verification.predictions,
                    "source": "program",
                    "rank": [3 if base_fit else 1, int(verified), -complexity],
                    "ancestry": f"round:{round_index}",
                }
            )
    return {
        "candidates": candidates,
        "errors": errors,
        "verified": verified,
        "feedback": feedback + ("\n" + "\n".join(errors) if errors else ""),
        "verification": verification.model_dump(mode="json") if verification else None,
    }


def select_predictions(task: ArcTask, candidates: list[dict]) -> list[dict]:
    selected = []
    for index, pair in enumerate(task.test):
        ordered, seen = [], set()
        for candidate in sorted(candidates, key=lambda c: c["rank"], reverse=True):
            prediction = candidate["predictions"][index]
            digest = content_hash(prediction)
            if digest not in seen:
                ordered.append(prediction)
                seen.add(digest)
            if len(ordered) == 2:
                break
        if not ordered:
            ordered.append(pair.input)
        if len(ordered) == 1:
            ordered.append(ordered[0])
        selected.append({"attempt_1": ordered[0], "attempt_2": ordered[1]})
    return selected


def score_predictions(tasks: list[ArcTask], submission: dict, completed: int) -> dict:
    """Post-hoc only: scores never reach model prompts, ranking or task scheduling."""
    if set(submission) != {task.task_id for task in tasks}:
        raise ValueError("submission does not cover the fixed cohort")
    correct, total, strict, macro = 0, 0, 0, 0.0
    details = []
    for task in tasks:
        attempts = submission[task.task_id]
        if len(attempts) != len(task.test):
            raise ValueError("submission test input count mismatch")
        hits = []
        for pair, attempt in zip(task.test, attempts, strict=True):
            if pair.output is None or set(attempt) != {"attempt_1", "attempt_2"}:
                raise ValueError("labelled scoring needs exactly two ordered attempts")
            validate_grid(attempt["attempt_1"])
            validate_grid(attempt["attempt_2"])
            hits.append(pair.output in [attempt["attempt_1"], attempt["attempt_2"]])
        correct += sum(hits)
        total += len(hits)
        strict += all(hits)
        macro += sum(hits) / len(hits)
        details.append({"task_id": task.task_id, "correct": sum(hits), "outputs": len(hits)})
    return {
        "correct_outputs": correct,
        "test_outputs": total,
        "exact_match": correct / total,
        "task_mean_exact_match": macro / len(tasks),
        "strict_correct_tasks": strict,
        "strict_task_accuracy": strict / len(tasks),
        "total_tasks": len(tasks),
        "completed_tasks": completed,
        "per_task": details,
        "goal_threshold_observed": len(tasks) == 20
        and completed == len(tasks)
        and correct / total >= 0.7
        and strict / len(tasks) >= 0.7,
        "model": "gpt-6-astra",
        "execution": "online_subscription_local_harness",
        "not_an_offline_nanbeige_score": True,
        "training_export_allowed": False,
    }


def write_outputs(run, labelled_tasks):
    states = [run.get(f"task:{t.task_id}") or {} for t in labelled_tasks]
    submission = {
        t.task_id: select_predictions(blind_task(t), s.get("candidates", []))
        for t, s in zip(labelled_tasks, states, strict=True)
    }
    completed = sum(bool(s.get("done")) for s in states)
    report = score_predictions(labelled_tasks, submission, completed)
    report["attempted_tasks"] = sum(s.get("round", 0) > 0 for s in states)
    report["responses"] = sum(s.get("round", 0) for s in states)
    report["usage"] = {
        name: sum(s.get("usage", {}).get(name, 0) for s in states)
        for name in ("input_tokens", "output_tokens", "reasoning_tokens")
    }
    report["task_solve_seconds"] = sum(s.get("elapsed", 0) for s in states)
    report["unknown_usage_calls"] = sum(s.get("unknown_usage_calls", 0) for s in states)
    atomic_json(run.root / "submission.json", submission)
    atomic_json(run.root / "report.json", report)
    return report
