from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass

from arc_agent.config import SolverConfig
from arc_agent.dsl import apply_program, search_programs, verify_program
from arc_agent.features import route_level
from arc_agent.hybrid import generate_hybrid_candidates
from arc_agent.llm import Generation, ModelAdapter
from arc_agent.models import (
    ArcTask,
    Attempt,
    Candidate,
    Grid,
    ModelResponse,
    Submission,
    TaskRun,
    Usage,
    grid_accuracies,
)
from arc_agent.sandbox import run_python_program
from arc_agent.skills import SkillCard, retrieve_cards

SEARCH_MODES = (
    "Object-centric search: segment objects and panels; reason about roles, relations, motion, "
    "containment, and which object acts as an instruction.",
    "Compression search: look for the shortest generative description using periodicity, symmetry, "
    "tiling, masks, logical overlays, counting, and error repair.",
    "Semantic search: treat shapes and colors as tokens or instructions; enumerate distinct rules "
    "and reject shortcuts that do not generalize to the probe input.",
)


@dataclass
class BudgetManager:
    total_seconds: float

    def __post_init__(self) -> None:
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed)

    def available(self, requested: float) -> float:
        return max(0.0, min(requested, self.remaining))


def _candidate_id(prefix: str, payload: str) -> str:
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:10]}"


def _grid_diff(predicted: Grid, expected: Grid, *, limit: int = 80) -> tuple[float, float, str]:
    expected_shape = (len(expected), len(expected[0]))
    predicted_shape = (len(predicted), len(predicted[0]))
    if predicted_shape != expected_shape:
        return 0.0, 0.0, f"expected shape {expected_shape}, got {predicted_shape}"
    differences = [
        (row, column, predicted[row][column], expected[row][column])
        for row in range(expected_shape[0])
        for column in range(expected_shape[1])
        if predicted[row][column] != expected[row][column]
    ]
    accuracy, balanced_accuracy = grid_accuracies(predicted, expected)
    detail = ", ".join(
        f"({row},{column})={actual}/{wanted}" for row, column, actual, wanted in differences[:limit]
    )
    if len(differences) > limit:
        detail += f", ... {len(differences) - limit} more"
    return (
        accuracy,
        balanced_accuracy,
        f"cell_accuracy={accuracy:.3f}; balanced_accuracy={balanced_accuracy:.3f}; "
        f"differing cells actual/expected: {detail}",
    )


def _verify_python(
    source: str, task: ArcTask
) -> tuple[bool, int, float, float, list[str], list[Grid]]:
    exact = 0
    pair_scores: list[float] = []
    balanced_scores: list[float] = []
    errors: list[str] = []
    for index, pair in enumerate(task.train):
        result = run_python_program(source, pair.input)
        if result.error:
            pair_scores.append(0.0)
            balanced_scores.append(0.0)
            errors.append(f"pair {index}: {result.error}")
        elif result.grid == pair.output:
            exact += 1
            pair_scores.append(1.0)
            balanced_scores.append(1.0)
        else:
            accuracy, balanced_accuracy, detail = _grid_diff(
                result.grid or [[]], pair.output or [[]]
            )
            pair_scores.append(accuracy)
            balanced_scores.append(balanced_accuracy)
            errors.append(f"pair {index}: {detail}")
    predictions: list[Grid] = []
    for index, pair in enumerate(task.test):
        result = run_python_program(source, pair.input)
        if result.error or result.grid is None:
            errors.append(f"test {index}: {result.error or 'no output'}")
            predictions = []
            break
        predictions.append(result.grid)
    return (
        exact == len(task.train) and not errors,
        exact,
        sum(pair_scores) / len(pair_scores),
        sum(balanced_scores) / len(balanced_scores),
        errors,
        predictions,
    )


def _from_generation(generation: Generation, task: ArcTask, *, round_index: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for generated in generation.candidates:
        if generated.program is not None:
            verification = verify_program(generated.program, task)
            predictions: list[Grid] = []
            try:
                predictions = [apply_program(generated.program, pair.input) for pair in task.test]
            except Exception as exc:
                verification.errors.append(f"test execution: {type(exc).__name__}: {exc}")
                predictions = []
            payload = generated.program.model_dump_json()
            candidates.append(
                Candidate(
                    candidate_id=_candidate_id(f"dsl-r{round_index}", payload),
                    source="llm_dsl",
                    hypothesis=generated.hypothesis,
                    program=generated.program,
                    predictions=predictions,
                    verified=verification.perfect and len(predictions) == len(task.test),
                    exact_train_pairs=verification.exact_pairs,
                    train_pairs=verification.total_pairs,
                    train_cell_accuracy=verification.cell_accuracy,
                    train_balanced_accuracy=verification.balanced_accuracy,
                    error="; ".join(verification.errors) or None,
                    score=(
                        verification.exact_pairs * 100.0
                        + verification.balanced_accuracy * 20.0
                        + verification.cell_accuracy * 2.0
                        - generated.program.complexity()
                    ),
                )
            )
        elif generated.python_source:
            verified, exact, cell_accuracy, balanced_accuracy, errors, predictions = (
                _verify_python(generated.python_source, task)
            )
            candidates.append(
                Candidate(
                    candidate_id=_candidate_id(f"python-r{round_index}", generated.python_source),
                    source="llm_python",
                    hypothesis=generated.hypothesis,
                    python_source=generated.python_source,
                    predictions=predictions,
                    verified=verified,
                    exact_train_pairs=exact,
                    train_pairs=len(task.train),
                    train_cell_accuracy=cell_accuracy,
                    train_balanced_accuracy=balanced_accuracy,
                    error="; ".join(errors) or None,
                    score=(
                        exact * 100.0
                        + balanced_accuracy * 20.0
                        + cell_accuracy * 2.0
                        - min(20.0, len(generated.python_source) / 1000)
                    ),
                )
            )
        elif len(generated.predictions) == len(task.test):
            source = "mlx_ttt" if generated.source == "mlx_ttt" else "direct"
            candidates.append(
                Candidate(
                    candidate_id=_candidate_id(
                        f"direct-r{round_index}", repr(generated.predictions)
                    ),
                    source=source,
                    hypothesis=generated.hypothesis,
                    predictions=generated.predictions,
                    verified=False,
                    train_pairs=len(task.train),
                    score=10.0 + 90.0 * generated.confidence,
                )
            )
    return candidates


def _feedback(candidates: Iterable[Candidate]) -> list[str]:
    ranked = sorted(
        (candidate for candidate in candidates if not candidate.verified),
        key=lambda candidate: (
            candidate.exact_train_pairs,
            candidate.train_balanced_accuracy,
            candidate.train_cell_accuracy,
            candidate.score,
        ),
        reverse=True,
    )
    feedback: list[str] = []
    seen_payloads: set[str] = set()
    for candidate in ranked:
        payload = candidate.python_source or (
            candidate.program.model_dump_json()
            if candidate.program
            else repr(candidate.symbolic_program or candidate.candidate_id)
        )
        if payload in seen_payloads:
            continue
        seen_payloads.add(payload)
        description = (
            f"{candidate.candidate_id} ({candidate.source}): matched "
            f"{candidate.exact_train_pairs}/{candidate.train_pairs} training pairs; "
            f"balanced accuracy={candidate.train_balanced_accuracy:.3f}; "
            f"raw cell accuracy={candidate.train_cell_accuracy:.3f}"
        )
        if candidate.hypothesis:
            description += f"\nHypothesis: {candidate.hypothesis[:500]}"
        if candidate.python_source:
            description += (
                f"\nPrior code to repair:\n```python\n{candidate.python_source[:1400]}\n```"
            )
        if candidate.error:
            description += f"\nVerifier counterexamples: {candidate.error[:1400]}"
        feedback.append(description)
        if len(feedback) == 2:
            break
    return feedback


def _prediction_key(predictions: list[Grid]) -> str:
    return repr(predictions)


def _fallbacks(task: ArcTask) -> tuple[list[Grid], list[Grid]]:
    identity = [[list(row) for row in pair.input] for pair in task.test]
    zeros = [[[0 for _ in row] for row in pair.input] for pair in task.test]
    return identity, zeros


def _build_attempts(task: ArcTask, candidates: Iterable[Candidate]) -> list[Attempt]:
    candidate_list = list(candidates)
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidate_list:
        if len(candidate.predictions) == len(task.test):
            grouped.setdefault(_prediction_key(candidate.predictions), []).append(candidate)
    consensus_ranked = sorted(
        grouped.values(),
        key=lambda group: (
            not any(candidate.verified for candidate in group),
            -len(group),
            -max(candidate.score for candidate in group),
            min(candidate.candidate_id for candidate in group),
        ),
    )
    score_ranked = sorted(
        candidate_list,
        key=lambda item: (not item.verified, -item.score, item.candidate_id),
    )
    prediction_sets: list[list[Grid]] = []
    seen: set[str] = set()
    if consensus_ranked:
        best = max(consensus_ranked[0], key=lambda candidate: candidate.score)
        prediction_sets.append(best.predictions)
        seen.add(_prediction_key(best.predictions))
    for candidate in score_ranked:
        if len(candidate.predictions) != len(task.test):
            continue
        key = _prediction_key(candidate.predictions)
        if key not in seen:
            prediction_sets.append(candidate.predictions)
            seen.add(key)
        if len(prediction_sets) == 2:
            break
    for fallback in _fallbacks(task):
        key = _prediction_key(fallback)
        if key not in seen:
            prediction_sets.append(fallback)
            seen.add(key)
        if len(prediction_sets) == 2:
            break
    if len(prediction_sets) == 1:
        prediction_sets.append(prediction_sets[0])
    return [
        Attempt(attempt_1=prediction_sets[0][index], attempt_2=prediction_sets[1][index])
        for index in range(len(task.test))
    ]


class ArcSolver:
    def __init__(
        self,
        config: SolverConfig,
        *,
        cards: list[SkillCard] | None = None,
        adapter: ModelAdapter | None = None,
        budget: BudgetManager | None = None,
    ) -> None:
        self.config = config
        self.cards = cards or []
        self.adapter = adapter
        self.budget = budget or BudgetManager(config.budget.total_seconds)

    def solve_task(self, task: ArcTask) -> TaskRun:
        started = time.monotonic()
        usage = Usage()
        model_responses: list[ModelResponse] = []
        deterministic = search_programs(task)
        candidates = list(deterministic)
        if self.config.hybrid.enabled:
            try:
                candidates.extend(generate_hybrid_candidates(task, self.config.hybrid))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                candidates.append(
                    Candidate(
                        candidate_id=_candidate_id("hybrid-error", message),
                        source="model_error",
                        hypothesis="Hybrid object-world-model path failed; retained other paths.",
                        train_pairs=len(task.train),
                        error=message,
                        score=-100.0,
                    )
                )
        initial_level = route_level(
            task, deterministic_found=any(candidate.verified for candidate in candidates)
        )
        final_level = initial_level
        timed_out = False

        if self.adapter is not None and not any(candidate.verified for candidate in candidates):
            # Deterministic search already exhausts the typed DSL. Model inference therefore
            # stays in one stable Python refinement mode, which also preserves prompt-cache
            # reuse across rounds.
            levels = [3]
            for level in levels:
                final_level = level
                per_task_limit = (
                    self.config.budget.level_2_seconds
                    if level == 2
                    else self.config.budget.level_3_seconds
                )
                call_limit = (
                    self.config.budget.level_2_calls
                    if level == 2
                    else self.config.budget.level_3_calls
                )
                task_deadline = time.monotonic() + self.budget.available(per_task_limit)
                skills = retrieve_cards(
                    self.cards, task, level=level, limit=self.config.skill_limit
                )
                for round_index in range(1, call_limit + 1):
                    if time.monotonic() >= task_deadline or self.budget.remaining <= 0:
                        timed_out = True
                        break
                    try:
                        independent = round_index <= min(
                            call_limit, self.config.budget.independent_calls
                        )
                        generation = self.adapter.generate(
                            task,
                            skills,
                            level=level,
                            feedback=None if independent else (_feedback(candidates) or None),
                            search_mode=(
                                SEARCH_MODES[(round_index - 1) % len(SEARCH_MODES)]
                            if independent
                            else "The prior candidate is disproven by the verifier. Returning "
                            "the same code is invalid. Derive a materially different rule from "
                            "all examples, then repair or replace the program."
                        ),
                        )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        candidates.append(
                            Candidate(
                                candidate_id=_candidate_id(f"model-error-r{round_index}", message),
                                source="model_error",
                                hypothesis="Model request failed; retained safe fallbacks.",
                                train_pairs=len(task.train),
                                error=message,
                                score=-100.0,
                            )
                        )
                        break
                    usage.add(generation.usage)
                    model_responses.append(
                        ModelResponse(
                            level=level,
                            round_index=round_index,
                            content=generation.raw_text,
                            reasoning=generation.reasoning_text,
                        )
                    )
                    round_candidates = _from_generation(generation, task, round_index=round_index)
                    if not generation.candidates:
                        round_candidates.append(
                            Candidate(
                                candidate_id=_candidate_id(
                                    f"parse-error-r{round_index}", generation.raw_text
                                ),
                                source="model_error",
                                hypothesis="No executable candidate could be parsed.",
                                train_pairs=len(task.train),
                                error=(
                                    "No structured Python candidate was completed. Put concise "
                                    "executable solve(grid) code in python_source before any "
                                    "explanation."
                                ),
                                score=-90.0,
                            )
                        )
                    candidates.extend(round_candidates)
                    if any(candidate.verified for candidate in round_candidates):
                        break
                if any(candidate.verified for candidate in candidates) or timed_out:
                    break

        attempts = _build_attempts(task, candidates)
        return TaskRun(
            task_id=task.task_id,
            initial_level=initial_level,
            final_level=final_level,
            attempts=attempts,
            candidates=candidates,
            model_responses=model_responses,
            usage=usage,
            elapsed_seconds=time.monotonic() - started,
            timed_out=timed_out,
        )

    def solve_tasks(self, tasks: Iterable[ArcTask]) -> tuple[list[TaskRun], Submission]:
        runs: list[TaskRun] = []
        submission: Submission = {}
        for task in tasks:
            run = self.solve_task(task)
            runs.append(run)
            submission[task.task_id] = run.attempts
        return runs, submission
