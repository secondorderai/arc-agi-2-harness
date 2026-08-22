"""Integration layer for object-program search and compact world-model proposals."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from arc_agent.config import HybridWorldModelConfig
from arc_agent.models import ArcTask, Candidate, Grid, grid_accuracies
from arc_agent.symbolic import SynthesisCandidate, synthesize


def _candidate_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _symbolic_candidate(candidate: SynthesisCandidate, task: ArcTask) -> Candidate:
    payload = candidate.program.to_json()
    return Candidate(
        candidate_id=_candidate_id("hybrid-symbolic", payload),
        source="hybrid_symbolic",
        hypothesis=f"Verified object program: {candidate.program.description()}",
        symbolic_program=candidate.program.to_dict(),
        predictions=[_copy_grid(grid) for grid in candidate.predictions],
        verified=candidate.verified and candidate.exact_pairs == len(task.train),
        exact_train_pairs=candidate.exact_pairs,
        train_pairs=candidate.total_pairs,
        train_cell_accuracy=candidate.train_cell_accuracy,
        train_balanced_accuracy=candidate.train_balanced_accuracy,
        score=(
            candidate.exact_pairs * 100.0
            + candidate.train_balanced_accuracy * 20.0
            + candidate.train_cell_accuracy * 2.0
            - candidate.description_length
        ),
    )


def symbolic_candidates(
    task: ArcTask,
    settings: HybridWorldModelConfig,
    *,
    operation_priors: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """Return bounded, demonstration-verified object-program candidates."""

    if not settings.symbolic_enabled:
        return []
    synthesized = synthesize(
        task,
        max_depth=settings.max_program_depth,
        max_nodes=settings.max_search_nodes,
        max_candidates=settings.max_symbolic_candidates,
        timeout_seconds=settings.search_time_seconds,
        operation_priors=operation_priors,
    )
    return [_symbolic_candidate(candidate, task) for candidate in synthesized]


def _load_world_model(settings: HybridWorldModelConfig) -> Any | None:
    if not settings.neural_enabled or not settings.checkpoint_path:
        return None
    checkpoint = Path(settings.checkpoint_path)
    if not checkpoint.is_file():
        return None
    if settings.neural_backend == "mlx":
        from arc_agent.world_model import MlxHybridWorldModel

        return MlxHybridWorldModel.load_checkpoint(checkpoint)
    from arc_agent.world_model import HybridWorldModel

    return HybridWorldModel.load_checkpoint(checkpoint)


def _learned_operation_priors(
    model: Any | None,
    task: ArcTask,
    *,
    top_k: int,
) -> dict[str, float]:
    """Aggregate the neural operation head over unlabelled test queries."""

    if model is None:
        return {}
    operation_map = {
        "tile": "repeat",
        "object_transform": "transform",
        "relation_follow": "relation_chain",
    }
    demonstrations = [(pair.input, pair.output or []) for pair in task.train]
    scores: defaultdict[str, float] = defaultdict(float)
    for pair in task.test:
        prediction = model.predict(
            pair.input,
            demonstrations,
            output_shape=(len(pair.input), len(pair.input[0])),
            top_k_programs=top_k,
        )
        for candidate in prediction.program_candidates:
            operation = operation_map.get(candidate.op, candidate.op)
            if operation != "unknown":
                scores[operation] += float(candidate.probability)
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {operation: score / total for operation, score in scores.items()}


def _shape_rules(task: ArcTask) -> tuple[tuple[str, Callable[[Grid], tuple[int, int]]], ...]:
    """Infer a small set of output-shape rules without reading test answers."""

    rules: list[tuple[str, Callable[[Grid], tuple[int, int]]]] = [
        ("same", lambda grid: (len(grid), len(grid[0])))
    ]
    output_shapes = {
        (len(pair.output or []), len((pair.output or [[]])[0])) for pair in task.train
    }
    if len(output_shapes) == 1:
        fixed = next(iter(output_shapes))
        rules.append(("fixed", lambda _grid, shape=fixed: shape))
    deltas = {
        (
            len(pair.output or []) - len(pair.input),
            len((pair.output or [[]])[0]) - len(pair.input[0]),
        )
        for pair in task.train
    }
    if len(deltas) == 1:
        row_delta, column_delta = next(iter(deltas))
        rules.append(
            (
                "delta",
                lambda grid, dr=row_delta, dc=column_delta: (
                    len(grid) + dr,
                    len(grid[0]) + dc,
                ),
            )
        )
    if all(
        len(pair.output or []) == len(pair.input[0])
        and len((pair.output or [[]])[0]) == len(pair.input)
        for pair in task.train
    ):
        rules.append(("transpose", lambda grid: (len(grid[0]), len(grid))))
    unique: list[tuple[str, Callable[[Grid], tuple[int, int]]]] = []
    # Preserve stable preference order while removing rules equivalent on all test inputs.
    emitted: set[tuple[tuple[int, int], ...]] = set()
    for name, rule in rules:
        shapes = tuple(rule(pair.input) for pair in task.test)
        if shapes in emitted or not all(1 <= side <= 30 for shape in shapes for side in shape):
            continue
        emitted.add(shapes)
        unique.append((name, rule))
    return tuple(unique)


def _training_accuracy(
    model: object,
    task: ArcTask,
    *,
    top_k: int,
) -> tuple[int, float, float]:
    """Leave-one-demonstration-out evidence for a direct neural predictor."""

    exact = 0
    raw_scores: list[float] = []
    balanced_scores: list[float] = []
    for index, pair in enumerate(task.train):
        demonstrations = [
            (other.input, other.output or [])
            for other_index, other in enumerate(task.train)
            if other_index != index
        ]
        if not demonstrations:
            demonstrations = [(pair.input, pair.output or [])]
        prediction = model.predict(
            pair.input,
            demonstrations,
            output_shape=(len(pair.output or []), len((pair.output or [[]])[0])),
            top_k_programs=top_k,
        )
        wanted = pair.output or []
        raw, balanced = grid_accuracies(prediction.grid, wanted)
        exact += int(prediction.grid == wanted)
        raw_scores.append(raw)
        balanced_scores.append(balanced)
    return (
        exact,
        sum(raw_scores) / len(raw_scores),
        sum(balanced_scores) / len(balanced_scores),
    )


def world_model_candidates(
    task: ArcTask,
    settings: HybridWorldModelConfig,
    *,
    model: Any | None = None,
) -> list[Candidate]:
    """Load a local checkpoint and turn neural outputs into auditable candidates."""

    if model is None:
        model = _load_world_model(settings)
    if model is None:
        return []
    exact, raw, balanced = _training_accuracy(model, task, top_k=settings.neural_top_k)
    verified = exact == len(task.train)
    if settings.require_demo_verification and not verified:
        return []
    demonstrations = [(pair.input, pair.output or []) for pair in task.train]
    candidates: list[Candidate] = []
    for rule_name, shape_rule in _shape_rules(task):
        predictions: list[Grid] = []
        program_summaries: list[str] = []
        for pair in task.test:
            prediction = model.predict(
                pair.input,
                demonstrations,
                output_shape=shape_rule(pair.input),
                top_k_programs=settings.neural_top_k,
            )
            predictions.append(prediction.grid)
            program_summaries.append(
                ", ".join(
                    f"{item.op}:{item.argument}@{item.probability:.3f}"
                    for item in prediction.program_candidates
                )
            )
        payload = f"{rule_name}:{predictions}:{program_summaries}"
        candidates.append(
            Candidate(
                candidate_id=_candidate_id("hybrid-world", payload),
                source="hybrid_world_model",
                hypothesis=(
                    f"Looped object-world model; shape_rule={rule_name}; operation priors="
                    + " | ".join(program_summaries)
                ),
                predictions=predictions,
                verified=verified,
                exact_train_pairs=exact,
                train_pairs=len(task.train),
                train_cell_accuracy=raw,
                train_balanced_accuracy=balanced,
                score=exact * 100.0 + balanced * 20.0 + raw * 2.0,
            )
        )
    return candidates


def generate_hybrid_candidates(
    task: ArcTask, settings: HybridWorldModelConfig
) -> list[Candidate]:
    """Generate symbolic and neural candidates, deduplicated by test behavior."""

    if not settings.enabled:
        return []
    model = None
    if settings.neural_enabled:
        try:
            model = _load_world_model(settings)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            # The exact symbolic branch remains usable when an optional local
            # accelerator/checkpoint cannot be loaded.
            model = None
    try:
        priors = _learned_operation_priors(model, task, top_k=settings.neural_top_k)
    except (OSError, RuntimeError, TypeError, ValueError):
        priors = {}
    candidates = symbolic_candidates(
        task,
        settings,
        operation_priors=priors,
    )
    with suppress(OSError, RuntimeError, TypeError, ValueError):
        candidates += world_model_candidates(task, settings, model=model)
    best_by_prediction: dict[str, Candidate] = {}
    for candidate in candidates:
        key = repr(candidate.predictions)
        previous = best_by_prediction.get(key)
        if previous is None or candidate.score > previous.score:
            best_by_prediction[key] = candidate
    return sorted(
        best_by_prediction.values(),
        key=lambda candidate: (not candidate.verified, -candidate.score, candidate.candidate_id),
    )


__all__ = [
    "generate_hybrid_candidates",
    "symbolic_candidates",
    "world_model_candidates",
]
