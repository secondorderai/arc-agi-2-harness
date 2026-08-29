from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Callable, Sequence

from arc_agent.models import ArcTask, Grid, grid_accuracies
from arc_agent.sandbox import UnsafeProgram
from arc_agent.v2_config import GuardConfig
from arc_agent.v2_models import InductionVerification, VerificationFailure
from arc_agent.v2_sandbox import run_induction_program, validate_induction_source

TrainPairs = list[tuple[Grid, Grid]]


def _copy(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _rotate(grid: Grid) -> Grid:
    return [list(row) for row in zip(*reversed(grid), strict=True)]


def _flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_vertical(grid: Grid) -> Grid:
    return [list(row) for row in reversed(grid)]


def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid, strict=True)]


def d4_transforms() -> list[tuple[str, Callable[[Grid], Grid]]]:
    def rotate_180(grid: Grid) -> Grid:
        return _rotate(_rotate(grid))

    def rotate_270(grid: Grid) -> Grid:
        return _rotate(rotate_180(grid))

    def anti_transpose(grid: Grid) -> Grid:
        return _flip_horizontal(_flip_vertical(_transpose(grid)))

    return [
        ("identity", _copy),
        ("rotate_90", _rotate),
        ("rotate_180", rotate_180),
        ("rotate_270", rotate_270),
        ("flip_horizontal", _flip_horizontal),
        ("flip_vertical", _flip_vertical),
        ("transpose", _transpose),
        ("anti_transpose", anti_transpose),
    ]


def _background(task: ArcTask) -> int:
    values = [cell for pair in task.train for row in pair.input for cell in row]
    return Counter(values).most_common(1)[0][0]


def color_permutations(task: ArcTask, *, count: int, seed: int) -> list[dict[int, int]]:
    if count <= 0:
        return []
    colors = sorted(
        {
            cell
            for pair in [*task.train, *task.test]
            for grid in (pair.input, pair.output)
            if grid is not None
            for row in grid
            for cell in row
        }
    )
    background = _background(task)
    foreground = [color for color in colors if color != background]
    if len(foreground) < 2:
        return []
    rng = random.Random(seed ^ int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16))
    mappings: list[dict[int, int]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(mappings) < count and attempts < count * 20:
        attempts += 1
        shuffled = list(foreground)
        rng.shuffle(shuffled)
        key = tuple(shuffled)
        if key == tuple(foreground) or key in seen:
            continue
        seen.add(key)
        mappings.append({background: background, **dict(zip(foreground, shuffled, strict=True))})
    return mappings


def _map_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def _grid_detail(actual: Grid | None, expected: Grid) -> tuple[float, float, str]:
    if actual is None:
        return 0.0, 0.0, "no grid returned"
    actual_shape = (len(actual), len(actual[0]))
    expected_shape = (len(expected), len(expected[0]))
    if actual_shape != expected_shape:
        return 0.0, 0.0, f"expected shape {expected_shape}, got {actual_shape}"
    raw, balanced = grid_accuracies(actual, expected)
    differences = [
        (row, column, actual[row][column], expected[row][column])
        for row in range(len(expected))
        for column in range(len(expected[0]))
        if actual[row][column] != expected[row][column]
    ]
    detail = ", ".join(
        f"({row},{column})={got}/{wanted}" for row, column, got, wanted in differences[:80]
    )
    if len(differences) > 80:
        detail += f", ... {len(differences) - 80} more"
    return raw, balanced, detail


def _task_grids(task: ArcTask) -> list[Grid]:
    grids: list[Grid] = []
    for pair in [*task.train, *task.test]:
        grids.append(pair.input)
        if pair.output is not None:
            grids.append(pair.output)
    return grids


def _pairs(task: ArcTask, transform: Callable[[Grid], Grid]) -> TrainPairs:
    return [(transform(pair.input), transform(pair.output or [[]])) for pair in task.train]


def _tests(task: ArcTask, transform: Callable[[Grid], Grid]) -> list[tuple[Grid, Grid | None]]:
    return [
        (transform(pair.input), transform(pair.output) if pair.output is not None else None)
        for pair in task.test
    ]


def verify_induction_program(
    source: str,
    task: ArcTask,
    *,
    guards: GuardConfig | None = None,
    include_test_labels: bool,
    run_transformations: bool = True,
    seed: int = 42,
) -> InductionVerification:
    settings = guards or GuardConfig()
    result = InductionVerification()
    try:
        validate_induction_source(
            source,
            forbidden_grids=_task_grids(task),
            forbidden_identifiers=[task.task_id],
            max_source_chars=settings.max_source_chars,
            max_literal_scalars=settings.max_literal_scalars,
        )
    except UnsafeProgram as exc:
        result.failures.append(VerificationFailure(case="static", error=str(exc)))
        return result
    result.static_safe = True

    raw_scores: list[float] = []
    balanced_scores: list[float] = []
    runtime_valid = True

    def check(
        *,
        case: str,
        train: Sequence[tuple[Grid, Grid]],
        input_grid: Grid,
        expected: Grid,
        category: str,
    ) -> None:
        nonlocal runtime_valid
        executed = run_induction_program(
            source,
            train,
            input_grid,
            timeout_seconds=settings.wall_timeout_seconds,
            validate_source=False,
        )
        result.total_cases += 1
        if category == "test":
            result.source_tests_total += 1
        elif category == "loo":
            result.leave_one_out_total += 1
        elif category == "transformed":
            result.transformed_total += 1
        exact = executed.error is None and executed.grid == expected
        if exact:
            result.exact_cases += 1
            raw_scores.append(1.0)
            balanced_scores.append(1.0)
            if category == "test":
                result.source_tests_exact += 1
            elif category == "loo":
                result.leave_one_out_exact += 1
            elif category == "transformed":
                result.transformed_exact += 1
            return
        raw, balanced, detail = _grid_detail(executed.grid, expected)
        raw_scores.append(raw)
        balanced_scores.append(balanced)
        runtime_valid = runtime_valid and executed.error is None
        if len(result.failures) < 64:
            result.failures.append(
                VerificationFailure(
                    case=case,
                    expected=expected,
                    actual=executed.grid,
                    error=executed.error,
                    detail=detail,
                )
            )

    base_train = [(pair.input, pair.output or [[]]) for pair in task.train]
    for index, pair in enumerate(task.train):
        check(
            case=f"demo_full_{index}",
            train=base_train,
            input_grid=pair.input,
            expected=pair.output or [[]],
            category="base",
        )
        if settings.require_leave_one_out:
            reduced = [item for item_index, item in enumerate(base_train) if item_index != index]
            check(
                case=f"demo_loo_{index}",
                train=reduced,
                input_grid=pair.input,
                expected=pair.output or [[]],
                category="loo",
            )

    predictions: list[Grid] = []
    for index, pair in enumerate(task.test):
        executed = run_induction_program(
            source,
            base_train,
            pair.input,
            timeout_seconds=settings.wall_timeout_seconds,
            validate_source=False,
        )
        if executed.error or executed.grid is None:
            runtime_valid = False
            if len(result.failures) < 64:
                result.failures.append(
                    VerificationFailure(case=f"test_execution_{index}", error=executed.error)
                )
            continue
        predictions.append(executed.grid)
        if include_test_labels:
            if pair.output is None:
                raise ValueError("training verification requires labelled test outputs")
            check(
                case=f"source_test_{index}",
                train=base_train,
                input_grid=pair.input,
                expected=pair.output,
                category="test",
            )
    result.predictions = predictions if len(predictions) == len(task.test) else []

    if run_transformations:
        transforms: list[tuple[str, Callable[[Grid], Grid]]] = []
        if settings.d4_transforms:
            transforms.extend(d4_transforms())
        for index, mapping in enumerate(
            color_permutations(task, count=settings.color_permutations, seed=seed)
        ):
            transforms.append(
                (
                    f"color_permutation_{index}",
                    lambda grid, m=mapping: _map_colors(grid, m),
                )
            )

        for transform_name, transform in transforms:
            transformed_train = _pairs(task, transform)
            for index, (input_grid, output_grid) in enumerate(transformed_train):
                reduced = [
                    item for item_index, item in enumerate(transformed_train) if item_index != index
                ]
                check(
                    case=f"{transform_name}_loo_{index}",
                    train=reduced,
                    input_grid=input_grid,
                    expected=output_grid,
                    category="transformed",
                )
            if include_test_labels:
                for index, (input_grid, expected) in enumerate(_tests(task, transform)):
                    if expected is None:
                        raise ValueError("training transformation requires labelled test outputs")
                    check(
                        case=f"{transform_name}_test_{index}",
                        train=transformed_train,
                        input_grid=input_grid,
                        expected=expected,
                        category="transformed",
                    )

    result.mean_cell_accuracy = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
    result.mean_balanced_accuracy = (
        sum(balanced_scores) / len(balanced_scores) if balanced_scores else 0.0
    )
    expected_predictions = len(task.test)
    result.accepted = bool(
        runtime_valid
        and result.static_safe
        and result.exact_cases == result.total_cases
        and len(result.predictions) == expected_predictions
        and (not include_test_labels or result.source_tests_exact == result.source_tests_total)
        and (
            not settings.require_leave_one_out
            or result.leave_one_out_exact == result.leave_one_out_total
        )
    )
    result.score = (
        result.exact_cases * 1_000.0
        + result.mean_balanced_accuracy * 100.0
        + result.mean_cell_accuracy * 10.0
    )
    return result


def verification_feedback(verification: InductionVerification) -> str:
    lines = [
        f"Accepted: {verification.accepted}",
        f"Exact cases: {verification.exact_cases}/{verification.total_cases}",
        (
            "Mean accuracy: "
            f"balanced={verification.mean_balanced_accuracy:.4f}, "
            f"raw={verification.mean_cell_accuracy:.4f}"
        ),
    ]
    for failure in verification.failures:
        lines.append(f"CASE {failure.case}")
        if failure.error:
            lines.append(f"Runtime/static error: {failure.error}")
        if failure.actual is not None:
            lines.append(f"Actual: {failure.actual}")
        if failure.expected is not None:
            lines.append(f"Expected: {failure.expected}")
        if failure.detail:
            lines.append(f"Diff: {failure.detail}")
    return "\n".join(lines)


def failed_induction_guards(
    verification: InductionVerification,
    task: ArcTask,
    *,
    guards: GuardConfig,
) -> list[dict[str, object]]:
    """Return durable, human-readable guard failures for a best-effort program."""
    failures = verification.failures
    result: list[dict[str, object]] = []

    def cases(prefix: str) -> list[str]:
        return sorted(failure.case for failure in failures if failure.case.startswith(prefix))

    static = cases("static")
    if not verification.static_safe or static:
        result.append({"guard": "static_safety", "failed_cases": static or ["static"]})

    parse = cases("response_parse")
    if parse:
        result.append({"guard": "structured_response", "failed_cases": parse})

    demo_full = cases("demo_full_")
    if demo_full:
        result.append(
            {
                "guard": "full_demonstrations_exact",
                "passed": max(0, len(task.train) - len(demo_full)),
                "required": len(task.train),
                "failed_cases": demo_full,
            }
        )

    source = sorted(
        failure.case
        for failure in failures
        if failure.case.startswith(("source_test_", "test_execution_"))
    )
    required_tests = len(task.test)
    if verification.source_tests_exact < required_tests or source:
        result.append(
            {
                "guard": "labelled_source_tests_exact",
                "passed": verification.source_tests_exact,
                "required": required_tests,
                "failed_cases": source,
            }
        )

    loo = cases("demo_loo_")
    required_loo = len(task.train) if guards.require_leave_one_out else 0
    if guards.require_leave_one_out and (
        verification.leave_one_out_exact < required_loo or loo
    ):
        result.append(
            {
                "guard": "leave_one_out_exact",
                "passed": verification.leave_one_out_exact,
                "required": required_loo,
                "failed_cases": loo,
            }
        )

    if guards.d4_transforms:
        for transform, _ in d4_transforms():
            transformed = cases(f"{transform}_")
            if transformed:
                result.append(
                    {
                        "guard": "d4_transform_exact",
                        "transform": transform,
                        "failed_cases": transformed,
                    }
                )

    color_names = sorted(
        {
            failure.case.split("_loo_", 1)[0].split("_test_", 1)[0]
            for failure in failures
            if failure.case.startswith("color_permutation_")
        }
    )
    for transform in color_names:
        result.append(
            {
                "guard": "color_permutation_exact",
                "transform": transform,
                "failed_cases": cases(f"{transform}_"),
            }
        )

    runtime = sorted(
        failure.case
        for failure in failures
        if failure.error and failure.case not in {*static, *parse}
    )
    if runtime:
        result.append({"guard": "runtime_and_grid_validation", "failed_cases": runtime})

    if not verification.accepted and not result:
        result.append(
            {
                "guard": "all_verifier_cases_exact",
                "passed": verification.exact_cases,
                "required": verification.total_cases,
                "failed_cases": sorted(failure.case for failure in failures),
            }
        )
    return result
