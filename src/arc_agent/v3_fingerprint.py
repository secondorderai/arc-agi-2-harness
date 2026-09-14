from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from typing import Any

from arc_agent.models import ArcTask, Grid
from arc_agent.object_ir import GridObject, extract_objects, infer_background
from arc_agent.v3_models import SignatureFamily, TaskFingerprint

FAMILIES: tuple[SignatureFamily, ...] = (
    "color-attribute",
    "rigid-geometry",
    "object-relational",
    "counting-compression",
    "repetition-pattern",
)


def blind_dataset_sha256(tasks: list[ArcTask]) -> str:
    payload = [
        {
            "task_id": task.task_id,
            "train": [
                {"input": pair.input, "output": pair.output} for pair in task.train
            ],
            "test_inputs": [pair.input for pair in task.test],
        }
        for task in sorted(tasks, key=lambda item: item.task_id)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rotate(grid: Grid) -> Grid:
    return [list(row) for row in zip(*reversed(grid), strict=True)]


def _flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _d4(grid: Grid) -> list[Grid]:
    rotations = [grid]
    for _ in range(3):
        rotations.append(_rotate(rotations[-1]))
    return [*rotations, *[_flip_horizontal(item) for item in rotations]]


def _symmetry(grid: Grid) -> tuple[float, float, float]:
    horizontal = float(grid == _flip_horizontal(grid))
    vertical = float(grid == list(reversed(grid)))
    rotational = float(grid == _rotate(_rotate(grid)))
    return horizontal, vertical, rotational


def _separator_lines(grid: Grid) -> int:
    rows = sum(len(set(row)) == 1 for row in grid)
    columns = sum(
        len({grid[row][column] for row in range(len(grid))}) == 1
        for column in range(len(grid[0]))
    )
    return rows + columns


def _color_map_possible(source: Grid, target: Grid) -> bool:
    if len(source) != len(target) or len(source[0]) != len(target[0]):
        return False
    mapping: dict[int, int] = {}
    for source_row, target_row in zip(source, target, strict=True):
        for left, right in zip(source_row, target_row, strict=True):
            if left in mapping and mapping[left] != right:
                return False
            mapping[left] = right
    return True


def _periodic_score(grid: Grid) -> float:
    height, width = len(grid), len(grid[0])
    best = 0.0
    for row_period in range(1, min(6, height) + 1):
        for column_period in range(1, min(6, width) + 1):
            if row_period == height and column_period == width:
                continue
            matches = sum(
                grid[row][column] == grid[row % row_period][column % column_period]
                for row in range(height)
                for column in range(width)
            )
            best = max(best, matches / (height * width))
    return best


def _relational_cues(objects: tuple[GridObject, ...]) -> int:
    cues = 0
    for left, right in combinations(objects, 2):
        same_row = left.bbox.center[0] == right.bbox.center[0]
        same_column = left.bbox.center[1] == right.bbox.center[1]
        row_gap = max(
            0,
            left.bbox.top - right.bbox.bottom - 1,
            right.bbox.top - left.bbox.bottom - 1,
        )
        column_gap = max(
            0,
            left.bbox.left - right.bbox.right - 1,
            right.bbox.left - left.bbox.right - 1,
        )
        cues += int(same_row or same_column or row_gap + column_gap <= 1)
    return cues


def fingerprint_task(task: ArcTask, *, include_test_inputs: bool = True) -> TaskFingerprint:
    pairs = task.train
    pair_count = len(pairs)
    input_cells: list[int] = []
    output_cells: list[int] = []
    input_colors: list[int] = []
    output_colors: list[int] = []
    input_components: list[int] = []
    output_components: list[int] = []
    shape_same = 0
    color_map = 0
    rigid = 0
    output_only = 0
    foreground_ratio: list[float] = []
    centroid_shift: list[float] = []
    row_scales: list[float] = []
    column_scales: list[float] = []
    symmetry_values: list[tuple[float, float, float]] = []
    separators: list[int] = []
    periodicity: list[float] = []
    input_relations: list[int] = []
    output_relations: list[int] = []

    for pair in pairs:
        output = pair.output or [[]]
        ih, iw = len(pair.input), len(pair.input[0])
        oh, ow = len(output), len(output[0])
        input_cells.append(ih * iw)
        output_cells.append(oh * ow)
        input_palette = {cell for row in pair.input for cell in row}
        output_palette = {cell for row in output for cell in row}
        input_colors.append(len(input_palette))
        output_colors.append(len(output_palette))
        input_background = infer_background(pair.input)
        output_background = infer_background(output)
        in_objects = extract_objects(pair.input, background=input_background)
        out_objects = extract_objects(output, background=output_background)
        input_components.append(len(in_objects))
        output_components.append(len(out_objects))
        shape_same += int((ih, iw) == (oh, ow))
        color_map += int(_color_map_possible(pair.input, output))
        rigid += int(any(candidate == output for candidate in _d4(pair.input)))
        output_only += int(bool(output_palette - input_palette))
        input_foreground = sum(cell != input_background for row in pair.input for cell in row)
        output_foreground = sum(cell != output_background for row in output for cell in row)
        foreground_ratio.append(output_foreground / max(1, input_foreground))
        if in_objects and out_objects:
            in_center = (
                sum(obj.center[0] for obj in in_objects) / len(in_objects),
                sum(obj.center[1] for obj in in_objects) / len(in_objects),
            )
            out_center = (
                sum(obj.center[0] for obj in out_objects) / len(out_objects),
                sum(obj.center[1] for obj in out_objects) / len(out_objects),
            )
            centroid_shift.append(
                abs(out_center[0] - in_center[0]) + abs(out_center[1] - in_center[1])
            )
        else:
            centroid_shift.append(0.0)
        row_scales.append(oh / ih)
        column_scales.append(ow / iw)
        symmetry_values.append(_symmetry(output))
        separators.append(_separator_lines(pair.input))
        periodicity.append(_periodic_score(output))
        input_relations.append(_relational_cues(in_objects))
        output_relations.append(_relational_cues(out_objects))

    test_pairs = task.test if include_test_inputs else []
    test_input_cells = [len(pair.input) * len(pair.input[0]) for pair in test_pairs]
    test_input_colors = [
        len({cell for row in pair.input for cell in row}) for pair in test_pairs
    ]
    test_input_components = [
        len(extract_objects(pair.input, background=infer_background(pair.input)))
        for pair in test_pairs
    ]
    test_input_symmetry = [max(_symmetry(pair.input)) for pair in test_pairs]
    test_input_periodicity = [_periodic_score(pair.input) for pair in test_pairs]

    def mean(values: list[float] | list[int]) -> float:
        return sum(values) / max(1, len(values))
    shape_ratio = shape_same / pair_count
    color_ratio = color_map / pair_count
    rigid_ratio = rigid / pair_count
    dimension_change = 1.0 - shape_ratio
    output_only_ratio = output_only / pair_count
    mean_symmetry = mean([max(value) for value in symmetry_values])
    mean_periodicity = mean(periodicity)
    mean_component_delta = mean(
        [abs(left - right) for left, right in zip(input_components, output_components, strict=True)]
    )
    mean_output_area = mean(output_cells)
    mean_input_area = mean(input_cells)
    compression = max(0.0, 1.0 - mean_output_area / max(1.0, mean_input_area))
    expansion = max(0.0, mean_output_area / max(1.0, mean_input_area) - 1.0)

    family_scores: dict[str, float] = {
        "color-attribute": 2.5 * color_ratio + shape_ratio + output_only_ratio,
        "rigid-geometry": 3.0 * rigid_ratio + shape_ratio + mean_symmetry,
        "object-relational": (
            shape_ratio
            + min(2.0, mean(centroid_shift) / 2.0)
            + min(2.0, mean_component_delta / 2.0)
        ),
        "counting-compression": (
            2.0 * dimension_change + min(2.0, compression * 2.0) + output_only_ratio
        ),
        "repetition-pattern": (
            min(2.0, expansion)
            + mean_periodicity
            + mean_symmetry
            + min(1.0, mean(separators) / 4.0)
        ),
    }
    predicted = max(FAMILIES, key=lambda family: (family_scores[family], -FAMILIES.index(family)))
    features = {
        "train_pairs": float(pair_count),
        "test_pairs": float(len(test_pairs)),
        "mean_input_cells": mean(input_cells),
        "mean_output_cells": mean(output_cells),
        "mean_input_colors": mean(input_colors),
        "mean_output_colors": mean(output_colors),
        "mean_input_components": mean(input_components),
        "mean_output_components": mean(output_components),
        "mean_input_relations": mean(input_relations),
        "mean_output_relations": mean(output_relations),
        "dimension_change_ratio": dimension_change,
        "color_map_ratio": color_ratio,
        "rigid_match_ratio": rigid_ratio,
        "output_only_color_ratio": output_only_ratio,
        "mean_foreground_ratio": mean(foreground_ratio),
        "mean_centroid_shift": mean(centroid_shift),
        "mean_row_scale": mean(row_scales),
        "mean_column_scale": mean(column_scales),
        "mean_output_symmetry": mean_symmetry,
        "mean_separator_lines": mean(separators),
        "mean_output_periodicity": mean_periodicity,
        "mean_test_input_cells": mean(test_input_cells),
        "mean_test_input_colors": mean(test_input_colors),
        "mean_test_input_components": mean(test_input_components),
        "mean_test_input_symmetry": mean(test_input_symmetry),
        "mean_test_input_periodicity": mean(test_input_periodicity),
    }
    return TaskFingerprint(
        task_id=task.task_id,
        predicted_family=predicted,
        family_scores=family_scores,
        features=features,
    )


def fingerprint_distance(left: TaskFingerprint, right: TaskFingerprint) -> float:
    keys = sorted(set(left.features) | set(right.features))
    return math.sqrt(
        sum(
            (
                (left.features.get(key, 0.0) - right.features.get(key, 0.0))
                / max(1.0, abs(left.features.get(key, 0.0)), abs(right.features.get(key, 0.0)))
            )
            ** 2
            for key in keys
        )
        / max(1, len(keys))
    )


def _meaningfully_different(left: TaskFingerprint, right: TaskFingerprint) -> bool:
    keys = (
        "mean_input_cells",
        "mean_input_colors",
        "mean_input_components",
        "mean_row_scale",
        "mean_column_scale",
    )
    return any(abs(left.features[key] - right.features[key]) > 1e-9 for key in keys)


def select_pilot_tasks(tasks: list[ArcTask]) -> tuple[list[ArcTask], dict[str, Any]]:
    """Select five reproducible, related-but-not-identical family pairs."""
    if len(tasks) < 10:
        raise ValueError("V3 pilot selection requires at least ten training tasks")
    ordered = sorted(tasks, key=lambda task: task.task_id)
    fingerprints = {
        task.task_id: fingerprint_task(task, include_test_inputs=False) for task in ordered
    }
    selected_ids: set[str] = set()
    selections: list[dict[str, Any]] = []

    for family in FAMILIES:
        ranked = sorted(
            (task for task in ordered if task.task_id not in selected_ids),
            key=lambda task: (-fingerprints[task.task_id].family_scores[family], task.task_id),
        )[:120]
        possible = [
            (
                fingerprint_distance(fingerprints[left.task_id], fingerprints[right.task_id]),
                left,
                right,
            )
            for left, right in combinations(ranked, 2)
            if _meaningfully_different(fingerprints[left.task_id], fingerprints[right.task_id])
        ]
        if not possible:
            raise ValueError(f"could not select a distinct task pair for family {family}")
        distance, left, right = min(
            possible,
            key=lambda item: (
                item[0]
                - 0.05
                * (
                    fingerprints[item[1].task_id].family_scores[family]
                    + fingerprints[item[2].task_id].family_scores[family]
                ),
                item[1].task_id,
                item[2].task_id,
            ),
        )
        selected_ids.update((left.task_id, right.task_id))
        selections.append(
            {
                "family": family,
                "task_ids": [left.task_id, right.task_id],
                "fingerprint_distance": distance,
                "family_scores": [
                    fingerprints[left.task_id].family_scores[family],
                    fingerprints[right.task_id].family_scores[family],
                ],
                "rationale": (
                    "closest high-confidence pair with different size, palette, or objects"
                ),
            }
        )

    by_id = {task.task_id: task for task in ordered}
    selected = [by_id[task_id] for row in selections for task_id in row["task_ids"]]
    family_counts = Counter(row["family"] for row in selections)
    if len(selected) != 10 or any(count != 1 for count in family_counts.values()):
        raise AssertionError("pilot selection must contain exactly one pair per family")
    return selected, {
        "schema_version": 1,
        "seed": 42,
        "selection_uses": "training demonstrations only",
        "pairs": selections,
        "task_ids": [task.task_id for task in selected],
        "fingerprints": {
            task.task_id: fingerprints[task.task_id].model_dump(mode="json") for task in selected
        },
    }
