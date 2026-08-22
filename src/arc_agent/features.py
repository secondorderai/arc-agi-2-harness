from __future__ import annotations

from collections import Counter

from arc_agent.dsl import connected_components, search_programs
from arc_agent.models import ArcTask


def task_features(task: ArcTask) -> dict[str, float]:
    inputs = [pair.input for pair in task.train]
    outputs = [pair.output for pair in task.train if pair.output is not None]
    input_colors = [len({cell for row in grid for cell in row}) for grid in inputs]
    output_colors = [len({cell for row in grid for cell in row}) for grid in outputs]
    backgrounds = [
        Counter(cell for row in grid for cell in row).most_common(1)[0][0] for grid in inputs
    ]
    component_counts = [
        len(connected_components(grid, background=background))
        for grid, background in zip(inputs, backgrounds, strict=True)
    ]
    dimension_changes = sum(
        len(pair.input) != len(pair.output or [])
        or len(pair.input[0]) != len((pair.output or [[]])[0])
        for pair in task.train
    )
    return {
        "train_pairs": float(len(task.train)),
        "test_pairs": float(len(task.test)),
        "mean_input_cells": sum(len(grid) * len(grid[0]) for grid in inputs) / len(inputs),
        "mean_input_colors": sum(input_colors) / len(input_colors),
        "mean_output_colors": sum(output_colors) / len(output_colors),
        "mean_components": sum(component_counts) / len(component_counts),
        "dimension_change_ratio": dimension_changes / len(task.train),
    }


def route_level(task: ArcTask, *, deterministic_found: bool | None = None) -> int:
    found = (
        bool(search_programs(task, limit=1)) if deterministic_found is None else deterministic_found
    )
    if found:
        return 1
    features = task_features(task)
    complexity = (
        features["mean_input_colors"]
        + features["mean_components"] * 0.75
        + features["dimension_change_ratio"] * 3
        + max(0.0, features["train_pairs"] - 3) * 0.5
    )
    return 2 if complexity <= 8.0 else 3
