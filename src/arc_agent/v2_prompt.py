from __future__ import annotations

import json
from collections import Counter

from arc_agent.dsl import connected_components
from arc_agent.features import task_features
from arc_agent.models import ArcPair, ArcTask, Grid
from arc_agent.v2_models import BankProgram

SEARCH_LENSES = (
    "Object-centric: infer objects, roles, containment, alignment, motion, and anchors.",
    "Relational: infer predicates between objects and conditional transformations.",
    "Geometric: test symmetry, rotations, reflections, scaling, tiling, rays, and paths.",
    "Symbolic: treat colours or shapes as instructions and infer their task-local meaning.",
    "Counting: infer frequencies, ordering, repetition counts, histograms, and packing.",
    "Compression: seek the shortest rule using panels, periodicity, masks, and overlays.",
)


def sanitize_task(task: ArcTask) -> ArcTask:
    return task.model_copy(
        update={
            "test": [ArcPair(input=pair.input, output=None) for pair in task.test],
        }
    )


def task_payload(task: ArcTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "train": [{"input": pair.input, "output": pair.output} for pair in task.train],
        "test": [{"input": pair.input} for pair in task.test],
    }


def _symmetry(grid: Grid) -> tuple[float, float, float]:
    horizontal = float(all(row == list(reversed(row)) for row in grid))
    vertical = float(grid == list(reversed(grid)))
    rotational = float(grid == [list(reversed(row)) for row in reversed(grid)])
    return horizontal, vertical, rotational


def episodic_features(task: ArcTask) -> dict[str, float]:
    features = task_features(task)
    inputs = [pair.input for pair in task.train]
    outputs = [pair.output for pair in task.train if pair.output is not None]
    backgrounds = [
        Counter(cell for row in grid for cell in row).most_common(1)[0][0] for grid in inputs
    ]
    components_8 = [
        len(connected_components(grid, background=background, connectivity=8))
        for grid, background in zip(inputs, backgrounds, strict=True)
    ]
    symmetries = [_symmetry(grid) for grid in inputs]
    input_sizes = [len(grid) * len(grid[0]) for grid in inputs]
    output_sizes = [len(grid) * len(grid[0]) for grid in outputs]
    separator_counts = [
        sum(len(set(row)) == 1 for row in grid)
        + sum(
            len({grid[row][column] for row in range(len(grid))}) == 1
            for column in range(len(grid[0]))
        )
        for grid in inputs
    ]
    features.update(
        {
            "mean_components_8": sum(components_8) / len(components_8),
            "horizontal_symmetry_ratio": sum(item[0] for item in symmetries) / len(symmetries),
            "vertical_symmetry_ratio": sum(item[1] for item in symmetries) / len(symmetries),
            "rotational_symmetry_ratio": sum(item[2] for item in symmetries) / len(symmetries),
            "mean_separator_lines": sum(separator_counts) / len(separator_counts),
            "mean_output_cells": sum(output_sizes) / len(output_sizes),
            "mean_area_ratio": sum(
                output / max(1, input_size)
                for output, input_size in zip(output_sizes, input_sizes, strict=True)
            )
            / len(input_sizes),
            "output_uses_new_color_ratio": sum(
                bool(
                    {cell for row in output for cell in row}
                    - {cell for row in input_grid for cell in row}
                )
                for input_grid, output in zip(inputs, outputs, strict=True)
            )
            / len(inputs),
        }
    )
    return features


def program_context(programs: list[BankProgram], *, limit: int) -> str:
    if not programs or limit <= 0:
        return "No earlier verified induction programs are available."
    sections: list[str] = []
    for index, program in enumerate(programs[:limit], start=1):
        sections.append(
            "\n".join(
                [
                    f"### Retrieved program {index}",
                    f"Hypothesis: {program.hypothesis}",
                    f"Tags: {', '.join(program.strategy_tags)}",
                    f"Invariants: {'; '.join(program.invariants)}",
                    "```python",
                    program.python_source,
                    "```",
                ]
            )
        )
    return "\n\n".join(sections)


def build_synthesis_prompt(
    task: ArcTask,
    *,
    phase: str,
    search_lens: str,
    feedback: str | None,
    retrieved_programs: list[BankProgram],
    retrieved_limit: int,
) -> str:
    if any(pair.output is not None for pair in task.test):
        task = sanitize_task(task)
    oracle_policy = (
        "This is a labelled training-set synthesis episode. Verifier feedback may reveal the "
        "held-out expected output, but embedding or recognizing a specific source grid is invalid."
        if phase == "training"
        else (
            "This is a frozen evaluation episode. Test outputs are unavailable; "
            "use demonstrations only."
        )
    )
    feedback_text = feedback or "No prior candidate exists. Derive a fresh general rule."
    return f"""Synthesize an ARC-AGI-2 induction program.

CONTRACT
- Define exactly: def solve(train, grid):
- train is a list of (input_grid, output_grid) pairs from the current task.
- Infer colours, dimensions, directions, counts, and object roles from train.
- Return one integer grid for grid.
- The program is tested on leave-one-demonstration-out episodes and transformed analogues.
- Do not memorize the displayed task or branch on literal grids.

POLICY
{oracle_policy}

SEARCH LENS
{search_lens}

TASK
{json.dumps(task_payload(task), separators=(",", ":"))}

RETRIEVED TRAINING PROGRAMS
Use these as transferable ideas only. Do not copy source-specific constants blindly.
{program_context(retrieved_programs, limit=retrieved_limit)}

VERIFIER FEEDBACK
{feedback_text}

Return one complete corrected program in the required structured schema.
"""
