from __future__ import annotations

import pytest

from arc_agent.dsl import (
    ProgramError,
    apply_program,
    program_family,
    search_programs,
    verify_program,
)
from arc_agent.models import ArcPair, ArcTask, Program


def test_core_operations():
    grid = [[1, 2], [3, 4]]
    assert apply_program(Program(op="rotate", args={"turns": 1}), grid) == [[3, 1], [4, 2]]
    assert apply_program(Program(op="flip", args={"axis": "horizontal"}), grid) == [
        [2, 1],
        [4, 3],
    ]
    assert apply_program(Program(op="scale", args={"rows": 2, "columns": 2}), [[1]]) == [
        [1, 1],
        [1, 1],
    ]
    assert apply_program(Program(op="tile", args={"rows": 2, "columns": 2}), [[1, 2]]) == [
        [1, 2, 1, 2],
        [1, 2, 1, 2],
    ]


def test_object_relational_operations():
    assert apply_program(
        Program(op="translate", args={"rows": 1, "columns": -1, "background": 0}),
        [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
    ) == [[0, 0, 0], [2, 0, 0], [0, 0, 0]]
    assert apply_program(
        Program(
            op="concat",
            args={
                "axis": "horizontal",
                "transform": "flip_horizontal",
                "order": "input_first",
            },
        ),
        [[1, 2], [3, 4]],
    ) == [[1, 2, 2, 1], [3, 4, 4, 3]]
    assert apply_program(
        Program(op="complete_symmetry", args={"axis": "horizontal", "background": 0}),
        [[2, 0, 0], [0, 0, 0]],
    ) == [[2, 0, 2], [0, 0, 0]]
    assert apply_program(
        Program(
            op="split_overlay",
            args={
                "axis": "vertical",
                "mode": "xor",
                "background": 0,
                "separator_color": 5,
                "output_color": 2,
            },
        ),
        [[1, 0, 5, 0, 3], [0, 1, 5, 0, 0]],
    ) == [[2, 2], [0, 2]]


def test_multicolor_eight_connected_component():
    program = Program(
        op="select_component",
        args={
            "background": 0,
            "criterion": "smallest",
            "connectivity": 8,
            "multicolor": True,
        },
    )
    grid = [[1, 0, 0, 0], [0, 2, 0, 3], [0, 0, 0, 3]]
    assert apply_program(program, grid) == [[1, 0], [0, 2]]


def test_move_color_to_contact_uses_anchor_distance():
    program = Program(
        op="move_color_to_contact",
        args={"moving_color": 2, "target_color": 8, "background": 0},
    )
    assert apply_program(
        program,
        [[0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 8, 0], [0, 0, 0, 0, 8, 0]],
    ) == [[0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 8, 0], [0, 0, 0, 0, 8, 0]]
    assert apply_program(
        program,
        [[0, 2, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0], [0, 8, 0]],
    ) == [[0, 0, 0], [0, 0, 0], [0, 2, 0], [0, 2, 0], [0, 8, 0]]


def test_pattern_mosaic_and_search():
    program = Program(
        op="pattern_mosaic",
        args={
            "background": 7,
            "empty_color": 0,
            "base_color": 7,
            "marker_color": 9,
            "row_order": [1, 0, 2],
            "column_order": [1, 0, 2],
            "output_rows": 16,
            "output_columns": 16,
        },
    )
    first = [
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 3, 7],
        [7, 7, 7, 3, 3, 3],
        [7, 7, 7, 7, 3, 7],
    ]
    second = [
        [7, 7, 7, 7, 7, 7],
        [7, 1, 7, 1, 7, 7],
        [7, 1, 1, 1, 7, 7],
        [7, 1, 7, 1, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
    ]
    expected_first = apply_program(program, first)
    expected_second = apply_program(program, second)
    assert expected_first[5][6] == 9
    assert expected_first[1][1] == 7
    task = ArcTask(
        task_id="pattern-mosaic",
        train=[
            ArcPair(input=first, output=expected_first),
            ArcPair(input=second, output=expected_second),
        ],
        test=[ArcPair(input=second)],
    )
    candidates = search_programs(task)
    assert candidates
    assert candidates[0].program is not None
    assert candidates[0].program.op == "pattern_mosaic"
    assert candidates[0].predictions == [expected_second]


def test_repair_periodic_panels_and_search():
    corrupted = [
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
        [3, 2, 1, 3, 1, 3, 1, 3, 3, 3, 1, 2, 3],
        [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ]
    expected = [row[:] for row in corrupted]
    expected[2][8] = 1
    program = Program(op="repair_periodic_panels")
    assert apply_program(program, corrupted) == expected
    task = ArcTask(
        task_id="periodic-repair",
        train=[ArcPair(input=corrupted, output=expected)],
        test=[ArcPair(input=corrupted)],
    )
    candidates = search_programs(task)
    assert candidates
    assert candidates[0].program is not None
    assert candidates[0].program.op == "repair_periodic_panels"


def test_select_unique_panel_per_band():
    grid = [[0 for _ in range(10)] for _ in range(7)]
    for row in (1, 2):
        for left, color in ((1, 1), (4, 1), (7, 2)):
            grid[row][left : left + 2] = [color, color]
    for row in (4, 5):
        for left, color in ((1, 3), (4, 4), (7, 4)):
            grid[row][left : left + 2] = [color, color]
    expected = [[0 for _ in range(4)] for _ in range(7)]
    for row in (1, 2):
        expected[row][1:3] = [2, 2]
    for row in (4, 5):
        expected[row][1:3] = [3, 3]
    assert apply_program(Program(op="select_unique_panel_per_band"), grid) == expected


def test_deduplicate_repeated_half_and_search():
    horizontal = [[4, 4, 4, 4], [6, 8, 6, 8]]
    vertical = [[2, 3], [4, 4], [2, 3], [4, 4]]
    program = Program(op="deduplicate_repeated_half")
    assert apply_program(program, horizontal) == [[4, 4], [6, 8]]
    assert apply_program(program, vertical) == [[2, 3], [4, 4]]
    task = ArcTask(
        task_id="repeated-half",
        train=[
            ArcPair(input=horizontal, output=[[4, 4], [6, 8]]),
            ArcPair(input=vertical, output=[[2, 3], [4, 4]]),
        ],
        test=[ArcPair(input=[[1, 2, 1, 2]])],
    )
    candidate = search_programs(task)[0]
    assert candidate.program is not None
    assert candidate.program.op == "deduplicate_repeated_half"
    assert candidate.predictions == [[[1, 2]]]


def test_fold_quadrants_overlap():
    grid = [
        [2, 2, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0],
        [2, 0, 0, 0, 0, 0, 2],
    ]
    assert apply_program(Program(op="fold_quadrants_overlap"), grid) == [
        [2, 2, 2],
        [0, 2, 2],
        [2, 0, 2],
    ]


def test_search_mirrors_grid_across_both_axes():
    source = [[1, 2], [3, 4]]
    expected = [
        [1, 2, 2, 1],
        [3, 4, 4, 3],
        [3, 4, 4, 3],
        [1, 2, 2, 1],
    ]
    task = ArcTask(
        task_id="double-mirror",
        train=[ArcPair(input=source, output=expected)],
        test=[ArcPair(input=source)],
    )
    candidates = search_programs(task)
    assert candidates
    assert candidates[0].predictions == [expected]


def test_frequency_marker_and_rare_color_replacement():
    marker_grid = [
        [3, 6, 4, 2, 4],
        [8, 4, 3, 3, 4],
        [5, 5, 5, 5, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    marked = apply_program(Program(op="mark_most_frequent_above_separator"), marker_grid)
    assert marked[-1] == [0, 0, 4, 0, 0]
    recolored = apply_program(
        Program(op="keep_most_frequent_colors", args={"keep_count": 2, "output_color": 7}),
        [[1, 2, 5], [1, 3, 3], [1, 3, 5]],
    )
    assert recolored == [[1, 7, 7], [1, 3, 3], [1, 3, 7]]


def test_split_overlay_supports_even_panels_and_neither():
    grid = [
        [0, 0, 9],
        [9, 9, 9],
        [0, 9, 0],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]
    program = Program(
        op="split_overlay",
        args={
            "axis": "horizontal",
            "mode": "neither",
            "background": 0,
            "separator_color": 0,
            "output_color": 2,
        },
    )
    assert apply_program(program, grid) == [[0, 2, 0], [0, 0, 0], [2, 0, 0]]


def test_pack_points_fill_between_and_expand_seed():
    scattered = [
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
        [0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
        [9, 3, 0, 0, 0, 0, 0, 0, 5, 0],
    ]
    assert apply_program(Program(op="pack_points_serpentine"), scattered) == [
        [9, 3, 4],
        [5, 8, 6],
        [2, 0, 0],
    ]
    assert apply_program(
        Program(op="fill_between_row_markers", args={"fill_color": 2}),
        [[0, 8, 0, 0, 8, 0], [0, 0, 8, 0, 0, 0]],
    ) == [[0, 8, 2, 2, 8, 0], [0, 0, 8, 0, 0, 0]]
    assert apply_program(
        Program(op="expand_seed_to_width"),
        [[3, 2, 3, 0, 0, 0, 0, 0]],
    ) == [[3, 2, 3, 3, 3, 3, 2, 3]]


def test_count_frequency_and_object_layout_primitives():
    assert apply_program(
        Program(op="summarize_nonbackground_count"),
        [[0, 4, 0], [4, 0, 4]],
    ) == [[4, 4, 4]]
    assert apply_program(
        Program(op="fill_with_most_frequent_color"),
        [[8, 8, 6], [4, 8, 9]],
    ) == [[8, 8, 8], [8, 8, 8]]
    assert apply_program(
        Program(op="frequency_histogram_columns"),
        [[1, 2, 3], [1, 2, 1]],
    ) == [[1, 2, 3], [1, 2, 0], [1, 0, 0]]
    assert apply_program(
        Program(op="recolor_singleton_components"),
        [[2, 2, 0], [0, 0, 2]],
    ) == [[2, 2, 0], [0, 0, 1]]
    assert apply_program(
        Program(op="move_center_block_to_corners"),
        [[0, 0, 0, 0], [0, 2, 3, 0], [0, 4, 9, 0], [0, 0, 0, 0]],
    ) == [[2, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 9]]


def test_periodic_symmetry_and_rectangle_completion_primitives():
    periodic = [
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4],
    ]
    periodic_output = [
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4],
    ]
    assert (
        apply_program(
            Program(op="complete_periodic_pattern", args={"mode": "tile"}),
            periodic,
        )
        == periodic_output
    )

    partial_symmetry = [
        [1, 0, 2, 0, 1],
        [0, 3, 0, 3, 0],
        [2, 0, 4, 0, 2],
        [0, 3, 0, 3, 0],
        [0, 0, 2, 0, 0],
    ]
    symmetric_output = [
        [1, 0, 2, 0, 1],
        [0, 3, 0, 3, 0],
        [2, 0, 4, 0, 2],
        [0, 3, 0, 3, 0],
        [1, 0, 2, 0, 1],
    ]
    assert (
        apply_program(Program(op="complete_bbox_symmetry"), partial_symmetry)
        == symmetric_output
    )

    rectangle = [[7 for _ in range(6)] for _ in range(6)]
    for column in range(1, 5):
        rectangle[1][column] = rectangle[4][column] = 6
    for row in range(1, 5):
        rectangle[row][1] = rectangle[row][4] = 6
    marked = apply_program(
        Program(op="mark_closed_square_corners", args={"output_color": 2}),
        rectangle,
    )
    assert {
        (row, column)
        for row, values in enumerate(marked)
        for column, value in enumerate(values)
        if value == 2
    } == {(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4)}

    for task_id, source, expected, operation in (
        ("periodic", periodic, periodic_output, "complete_periodic_pattern"),
        ("symmetric", partial_symmetry, symmetric_output, "complete_bbox_symmetry"),
        ("rectangle", rectangle, marked, "mark_closed_square_corners"),
    ):
        task = ArcTask(
            task_id=task_id,
            train=[ArcPair(input=source, output=expected)],
            test=[ArcPair(input=source)],
        )
        candidates = search_programs(task)
        assert any(
            candidate.program and candidate.program.op == operation
            for candidate in candidates
        )


def test_search_identity(identity_task):
    candidates = search_programs(identity_task)
    assert candidates
    assert candidates[0].verified
    assert candidates[0].predictions == [[[2, 0], [0, 2]]]


def test_search_rotation_and_recolor():
    task = ArcTask(
        task_id="rotate-recolor",
        train=[
            ArcPair(input=[[1, 0], [0, 0]], output=[[0, 2], [0, 0]]),
            ArcPair(input=[[0, 1], [0, 0]], output=[[0, 0], [0, 2]]),
        ],
        test=[ArcPair(input=[[0, 0], [1, 0]])],
    )
    candidates = search_programs(task)
    assert candidates
    assert candidates[0].predictions == [[[2, 0], [0, 0]]]
    assert program_family(candidates[0].program) in {"color-mapping", "spatial-transform"}


def test_verifier_returns_structured_diff(identity_task):
    result = verify_program(Program(op="rotate", args={"turns": 1}), identity_task)
    assert not result.perfect
    assert "pair 0" in result.errors[0]


def test_program_rejects_unused_arguments():
    with pytest.raises(ProgramError, match="unexpected args"):
        apply_program(Program(op="crop_background", args={"mapping": {"2": 5}}), [[0, 2]])


def test_search_ignores_oversized_generated_outputs():
    task = ArcTask(
        task_id="large-grid",
        train=[ArcPair(input=[[1] * 20], output=[[1] * 20])],
        test=[ArcPair(input=[[1] * 20])],
    )
    assert search_programs(task)
