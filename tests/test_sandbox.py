from __future__ import annotations

from arc_agent.sandbox import run_python_program


def test_safe_program_runs():
    result = run_python_program(
        "def solve(grid):\n    return [list(reversed(row)) for row in grid]\n",
        [[1, 2], [3, 4]],
    )
    assert result.error is None
    assert result.grid == [[2, 1], [4, 3]]


def test_safe_helpers_and_dict_methods_run():
    result = run_python_program(
        "def most_common(values):\n"
        "    counts = {}\n"
        "    for value in values:\n"
        "        counts[value] = counts.get(value, 0) + 1\n"
        "    return max(counts, key=counts.get)\n"
        "def solve(grid):\n"
        "    color = most_common([value for row in grid for value in row])\n"
        "    return [[color for value in row] for row in grid]\n",
        [[1, 0], [0, 0]],
    )
    assert result.error is None
    assert result.grid == [[0, 0], [0, 0]]


def test_import_and_dunder_are_rejected():
    imported = run_python_program("import os\ndef solve(grid):\n    return grid\n", [[1]])
    assert "disallowed" in (imported.error or "")
    dunder = run_python_program("def solve(grid):\n    return (1).__class__\n", [[1]])
    assert "allowed" in (dunder.error or "")


def test_infinite_loop_times_out():
    result = run_python_program(
        "def solve(grid):\n    while True:\n        pass\n",
        [[1]],
        timeout_seconds=0.2,
    )
    assert "timed out" in (result.error or "")
