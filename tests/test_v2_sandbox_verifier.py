from __future__ import annotations

from arc_agent.models import ArcPair, ArcTask
from arc_agent.v2_config import GuardConfig
from arc_agent.v2_prompt import build_synthesis_prompt, sanitize_task
from arc_agent.v2_sandbox import run_induction_program
from arc_agent.v2_verifier import verify_induction_program


def test_induction_sandbox_runs_two_argument_program() -> None:
    result = run_induction_program(
        "def solve(train, grid):\n    return [row[:] for row in grid]",
        [([[1]], [[1]])],
        [[2, 0], [0, 2]],
    )
    assert result.error is None
    assert result.grid == [[2, 0], [0, 2]]


def test_induction_sandbox_rejects_embedded_source_grid() -> None:
    source_grid = [[1, 2], [3, 4]]
    result = run_induction_program(
        "def solve(train, grid):\n    return [[1,2],[3,4]]",
        [([[0]], [[0]])],
        [[0]],
        forbidden_grids=[source_grid],
    )
    assert "embedded" in (result.error or "")


def test_induction_sandbox_rejects_literal_grid_branch() -> None:
    result = run_induction_program(
        "def solve(train, grid):\n    if grid == [[1]]:\n        return [[2]]\n    return grid",
        [([[1]], [[2]])],
        [[1]],
    )
    assert "compared" in (result.error or "")


def test_identity_program_passes_full_training_guard(identity_task: ArcTask) -> None:
    verification = verify_induction_program(
        "def solve(train, grid):\n    return [row[:] for row in grid]",
        identity_task,
        guards=GuardConfig(d4_transforms=True, color_permutations=2),
        include_test_labels=True,
    )
    assert verification.accepted
    assert verification.source_tests_exact == verification.source_tests_total == 1
    assert verification.leave_one_out_exact == verification.leave_one_out_total == 1
    assert verification.transformed_exact == verification.transformed_total


def test_lookup_memorizer_fails_leave_one_out() -> None:
    task = ArcTask(
        task_id="recolor",
        train=[
            ArcPair(input=[[1, 0]], output=[[2, 0]]),
            ArcPair(input=[[0, 1]], output=[[0, 2]]),
        ],
        test=[ArcPair(input=[[1, 1]], output=[[2, 2]])],
    )
    verification = verify_induction_program(
        "def solve(train, grid):\n"
        "    for inp, out in train:\n"
        "        if inp == grid:\n"
        "            return out\n"
        "    return grid",
        task,
        guards=GuardConfig(d4_transforms=False, color_permutations=0),
        include_test_labels=True,
    )
    assert not verification.accepted
    assert verification.leave_one_out_exact == 0


def test_evaluation_prompt_never_contains_test_output() -> None:
    task = ArcTask(
        task_id="blind",
        train=[ArcPair(input=[[1]], output=[[2]])],
        test=[ArcPair(input=[[3, 3]], output=[[9, 9]])],
    )
    prompt = build_synthesis_prompt(
        sanitize_task(task),
        phase="evaluation",
        search_lens="test lens",
        feedback=None,
        retrieved_programs=[],
        retrieved_limit=0,
    )
    assert "[[9,9]]" not in prompt.replace(" ", "")
    assert '"output":null' not in prompt
    assert '"test":[{"input":[[3,3]]}]' in prompt
