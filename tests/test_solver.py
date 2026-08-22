from __future__ import annotations

from arc_agent.config import SolverConfig
from arc_agent.llm import (
    GeneratedCandidate,
    Generation,
    StaticAdapter,
    build_prompt,
    parse_candidates,
    render_task_data_url,
    structural_summary,
)
from arc_agent.models import Program, grid_accuracies
from arc_agent.solver import ArcSolver


class FailingAdapter:
    def generate(self, *args, **kwargs):
        del args, kwargs
        raise ConnectionError("server unavailable")


def test_hybrid_configuration_is_opt_in():
    default = SolverConfig()
    assert not default.hybrid.enabled
    configured = SolverConfig.model_validate(
        {
            "hybrid": {
                "enabled": True,
                "max_program_depth": 4,
                "max_search_nodes": 1234,
            }
        }
    )
    assert configured.hybrid.enabled
    assert configured.hybrid.max_program_depth == 4
    assert configured.hybrid.max_search_nodes == 1234


def test_hybrid_symbolic_candidates_are_integrated(identity_task):
    config = SolverConfig.model_validate(
        {
            "model": {"enabled": False},
            "hybrid": {
                "enabled": True,
                "symbolic_enabled": True,
                "neural_enabled": False,
                "max_program_depth": 1,
                "max_search_nodes": 200,
                "search_time_seconds": 1,
            },
        }
    )
    run = ArcSolver(config).solve_task(identity_task)
    hybrid = [candidate for candidate in run.candidates if candidate.source == "hybrid_symbolic"]
    assert hybrid
    assert hybrid[0].verified
    assert hybrid[0].symbolic_program is not None
    assert hybrid[0].predictions == [[[2, 0], [0, 2]]]


def test_candidate_parser_normalizes_program_list():
    candidates = parse_candidates(
        '{"candidates":[{"hypothesis":"two steps","program":['
        '{"op":"rotate","args":{"turns":1}},{"op":"flip",'
        '"args":{"axis":"horizontal"}}]}]}'
    )
    assert candidates[0].program is not None
    assert candidates[0].program.op == "compose"
    assert len(candidates[0].program.steps) == 2


def test_balanced_accuracy_does_not_reward_background_dominance():
    expected = [[0, 0, 0], [0, 0, 0], [0, 0, 2]]
    raw, balanced = grid_accuracies([[0, 0, 0] for _ in range(3)], expected)
    assert raw == 8 / 9
    assert balanced == 0.5


def test_candidate_parser_normalizes_compose_key():
    candidates = parse_candidates(
        '{"candidates":[{"hypothesis":"two steps","program":{"compose":['
        '{"op":"rotate","args":{"turns":1}},'
        '{"op":"flip","args":{"axis":"horizontal"}}]}}]}'
    )
    assert candidates[0].program is not None
    assert candidates[0].program.op == "compose"
    assert [step.op for step in candidates[0].program.steps] == ["rotate", "flip"]


def test_candidate_parser_repairs_unclosed_python_source():
    candidates = parse_candidates(
        '{"candidates":[{"hypothesis":"identity",'
        '"python_source":"def solve(grid):\\n    return grid}]}'
    )
    assert candidates[0].python_source == "def solve(grid):\n    return grid"


def test_candidate_parser_accepts_fenced_python_without_json():
    candidates = parse_candidates(
        "HYPOTHESIS: mirror every row\n"
        "```python\n"
        "def solve(grid):\n"
        "    return [row[::-1] for row in grid]\n"
        "```"
    )
    assert candidates[0].hypothesis == "mirror every row"
    assert candidates[0].python_source == (
        "def solve(grid):\n    return [row[::-1] for row in grid]"
    )


def test_candidate_parser_accepts_direct_structured_python():
    candidates = parse_candidates(
        '{"hypothesis":"identity","python_source":"def solve(grid):\\n    return grid"}'
    )
    assert candidates[0].hypothesis == "identity"
    assert candidates[0].python_source == "def solve(grid):\n    return grid"


def test_candidate_parser_accepts_python_only_structured_response():
    candidates = parse_candidates('{"python_source":"def solve(grid):\\n    return grid"}')
    assert candidates[0].hypothesis == ""
    assert candidates[0].python_source == "def solve(grid):\n    return grid"


def test_candidate_parser_recovers_complete_python_from_truncated_json():
    candidates = parse_candidates(
        '{"python_source":"def solve(grid):\\n    result = grid[::-1]\\n    return result'
    )
    assert candidates[0].python_source == (
        "def solve(grid):\n    result = grid[::-1]\n    return result"
    )


def test_candidate_parser_rejects_incomplete_python_from_truncated_json():
    text = '{"python_source":"def solve(grid):\\n    for row in grid:'
    try:
        parse_candidates(text)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete Python must not be recovered")


def test_candidate_parser_recovers_complete_python_before_prose():
    candidates = parse_candidates(
        "Failed to parse response: </think>\n\n"
        "def solve(grid):\n    out = grid[::-1]\n    return out\n"
        "```\n\nWait, let us analyze the first example."
    )
    assert candidates[0].python_source == (
        "def solve(grid):\n    out = grid[::-1]\n    return out"
    )


def test_level_three_prompt_is_python_only(hard_task):
    prompt = build_prompt(hard_task, [], level=3, feedback=None)
    assert "python_source" in prompt
    assert "Supported DSL operations" not in prompt
    assert "Spend output tokens on executable code" in prompt
    assert "no comments, docstrings" in prompt


def test_prompt_includes_independent_search_lens(hard_task):
    prompt = build_prompt(
        hard_task,
        [],
        level=3,
        feedback=None,
        search_mode="Object-centric independent search",
    )
    assert "Object-centric independent search" in prompt


def test_task_visualization_is_embedded_png(hard_task):
    data_url = render_task_data_url(hard_task)
    assert data_url.startswith("data:image/png;base64,")
    assert len(data_url) > 200


def test_structural_summary_describes_geometry_without_test_outputs(hard_task):
    summary = structural_summary(hard_task)
    assert "train1 input(3x3" in summary
    assert "components=[" in summary
    assert "delta(resize 3x3->2x2)" in summary
    assert "test1 input(2x2" in summary


def test_structural_summary_is_an_optional_prompt_lens(hard_task):
    default_prompt = build_prompt(hard_task, [], level=3, feedback=None)
    structured_prompt = build_prompt(
        hard_task,
        [],
        level=3,
        feedback=None,
        structural_summary_enabled=True,
    )
    assert "DETERMINISTIC STRUCTURAL SUMMARY" not in default_prompt
    assert "DETERMINISTIC STRUCTURAL SUMMARY" in structured_prompt


def test_deterministic_solver_makes_two_attempts(identity_task):
    config = SolverConfig.model_validate(
        {"name": "test", "model": {"enabled": False, "model": "disabled"}}
    )
    run = ArcSolver(config).solve_task(identity_task)
    assert run.final_level == 1
    assert run.attempts[0].attempt_1 == [[2, 0], [0, 2]]
    assert len(run.attempts) == 1


def test_llm_candidate_is_verified_and_used(hard_task):
    generation = Generation(
        candidates=[
            GeneratedCandidate(
                hypothesis="fallback to identity for the synthetic test",
                program=Program(op="identity"),
            ),
            GeneratedCandidate(
                hypothesis="pure program",
                python_source=(
                    "def solve(grid):\n"
                    "    return [list(reversed(row[1:])) for row in reversed(grid[1:])]\n"
                ),
            ),
        ]
    )
    config = SolverConfig()
    run = ArcSolver(config, adapter=StaticAdapter([generation])).solve_task(hard_task)
    assert any(candidate.verified for candidate in run.candidates)
    assert run.attempts[0].attempt_1 == [[4]]


def test_partial_python_candidate_records_cell_accuracy(hard_task):
    generation = Generation(
        candidates=[
            GeneratedCandidate(
                hypothesis="change one cell",
                python_source=(
                    "def solve(grid):\n    return [[0 for x in row[1:]] for row in grid[1:]]"
                ),
            )
        ]
    )
    config = SolverConfig.model_validate({"budget": {"level_2_calls": 1, "level_3_calls": 1}})
    run = ArcSolver(config, adapter=StaticAdapter([generation])).solve_task(hard_task)
    partial = next(candidate for candidate in run.candidates if candidate.source == "llm_python")
    assert 0.0 <= partial.train_cell_accuracy < 1.0
    assert 0.0 <= partial.train_balanced_accuracy < 1.0
    assert "differing cells" in (partial.error or "")
    assert partial.predictions == [[[0]]]
    assert run.attempts[0].attempt_1 == [[0]]


def test_failed_generation_falls_back(hard_task):
    config = SolverConfig.model_validate(
        {
            "budget": {"level_2_calls": 1, "level_3_calls": 1},
            "model": {"enabled": True},
        }
    )
    run = ArcSolver(
        config,
        adapter=StaticAdapter([Generation(), Generation()]),
    ).solve_task(hard_task)
    assert run.attempts[0].attempt_1 == hard_task.test[0].input
    assert run.attempts[0].attempt_2 == [[0, 0], [0, 0]]


def test_model_error_is_recorded_and_falls_back(hard_task):
    config = SolverConfig.model_validate(
        {
            "budget": {"level_2_calls": 1, "level_3_calls": 1},
            "model": {"enabled": True},
        }
    )
    run = ArcSolver(config, adapter=FailingAdapter()).solve_task(hard_task)
    assert any(candidate.source == "model_error" for candidate in run.candidates)
    assert run.attempts[0].attempt_1 == hard_task.test[0].input


def test_parse_error_is_recorded_for_refinement(hard_task):
    config = SolverConfig.model_validate(
        {"budget": {"level_3_calls": 1}, "model": {"enabled": True}}
    )
    run = ArcSolver(
        config,
        adapter=StaticAdapter([Generation(raw_text="not valid JSON")]),
    ).solve_task(hard_task)
    assert any(candidate.source == "model_error" for candidate in run.candidates)


def test_consensus_prediction_is_first_attempt(hard_task):
    shared = [[[9]]]
    alternative = [[[8]]]
    generations = [
        Generation(candidates=[GeneratedCandidate(predictions=shared)]),
        Generation(candidates=[GeneratedCandidate(predictions=shared)]),
        Generation(candidates=[GeneratedCandidate(predictions=alternative)]),
    ]
    config = SolverConfig.model_validate({"budget": {"level_3_calls": 3, "independent_calls": 3}})
    run = ArcSolver(config, adapter=StaticAdapter(generations)).solve_task(hard_task)
    assert run.attempts[0].attempt_1 == [[9]]
    assert run.attempts[0].attempt_2 == [[8]]
