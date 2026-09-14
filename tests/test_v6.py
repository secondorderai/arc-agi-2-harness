from __future__ import annotations

import json
import time

import pytest
from pydantic import ValidationError

from arc_agent.models import ArcTask
from arc_agent.v2_codex import _snapshot
from arc_agent.v4_state import ExperimentRun, blind_task
from arc_agent.v6_eval import (
    EvalConfig,
    SolverResponse,
    decode_predictions,
    evaluate_candidate,
    initial_prompt,
    score_predictions,
    select_predictions,
    write_outputs,
)
from arc_agent.v6_runner import fixed_cohort, run_round


@pytest.fixture
def task():
    return ArcTask.model_validate(
        {
            "task_id": "fixture",
            "train": [
                {"input": [[1, 0]], "output": [[0, 1]]},
                {"input": [[2, 0, 0]], "output": [[0, 0, 2]]},
            ],
            "test": [{"input": [[3, 0]], "output": [[0, 3]]}],
        }
    )


@pytest.fixture
def response():
    return SolverResponse(
        symbolic_model="Reflect rows left-to-right; preserve colors.",
        python_source="def solve(train, grid):\n    return [list(reversed(row)) for row in grid]",
        direct_predictions=[["03"]],
        alternative_predictions=None,
    )


def test_fixed_original_cohort():
    tasks, split = fixed_cohort(EvalConfig())
    assert len(tasks) == 20 and sum(len(t.test) for t in tasks) == 22
    assert [t.task_id for t in tasks] == split["groups"]["development"][:20]
    assert not set(t.task_id for t in tasks) & set(split["groups"]["training"])


@pytest.mark.parametrize(
    "values",
    [
        {"teacher": {"model": "gpt-5.6-sol"}},
        {"teacher": {"auth_mode": "api_key"}},
        {"cohort": "easiest_twenty"},
        {"target": 0.1},
        {"global_seconds": 43201},
    ],
)
def test_no_model_billing_or_easy_cohort_fallback(values):
    with pytest.raises(ValidationError):
        EvalConfig(**values)


@pytest.mark.parametrize("raw", [[["1x"]], [["1", "22"]], [], [[""]], [["1"]] * 2])
def test_invalid_predictions(raw):
    with pytest.raises(ValueError):
        decode_predictions(raw, 1)


def test_solver_and_verifier_label_boundaries(task, response):
    payload = json.loads(initial_prompt(task))
    assert all(set(p) == {"input"} for p in payload["test"])
    assert "fixture" not in json.dumps(payload)
    with pytest.raises(ValueError, match="label-blind"):
        evaluate_candidate(response, task, 0)
    result = evaluate_candidate(response, blind_task(task), 0)
    assert result["verified"]
    assert result["verification"]["source_tests_total"] == 0
    assert result["verification"]["leave_one_out_exact"] == 2
    assert "source_test_" not in result["feedback"]


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef solve(train, grid):\n    return grid",
        "def solve(train, grid):\n    open('/private').read()\n    return grid",
        "def solve(train, grid):\n    return grid.__class__",
    ],
)
def test_unsafe_code_never_becomes_a_program_candidate(task, response, source):
    response.python_source = source
    result = evaluate_candidate(response, blind_task(task), 0)
    assert not result["verified"]
    assert all(c["source"] != "program" for c in result["candidates"])


def test_direct_and_program_dedup_without_answer_oracle(task, response):
    response.alternative_predictions = [["30"]]
    result = evaluate_candidate(response, blind_task(task), 0)
    selected = select_predictions(blind_task(task), result["candidates"])
    assert selected == [{"attempt_1": [[0, 3]], "attempt_2": [[3, 0]]}]
    altered = task.model_copy(deep=True)
    altered.test[0].output = [[9]]
    assert select_predictions(blind_task(altered), result["candidates"]) == selected


def test_scoring_counts_every_output_and_task(task):
    other = task.model_copy(deep=True)
    other.task_id = "other"
    other.test.append(other.test[0])
    submission = {
        task.task_id: [{"attempt_1": [[0, 3]], "attempt_2": [[9]]}],
        other.task_id: [
            {"attempt_1": [[9]], "attempt_2": [[0, 3]]},
            {"attempt_1": [[9]], "attempt_2": [[8]]},
        ],
    }
    report = score_predictions([task, other], submission, completed=1)
    assert report["exact_match"] == 2 / 3
    assert report["task_mean_exact_match"] == 0.75
    assert report["strict_task_accuracy"] == 0.5
    assert not report["goal_threshold_observed"]
    assert report["test_outputs"] == 3
    assert not report["training_export_allowed"]


def test_complete_fallbacks_exist_before_any_model_call(tmp_path, task):
    run = ExperimentRun(tmp_path, {"v": 6})
    report = write_outputs(run, [task])
    assert report["completed_tasks"] == 0
    assert not report["goal_threshold_observed"]
    saved = json.loads((tmp_path / "submission.json").read_text())
    assert set(saved[task.task_id][0]) == {"attempt_1", "attempt_2"}
    run.close()


class FakeSolver:
    def __init__(self, response, interrupt=False):
        self.response, self.interrupt = response, interrupt
        self.created, self.retrieved = 0, 0

    def create(self, **kwargs):
        self.created += 1
        snapshot = _snapshot(response_id="id", status="in_progress", thread_id="t", turn_id="u")
        kwargs["checkpoint"](snapshot)
        if self.interrupt:
            raise KeyboardInterrupt
        return snapshot

    def retrieve(self, response_id, *, request):
        self.retrieved += 1
        return _snapshot(
            response_id="id",
            status="completed",
            thread_id="t",
            turn_id="u",
            text=self.response.model_dump_json(),
        )

    def cancel(self, response_id):
        pass


def test_inflight_and_completed_resume_no_regeneration(tmp_path, task, response):
    cfg = EvalConfig(poll_seconds=0.001)
    solver = FakeSolver(response, interrupt=True)
    run = ExperimentRun(tmp_path, {"v": 6})
    with pytest.raises(KeyboardInterrupt):
        run_round(blind_task(task), cfg, run, solver, time.time() + 100)
    run.close()
    solver.interrupt = False
    run = ExperimentRun(tmp_path, {"v": 6}, resume=True)
    run_round(blind_task(task), cfg, run, solver, time.time() + 100)
    state = run.get("task:fixture")
    assert state["done"] and state["status"] == "verified"
    run_round(blind_task(task), cfg, run, solver, time.time() + 100)
    assert solver.created == solver.retrieved == 1
    record = run.artifacts.get(state["records"][0])
    assert record["split"] == "development" and not record["training_export_allowed"]
    run.close()


def test_scoring_cannot_change_a_candidate(task, response):
    result = evaluate_candidate(response, blind_task(task), 0)
    before = json.dumps(result, sort_keys=True)
    submission = {task.task_id: select_predictions(blind_task(task), result["candidates"])}
    score_predictions([task], submission, completed=1)
    assert json.dumps(result, sort_keys=True) == before
