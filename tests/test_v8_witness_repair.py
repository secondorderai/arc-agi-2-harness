"""Stage-specific repair checks; synthetic fixtures, never training examples."""

import json
from contextlib import closing

import pytest
from pydantic import ValidationError
from test_v7 import IDENTITY
from test_v7 import artifact as artifact
from test_v7 import task as task
from test_v8 import FakeStudent

from arc_agent.v4_config import V4Config, content_hash
from arc_agent.v4_state import ExperimentRun
from arc_agent.v8_cli import experiment_binding, matching_smoke
from arc_agent.v8_config import V8Config
from arc_agent.v8_runner import advance_stage, state_key

UNSAFE = "def solve(train, grid):\n    raise ValueError('bad')\n"


def config():
    return V8Config(max_cycles=4, witness_repair_policy="repair_static_errors")


def static_failure(run, reasoner, task, cfg):
    reasoner.source = UNSAFE
    advance_stage(task, "symbolic", 42, cfg, run, reasoner)
    return advance_stage(task, "symbolic", 42, cfg, run, reasoner)


def test_static_repair_retains_grounded_state_and_actual_code_error(tmp_path, task, artifact):
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        reasoner, cfg = FakeStudent(artifact), config()
        state = static_failure(run, reasoner, task, cfg)
        original = artifact.symbolic.model_dump(mode="json")
        assert state["stage"] == "witness" and state["status"] == "running"
        assert state["symbolic"] == original and state["grounded_states"] == 1
        assert state["cycle"] == 1 and state["request_count"] == 2
        assert state["feedback"]["symbolic_state_sha256"] == content_hash(original)
        assert state["feedback"]["failures"][0]["error"] == "disallowed syntax: Raise"
        assert state["previous_source"] == UNSAFE and not state["candidates"]
        reasoner.source = IDENTITY
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert [c["stage"] for c in reasoner.calls] == ["symbolic", "witness", "witness"]
        call = reasoner.calls[-1]
        payload = json.loads(call["messages"][-1]["content"])
        assert payload["symbolic_model"] == original
        assert payload["previous_witness"] == UNSAFE
        assert payload["visible_verifier_feedback"]["repair_scope"] == "witness_only"
        assert "Never emit raise, assert, try/except" in call["messages"][0]["content"]
        assert "test_0_output" not in payload["grids"]
        assert state["status"] == "complete" and state["cycle"] == 2
        assert state["symbolic"] == original and state["grounded_states"] == 1
        assert state["request_count"] == state["valid_responses"] == 3
        assert state["generated_tokens"] == 300 and state["elapsed_seconds"] > 0


@pytest.mark.parametrize("policy", ["revise_symbolic", "repair_static_errors"])
def test_visible_counterexample_still_revises_symbolic(tmp_path, task, artifact, policy):
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        reasoner, cfg = FakeStudent(artifact), V8Config(witness_repair_policy=policy)
        reasoner.source = "def solve(train, grid):\n    return [[0 for c in row] for row in grid]\n"
        advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["stage"] == "symbolic" and state["status"] == "running"
        assert state["feedback"]["failures"][0]["expected"] == task.train[0].output
        assert "repair_scope" not in state["feedback"]


def test_legacy_static_failure_still_revises_symbols(tmp_path, task, artifact):
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        state = static_failure(run, FakeStudent(artifact), task, V8Config())
        assert state["stage"] == "symbolic" and "repair_scope" not in state["feedback"]


def test_repeated_code_failures_exhaust_the_same_cycle_allowance(tmp_path, task, artifact):
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        reasoner, cfg = FakeStudent(artifact), config()
        state = static_failure(run, reasoner, task, cfg)
        for _ in range(3):
            state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["status"] == "exhausted" and state["cycle"] == 4
        assert state["request_count"] == state["valid_responses"] == 5
        assert state["grounded_states"] == 1 and not state["candidates"]
        assert advance_stage(task, "symbolic", 42, cfg, run, reasoner) == state
        assert len(reasoner.calls) == 5


@pytest.mark.parametrize("mutation", ["hash", "meaning", "grounding"])
def test_corrupt_repair_state_fails_before_dispatch(tmp_path, task, artifact, mutation):
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        reasoner, cfg = FakeStudent(artifact), config()
        state = static_failure(run, reasoner, task, cfg)
        if mutation == "hash":
            state["feedback"]["symbolic_state_sha256"] = "0" * 64
        elif mutation == "meaning":
            state["symbolic"]["definitions"][0]["provisional_meaning"] = "different meaning"
        else:
            state["symbolic"]["entities"][0]["cells"][0]["color"] = 9
        run.save(state_key(task.task_id, "symbolic", 42), state)
        with pytest.raises(ValueError):
            advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert len(reasoner.calls) == 2


def test_code_repair_saved_response_replays_without_redispatch(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner, cfg = FakeStudent(artifact), config()
    state = static_failure(run, reasoner, task, cfg)
    symbols = state["symbolic"]
    original_save = run.save

    def crash_after_response(key, value):
        original_save(key, value)
        if value.get("response"):
            raise KeyboardInterrupt

    run.save = crash_after_response
    reasoner.source = IDENTITY
    with pytest.raises(KeyboardInterrupt):
        advance_stage(task, "symbolic", 42, cfg, run, reasoner)
    run.close()
    with closing(ExperimentRun(tmp_path, {"v": 8}, resume=True)) as run:
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["status"] == "complete" and len(reasoner.calls) == 3
        assert state["request_count"] == 3 and state["symbolic"] == symbols
        assert state["generated_tokens"] == 300


def test_unknown_code_repair_is_counted_and_never_redispatched(tmp_path, task, artifact):
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        reasoner, cfg = FakeStudent(artifact), config()
        static_failure(run, reasoner, task, cfg)
        reasoner.interrupt = True
        with pytest.raises(KeyboardInterrupt):
            advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        reasoner.interrupt = False
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["status"] == "timed_out" and len(reasoner.calls) == 3
        assert state["request_count"] == 3 and state["valid_responses"] == 2
        assert state["unknown_usage_calls"] == 1


def test_repair_policy_is_bound_into_current_source_smoke(monkeypatch):
    import arc_agent.v8_cli as cli

    monkeypatch.setattr(cli, "source_hash", lambda: "frozen-source")
    monkeypatch.setattr(cli, "file_hash", lambda path: "manifest")
    old = experiment_binding(V8Config(), V4Config(), {})
    new = experiment_binding(config(), V4Config(), {})
    report = {**old, "passed": True, "context_tokens": 4096}
    assert matching_smoke(report, old, 4096)
    assert not matching_smoke(report, new, 4096)
    with pytest.raises(ValidationError):
        V8Config(witness_repair_policy="ignore_sandbox")
