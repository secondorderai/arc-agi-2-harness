"""Saved-response replay tests; no model inference or training."""

import copy
import importlib.util
from contextlib import closing
from pathlib import Path

import pytest
from test_v7 import IDENTITY
from test_v7 import artifact as artifact
from test_v7 import task as task
from test_v8 import FakeStudent
from test_v8_cell_protocol import PackedStudent
from test_v8_reference_repair import PatchStudent

from arc_agent.v4_state import ExperimentRun
from arc_agent.v8_config import V8Config
from arc_agent.v8_runner import advance_stage, state_key

SPEC = importlib.util.spec_from_file_location(
    "baseline_replay", Path(__file__).parents[1] / "scripts/replay_nanbeige_baseline.py"
)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class RecordedStudent(FakeStudent):
    def prepare_stage(self, messages, *, seed, task, stage):
        return {
            **super().prepare_stage(messages, seed=seed, task=task, stage=stage),
            "_stage": stage,
        }


class RecordedPatchStudent(PatchStudent):
    def prepare_stage(self, messages, *, seed, task, stage):
        return {
            **super().prepare_stage(messages, seed=seed, task=task, stage=stage),
            "_stage": stage,
        }


def record(tmp_path, task, artifact, *, unsafe=False, mode="symbolic"):
    config = V8Config(max_cycles=4, witness_repair_policy="repair_static_errors")
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        reasoner = RecordedStudent(artifact)
        if unsafe:
            reasoner.source = "def solve(train, grid):\n    raise ValueError('bad')\n"
        state = advance_stage(task, mode, 42, config, run, reasoner)
        if mode == "symbolic":
            state = advance_stage(task, mode, 42, config, run, reasoner)
            if unsafe:
                reasoner.source = IDENTITY
                state = advance_stage(task, mode, 42, config, run, reasoner)
        work = {
            f"{state_key(task.task_id, mode, 42)}:{i}": run.get(
                f"{state_key(task.task_id, mode, 42)}:{i}"
            )
            for i in range(state["step"])
        }
    return state, work, config


@pytest.mark.parametrize(
    "unsafe,mode", [(False, "direct"), (False, "symbolic"), (True, "symbolic")]
)
def test_replays_exact_semantics_and_terminal_noop(tmp_path, task, artifact, unsafe, mode):
    state, work, config = record(tmp_path, task, artifact, unsafe=unsafe, mode=mode)
    before = copy.deepcopy((state, work))
    result = replay.replay_state(task, mode, 42, state, work, tmp_path / "artifacts", config)
    assert result["semantic_state_reproduced"] and result["terminal_noop_verified"]
    assert result["replayed_steps"] == (3 if unsafe else 1 if mode == "direct" else 2)
    assert (state, work) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("cycle", 3),
        ("grounded_states", 0),
        ("feedback", {}),
        ("symbolic", None),
        ("candidates", []),
        ("attempts", []),
        ("generated_tokens", 1),
    ],
)
def test_rejects_changed_state(tmp_path, task, artifact, field, value):
    state, work, config = record(tmp_path, task, artifact)
    state[field] = value
    with pytest.raises(ValueError, match="replayed state differs"):
        replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)


def test_changed_prompt_cannot_replay(tmp_path, task, artifact):
    state, work, config = record(tmp_path, task, artifact)
    first = work[f"{state_key(task.task_id, 'symbolic', 42)}:0"]
    first["messages"][0]["content"] += " altered"
    with pytest.raises(ValueError, match="prompt changed"):
        replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)


def test_corrupt_artifacts_cannot_replay(tmp_path, task, artifact):
    state, work, config = record(tmp_path, task, artifact)
    (tmp_path / "artifacts" / f"{state['artifact_refs'][0]}.json").write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)


@pytest.mark.parametrize("status,unknown", [("timed_out", 0), ("context_limit", 0), ("running", 1)])
def test_timing_and_unknown_calls_are_not_invented(tmp_path, task, artifact, status, unknown):
    state, work, config = record(tmp_path, task, artifact)
    state.update(status=status, unknown_usage_calls=unknown)
    result = replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)
    assert not result["semantic_state_reproduced"]
    assert result["replayed_steps"] == 0 and "unsupported_reason" in result


def test_missing_response_is_not_regenerated(tmp_path, task, artifact):
    state, work, config = record(tmp_path, task, artifact)
    work[f"{state_key(task.task_id, 'symbolic', 42)}:0"]["response"] = None
    result = replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)
    assert not result["semantic_state_reproduced"] and result["replayed_steps"] == 0


def test_no_inference_guard():
    with pytest.raises(AssertionError, match="prepare a new"):
        replay.NoInference().prepare_stage()
    with pytest.raises(AssertionError, match="never call a model"):
        replay.NoInference().complete()


@pytest.mark.parametrize("wire", ["triples", "patch"])
def test_compact_cells_and_reference_repairs_replay(tmp_path, task, artifact, wire):
    config = V8Config(
        max_cycles=4,
        structured_style="reference_repair_bounded_ws",
        symbolic_cell_encoding="triples_v1" if wire == "triples" else "named_fields",
    )
    if wire == "patch":
        artifact.symbolic.hypotheses[0].ordered_actions = ["apply_preserve"]
    reasoner = PackedStudent(artifact) if wire == "triples" else RecordedPatchStudent(artifact)
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        for _ in range(2 if wire == "triples" else 3):
            state = advance_stage(task, "symbolic", 42, config, run, reasoner)
        key = state_key(task.task_id, "symbolic", 42)
        work = {f"{key}:{i}": run.get(f"{key}:{i}") for i in range(state["step"])}
    result = replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)
    assert result["semantic_state_reproduced"] and result["terminal_noop_verified"]


def test_stage_metadata_must_match_reconstructed_transition(tmp_path, task, artifact):
    state, work, config = record(tmp_path, task, artifact)
    work[f"{state_key(task.task_id, 'symbolic', 42)}:0"]["prepared"]["_stage"] = "direct"
    with pytest.raises(ValueError, match="request stage differs"):
        replay.replay_state(task, "symbolic", 42, state, work, tmp_path / "artifacts", config)
