from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from arc_agent.models import ArcTask
from arc_agent.v2_codex import _snapshot
from arc_agent.v4_config import content_hash
from arc_agent.v4_state import ExperimentRun
from arc_agent.v5_config import V5Config
from arc_agent.v5_pipeline import collect_task, export_examples
from arc_agent.v5_symbolic import teaching_task
from arc_agent.v7_cli import pilot_report
from arc_agent.v7_dataset import export_verified_examples
from arc_agent.v7_symbolic import (
    BridgedArtifact,
    PythonTrainingContract,
    output_schema,
    verify_artifact,
)

IDENTITY = (
    "def preserve(grid):\n    return [row[:] for row in grid]\n\n"
    "def solve(train, grid):\n    return preserve(grid)\n"
)


@pytest.fixture
def task():
    return ArcTask.model_validate(
        {
            "task_id": "fixture",
            "train": [
                {"input": [[1, 0]], "output": [[1, 0]]},
                {"input": [[2, 2]], "output": [[2, 2]]},
            ],
            "test": [{"input": [[3]], "output": [[8]]}],
        }
    )


@pytest.fixture
def artifact():
    return BridgedArtifact.model_validate(
        {
            "symbolic": {
                "entities": [
                    {
                        "id": "a",
                        "grid": "train_0_input",
                        "cells": [{"row": 0, "column": 0, "color": 1}],
                        "proposed_role": "possible foreground",
                    }
                ],
                "definitions": [
                    {"id": "preserve", "provisional_meaning": "retain cells", "depends_on": []}
                ],
                "relations": [],
                "hypotheses": [
                    {
                        "id": "h",
                        "statement": "possibly unchanged",
                        "concepts": ["preserve"],
                        "ordered_actions": ["preserve"],
                        "status": "proposed",
                        "counterevidence": [],
                    }
                ],
                "unresolved": ["other examples may refute this"],
                "proposed_transfer_skill": "test identity before proposing changes",
            },
            "witness": {
                "hypothesis_id": "h",
                "python_source": IDENTITY,
                "action_bindings": [{"action_id": "preserve", "function": "preserve"}],
            },
        }
    )


class Teacher:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []
        self.interrupt = False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = kwargs["round_index"]
        pending = _snapshot(
            response_id=f"r{index}", status="in_progress", thread_id="t", turn_id=str(index)
        )
        kwargs["checkpoint"](pending)
        if self.interrupt:
            raise KeyboardInterrupt
        return pending

    def retrieve(self, response_id, *, request):
        index = int(response_id[1:])
        return _snapshot(
            response_id=response_id,
            status="completed",
            thread_id="t",
            turn_id=str(index),
            text=self.artifacts[index].model_dump_json(),
        )

    def cancel(self, response_id):
        pass


def split_for(task):
    return {
        "groups": {"training": [task.task_id], "development": [], "lockbox": []},
        "task_hashes": {task.task_id: content_hash(task.model_dump(mode="json"))},
    }


def test_python_witness_links_are_not_semantic_proof(task, artifact):
    result = verify_artifact(artifact, *teaching_task(task))
    assert result["accepted"] and result["grounding_valid"]
    assert result["binding_validation"] == "static_reachable_action_functions"
    assert result["free_form_semantics_verified"] is False
    assert result["action_order_execution_verified"] is False
    assert result["visible_feedback"]["train_predictions"] == [[[1, 0]]]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unreachable",
        "action",
        "order",
        "refuted",
        "import",
        "lookup",
        "module_expression",
        "defaults",
        "duplicate",
        "nested",
    ],
)
def test_invalid_witness_rejected(task, artifact, mutation):
    witness = artifact.witness
    if mutation == "missing":
        witness.action_bindings[0].function = "unknown"
    elif mutation == "unreachable":
        witness.python_source = IDENTITY.replace(
            "return preserve(grid)", "return [r[:] for r in grid]"
        )
    elif mutation == "action":
        witness.action_bindings[0].action_id = "unknown"
    elif mutation == "order":
        witness.action_bindings.append(witness.action_bindings[0])
    elif mutation == "refuted":
        artifact.symbolic.hypotheses[0].status = "refuted"
    elif mutation == "import":
        witness.python_source = "import os\n" + IDENTITY
    elif mutation == "lookup":
        witness.python_source = IDENTITY.replace("return preserve(grid)", "return [[1, 0]]")
    elif mutation == "module_expression":
        witness.python_source = IDENTITY + "solve([], [])\n"
    elif mutation == "defaults":
        witness.python_source = IDENTITY.replace("def preserve(grid):", "def preserve(grid=[]):")
    elif mutation == "duplicate":
        witness.python_source = IDENTITY + "\ndef preserve(grid):\n    return grid\n"
    elif mutation == "nested":
        witness.python_source = IDENTITY.replace(
            "    return preserve(grid)",
            "    def nested():\n        return grid\n    return preserve(grid)",
        )
    result = verify_artifact(artifact, *teaching_task(task))
    assert not result["accepted"]
    assert not result["visible_verified"]


def test_sealed_pair_never_in_witness_train(monkeypatch, task, artifact):
    import arc_agent.v7_symbolic as module

    original = module.run_induction_program
    calls = []

    def inspect(source, train, grid, **kwargs):
        calls.append((train, grid))
        assert train == [(task.train[0].input, task.train[0].output)]
        return original(source, train, grid, **kwargs)

    monkeypatch.setattr(module, "run_induction_program", inspect)
    assert verify_artifact(artifact, *teaching_task(task))["accepted"]
    assert len(calls) == 2
    assert calls[-1][1] == task.train[-1].input


def test_visible_failure_does_not_execute_sealed_input(monkeypatch, task, artifact):
    import arc_agent.v7_symbolic as module

    original = module.run_induction_program
    calls = []

    def inspect(source, train, grid, **kwargs):
        calls.append(grid)
        return original(source, train, grid, **kwargs)

    monkeypatch.setattr(module, "run_induction_program", inspect)
    task.train[0].output = [[9, 0]]
    result = verify_artifact(artifact, *teaching_task(task))
    assert not result["visible_verified"]
    assert calls == [task.train[0].input]
    assert result["visible_feedback"]["failures"][0]["expected"] == [[9, 0]]


def test_changed_test_labels_have_no_effect(task, artifact):
    before = verify_artifact(artifact, *teaching_task(task))
    task.test[0].output = [[9, 8, 7]]
    assert verify_artifact(artifact, *teaching_task(task)) == before


def test_changed_sealed_labels_do_not_change_visible_feedback(task, artifact):
    before = verify_artifact(artifact, *teaching_task(task))
    task.train[-1].output = [[7, 8, 9]]
    after = verify_artifact(artifact, *teaching_task(task))
    assert before["visible_feedback"] == after["visible_feedback"]
    assert before["accepted"] and not after["accepted"]


def test_answer_fields_forbidden(artifact):
    value = artifact.model_dump(mode="json")
    value["symbolic"]["python_source"] = IDENTITY
    with pytest.raises(ValidationError):
        BridgedArtifact.model_validate(value)
    assert output_schema()["additionalProperties"] is False


def test_sealed_failure_stops_without_feedback(tmp_path, task, artifact):
    task.train[-1].output = [[7, 8, 9]]
    teacher = Teacher([artifact])
    cfg, contract = V5Config(poll_seconds=0.001), PythonTrainingContract()
    run = ExperimentRun(tmp_path, {"v": 7})
    try:
        state = collect_task(task, cfg, run, teacher, contract=contract)
        assert state["status"] == "heldout_rejected"
        assert state["feedback"] is None
        assert len(teacher.calls) == 1
        assert "7,8,9" not in teacher.calls[0]["prompt"]
        report = export_examples(run, cfg, [task], split_for(task), contract=contract)
        assert report["full_symbolic_examples"] == 0
    finally:
        run.close()


def test_repair_continuity_export_and_resume(tmp_path, task, artifact):
    bad = artifact.model_copy(deep=True)
    bad.witness.python_source = IDENTITY.replace(
        "return [row[:] for row in grid]", "return [[0 for c in row] for row in grid]"
    )
    teacher = Teacher([bad, artifact])
    cfg, contract = V5Config(poll_seconds=0.001), PythonTrainingContract()
    run = ExperimentRun(tmp_path, {"v": 7})
    teacher.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        collect_task(task, cfg, run, teacher, contract=contract)
    run.close()
    teacher.interrupt = False
    run = ExperimentRun(tmp_path, {"v": 7}, resume=True)
    try:
        state = collect_task(task, cfg, run, teacher, contract=contract)
        assert state["status"] == "accepted" and state["round"] == 2
        assert len(teacher.calls) == 2
        assert teacher.calls[1]["previous_response_id"] == "r0"
        assert "visible_verifier_feedback" in teacher.calls[1]["prompt"]
        report = export_examples(run, cfg, [task], split_for(task), contract=contract)
        assert report["full_symbolic_examples"] == 1
        examples = json.loads((tmp_path / "symbolic-sft.json").read_text())
        repair = next(e for e in examples if e["category"] == "repair")
        assert "previous_symbolic_model" in repair["prompt"][-1]["content"]
        assert IDENTITY not in repair["completion"][0]["content"]
        assert "python_source" not in repair["completion"][0]["content"]
        assert (
            json.loads(repair["completion"][0]["content"])["hypotheses"][0]["status"] == "proposed"
        )
        metrics = pilot_report(run, [task], report)
        assert metrics["dispatched_calls"] == metrics["valid_structured_responses"] == 2
        assert metrics["accepted_categories"] == {"repair": 1}
        assert not metrics["data_quality_gate_passed"]  # only one source task, no interpretation
        collect_task(task, cfg, run, teacher, contract=contract)
        assert len(teacher.calls) == 2
    finally:
        run.close()


def test_missing_response_counts_in_validity_denominator(tmp_path, task):
    run = ExperimentRun(tmp_path, {"v": 7})
    try:
        run.save("task:fixture", {"status": "timed_out", "round": 0, "records": []})
        run.save("call:fixture:0", {"snapshot": {"status": "in_progress"}})
        report = pilot_report(run, [task], {})
        assert report["dispatched_calls"] == 1
        assert report["completed_responses"] == 0
        assert report["structured_valid_rate_per_dispatch"] == 0
        assert not report["data_quality_gate_passed"]
    finally:
        run.close()


@pytest.mark.parametrize("tamper", [None, "prompt", "target", "response", "feedback", "split"])
def test_export_audit_reconstructs_prompt_and_teacher_target(tmp_path, task, artifact, tamper):
    run = ExperimentRun(tmp_path, {"v": 7})
    cfg = V5Config(poll_seconds=0.001)
    split = split_for(task)
    try:
        state = collect_task(task, cfg, run, Teacher([artifact]), contract=PythonTrainingContract())
        record = run.artifacts.get(state["records"][0])
        if tamper == "prompt":
            record["prompt"][-1]["content"] += "sealed output: [[2,2]]"
        elif tamper == "target":
            record["artifact"]["symbolic"]["unresolved"] = ["invented by another author"]
        elif tamper == "response":
            record["raw_response_ref"] = run.artifacts.put({"output_text": "invented"})
        elif tamper == "feedback":
            record["verification"]["visible_feedback"]["train_predictions"] = [[[2, 2]]]
        elif tamper == "split":
            split["groups"]["training"] = []
            split["groups"]["development"] = [task.task_id]
        if tamper:
            state["records"] = [run.artifacts.put(record)]
            run.save("task:fixture", state)
            with pytest.raises(ValueError, match="training audit"):
                export_verified_examples(run, cfg, [task], split)
            assert not (tmp_path / "symbolic-sft.json").exists()
        else:
            report = export_verified_examples(run, cfg, [task], split)
            assert report["full_symbolic_examples"] == 1
    finally:
        run.close()


def test_witness_runtime_failure_is_reported(task, artifact):
    artifact.witness.python_source = IDENTITY.replace(
        "return [row[:] for row in grid]", "return 1 / 0"
    )
    result = verify_artifact(artifact, *teaching_task(task))
    assert result["grounding_valid"] and not result["visible_verified"]
    assert "ZeroDivisionError" in result["visible_feedback"]["failures"][0]["error"]


def test_native_student_labels_exclude_python_witness(tmp_path, task, artifact):
    from pathlib import Path

    from arc_agent.v5_dataset import prepare_student

    run = ExperimentRun(tmp_path, {"v": 7})
    cfg = V5Config(poll_seconds=0.001)
    try:
        collect_task(task, cfg, run, Teacher([artifact]), contract=PythonTrainingContract())
        export_verified_examples(run, cfg, [task], split_for(task))
        report = prepare_student(tmp_path, Path(".runtime/nanbeige/model-hf"))
        assert report["examples"] == 2 and report["loss_mask_data_gate_passed"]
        assert not report["trainer_loss_mask_integration_verified"]
        for example in json.loads((tmp_path / "symbolic-sft.json").read_text()):
            assert "python_source" not in json.dumps(example)
            assert "action_bindings" not in json.dumps(example)
        for row in json.loads((tmp_path / "nanbeige-symbolic-tokenized.json").read_text()):
            assert set(row["labels"][: row["prompt_tokens"]]) == {-100}
            assert row["labels"][row["prompt_tokens"] :] == row["input_ids"][row["prompt_tokens"] :]
    finally:
        run.close()
