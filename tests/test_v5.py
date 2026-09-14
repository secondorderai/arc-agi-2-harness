from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from arc_agent.models import ArcTask
from arc_agent.v2_codex import _snapshot
from arc_agent.v2_openai import ConfigurationError
from arc_agent.v4_config import content_hash
from arc_agent.v4_state import ExperimentRun
from arc_agent.v5_config import TeacherConfig, V5Config
from arc_agent.v5_pipeline import collect_task, export_examples, student_prompt
from arc_agent.v5_symbolic import TeacherArtifact, output_schema, teaching_task, verify_artifact
from arc_agent.v5_teacher import AstraTeacher


@pytest.fixture
def task():
    return ArcTask.model_validate(
        {
            "task_id": "fixture",
            "train": [
                {"input": [[1, 0]], "output": [[1, 0]]},
                {"input": [[2, 2]], "output": [[2, 2]]},
            ],
            "test": [{"input": [[3]], "output": [[3]]}],
        }
    )


@pytest.fixture
def artifact():
    return TeacherArtifact.model_validate(
        {
            "symbolic": {
                "entities": [
                    {
                        "id": "a",
                        "grid": "train_0_input",
                        "cells": [{"row": 0, "column": 0, "color": 1}],
                        "proposed_role": "foreground",
                    }
                ],
                "definitions": [
                    {"id": "preserve", "provisional_meaning": "retain every cell", "depends_on": []}
                ],
                "relations": [],
                "hypotheses": [
                    {
                        "id": "h",
                        "statement": "input is unchanged",
                        "concepts": ["preserve"],
                        "ordered_actions": ["preserve"],
                        "status": "proposed",
                        "counterevidence": [],
                    }
                ],
                "unresolved": ["identity may fail on other tasks"],
                "proposed_transfer_skill": "check the unchanged-input hypothesis first",
            },
            "witness": {"hypothesis_id": "h", "program_json": '{"op":"identity"}'},
        }
    )


@pytest.mark.parametrize(
    "values",
    [
        {"teacher": {"model": "gpt-5.6-sol"}},
        {"teacher": {"auth_mode": "api_key"}},
        {"teacher": {"model": "Nanbeige/Nanbeige4.2-3B"}},
        {"student": "Qwen/Qwen3-4B"},
        {"student_revision": "main"},
        {"budget_usd": 100},
    ],
)
def test_lineage_fail_closed(values):
    with pytest.raises(ValidationError):
        V5Config(**values)


def test_holdout_and_test_labels_absent(task):
    visible, heldout = teaching_task(task)
    assert len(visible.train) == 1
    assert visible.test[0].output is None
    assert heldout.train[0] == task.train[-1]
    prompt = student_prompt(visible)
    assert "train_1" not in prompt[-1]["content"]
    assert "test_0_output" not in prompt[-1]["content"]
    assert task.task_id not in prompt[-1]["content"]


def test_verified_symbolic_artifact_is_not_semantic_proof(task, artifact):
    result = verify_artifact(artifact, *teaching_task(task))
    assert result["accepted"]
    assert not result["free_form_semantics_verified"]


@pytest.mark.parametrize(
    "mutation",
    [
        "color",
        "bounds",
        "grid",
        "duplicate",
        "undefined",
        "cycle",
        "relation",
        "witness",
        "refuted",
        "program",
    ],
)
def test_invalid_artifact_rejected(task, artifact, mutation):
    state = artifact.symbolic
    if mutation == "color":
        state.entities[0].cells[0].color = 8
    elif mutation == "bounds":
        state.entities[0].cells[0].row = 10
    elif mutation == "grid":
        state.entities[0].grid = "train_1_output"
    elif mutation == "duplicate":
        state.entities.append(state.entities[0])
    elif mutation == "undefined":
        state.hypotheses[0].ordered_actions = ["missing"]
    elif mutation == "cycle":
        state.definitions[0].depends_on = ["preserve"]
    elif mutation == "relation":
        from arc_agent.v5_symbolic import Relation

        state.relations = [Relation(subject="a", predicate="left_of", object="a")]
    elif mutation == "witness":
        artifact.witness.hypothesis_id = "missing"
    elif mutation == "refuted":
        state.hypotheses[0].status = "refuted"
    elif mutation == "program":
        artifact.witness.program_json = '{"op":"exec","args":{"code":"bad"}}'
    assert not verify_artifact(artifact, *teaching_task(task))["accepted"]


def test_answer_fields_forbidden(artifact):
    value = artifact.model_dump(mode="json")
    value["symbolic"]["predictions"] = [[[1]]]
    with pytest.raises(ValidationError):
        TeacherArtifact.model_validate(value)
    assert output_schema()["additionalProperties"] is False


class FakeTeacher:
    def __init__(self, artifact):
        self.artifact = artifact
        self.created = 0
        self.prompts = []
        self.interrupt_after_checkpoint = False

    def create(self, **kwargs):
        self.created += 1
        self.prompts.append(kwargs["prompt"])
        pending = _snapshot(response_id="r", status="in_progress", thread_id="t", turn_id="u")
        kwargs["checkpoint"](pending)
        if self.interrupt_after_checkpoint:
            raise KeyboardInterrupt
        return pending

    def retrieve(self, response_id, *, request):
        return _snapshot(
            response_id="r",
            status="completed",
            thread_id="t",
            turn_id="u",
            text=self.artifact.model_dump_json(),
        )

    def cancel(self, response_id):
        pass


def split_for(task):
    return {
        "groups": {"training": [task.task_id], "development": [], "lockbox": []},
        "task_hashes": {task.task_id: content_hash(task.model_dump(mode="json"))},
    }


def test_resume_and_symbolic_only_export(tmp_path, task, artifact):
    config = V5Config(poll_seconds=0.001)
    teacher = FakeTeacher(artifact)
    teacher.interrupt_after_checkpoint = True
    run = ExperimentRun(tmp_path, {"v": 5})
    with pytest.raises(KeyboardInterrupt):
        collect_task(task, config, run, teacher)
    run.close()
    run = ExperimentRun(tmp_path, {"v": 5}, resume=True)
    teacher.interrupt_after_checkpoint = False
    result = collect_task(task, config, run, teacher)
    assert result["status"] == "accepted"
    collect_task(task, config, run, teacher)
    assert teacher.created == 1
    report = export_examples(run, config, [task], split_for(task))
    assert report["examples"] == 2
    assert report["grounding_examples"] == report["full_symbolic_examples"] == 1
    exported = json.loads((tmp_path / "symbolic-sft.json").read_text())
    grounding = json.loads(exported[0]["completion"][0]["content"])
    assert "proposed_role" not in grounding["entities"][0]
    completion = json.loads(exported[1]["completion"][0]["content"])
    assert "witness" not in completion and "predictions" not in completion
    assert completion["hypotheses"][0]["status"] == "proposed"
    # Atomic export is repeatable, not append-only duplication.
    export_examples(run, config, [task], split_for(task))
    assert json.loads((tmp_path / "symbolic-sft.json").read_text()) == exported
    run.close()


def test_holdout_failure_stops_without_feedback_or_export(tmp_path, task, artifact):
    task.train[-1].output = [[9]]
    config = V5Config(poll_seconds=0.001)
    teacher = FakeTeacher(artifact)
    run = ExperimentRun(tmp_path, {"v": 5})
    result = collect_task(task, config, run, teacher)
    assert result["status"] == "heldout_rejected"
    assert teacher.created == 1
    assert result["feedback"] is None
    report = export_examples(run, config, [task], split_for(task))
    assert report["full_symbolic_examples"] == 0
    assert report["grounding_examples"] == 1
    run.close()


@pytest.mark.parametrize("bad", ["split", "teacher", "student", "hash"])
def test_export_rejects_contamination(tmp_path, task, artifact, bad):
    config = V5Config(poll_seconds=0.001)
    run = ExperimentRun(tmp_path, {"v": 5})
    state = collect_task(task, config, run, FakeTeacher(artifact))
    record = run.artifacts.get(state["records"][0])
    field, value = {
        "split": ("split", "development"),
        "teacher": ("teacher_model", "other"),
        "student": ("student", "other"),
        "hash": ("task_hash", "changed"),
    }[bad]
    record[field] = value
    state["records"] = [run.artifacts.put(record)]
    run.save(f"task:{task.task_id}", state)
    with pytest.raises(ValueError, match="isolation"):
        export_examples(run, config, [task], split_for(task))
    run.close()


class FakeRPC:
    latest_rate_limits = None

    def __init__(self, available=True, auth="chatgpt"):
        self.available, self.auth, self.params = available, auth, []

    def request(self, method, params, **kwargs):
        self.params.append((method, params))
        if method == "account/read":
            return {"account": {"type": self.auth}}
        if method == "model/list":
            return {
                "data": [
                    {
                        "id": "gpt-6-astra",
                        "supportedReasoningEfforts": [{"reasoningEffort": "xhigh"}],
                    }
                ]
                if self.available
                else []
            }
        if method == "account/rateLimits/read":
            return {}
        if method == "turn/start":
            return {"turn": {"id": "u"}}
        raise AssertionError(method)


def test_subscription_exact_model_and_schema(tmp_path):
    rpc = FakeRPC()
    teacher = AstraTeacher(settings=TeacherConfig(), workspace=tmp_path, rpc=rpc)
    assert teacher.preflight()["available"]
    teacher._start_turn(
        thread_id="t", prompt="fixture", request_key="key", token_limit=8192, model="gpt-6-astra"
    )
    params = rpc.params[-1][1]
    assert params["model"] == "gpt-6-astra"
    assert params["outputSchema"] == output_schema()
    profile = teacher._thread_params("gpt-6-astra")["config"]
    assert profile["default_permissions"] == "arc-symbolic-teacher"
    assert profile["permissions.arc-symbolic-teacher.network.enabled"] is False
    assert set(profile["permissions.arc-symbolic-teacher.filesystem"].values()) == {"read"}
    assert "sandboxPolicy" not in params
    assert teacher._thread_params("gpt-6-astra")["modelProvider"] == "openai"


@pytest.mark.parametrize("available,auth", [(False, "chatgpt"), (True, "apiKey")])
def test_no_model_or_billing_fallback(tmp_path, available, auth):
    teacher = AstraTeacher(
        settings=TeacherConfig(), workspace=tmp_path, rpc=FakeRPC(available, auth)
    )
    with pytest.raises(ConfigurationError):
        teacher.preflight()


def test_native_student_tokenization_and_mask(tmp_path, task, artifact):
    from pathlib import Path

    from tokenizers import Tokenizer

    from arc_agent.v4_reasoner import NativeTemplate
    from arc_agent.v5_dataset import prepare_student, tokenize_example

    tokenizer_dir = Path(".runtime/nanbeige/model-hf")
    if not (tokenizer_dir / "tokenizer.json").is_file():
        pytest.skip("requires the pinned local Nanbeige tokenizer, not model weights")
    config = V5Config(poll_seconds=0.001)
    run = ExperimentRun(tmp_path, {"v": 5})
    collect_task(task, config, run, FakeTeacher(artifact))
    export_examples(run, config, [task], split_for(task))
    report = prepare_student(tmp_path, tokenizer_dir)
    assert report["loss_mask_data_gate_passed"]
    assert report["examples"] == 2
    rows = json.loads((tmp_path / "nanbeige-symbolic-tokenized.json").read_text())
    for row in rows:
        n = row["prompt_tokens"]
        assert all(label == -100 for label in row["labels"][:n])
        assert row["labels"][n:] == row["input_ids"][n:]
        assert n > 0 and row["completion_tokens"] > 0
    example = json.loads((tmp_path / "symbolic-sft.json").read_text())[0]
    with pytest.raises(ValueError, match="never truncate"):
        tokenize_example(
            example,
            NativeTemplate(tokenizer_dir / "tokenizer_config.json"),
            Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json")),
            1,
        )
    # Files changed after export cannot silently enter training.
    (tmp_path / "symbolic-sft.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        prepare_student(tmp_path, tokenizer_dir)
    run.close()


def test_symbolic_only_keeps_checked_observations(tmp_path, task, artifact):
    artifact.witness = None
    config = V5Config(poll_seconds=0.001)
    run = ExperimentRun(tmp_path, {"v": 5})
    teacher = FakeTeacher(artifact)
    assert collect_task(task, config, run, teacher)["status"] == "symbolic_only"
    report = export_examples(run, config, [task], split_for(task))
    assert report["grounding_examples"] == 1
    assert report["full_symbolic_examples"] == 0
    assert teacher.created == 1
    run.close()


@pytest.mark.parametrize(
    "override",
    [
        {"model": "gpt-5.6-sol"},
        {"modelProvider": "other"},
        {"activePermissionProfile": None},
        {"sandbox": {"type": "readOnly", "networkAccess": True}},
        {"instructionSources": ["/untrusted/AGENTS.md"]},
        {"approvalPolicy": "on-request"},
    ],
)
def test_effective_provider_and_permissions_fail_closed(tmp_path, override):
    class ProfileRPC(FakeRPC):
        def request(self, method, params, **kwargs):
            return {
                "model": "gpt-6-astra",
                "modelProvider": "openai",
                "activePermissionProfile": {"id": "arc-symbolic-teacher"},
                "sandbox": {"type": "readOnly", "networkAccess": False},
                "approvalPolicy": "never",
                "instructionSources": [],
                **override,
            }

    teacher = AstraTeacher(settings=TeacherConfig(), workspace=tmp_path, rpc=ProfileRPC())
    with pytest.raises(ConfigurationError, match="unexpected teacher"):
        teacher._request("thread/start", teacher._thread_params("gpt-6-astra"))
