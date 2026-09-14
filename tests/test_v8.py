from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_v7 import IDENTITY
from test_v7 import artifact as artifact
from test_v7 import task as task

from arc_agent.v4_config import V4Config, content_hash
from arc_agent.v4_reasoner import ContextOverflow, NativeTemplate
from arc_agent.v4_runtime import prompt_cache_arguments
from arc_agent.v4_state import ExperimentRun, atomic_json, blind_task
from arc_agent.v5_pipeline import student_prompt
from arc_agent.v8_cli import write_report
from arc_agent.v8_config import V8Config
from arc_agent.v8_dataset import prepare_compact_student
from arc_agent.v8_eval import check_witness, score_submission
from arc_agent.v8_reasoner import (
    PYTHON_GRAMMAR,
    StudentReasoner,
    compact_symbolic_prompt,
    direct_prompt,
    grounded_schema,
    symbolic_prompt,
    witness_prompt,
)
from arc_agent.v8_runner import advance_stage, grounding_feedback, initial_state, state_key


class FakeStudent:
    def __init__(self, artifact):
        self.artifact = artifact
        self.calls = []
        self.interrupt = False
        self.bad_json = False
        self.source = IDENTITY

    def prepare_stage(self, messages, *, seed, task, stage):
        assert all(p.output is None for p in task.test)
        return {"messages": messages, "stage": stage, "seed": seed, "n_predict": 2048}

    def complete(self, prepared, *, timeout):
        self.calls.append(prepared)
        if self.interrupt:
            raise KeyboardInterrupt
        value = {
            "symbolic": self.artifact.symbolic.model_dump(mode="json"),
            "witness": {"python_source": self.source},
            "direct": {"predictions": [{"attempt_1": ["3"], "attempt_2": "same_as_1"}]},
        }[prepared["stage"]]
        return {
            "final": "broken" if self.bad_json else json.dumps(value),
            "reasoning": "",
            "prompt_tokens": 50,
            "generated_tokens": 100,
            "elapsed_seconds": 0.01,
            "raw": {"content": json.dumps(value)},
        }


def test_symbolic_decoding_bounds_come_only_from_visible_grids(task):
    schema = grounded_schema(task)
    variants = schema["$defs"]["Entity"]["oneOf"]
    keyed = {e["properties"]["grid"]["const"]: e for e in variants}
    assert "test_0_output" not in keyed
    entity = keyed["train_0_input"]["properties"]["cells"]["items"]["properties"]
    assert entity["row"]["maximum"] == 0 and entity["column"]["maximum"] == 1
    task.test[0].output = [[9] * 8] * 8
    assert grounded_schema(task) == schema


def test_prompt_cache_opt_out_preserves_legacy_defaults():
    assert prompt_cache_arguments(V4Config()) == []
    assert prompt_cache_arguments(V4Config(prompt_cache_mib=0)) == ["--cache-ram", "0"]
    with pytest.raises(ValidationError):
        V4Config(prompt_cache_mib=8192)


def test_symbolic_compiler_feedback_explains_invalid_cells_and_links(task, artifact):
    symbols = artifact.symbolic
    symbols.entities[0].cells[0].color = 9
    symbols.hypotheses[0].ordered_actions = ["undefined_action"]
    feedback = grounding_feedback(symbols, task, ValueError("invalid"))
    assert feedback["diagnostics"][0]["observed_color"] == 1
    assert feedback["diagnostics"][-1]["undefined_concepts_or_actions"] == ["undefined_action"]
    assert "test_0_output" not in feedback["grid_shapes"]


def test_prompts_blind_lossless_and_do_not_mutate(task, artifact):
    original = student_prompt(blind_task(task))
    copied = json.loads(json.dumps(original))
    compact = compact_symbolic_prompt(original)
    assert original == copied
    evidence = json.loads(compact[-1]["content"])
    assert evidence["grids"]["train_0_input"] == ["10"]
    assert "test_0_output" not in evidence["grids"]
    variants = [
        direct_prompt(task),
        symbolic_prompt(task),
        witness_prompt(task, artifact.symbolic.model_dump()),
    ]
    task.test[0].output = [[8, 9, 8, 9]]
    assert variants == [
        direct_prompt(task),
        symbolic_prompt(task),
        witness_prompt(task, artifact.symbolic.model_dump()),
    ]
    assert task.task_id not in json.dumps(variants)


def test_two_stage_state_witness_and_completed_resume(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner, cfg = FakeStudent(artifact), V8Config()
    try:
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["stage"] == "witness" and state["grounded_states"] == 1
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["status"] == "complete" and state["valid_responses"] == 2
        assert state["attempts"] == [{"attempt_1": [[3]], "attempt_2": [[3]]}]
        assert len(reasoner.calls) == 2
        assert advance_stage(task, "symbolic", 42, cfg, run, reasoner) == state
        assert len(reasoner.calls) == 2
        prompt = json.loads(reasoner.calls[1]["messages"][-1]["content"])
        assert prompt["symbolic_model"] == artifact.symbolic.model_dump(mode="json")
    finally:
        run.close()


def test_response_saved_before_parse_can_resume_without_generation(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    save = run.save

    def crash(key, value):
        save(key, value)
        if value.get("response"):
            raise KeyboardInterrupt

    run.save = crash
    reasoner, cfg = FakeStudent(artifact), V8Config()
    with pytest.raises(KeyboardInterrupt):
        advance_stage(task, "direct", 42, cfg, run, reasoner)
    run.close()
    run = ExperimentRun(tmp_path, {"v": 8}, resume=True)
    try:
        state = advance_stage(task, "direct", 42, cfg, run, reasoner)
        assert len(reasoner.calls) == 1 and state["status"] == "complete"
        assert state["request_count"] == 1 and state["generated_tokens"] == 100
    finally:
        run.close()


def test_uncertain_local_call_is_not_duplicated(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner, cfg = FakeStudent(artifact), V8Config()
    reasoner.interrupt = True
    try:
        with pytest.raises(KeyboardInterrupt):
            advance_stage(task, "direct", 42, cfg, run, reasoner)
        reasoner.interrupt = False
        state = advance_stage(task, "direct", 42, cfg, run, reasoner)
        assert len(reasoner.calls) == 1
        assert state["status"] == "timed_out" and state["unknown_usage_calls"] == 1
        assert state["request_count"] == 1
    finally:
        run.close()


def test_failed_witness_becomes_visible_revision_context(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner, cfg = FakeStudent(artifact), V8Config()
    reasoner.source = IDENTITY.replace(
        "return [row[:] for row in grid]", "return [[0 for c in row] for row in grid]"
    )
    try:
        advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["stage"] == "symbolic" and state["status"] == "running"
        assert state["feedback"]["failures"][0]["expected"] == task.train[0].output
        reasoner.source = IDENTITY
        advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        messages = json.loads(reasoner.calls[-1]["messages"][-1]["content"])
        assert messages["previous_symbolic_model"]["hypotheses"][0]["status"] == "proposed"
        assert "visible_verifier_feedback" in messages
        state = advance_stage(task, "symbolic", 42, cfg, run, reasoner)
        assert state["status"] == "complete" and state["cycle"] == 2
    finally:
        run.close()


def test_grounding_failure_not_sent_to_witness(tmp_path, task, artifact):
    artifact.symbolic.entities[0].cells[0].color = 9
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner = FakeStudent(artifact)
    try:
        state = advance_stage(task, "symbolic", 42, V8Config(), run, reasoner)
        assert state["stage"] == "symbolic" and state["grounded_states"] == 0
        assert state["valid_responses"] == 1
        assert "contradicts" in state["feedback"]["error"]
    finally:
        run.close()


def test_missing_artifact_is_fatal_not_regenerated(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner = FakeStudent(artifact)
    try:
        state = initial_state(task, "symbolic", 42)
        state["artifact_refs"] = ["0" * 64]
        run.save(state_key(task.task_id, "symbolic", 42), state)
        with pytest.raises(FileNotFoundError):
            advance_stage(task, "symbolic", 42, V8Config(), run, reasoner)
        assert not reasoner.calls
    finally:
        run.close()


def test_deadline_prevents_dispatch(tmp_path, task, artifact):
    run = ExperimentRun(tmp_path, {"v": 8})
    reasoner = FakeStudent(artifact)
    try:
        state = advance_stage(
            task, "direct", 42, V8Config(), run, reasoner, global_deadline=time.monotonic() - 1
        )
        assert state["status"] == "timed_out" and not reasoner.calls
    finally:
        run.close()


def test_verifier_refuses_test_oracle(task):
    with pytest.raises(ValueError, match="test labels"):
        check_witness(IDENTITY, task)
    result, feedback, candidate = check_witness(IDENTITY, blind_task(task))
    assert result["accepted"] and not feedback["failures"]
    assert candidate["predictions"] == [[[3]]]


def test_full_denominator_and_inflight_dispatch_reporting(tmp_path, task):
    run = ExperimentRun(tmp_path, {"v": 8})
    cfg = V8Config()
    try:
        key = state_key(task.task_id, "symbolic", 42)
        run.save(key, initial_state(task, "symbolic", 42))
        run.save(key + ":0", {"dispatched_at": time.time(), "response": None})
        report = write_report(run, [task], cfg, elapsed=5, evidence={})
        metrics = report["metrics"]["symbolic:42"]
        assert metrics["dispatches"] == metrics["unknown_usage_calls"] == 1
        assert metrics["valid_response_rate"] == 0
        assert metrics["test_outputs"] == 1 and metrics["correct_outputs"] == 0
        assert not report["complete"]
        assert len(list(tmp_path.glob("submission-*.json"))) == 6
        with pytest.raises(ValueError, match="entire fixed cohort"):
            score_submission([task], {})
    finally:
        run.close()


def test_compact_student_targets_unchanged_and_masked(tmp_path, task, artifact):
    source, destination = tmp_path / "source", tmp_path / "view"
    example = {
        "prompt": student_prompt(blind_task(task)),
        "completion": [{"role": "assistant", "content": artifact.symbolic.model_dump_json()}],
        "source_task_id": task.task_id,
        "category": "interpretation",
        "record_ref": "a" * 64,
    }
    atomic_json(source / "symbolic-sft.json", [example])
    atomic_json(
        source / "dataset-report.json",
        {
            "dataset_hash": content_hash([example]),
            "teacher": "gpt-6-astra",
            "student": "Nanbeige/Nanbeige4.2-3B",
        },
    )
    atomic_json(tmp_path / "split.json", {"groups": {"training": [task.task_id]}})
    report = prepare_compact_student(
        source, destination, tmp_path / "split.json", Path(".runtime/nanbeige/model-hf")
    )
    exported = json.loads((destination / "symbolic-sft.json").read_text())
    assert exported[0]["completion"] == example["completion"]
    assert report["loss_mask_data_gate_passed"] and report["view_manifest"]["target_unchanged"]
    rows = json.loads((destination / "nanbeige-symbolic-tokenized.json").read_text())
    assert set(rows[0]["labels"][: rows[0]["prompt_tokens"]]) == {-100}
    atomic_json(tmp_path / "split.json", {"groups": {"training": []}})
    with pytest.raises(ValueError, match="lockbox"):
        prepare_compact_student(
            source, tmp_path / "bad", tmp_path / "split.json", Path(".runtime/nanbeige/model-hf")
        )


def test_native_symbolic_contract_has_empty_thinking_prefix_and_no_shrink(task):
    config = V4Config(context_tokens=8192)
    reasoner = object.__new__(StudentReasoner)
    reasoner.config = config
    reasoner.template = NativeTemplate(Path(".runtime/nanbeige/model-hf/tokenizer_config.json"))
    length = [100]

    def transport(request):
        assert request.url.path == "/tokenize"
        return httpx.Response(200, json={"tokens": [1] * length[0]})

    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(transport)
    )
    try:
        result = reasoner.prepare_stage(
            symbolic_prompt(task), seed=42, task=blind_task(task), stage="symbolic"
        )
        assert "</think>" in result["_rendered_prompt"]
        assert "json_schema" in result and "grammar_lazy" not in result
        assert result["n_predict"] == 2048
        witness = reasoner.prepare_stage(
            witness_prompt(task, {}), seed=42, task=blind_task(task), stage="witness"
        )
        assert witness["grammar"] == PYTHON_GRAMMAR and witness["reasoning_budget_tokens"] == 384
        length[0] = 7000
        with pytest.raises(ContextOverflow):
            reasoner.prepare_stage(
                symbolic_prompt(task), seed=42, task=blind_task(task), stage="symbolic"
            )
    finally:
        reasoner.close()


@pytest.mark.parametrize(
    "mutation",
    [{"seeds": [42]}, {"model": "gpt-6-astra"}, {"modes": ["symbolic"]}, {"budget_usd": 20}],
)
def test_comparison_config_fails_closed(mutation):
    with pytest.raises(ValidationError):
        V8Config(**mutation)
