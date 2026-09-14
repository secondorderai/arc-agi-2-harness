"""Reference patches are model choices over an unchanged base, never implicit corrections."""

import copy
import json
from pathlib import Path

import httpx
import pytest
from test_v7 import artifact as artifact
from test_v7 import task as task
from test_v8 import FakeStudent
from test_v8_grammar import CONVERTER
from test_v8_grammar import accepts as accepts

from arc_agent.v4_config import V4Config, content_hash
from arc_agent.v4_state import ExperimentRun
from arc_agent.v5_symbolic import SymbolicModel, validate_symbols
from arc_agent.v8_config import V8Config
from arc_agent.v8_grammar import compact_symbolic_grammar
from arc_agent.v8_reasoner import StudentReasoner, symbolic_prompt
from arc_agent.v8_reference_repair import (
    apply_reference_patch,
    broken_references,
    prepare_reference_repair,
    reference_patch_schema,
)
from arc_agent.v8_runner import advance_stage, state_key


@pytest.fixture
def base(artifact):
    value = artifact.symbolic.model_dump(mode="json")
    value["hypotheses"][0]["ordered_actions"] = ["apply_preserve"]
    return value


@pytest.fixture
def patch():
    return {"hypotheses": [{"id": "h", "ordered_actions": ["preserve"]}]}


def test_patch_changes_only_broken_fields(base, patch, task):
    original = copy.deepcopy(base)
    result = apply_reference_patch(base, patch, content_hash(base))
    assert base == original
    expected = copy.deepcopy(base)
    expected["hypotheses"][0]["ordered_actions"] = ["preserve"]
    assert result == expected
    validate_symbols(SymbolicModel.model_validate(result), task)
    assert broken_references(base) == {"h": {"ordered_actions"}}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["hypotheses"][0].update(ordered_actions=["invented"]),
        lambda p: p["hypotheses"][0].update(concepts=[]),
        lambda p: p["hypotheses"][0].update(ordered_actions=[]),
        lambda p: p["hypotheses"][0].update(id="wrong"),
        lambda p: p["hypotheses"].append(copy.deepcopy(p["hypotheses"][0])),
        lambda p: p.update(hypotheses=[]),
        lambda p: p.update(definitions=[]),
        lambda p: p["hypotheses"][0].update(statement="silently changed"),
        lambda p: p["hypotheses"][0].update(ordered_actions=None),
        lambda p: p["hypotheses"][0].pop("ordered_actions"),
    ],
)
def test_invalid_or_expansive_patch_rejected(base, patch, mutate):
    mutate(patch)
    with pytest.raises(ValueError):
        apply_reference_patch(base, patch, content_hash(base))


def test_stale_base_fails_closed(base, patch):
    expected = content_hash(base)
    base["unresolved"].append("new evidence")
    with pytest.raises(RuntimeError, match="stale-state"):
        apply_reference_patch(base, patch, expected)


def test_native_patch_grammar_uses_declared_names_and_preserves_good_fields(accepts, base, patch):
    grammar = compact_symbolic_grammar(reference_patch_schema(base), CONVERTER)
    assert accepts(grammar, json.dumps(patch, separators=(",", ":"))), accepts.last_result
    for value in ["invented", "apply preserve"]:
        patch["hypotheses"][0]["ordered_actions"] = [value]
        assert not accepts(grammar, json.dumps(patch, separators=(",", ":")))
    patch["hypotheses"][0].update(ordered_actions=["preserve"], concepts=[])
    assert not accepts(grammar, json.dumps(patch, separators=(",", ":")))


def test_prompt_patch_keeps_raw_evidence_and_immutable_state(base, task):
    prompt = symbolic_prompt(task, base, {"error": "undefined symbolic concept"})
    original = copy.deepcopy(prompt)
    prepared = prepare_reference_repair(prompt)
    assert prompt == original
    assert prepared["messages"][-1] == prompt[-1]
    assert prepared["metadata"]["_reference_base"] == base
    assert prepared["metadata"]["_reference_base_sha256"] == content_hash(base)
    assert "Repair ONLY" in prepared["messages"][0]["content"]
    assert "test_0_output" not in json.loads(prepared["messages"][-1]["content"])["grids"]
    task.test[0].output = [[9, 8, 7]]
    assert (
        prepare_reference_repair(
            symbolic_prompt(task, base, {"error": "undefined symbolic concept"})
        )
        == prepared
    )


def test_only_hypothesis_reference_failures_are_eligible(base, task):
    assert prepare_reference_repair(symbolic_prompt(task)) is None
    assert (
        prepare_reference_repair(
            symbolic_prompt(task, base, {"error": "entity contradicts authoritative cell"})
        )
        is None
    )
    base["definitions"][0]["depends_on"] = ["missing_definition"]
    assert not broken_references(base)
    assert (
        prepare_reference_repair(
            symbolic_prompt(task, base, {"error": "undefined symbolic concept"})
        )
        is None
    )
    with pytest.raises(ValueError, match="no eligible"):
        reference_patch_schema(base)


def test_partial_multiple_hypotheses_and_meanings_stay_unchanged(base, patch):
    base["hypotheses"].append(
        {**copy.deepcopy(base["hypotheses"][0]), "id": "valid", "ordered_actions": ["preserve"]}
    )
    result = apply_reference_patch(base, patch, content_hash(base))
    assert result["hypotheses"][1] == base["hypotheses"][1]
    assert result["definitions"] == base["definitions"]
    assert result["entities"] == base["entities"]


def test_patched_names_do_not_certify_grounding(base, patch, task):
    base["entities"][0]["cells"][0]["color"] = 9
    result = apply_reference_patch(base, patch, content_hash(base))
    with pytest.raises(ValueError, match="contradicts"):
        validate_symbols(SymbolicModel.model_validate(result), task)


class PatchStudent(FakeStudent):
    def prepare_stage(self, messages, *, seed, task, stage):
        prepared = super().prepare_stage(messages, seed=seed, task=task, stage=stage)
        repair = prepare_reference_repair(messages) if stage == "symbolic" else None
        if repair:
            prepared.update(repair["metadata"])
        return prepared

    def complete(self, prepared, *, timeout):
        if prepared.get("_response_format") != "symbolic_reference_patch":
            return super().complete(prepared, timeout=timeout)
        self.calls.append(prepared)
        text = json.dumps({"hypotheses": [{"id": "h", "ordered_actions": ["preserve"]}]})
        return {
            "final": text,
            "reasoning": "",
            "prompt_tokens": 50,
            "generated_tokens": 100,
            "elapsed_seconds": 0.01,
            "raw": {"content": text},
        }


@pytest.mark.parametrize("crash_after_patch_response", [False, True])
def test_patch_witness_resume_and_accounting(tmp_path, task, artifact, crash_after_patch_response):
    artifact.symbolic.hypotheses[0].ordered_actions = ["apply_preserve"]
    reasoner = PatchStudent(artifact)
    config = V8Config(structured_style="reference_repair")
    run = ExperimentRun(tmp_path, {"protocol": "reference_repair"})
    try:
        first = advance_stage(task, "symbolic", 42, config, run, reasoner)
        assert first["grounded_states"] == 0 and first["valid_responses"] == 1
        if crash_after_patch_response:
            save = run.save

            def crashing_save(key, value):
                save(key, value)
                if value.get("response") and value.get("prepared", {}).get("_response_format"):
                    raise KeyboardInterrupt

            run.save = crashing_save
            with pytest.raises(KeyboardInterrupt):
                advance_stage(task, "symbolic", 42, config, run, reasoner)
            run.close()
            run = ExperimentRun(tmp_path, {"protocol": "reference_repair"}, resume=True)
        state = advance_stage(task, "symbolic", 42, config, run, reasoner)
        assert state["stage"] == "witness" and state["grounded_states"] == 1
        assert state["valid_responses"] == state["request_count"] == 2
        patches = [run.artifacts.get(ref) for ref in state["artifact_refs"]]
        assert sum(p.get("kind") == "symbolic_reference_patch" for p in patches) == 1
        state = advance_stage(task, "symbolic", 42, config, run, reasoner)
        assert state["status"] == "complete" and len(reasoner.calls) == 3
        assert state["generated_tokens"] == 300 and state["valid_responses"] == 3
        assert advance_stage(task, "symbolic", 42, config, run, reasoner) == state
        assert run.get(state_key(task.task_id, "symbolic", 42)) == state
    finally:
        run.close()


def test_both_broken_fields_must_be_supplied(base, patch):
    base["hypotheses"][0]["concepts"] = ["also_missing"]
    with pytest.raises(ValueError, match="omits a broken"):
        apply_reference_patch(base, patch, content_hash(base))
    patch["hypotheses"][0]["concepts"] = ["preserve"]
    assert not broken_references(apply_reference_patch(base, patch, content_hash(base)))


@pytest.mark.parametrize("style", ["reference_repair", "reference_repair_bounded_ws"])
@pytest.mark.parametrize("cell_encoding", ["named_fields", "triples_v1"])
@pytest.mark.parametrize("output_tokens", [2048, 4096])
def test_native_request_response_and_independent_audit_share_patch_contract(
    base, patch, task, style, cell_encoding, output_tokens
):
    from test_nanbeige_baseline_audit import audit

    manifest = json.loads(Path(".runtime/nanbeige/manifest.json").read_text())
    reasoner = StudentReasoner(
        V4Config(context_tokens=12288, max_new_tokens=output_tokens),
        manifest,
        structured_style=style,
        symbolic_cell_encoding=cell_encoding,
    )
    reasoner.client.close()

    def transport(request):
        data = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1] * 100})
        assert request.url.path == "/completion"
        assert not any(key.startswith("_") for key in data)
        assert (
            data["n_predict"] == output_tokens and "grammar" in data and "grammar_lazy" not in data
        )
        return httpx.Response(
            200, json={"content": json.dumps(patch), "tokens_predicted": 40, "truncated": False}
        )

    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(transport)
    )
    try:
        messages = symbolic_prompt(task, base, {"error": "undefined symbolic concept"})
        prepared = reasoner.prepare_stage(messages, seed=43, task=task, stage="symbolic")
        assert prepared["_response_format"] == "symbolic_reference_patch"
        assert prepared["_reference_base_sha256"] == content_hash(base)
        assert (
            "Repair ONLY" in prepared["_rendered_prompt"]
            and "</think>" in prepared["_rendered_prompt"]
        )
        response = reasoner.complete(prepared, timeout=10)
        assert response["reasoning"] == "" and json.loads(response["final"]) == patch
        call = {"prepared": prepared, "response": response, "messages": messages}
        assert audit.valid_response(call, 1)
        broken = copy.deepcopy(call)
        broken["prepared"]["_reference_base"]["unresolved"].append("changed")
        with pytest.raises(ValueError, match="authoritative input"):
            audit.valid_response(broken, 1)
        bad_payload = {"hypotheses": [{"id": "h", "ordered_actions": ["invented"]}]}
        call["response"]["final"] = json.dumps(bad_payload)
        call["response"]["raw"]["content"] = json.dumps(bad_payload)
        assert not audit.valid_response(call, 1)
    finally:
        reasoner.close()
