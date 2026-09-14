"""Packed symbolic output remains canonical before grounding, repair and witnessing."""

import copy
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from test_nanbeige_baseline_audit import audit
from test_nanbeige_cell_codec import codec
from test_v7 import artifact as artifact
from test_v7 import task as task
from test_v8 import FakeStudent
from test_v8_grammar import CONVERTER
from test_v8_grammar import accepts as accepts

from arc_agent.v4_adaptive import AdaptiveSession
from arc_agent.v4_config import V4Config
from arc_agent.v4_state import ExperimentRun
from arc_agent.v8_cli import experiment_binding, matching_smoke
from arc_agent.v8_codec import (
    TRIPLES_FORMAT,
    TRIPLES_INSTRUCTION,
    pack_symbolic_cells,
    packed_symbolic_schema,
    unpack_symbolic_cells,
)
from arc_agent.v8_config import V8Config
from arc_agent.v8_grammar import compact_symbolic_grammar
from arc_agent.v8_reasoner import StudentReasoner, StudentSession, grounded_schema, symbolic_prompt
from arc_agent.v8_runner import advance_stage, state_key


class PackedStudent(FakeStudent):
    def prepare_stage(self, messages, *, seed, task, stage):
        prepared = super().prepare_stage(messages, seed=seed, task=task, stage=stage)
        prepared.update(_stage=stage, _prompt_tokens=50)
        if stage == "symbolic":
            prepared["_response_format"] = TRIPLES_FORMAT
        return prepared

    def complete(self, prepared, *, timeout):
        response = super().complete(prepared, timeout=timeout)
        if prepared["stage"] == "symbolic":
            response["final"] = json.dumps(
                pack_symbolic_cells(self.artifact.symbolic.model_dump(mode="json"))
            )
        response["raw"] = {
            "content": ("" if prepared["stage"] == "symbolic" else "</think>") + response["final"],
            "tokens_predicted": response["generated_tokens"],
            "truncated": False,
        }
        return response


def test_codec_matches_measured_prototype_exactly(task, artifact, accepts):
    state = artifact.symbolic.model_dump(mode="json")
    packed = pack_symbolic_cells(state)
    assert packed == codec.packed_state(state)
    assert unpack_symbolic_cells(packed) == codec.unpacked_state(packed) == state
    schema = packed_symbolic_schema(grounded_schema(task))
    assert schema == codec.packed_schema(grounded_schema(task))
    packed["version"] = packed.pop("version")
    assert accepts(
        compact_symbolic_grammar(schema, CONVERTER), json.dumps(packed, separators=(",", ":"))
    ), accepts.last_result


def test_packed_output_decodes_before_grounding_witnessing_and_audit(tmp_path, task, artifact):
    cfg = V8Config(symbolic_cell_encoding="triples_v1")
    student = PackedStudent(artifact)
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        state = advance_stage(task, "symbolic", 42, cfg, run, student)
        assert state["symbolic"] == artifact.symbolic.model_dump(mode="json")
        assert state["grounded_states"] == 1 and state["stage"] == "witness"
        artifacts = [run.artifacts.get(ref) for ref in state["artifact_refs"]]
        decoding = [a for a in artifacts if a.get("kind") == "symbolic_cell_decode"]
        assert len(decoding) == 1 and decoding[0]["canonical"] == state["symbolic"]
        assert run.artifacts.get(decoding[0]["source_response_ref"])["final"] != json.dumps(
            state["symbolic"]
        )
        state = advance_stage(task, "symbolic", 42, cfg, run, student)
        assert state["status"] == "complete"
        evidence = json.loads(student.calls[-1]["messages"][-1]["content"])
        assert evidence["symbolic_model"] == state["symbolic"]
        assert isinstance(evidence["symbolic_model"]["entities"][0]["cells"][0], dict)
        key = state_key(task.task_id, "symbolic", 42)
        work = {f"{key}:{step}": run.get(f"{key}:{step}") for step in range(state["step"])}
        stats = audit.state_statistics(
            task.task_id,
            "symbolic",
            42,
            state,
            work,
            tmp_path / "artifacts",
            outputs=1,
            symbolic_cell_encoding="triples_v1",
        )
        assert stats["valid_responses"] == stats["dispatches"] == 2
        with pytest.raises(ValueError, match="differs from the frozen protocol"):
            audit.state_statistics(
                task.task_id, "symbolic", 42, state, work, tmp_path / "artifacts", outputs=1
            )
        altered = copy.deepcopy(state)
        altered["artifact_refs"].remove(audit.digest(decoding[0]))
        with pytest.raises(ValueError, match="decoding evidence"):
            audit.state_statistics(
                task.task_id,
                "symbolic",
                42,
                altered,
                work,
                tmp_path / "artifacts",
                outputs=1,
                symbolic_cell_encoding="triples_v1",
            )


def test_packed_cells_do_not_bypass_grounding(tmp_path, task, artifact):
    artifact.symbolic.entities[0].cells[0].color = 9
    with closing(ExperimentRun(tmp_path, {"v": 8})) as run:
        state = advance_stage(
            task,
            "symbolic",
            42,
            V8Config(symbolic_cell_encoding="triples_v1"),
            run,
            PackedStudent(artifact),
        )
        assert state["valid_responses"] == 1 and state["grounded_states"] == 0
        assert "contradicts" in state["feedback"]["error"]


def test_packed_saved_response_replays_without_duplicate_generation(tmp_path, task, artifact):
    cfg, student = V8Config(symbolic_cell_encoding="triples_v1"), PackedStudent(artifact)
    run = ExperimentRun(tmp_path, {"v": 8})
    save = run.save

    def interrupt(key, value):
        save(key, value)
        if value.get("response"):
            raise KeyboardInterrupt

    run.save = interrupt
    with pytest.raises(KeyboardInterrupt):
        advance_stage(task, "symbolic", 42, cfg, run, student)
    run.close()
    with closing(ExperimentRun(tmp_path, {"v": 8}, resume=True)) as run:
        state = advance_stage(task, "symbolic", 42, cfg, run, student)
        assert state["grounded_states"] == 1 and len(student.calls) == 1
        assert state["request_count"] == state["valid_responses"] == 1
        assert state["symbolic"] == artifact.symbolic.model_dump(mode="json")


def test_native_packed_request_preserves_raw_reply_and_audit_contract(task, artifact):
    manifest = json.loads(Path(".runtime/nanbeige/manifest.json").read_text())
    reasoner = StudentReasoner(
        V4Config(),
        manifest,
        structured_style="reference_repair_bounded_ws",
        symbolic_cell_encoding="triples_v1",
    )
    reasoner.client.close()
    payload = pack_symbolic_cells(artifact.symbolic.model_dump(mode="json"))
    final = json.dumps(payload)

    def transport(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1] * 100})
        assert body["n_predict"] == 2048 and "grammar_lazy" not in body
        assert "grammar" in body and not any(k.startswith("_") for k in body)
        return httpx.Response(
            200, json={"content": final, "tokens_predicted": 100, "truncated": False}
        )

    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(transport)
    )
    try:
        messages = symbolic_prompt(task)
        original = copy.deepcopy(messages)
        prepared = reasoner.prepare_stage(messages, seed=42, task=task, stage="symbolic")
        assert messages == original and prepared["_response_format"] == TRIPLES_FORMAT
        assert TRIPLES_INSTRUCTION in prepared["_rendered_prompt"]
        response = reasoner.complete(prepared, timeout=10)
        assert response["final"] == final and response["raw"]["content"] == final
        call = {"messages": messages, "prepared": prepared, "response": response}
        assert audit.valid_response(call, 1)
        named = artifact.symbolic.model_dump(mode="json")
        call["response"]["final"] = call["response"]["raw"]["content"] = json.dumps(named)
        assert not audit.valid_response(call, 1)
    finally:
        reasoner.close()


def test_context_expansion_keeps_cell_encoding(monkeypatch):
    import arc_agent.v8_reasoner as module

    session = object.__new__(StudentSession)
    session.symbolic_cell_encoding = "triples_v1"
    session.structured_style = "reference_repair_bounded_ws"
    session.manifest = {}
    cfg = object()

    def start(self, context):
        self.reasoner = SimpleNamespace(config=cfg, close=lambda: None)

    captures = []

    def reasoner(config, manifest, **kwargs):
        captures.append(kwargs)
        return SimpleNamespace(config=config)

    monkeypatch.setattr(AdaptiveSession, "_start", start)
    monkeypatch.setattr(module, "StudentReasoner", reasoner)
    session._start(8192)
    session._start(12288)
    assert all(c["symbolic_cell_encoding"] == "triples_v1" for c in captures)


@pytest.mark.parametrize(
    "config_encoding,identity_encoding,valid",
    [
        (None, None, True),
        ("triples_v1", "triples_v1", True),
        ("triples_v1", None, False),
        (None, "triples_v1", False),
        ("lossy", "lossy", False),
    ],
)
def test_cell_encoding_identity_is_explicit(tmp_path, config_encoding, identity_encoding, valid):
    identity = {"config": {}}
    if config_encoding is not None:
        identity["config"]["symbolic_cell_encoding"] = config_encoding
    if identity_encoding is not None:
        identity["symbolic_cell_encoding"] = identity_encoding
    if valid:
        audit.verify_format_archive(tmp_path, identity)
    else:
        with pytest.raises(ValueError, match="symbolic cell encoding identity"):
            audit.verify_format_archive(tmp_path, identity)


def test_smoke_encoding_isolation_and_unknown_encoding_rejection(monkeypatch):
    import arc_agent.v8_cli as cli

    monkeypatch.setattr(cli, "source_hash", lambda: "source")
    monkeypatch.setattr(cli, "file_hash", lambda path: "manifest")
    runtime = V4Config()
    named = experiment_binding(V8Config(), runtime, {})
    packed = experiment_binding(V8Config(symbolic_cell_encoding="triples_v1"), runtime, {})
    report = {**named, "passed": True, "context_tokens": 4096}
    assert not matching_smoke(report, packed, 4096)
    with pytest.raises(ValidationError):
        V8Config(symbolic_cell_encoding="lossy")
    with pytest.raises(ValueError, match="unknown symbolic cell encoding"):
        StudentReasoner(V4Config(), {}, symbolic_cell_encoding="lossy")
