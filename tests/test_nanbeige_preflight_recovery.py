"""Narrow pre-model exception must never conceal inference or telemetry failures."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "preflight_recovery", Path(__file__).parents[1] / "scripts/recover_nanbeige_preflight.py"
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def empty_segment():
    return {
        "context_tokens": 12288,
        "path": "runs/v8/untrained-output4k-baseline-01/runtime-segments/001-context-12288",
        "offline_enforced": False,
        "metal_verified": False,
        "memory_safe": False,
        "prompt_cache_mib": 0,
        "prompt_cache_disabled_verified": False,
        "warning_pressure_policy": "record",
        "warning_pressure_samples": 0,
        "guard_error": None,
        "peak_server_rss_bytes": 0,
        "swap_growth_bytes": 0,
        "peak_pressure_level": 0,
        "initial_pressure_level": None,
        "initial_swap_used_bytes": None,
        "sample_count": 0,
    }


def test_only_exact_empty_preflight_is_classifiable(tmp_path):
    segment = empty_segment()
    before = copy.deepcopy(segment)
    recovery.no_execution_segment(tmp_path, segment)
    assert segment == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_count", 1),
        ("peak_server_rss_bytes", 4096),
        ("initial_pressure_level", 1),
        ("initial_swap_used_bytes", 0),
        ("guard_error", "OOM"),
        ("memory_safe", True),
        ("offline_enforced", True),
        ("context_tokens", 16384),
        ("swap_growth_bytes", 1),
        ("warning_pressure_policy", "stop"),
        ("unexpected", True),
        ("path", "runs/v8/another/runtime-segments/001-context-12288"),
    ],
)
def test_rejects_nonexact_or_executed_segment(tmp_path, field, value):
    segment = empty_segment()
    segment[field] = value
    with pytest.raises(ValueError, match="exact approved"):
        recovery.no_execution_segment(tmp_path, segment)


def test_rejects_runtime_files_even_if_counters_are_zero(tmp_path):
    (tmp_path / "runtime-segments/001-context-12288").mkdir(parents=True)
    with pytest.raises(ValueError, match="runtime files"):
        recovery.no_execution_segment(tmp_path, empty_segment())


def test_rejects_dangling_segment_symlink(tmp_path):
    folder = tmp_path / "runtime-segments"
    folder.mkdir()
    (folder / "001-context-12288").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="runtime files"):
        recovery.no_execution_segment(tmp_path, empty_segment())


def proofs(monkeypatch):
    work = {"original": {"response": "preserved"}}
    monkeypatch.setattr(recovery, "REPLAY_WORK", recovery.audit.digest(work))
    before = {
        "identity_hash": recovery.IDENTITY,
        "provisional_live_snapshot": False,
        "metrics": {"direct:42": {"unreconciled_dispatches": 0, "dispatches": 20}},
    }
    after = copy.deepcopy(before)
    replay = {
        "identity_hash": recovery.IDENTITY,
        "provisional_live_snapshot": False,
        "stages_replayed": 40,
        "states_reproduced": 40,
        "new_model_calls": 0,
        "work_snapshot_hash": recovery.REPLAY_WORK,
    }
    return before, after, replay, work


def test_proofs_require_unchanged_exact_work(monkeypatch):
    values = proofs(monkeypatch)
    recovery.validate_proofs(*values)
    values[3]["extra_dispatch"] = {}
    with pytest.raises(ValueError, match="snapshot changed"):
        recovery.validate_proofs(*values)


@pytest.mark.parametrize(
    "mutation", ["new_call", "pending", "different_identity", "live", "new_inference"]
)
def test_rejects_ambiguous_or_changed_proof(monkeypatch, mutation):
    before, after, replay, work = proofs(monkeypatch)
    if mutation == "new_call":
        after["metrics"]["direct:42"]["dispatches"] += 1
    elif mutation == "pending":
        before["metrics"]["direct:42"]["unreconciled_dispatches"] = 1
        after = copy.deepcopy(before)
    elif mutation == "different_identity":
        replay["identity_hash"] = "another"
    elif mutation == "live":
        after["provisional_live_snapshot"] = True
    else:
        replay["new_model_calls"] = 1
    with pytest.raises(ValueError):
        recovery.validate_proofs(before, after, replay, work)


def test_create_only_output_does_not_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    recovery.write_new(path, {"proof": 1})
    with pytest.raises(FileExistsError):
        recovery.write_new(path, {"proof": 2})
    assert json.loads(path.read_text()) == {"proof": 1}


def test_audit_requires_externally_pinned_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "review", lambda **kw: pytest.fail("receipt must be pinned"))
    monkeypatch.setattr("sys.argv", ["recovery", "audit", "--output", str(tmp_path / "new.json")])
    with pytest.raises(ValueError, match="receipt hash"):
        recovery.main()


def receipt_fixture(tmp_path):
    work = {f"call:{i}": {"response": i} for i in range(40)}
    work.update({f"terminal:{i}": {"status": "complete"} for i in range(20)})
    for name in recovery.PROOFS:
        (tmp_path / name).write_text("sealed evidence")
    segments = [{"original": "preserved"}, empty_segment()]
    receipt = {
        "schema": "approved_pre_model_recovery_v1",
        "identity_hash": recovery.IDENTITY,
        "recovery_script_sha256": recovery.audit.file_hash(recovery.__file__),
        "proof_hashes": {p: recovery.audit.file_hash(tmp_path / p) for p in recovery.PROOFS},
        "original_segments": copy.deepcopy(segments),
        "completed_call_hashes": {
            k: recovery.audit.digest(v) for k, v in work.items() if k.startswith("call:")
        },
        "terminal_state_hashes": {
            k: recovery.audit.digest(v) for k, v in work.items() if k.startswith("terminal:")
        },
    }
    return receipt, work, segments


def test_receipt_preserves_completed_calls_and_original_segments(tmp_path):
    receipt, work, segments = receipt_fixture(tmp_path)
    original = copy.deepcopy((receipt, work, segments))
    recovery.validate_receipt(tmp_path, receipt, work, segments)
    assert (receipt, work, segments) == original


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "script",
        "proof",
        "proof_inventory",
        "segment",
        "call",
        "call_inventory",
        "terminal",
        "terminal_inventory",
    ],
)
def test_receipt_rejects_changed_evidence(tmp_path, mutation):
    receipt, work, segments = receipt_fixture(tmp_path)
    if mutation == "identity":
        receipt["identity_hash"] = "another"
    elif mutation == "script":
        receipt["recovery_script_sha256"] = "another"
    elif mutation == "proof":
        (tmp_path / recovery.PROOFS[0]).write_text("changed")
    elif mutation == "proof_inventory":
        receipt["proof_hashes"].pop(recovery.PROOFS[0])
    elif mutation == "segment":
        segments[0]["original"] = "changed"
    elif mutation == "call":
        work["call:0"]["response"] = "changed"
    elif mutation == "call_inventory":
        receipt["completed_call_hashes"].pop("call:0")
    elif mutation == "terminal":
        work["terminal:0"]["status"] = "running"
    else:
        receipt["terminal_state_hashes"].pop("terminal:0")
    with pytest.raises(ValueError):
        recovery.validate_receipt(tmp_path, receipt, work, segments)


def test_resume_requires_offline_wrapper_before_review(monkeypatch, tmp_path):
    import arc_agent.v4_runtime as runtime

    monkeypatch.setattr(runtime, "external_network_denied", lambda: False)
    monkeypatch.setattr(recovery, "review", lambda **kw: pytest.fail("must not touch run"))
    monkeypatch.setattr("sys.argv", ["recovery", "resume", "--output", str(tmp_path / "new.json")])
    with pytest.raises(ValueError, match="run untouched"):
        recovery.main()
    assert not (tmp_path / "new.json").exists()


def execution_fixture(tmp_path):
    from arc_agent.v4_cli import runtime_evidence

    path = tmp_path / "runtime-segments/000-context-12288"
    path.mkdir(parents=True)
    samples = [{"pressure_level": 1, "swap_used_bytes": 1000, "server_rss_bytes": 5000}]
    baseline = {**samples[0], "warning_pressure_policy": "record"}
    config = SimpleNamespace(
        max_swap_growth_gib=1.0, prompt_cache_mib=0, warning_pressure_policy="record"
    )
    (path / "memory.jsonl").write_text(json.dumps(samples[0]) + "\n")
    (path / "memory-baseline.json").write_text(json.dumps(baseline))
    (path / "server.log").write_text(
        "ggml_metal_init offloaded 45/45 layers to GPU\nprompt cache is disabled\n"
    )
    server = SimpleNamespace(
        samples=samples,
        baseline=baseline,
        log_start_offset=0,
        guard_error=None,
        config=config,
        offline_verified=True,
    )
    segment = {"context_tokens": 12288, "path": str(path), **runtime_evidence(server, path)}
    return path, config, segment


def test_actual_execution_reconstructed_without_mutation(tmp_path):
    _, config, segment = execution_fixture(tmp_path)
    before = copy.deepcopy(segment)
    recovery.inspect_execution(tmp_path, [segment, empty_segment()], config)
    assert segment == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "critical",
        "unknown",
        "nan",
        "swap",
        "count",
        "guard",
        "offline",
        "metal",
        "extra_empty",
    ],
)
def test_other_runtime_failure_is_never_excluded(tmp_path, mutation):
    path, config, segment = execution_fixture(tmp_path)
    sample = json.loads((path / "memory.jsonl").read_text())
    segments = [segment, empty_segment()]
    if mutation == "missing":
        sample["server_rss_bytes"] = None
    elif mutation == "critical":
        sample["pressure_level"] = 4
    elif mutation == "unknown":
        sample["pressure_level"] = 0
    elif mutation == "nan":
        sample["swap_used_bytes"] = float("nan")
    elif mutation == "swap":
        sample["swap_used_bytes"] += 2 * 1024**3
    elif mutation == "count":
        segment["sample_count"] = 0
    elif mutation == "guard":
        segment["guard_error"] = "disk failure"
    elif mutation == "offline":
        segment["offline_enforced"] = False
    elif mutation == "metal":
        (path / "server.log").write_text("no GPU\n")
    else:
        segments.append(
            {**empty_segment(), "path": str(tmp_path / "runtime-segments/002-context-12288")}
        )
    (path / "memory.jsonl").write_text(json.dumps(sample) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.inspect_execution(tmp_path, segments, config)
