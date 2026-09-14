"""Local-only assembly fixtures. Synthetic approvals/evidence never authorize a real job."""

import copy
import fcntl
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_nanbeige_training_preflight as evidence_fixtures

validated_fixture = evidence_fixtures.bundle

SPEC = importlib.util.spec_from_file_location(
    "bundle_builder", Path(__file__).parents[1] / "scripts/prepare_nanbeige_bundle.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def inputs(validated_fixture, monkeypatch):
    root, _ = validated_fixture
    student, baseline = root / "student", root / "baseline"
    student.mkdir()
    baseline.mkdir()
    (baseline / ".lock").touch()
    for name in (
        "symbolic-sft.json",
        "nanbeige-symbolic-tokenized.json",
        "student-data-report.json",
    ):
        shutil.copyfile(root / name, student / name)
    shutil.copyfile(root / "source-dataset-report.json", student / "dataset-report.json")
    shutil.copyfile(root / "baseline-identity.json", baseline / "identity.json")
    shutil.copyfile(root / "baseline-report.json", baseline / "report.json")
    (student / "submission.json").write_text("not permitted in a training bundle")

    def fake_native(folder, model_dir):
        examples, rows, _ = builder.preflight.validate_data_evidence(folder)
        return {
            "examples": len(examples),
            "schedule_draws": 2 * len(rows),
            "native_tokenization_verified": True,
            "data_evidence_verified": True,
            "weights_loaded": False,
            "training_launched": False,
            "upload_performed": False,
        }

    monkeypatch.setattr(builder, "verify_native", fake_native)
    monkeypatch.setattr(
        builder,
        "audit_stopped_baseline",
        lambda *args: json.loads((root / "baseline-audit.json").read_text()),
    )
    return {
        "student_data": student,
        "curriculum_audit": root / "curriculum-audit.json",
        "trainer_probe": root / "trainer-probe.json",
        "split": root / "split.json",
        "baseline": baseline,
        "destination": root / "draft",
        "model_dir": root / "model",
    }


def seal_args(inputs):
    draft = inputs["destination"]
    return {
        "draft": draft,
        "draft_hash": builder.preflight.file_hash(draft / "draft-manifest.json"),
        "baseline": inputs["baseline"],
        "data": draft.parent / "data",
        "destination": draft.parent / "sealed",
        "model_dir": inputs["model_dir"],
    }


def test_data_draft_is_exact_and_not_a_training_gate(inputs):
    before = inputs["student_data"].joinpath("symbolic-sft.json").read_bytes()
    draft = builder.stage(**inputs)
    assert draft["baseline_gate_passed"] is False
    assert draft["compute_approval_ref"] is None and draft["output_use_permission_ref"] is None
    assert draft["training_launched"] is False and draft["upload_performed"] is False
    assert set(draft["files"]) == builder.preflight.DATA_EVIDENCE_FILES | builder.SCRIPT_FILES
    assert not (inputs["destination"] / "bundle-manifest.json").exists()
    assert not (inputs["destination"] / "submission.json").exists()
    assert (inputs["destination"] / "symbolic-sft.json").read_bytes() == before
    assert builder.verify_draft(inputs["destination"], seal_args(inputs)["draft_hash"]) == draft


def test_sealed_evidence_still_needs_explicit_approval(inputs):
    builder.stage(**inputs)
    args = seal_args(inputs)
    receipt = builder.seal(**args)
    assert receipt["baseline_gate_passed"] and receipt["job_launch_authorized"] is False
    assert receipt["status"] == "evidence_sealed_pending_approval"
    manifest, _, _ = builder.preflight.validate_bundle(
        args["destination"], receipt["manifest_sha256"], require_approval=False
    )
    assert "baseline-audit.json" in manifest["files"] and not manifest["gpu_launches"]
    with pytest.raises(ValueError, match="approval"):
        builder.preflight.validate_bundle(
            args["destination"], receipt["manifest_sha256"], require_approval=True
        )
    with pytest.raises(ValueError, match="new destination"):
        builder.seal(**args)


def test_live_baseline_cannot_be_sealed(inputs):
    builder.stage(**inputs)
    args = seal_args(inputs)
    with (inputs["baseline"] / ".lock").open("r") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still owned"):
            builder.seal(**args)
    assert not args["destination"].exists()


@pytest.mark.parametrize("fault", ["incomplete", "identity", "provisional", "audit_missing"])
def test_sealing_rejects_unfinished_or_inconsistent_baseline(inputs, monkeypatch, fault):
    builder.stage(**inputs)
    args = seal_args(inputs)
    if fault == "incomplete":
        path = inputs["baseline"] / "report.json"
        value = json.loads(path.read_text())
        value["complete"] = False
        path.write_text(json.dumps(value))
    elif fault == "identity":
        path = inputs["baseline"] / "identity.json"
        value = json.loads(path.read_text())
        value["unexpected_change"] = True
        path.write_text(json.dumps(value))
    else:
        monkeypatch.setattr(
            builder,
            "audit_stopped_baseline",
            lambda *args: {"complete": fault == "provisional", "provisional_live_snapshot": True},
        )
    with pytest.raises(ValueError):
        builder.seal(**args)
    assert not args["destination"].exists()


@pytest.mark.parametrize("fault", ["manifest", "data", "symlink", "tooling"])
def test_changed_draft_rejected(inputs, fault):
    builder.stage(**inputs)
    args = seal_args(inputs)
    if fault == "manifest":
        args["draft_hash"] = "wrong"
    elif fault == "symlink":
        target = args["draft"] / "symbolic-sft.json"
        target.rename(args["draft"] / "saved-symbolic.json")
        target.symlink_to(args["draft"] / "saved-symbolic.json")
    else:
        name = "symbolic-sft.json" if fault == "data" else "validate_nanbeige_training.py"
        (args["draft"] / name).write_text("changed")
    with pytest.raises(ValueError):
        builder.seal(**args)
    assert not args["destination"].exists()


def test_destination_and_disk_guards(inputs, monkeypatch):
    inputs["destination"].mkdir()
    with pytest.raises(ValueError, match="new destination"):
        builder.stage(**inputs)
    inputs["destination"] = inputs["destination"].parent / "low-disk"
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _: SimpleNamespace(free=9 * 1024**3))
    with pytest.raises(ValueError, match="10 GiB"):
        builder.stage(**inputs)
    assert not inputs["destination"].exists()


@pytest.mark.parametrize("fault", ["provisional", "missing", "identity", "metrics"])
def test_preflight_requires_independent_audit_even_if_hashes_match(validated_fixture, fault):
    root, _ = validated_fixture
    path = root / "baseline-audit.json"
    audit = json.loads(path.read_text())
    if fault == "provisional":
        audit["provisional_live_snapshot"] = True
    elif fault == "missing":
        audit.pop("stored_evidence_verified")
    elif fault == "identity":
        audit["identity_hash"] = "wrong"
    else:
        audit["metrics"]["direct:42"]["correct_outputs"] = 1
    path.write_text(json.dumps(audit))
    manifest = copy.deepcopy(json.loads((root / "bundle-manifest.json").read_text()))
    manifest["files"]["baseline-audit.json"] = builder.preflight.file_hash(path)
    (root / "bundle-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="audit"):
        builder.preflight.validate_bundle(
            root, builder.preflight.file_hash(root / "bundle-manifest.json"), require_approval=False
        )
