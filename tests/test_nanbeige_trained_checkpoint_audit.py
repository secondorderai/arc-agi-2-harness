"""Synthetic stored-provenance fixtures, never real training or launch permission."""

import copy
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest
import test_nanbeige_training_preflight as evidence_fixtures

validated_fixture = evidence_fixtures.bundle
SPEC = importlib.util.spec_from_file_location(
    "trained_checkpoint_audit",
    Path(__file__).parents[1] / "scripts/audit_nanbeige_trained_checkpoint.py",
)
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)
job = auditor.job


def write_json(path, value):
    path.write_text(json.dumps(value))
    return job.file_hash(path)


@pytest.fixture
def evidence(validated_fixture):
    root, _ = validated_fixture
    manifest = job.read_json(root / "bundle-manifest.json")
    manifest.update(
        compute_approval_ref="synthetic-test-not-real-approval",
        output_use_permission_ref="synthetic-test-not-real-permission",
        tooling_hashes={},
        gpu_launches={},
    )
    for name in ("validate_nanbeige_training.py", "train_nanbeige_symbolic.py"):
        shutil.copyfile(auditor.SCRIPTS / name, root / name)
        manifest["tooling_hashes"][name] = job.file_hash(root / name)
    for mode in ("train", "compatibility"):
        manifest["gpu_launches"][mode] = {
            "compute_approval_ref": "synthetic-test-not-real-approval",
            "output_use_permission_ref": "synthetic-test-not-real-permission",
            "model_repo": "fixture/private-model",
            "trackio_space": "fixture/private-space",
            "trackio_dataset": "fixture/private-metrics",
            "run_id": f"fixture-{mode}",
            "hardware": "fixture-gpu",
            "script_sha256": manifest["tooling_hashes"]["train_nanbeige_symbolic.py"],
            "timeout_seconds": 7200,
        }
    bundle_hash = write_json(root / "bundle-manifest.json", manifest)
    examples = job.read_json(root / "symbolic-sft.json")
    rows = job.read_json(root / "nanbeige-symbolic-tokenized.json")
    schedule = auditor.preflight.balanced_schedule(rows)
    final_step = math.ceil(len(schedule) / 16)
    binding = {
        "model": job.MODEL,
        "revision": job.REVISION,
        "dataset_hash": job.digest(examples),
        "environment": {**auditor.ENVIRONMENT, "torch": "2.11.0+cu128"},
        "preflight_hash": manifest["tooling_hashes"]["validate_nanbeige_training.py"],
        "script_hash": manifest["tooling_hashes"]["train_nanbeige_symbolic.py"],
        "gpu": "synthetic fixture, not real hardware",
        "cuda": "synthetic-fixture",
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "effective_batch": 16,
        "max_length": 8192,
        "seed": 42,
    }
    checkpoint = root / f"checkpoint-{final_step}"
    checkpoint.mkdir()
    for name in job.CHECKPOINT_REQUIRED - {"run-identity.json", "data-cursor.json"}:
        (checkpoint / name).write_text("synthetic fixture; not real model or optimizer tensors")
    write_json(checkpoint / "trainer_state.json", {"global_step": final_step})
    write_json(
        checkpoint / "adapter_config.json",
        {
            "base_model_name_or_path": job.MODEL,
            "revision": job.REVISION,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "target_modules": job.TARGETS,
            "bias": "none",
        },
    )
    identity = {
        **binding,
        "mode": "train",
        "run_id": "fixture-train",
        "bundle_hash": bundle_hash,
        "schedule_hash": job.digest(schedule),
    }
    checkpoint_hash = job.seal_checkpoint(checkpoint, identity, schedule, final_step)

    def report_for(mode, step, checksum):
        suffix = f"{'train' if mode == 'train' else 'reference'}/checkpoint-{step}"
        ref = {
            "repo": "fixture/private-model",
            "revision": "a" * 40,
            "path": f"runs/fixture-{mode}/{suffix}",
            "manifest_sha256": checksum,
        }
        return {
            "binding": binding,
            "mode": mode,
            "passed": True,
            "student_trained": mode == "train",
            "promotion_proven": False,
            "official_weights_loaded": True,
            "gpu_training_verified": True,
            "real_masks_verified": True,
            "shared_layers_verified": True,
            "adapter_save_reload_verified": True,
            "durable_checkpoints_verified": True,
            "exact_resume_verified": mode == "compatibility",
            "padded_8k_forward_backward_verified": mode == "compatibility",
            "unique_examples": len(rows),
            "scheduled_draws": len(schedule) if mode == "train" else 32,
            "checkpoint_refs": {suffix: ref},
            "final_checkpoint_ref": ref,
            "optimizer_updates_this_invocation": step,
            "recovered_completed_checkpoint": False,
            "elapsed_seconds": 30,
            "peak_gpu_allocated_bytes": 1024,
            "peak_gpu_reserved_bytes": 2048,
        }

    report_path = root / "training-report.json"
    compatibility_path = root / "compatibility-report.json"
    return {
        "bundle": root,
        "bundle_hash": bundle_hash,
        "report_path": report_path,
        "report_hash": write_json(report_path, report_for("train", final_step, checkpoint_hash)),
        "checkpoint": checkpoint,
        "compatibility_path": compatibility_path,
        "compatibility_hash": write_json(
            compatibility_path, report_for("compatibility", 2, "b" * 64)
        ),
    }


def change_report(evidence, mutate, *, compatibility=False):
    path_key, hash_key = (
        ("compatibility_path", "compatibility_hash")
        if compatibility
        else ("report_path", "report_hash")
    )
    value = job.read_json(evidence[path_key])
    mutate(value)
    evidence[hash_key] = write_json(evidence[path_key], value)


def test_stored_provenance_is_read_only_and_not_model_validation(evidence):
    before = {str(p): job.file_hash(p) for p in evidence["bundle"].rglob("*") if p.is_file()}
    result = auditor.audit(**evidence)
    assert result["stored_training_evidence_verified"] is True
    for key in (
        "tensor_payload_validated",
        "gpu_execution_replayed",
        "remote_availability_verified",
        "inference_ready",
        "training_launched",
        "promotion_proven",
    ):
        assert result[key] is False
    assert result["reported_optimizer_updates_this_invocation"] == 1
    assert before == {
        str(p): job.file_hash(p) for p in evidence["bundle"].rglob("*") if p.is_file()
    }


def test_completed_recovery_never_claims_new_training(evidence):
    change_report(
        evidence,
        lambda value: value.update(
            optimizer_updates_this_invocation=0,
            recovered_completed_checkpoint=True,
            gpu_training_verified=False,
        ),
    )
    result = auditor.audit(**evidence)
    assert result["reported_completed_checkpoint_recovery"] is True
    assert result["reported_optimizer_updates_this_invocation"] == 0
    assert result["gpu_execution_replayed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "compatibility"),
        ("student_trained", False),
        ("passed", False),
        ("promotion_proven", True),
        ("official_weights_loaded", False),
        ("real_masks_verified", False),
        ("shared_layers_verified", False),
        ("adapter_save_reload_verified", False),
        ("durable_checkpoints_verified", False),
        ("scheduled_draws", 32),
        ("unique_examples", 100),
        ("gpu_training_verified", False),
        ("optimizer_updates_this_invocation", True),
        ("optimizer_updates_this_invocation", 0),
        ("recovered_completed_checkpoint", True),
        ("elapsed_seconds", float("nan")),
        ("peak_gpu_allocated_bytes", 9999),
    ],
)
def test_report_claims_cannot_replace_evidence(evidence, field, value):
    change_report(evidence, lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        auditor.audit(**evidence)


@pytest.mark.parametrize("fault", ["repo", "path", "revision", "hash", "ledger"])
def test_changed_durable_reference_is_rejected(evidence, fault):
    def mutate(report):
        reference = copy.deepcopy(report["final_checkpoint_ref"])
        if fault == "ledger":
            report["checkpoint_refs"] = {}
        else:
            field = "manifest_sha256" if fault == "hash" else fault
            reference[field] = {
                "repo": "another/model",
                "path": "../checkpoint-1",
                "revision": "main",
                "hash": "c" * 64,
            }[fault]
            report["final_checkpoint_ref"] = reference
            report["checkpoint_refs"]["train/checkpoint-1"] = reference

    change_report(evidence, mutate)
    with pytest.raises(ValueError):
        auditor.audit(**evidence)


@pytest.mark.parametrize(
    "file", ["optimizer.pt", "rng_state.pth", "adapter_model.safetensors", "data-cursor.json"]
)
def test_changed_checkpoint_member_rejected(evidence, file):
    (evidence["checkpoint"] / file).write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        auditor.audit(**evidence)


def test_unlisted_checkpoint_artifact_rejected(evidence):
    (evidence["checkpoint"] / "unaccounted.json").write_text("{}")
    with pytest.raises(ValueError, match="unaccounted"):
        auditor.audit(**evidence)


def test_symlink_report_rejected(evidence):
    alias = evidence["bundle"] / "linked-report.json"
    alias.symlink_to(evidence["report_path"])
    evidence["report_path"] = alias
    with pytest.raises(ValueError, match="regular"):
        auditor.audit(**evidence)


def test_changed_training_binding_rejected(evidence):
    change_report(evidence, lambda report: report["binding"].update(model="another/model"))
    with pytest.raises(ValueError, match="binding"):
        auditor.audit(**evidence)


def test_non_gpu_compatibility_report_rejected(evidence):
    change_report(
        evidence, lambda report: report.update(gpu_training_verified=False), compatibility=True
    )
    with pytest.raises(ValueError, match="compatibility"):
        auditor.audit(**evidence)


def test_unapproved_bundle_rejected_before_checkpoint_admission(evidence):
    path = evidence["bundle"] / "bundle-manifest.json"
    value = job.read_json(path)
    value["output_use_permission_ref"] = None
    evidence["bundle_hash"] = write_json(path, value)
    with pytest.raises(ValueError, match="permission"):
        auditor.audit(**evidence)


def test_report_hash_is_required(evidence):
    evidence["report_hash"] = "d" * 64
    with pytest.raises(ValueError, match="hash"):
        auditor.audit(**evidence)
