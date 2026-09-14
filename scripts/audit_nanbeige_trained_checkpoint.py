"""Local, read-only audit of a future trained checkpoint's stored provenance.

No network, model loading, pickle deserialization, conversion, training or approval.
This checks bytes and cross-artifact consistency, not GPU execution, tensor quality,
remote availability or ARC promotion. It is not a trained-model inference entry point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def local_tool(name):
    # Load only repository-owned audit code, never code supplied by an evidence bundle.
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = local_tool("validate_nanbeige_training.py")
job = local_tool("train_nanbeige_symbolic.py")
ENVIRONMENT = {
    "torch": "2.11.0",
    "transformers": "4.57.6",
    "trl": "0.25.1",
    "peft": "0.18.0",
    "accelerate": "1.12.0",
    "datasets": "4.4.1",
}


def checked_file(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular evidence file: {path.name}")
    actual = job.file_hash(path)
    if expected is not None and (
        not isinstance(expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
        or actual != expected
    ):
        raise ValueError(f"evidence hash mismatch: {path.name}")
    return actual


def check_reference(reference, launch, suffix):
    if (
        not isinstance(reference, dict)
        or set(reference) != {"repo", "revision", "path", "manifest_sha256"}
        or reference.get("repo") != launch["model_repo"]
        or reference.get("path") != f"runs/{launch['run_id']}/{suffix}"
        or not isinstance(reference.get("revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", reference["revision"])
        or not isinstance(reference.get("manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", reference["manifest_sha256"])
    ):
        raise ValueError("checkpoint reference differs from the approved immutable destination")


def check_final_reference(report, launch, suffix):
    reference = report.get("final_checkpoint_ref")
    check_reference(reference, launch, suffix)
    if report.get("checkpoint_refs", {}).get(suffix) != reference:
        raise ValueError("final checkpoint is absent from the durable checkpoint ledger")
    return reference


def check_binding(binding, examples, preflight_hash, script_hash):
    expected = {
        "model": job.MODEL,
        "revision": job.REVISION,
        "dataset_hash": job.digest(examples),
        "preflight_hash": preflight_hash,
        "script_hash": script_hash,
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "effective_batch": 16,
        "max_length": 8192,
        "seed": 42,
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != set(expected) | {"environment", "gpu", "cuda"}
        or any(binding.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("trained report differs from the pinned data/model/training binding")
    environment = binding["environment"]
    if (
        not isinstance(environment, dict)
        or set(environment) != set(ENVIRONMENT)
        or any(
            not isinstance(environment[key], str) or environment[key].split("+", 1)[0] != version
            for key, version in ENVIRONMENT.items()
        )
        or any(
            not isinstance(binding[key], str) or not binding[key].strip() for key in ("gpu", "cuda")
        )
    ):
        raise ValueError("missing or incompatible recorded training environment")


def check_training_report(report, rows, schedule):
    required = (
        "passed",
        "student_trained",
        "official_weights_loaded",
        "real_masks_verified",
        "shared_layers_verified",
        "adapter_save_reload_verified",
        "durable_checkpoints_verified",
    )
    if (
        report.get("mode") != "train"
        or any(report.get(key) is not True for key in required)
        or report.get("promotion_proven") is not False
        or report.get("unique_examples") != len(rows)
        or report.get("scheduled_draws") != len(schedule)
    ):
        raise ValueError("require a completed SFT report, never a probe or promotion assertion")
    updates = report.get("optimizer_updates_this_invocation")
    recovered = report.get("recovered_completed_checkpoint")
    if (
        type(updates) is not int
        or type(recovered) is not bool
        or not 0 <= updates <= math.ceil(len(schedule) / 16)
        or recovered != (updates == 0)
        or report.get("gpu_training_verified") is not (updates > 0)
    ):
        raise ValueError("report-only recovery must not claim new GPU optimizer updates")
    for key in ("elapsed_seconds", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
        value = report.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"missing or invalid training telemetry: {key}")
    if report["peak_gpu_allocated_bytes"] > report["peak_gpu_reserved_bytes"]:
        raise ValueError("allocated GPU memory exceeds reserved memory")


def audit(
    bundle,
    bundle_hash,
    report_path,
    report_hash,
    checkpoint,
    compatibility_path,
    compatibility_hash,
):
    bundle, checkpoint = Path(bundle), Path(checkpoint)
    if any(path.is_symlink() or not path.is_dir() for path in (bundle, checkpoint)):
        raise ValueError("bundle and checkpoint must be regular local directories")
    inputs = {
        bundle / "bundle-manifest.json": bundle_hash,
        Path(report_path): report_hash,
        Path(compatibility_path): compatibility_hash,
    }
    for path, expected in inputs.items():
        checked_file(path, expected)
    manifest, examples, rows = preflight.validate_bundle(bundle, bundle_hash, require_approval=True)
    tool_hashes = {}
    for name in ("validate_nanbeige_training.py", "train_nanbeige_symbolic.py"):
        expected = checked_file(SCRIPTS / name)
        if manifest.get("tooling_hashes", {}).get(name) != expected:
            raise ValueError("training tooling differs from the locally audited implementation")
        checked_file(bundle / name, expected)
        inputs[bundle / name] = expected
        tool_hashes[name] = expected
    launch = job.check_launch(manifest, "train", tool_hashes["train_nanbeige_symbolic.py"])
    probe_launch = job.check_launch(
        manifest, "compatibility", tool_hashes["train_nanbeige_symbolic.py"]
    )
    report = job.read_json(report_path)
    compatibility = job.read_json(compatibility_path)
    binding = report.get("binding")
    check_binding(
        binding,
        examples,
        tool_hashes["validate_nanbeige_training.py"],
        tool_hashes["train_nanbeige_symbolic.py"],
    )
    job.check_compatibility(compatibility, binding)
    if (
        compatibility.get("student_trained") is not False
        or compatibility.get("promotion_proven") is not False
        or compatibility.get("scheduled_draws") != 32
        or compatibility.get("optimizer_updates_this_invocation") != 2
        or compatibility.get("recovered_completed_checkpoint") is not False
    ):
        raise ValueError("compatibility must contain its actual two-update probe")
    check_final_reference(compatibility, probe_launch, "reference/checkpoint-2")
    schedule = preflight.balanced_schedule(rows)
    check_training_report(report, rows, schedule)
    final_step = math.ceil(len(schedule) / 16)
    if checkpoint.name != f"checkpoint-{final_step}":
        raise ValueError("require the final checkpoint, not a partial or compatibility adapter")
    reference = check_final_reference(report, launch, f"train/checkpoint-{final_step}")
    checkpoint_hash = reference["manifest_sha256"]
    checked_file(checkpoint / "checkpoint-manifest.json", checkpoint_hash)
    checkpoint_manifest = job.read_json(checkpoint / "checkpoint-manifest.json")
    if set(p.name for p in checkpoint.iterdir()) != set(checkpoint_manifest["files"]) | {
        "checkpoint-manifest.json"
    }:
        raise ValueError("checkpoint directory contains unaccounted artifacts")
    for name, expected in checkpoint_manifest["files"].items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("invalid checkpoint member path")
        inputs[checkpoint / name] = expected
        checked_file(checkpoint / name, expected)
    identity = {
        **binding,
        "mode": "train",
        "run_id": launch["run_id"],
        "bundle_hash": bundle_hash,
        "schedule_hash": job.digest(schedule),
    }
    cursor = job.validate_checkpoint(checkpoint, checkpoint_hash, identity, schedule)
    if cursor["next_draw"] != len(schedule) or cursor["global_step"] != final_step:
        raise ValueError("training schedule is not complete")
    inputs[checkpoint / "checkpoint-manifest.json"] = checkpoint_hash
    inputs.update({bundle / name: expected for name, expected in manifest["files"].items()})
    for path, expected in inputs.items():
        checked_file(path, expected)
    return {
        "scope": "local checkpoint bytes and stored provenance consistency only",
        "stored_training_evidence_verified": True,
        "model": job.MODEL,
        "revision": job.REVISION,
        "bundle_sha256": bundle_hash,
        "training_report_sha256": report_hash,
        "compatibility_report_sha256": compatibility_hash,
        "checkpoint_ref": reference,
        "adapter_sha256": checkpoint_manifest["files"]["adapter_model.safetensors"],
        "training_identity_hash": job.digest(identity),
        "schedule_draws": len(schedule),
        "global_step": final_step,
        "reported_optimizer_updates_this_invocation": report["optimizer_updates_this_invocation"],
        "reported_completed_checkpoint_recovery": report["recovered_completed_checkpoint"],
        "auditor_sha256": checked_file(__file__),
        "remote_availability_verified": False,
        "gpu_execution_replayed": False,
        "tensor_payload_validated": False,
        "inference_ready": False,
        "training_launched": False,
        "promotion_proven": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "report", "checkpoint", "compatibility"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        if name != "checkpoint":
            parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, help="New audit file; never overwrite")
    args = parser.parse_args()
    result = audit(
        args.bundle,
        args.bundle_sha256,
        args.report,
        args.report_sha256,
        args.checkpoint,
        args.compatibility,
        args.compatibility_sha256,
    )
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
