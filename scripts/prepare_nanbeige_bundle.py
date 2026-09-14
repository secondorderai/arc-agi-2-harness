"""Stage verified symbolic data locally; seal evidence only after an audited baseline.

No model weights, uploads, credentials, paid jobs, training or implicit approval. All
destinations must be new. A draft is intentionally not a launchable training bundle.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SCRIPT_FILES = {
    "prepare_nanbeige_bundle.py",
    "validate_nanbeige_training.py",
    "train_nanbeige_symbolic.py",
}
spec = importlib.util.spec_from_file_location(
    "bundle_preflight", SCRIPTS / "validate_nanbeige_training.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def copy_regular(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"expected regular evidence file: {source.name}")
    before = preflight.file_hash(source)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if preflight.file_hash(destination) != before or preflight.file_hash(source) != before:
        raise ValueError(f"source changed while copying: {source.name}")


def create_destination(destination, sources):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("preserve existing artifacts; choose a new destination")
    if not destination.parent.is_dir():
        raise ValueError("destination parent must already exist")
    reserve = 10 * 1024**3 + sum(Path(path).stat().st_size for path in sources)
    if shutil.disk_usage(destination.parent).free < reserve:
        raise ValueError("preserve at least 10 GiB free disk after staging")
    destination.mkdir()


def verify_native(root, model_dir):
    examples, rows, _ = preflight.validate_data_evidence(root)
    for name, expected in {**preflight.MODEL_CODE, **preflight.TOKENIZER_HASHES}.items():
        if preflight.file_hash(model_dir / name) != expected:
            raise ValueError(f"pinned model/tokenizer changed: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    preflight.validate_native(examples, rows, tokenizer)
    schedule = preflight.balanced_schedule(rows)
    return {
        "data_evidence_verified": True,
        "native_tokenization_verified": True,
        "examples": len(examples),
        "dataset_hash": preflight.digest(examples),
        "categories": dict(Counter(row["category"] for row in rows)),
        "schedule_draws": len(schedule),
        "schedule_hash": preflight.digest(schedule),
        "per_row_exposure_counts": dict(Counter(draw["row"] for draw in schedule)),
        "weights_loaded": False,
        "training_launched": False,
        "upload_performed": False,
        "promotion_proven": False,
    }


def stage(student_data, curriculum_audit, trainer_probe, split, baseline, destination, model_dir):
    sources = {
        name: student_data / name
        for name in (
            "symbolic-sft.json",
            "nanbeige-symbolic-tokenized.json",
            "student-data-report.json",
        )
    }
    sources.update(
        {
            "source-dataset-report.json": student_data / "dataset-report.json",
            "curriculum-audit.json": curriculum_audit,
            "trainer-probe.json": trainer_probe,
            "split.json": split,
            **{name: SCRIPTS / name for name in SCRIPT_FILES},
        }
    )
    identity = preflight.read_json(baseline / "identity.json")
    frozen_split = preflight.read_json(split)
    if (
        identity["student"] != "untrained_nanbeige"
        or identity["split_hash"] != preflight.file_hash(split)
        or identity["cohort"] != frozen_split["groups"]["development"][:20]
    ):
        raise ValueError("draft must reference the matching frozen untrained baseline")
    create_destination(destination, sources.values())
    # A failed build leaves its files for inspection, but no successful manifest marker.
    for name, source in sources.items():
        copy_regular(source, destination / name)
    report = verify_native(destination, model_dir)
    manifest = {
        "kind": "nanbeige_training_data_draft",
        "version": 1,
        "student": preflight.MODEL,
        "student_revision": preflight.REVISION,
        "teacher": preflight.TEACHER,
        "files": {name: preflight.file_hash(destination / name) for name in sorted(sources)},
        "baseline_workspace": str(baseline.resolve()),
        "baseline_identity_hash": preflight.digest(identity),
        "pending_evidence": [
            "baseline-report.json",
            "baseline-identity.json",
            "baseline-audit.json",
        ],
        "baseline_gate_passed": False,
        "compute_approval_ref": None,
        "output_use_permission_ref": None,
        **report,
    }
    write_new_json(destination / "draft-manifest.json", manifest)
    return manifest


def verify_draft(root, expected_hash):
    if preflight.file_hash(root / "draft-manifest.json") != expected_hash:
        raise ValueError("draft manifest changed")
    manifest = preflight.read_json(root / "draft-manifest.json")
    if (
        manifest.get("kind") != "nanbeige_training_data_draft"
        or manifest.get("version") != 1
        or manifest.get("student") != preflight.MODEL
        or manifest.get("student_revision") != preflight.REVISION
        or manifest.get("teacher") != preflight.TEACHER
        or set(manifest.get("files", {})) != preflight.DATA_EVIDENCE_FILES | SCRIPT_FILES
    ):
        raise ValueError("invalid data-only draft lineage or file allowlist")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or preflight.file_hash(path) != expected:
            raise ValueError(f"draft artifact changed: {name}")
    for name in SCRIPT_FILES:
        if preflight.file_hash(SCRIPTS / name) != manifest["files"][name]:
            raise ValueError("staged tooling changed; build a new draft instead of rewriting it")
    preflight.validate_data_evidence(root)
    return manifest


def audit_stopped_baseline(baseline, split, data):
    frozen = baseline / "frozen/src"
    if not (frozen / "arc_agent/v8_reasoner.py").is_file():
        raise ValueError("extract the baseline source archive for reproducible auditing first")
    environment = dict(os.environ, PYTHONPATH=str(frozen.resolve()))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_nanbeige_baseline.py"),
            "--workspace",
            str(baseline),
            "--split",
            str(split),
            "--data",
            str(data),
        ],
        env=environment,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def seal(draft, draft_hash, baseline, data, destination, model_dir):
    manifest = verify_draft(draft, draft_hash)
    # Hold the reader lock while auditing AND copying, so a resumed writer cannot race us.
    with (baseline / ".lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("baseline is still owned; never seal a live run") from None
        identity = preflight.read_json(baseline / "identity.json")
        report = preflight.read_json(baseline / "report.json")
        if preflight.digest(identity) != manifest["baseline_identity_hash"]:
            raise ValueError("baseline identity differs from the staged comparison")
        preflight.validate_baseline(report, identity)
        audited = audit_stopped_baseline(baseline, draft / "split.json", data)
        if (
            audited.get("complete") is not True
            or audited.get("provisional_live_snapshot") is not False
        ):
            raise ValueError("a complete non-live independent baseline audit is required")
        sources = {name: draft / name for name in manifest["files"]}
        sources.update(
            {
                "baseline-report.json": baseline / "report.json",
                "baseline-identity.json": baseline / "identity.json",
            }
        )
        create_destination(destination, sources.values())
        for name, source in sources.items():
            copy_regular(source, destination / name)
        write_new_json(destination / "baseline-audit.json", audited)
    native = verify_native(destination, model_dir)
    sealed = {
        "student": preflight.MODEL,
        "student_revision": preflight.REVISION,
        "teacher": preflight.TEACHER,
        "source_draft_sha256": draft_hash,
        "files": {
            name: preflight.file_hash(destination / name)
            for name in sorted(preflight.BUNDLE_EVIDENCE_FILES)
        },
        "tooling_hashes": {
            name: preflight.file_hash(destination / name) for name in sorted(SCRIPT_FILES)
        },
        "compute_approval_ref": None,
        "output_use_permission_ref": None,
        "gpu_launches": {},
    }
    write_new_json(destination / "bundle-manifest.json", sealed)
    bundle_hash = preflight.file_hash(destination / "bundle-manifest.json")
    preflight.validate_bundle(destination, bundle_hash, require_approval=False)
    receipt = {
        **native,
        "status": "evidence_sealed_pending_approval",
        "baseline_gate_passed": True,
        "manifest_sha256": bundle_hash,
        "preflight_sha256": sealed["tooling_hashes"]["validate_nanbeige_training.py"],
        "job_launch_authorized": False,
    }
    write_new_json(destination / "assembly-report.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument(
        "--student-data", type=Path, default=Path("runs/v8/student-data-pilot-final")
    )
    staging.add_argument(
        "--curriculum-audit",
        type=Path,
        default=Path("runs/v7/astra-symbolic-bridge-01/final-curriculum-audit.json"),
    )
    staging.add_argument(
        "--trainer-probe", type=Path, default=Path("runs/v8/trainer-probe-final/report.json")
    )
    staging.add_argument("--split", type=Path, default=Path("runs/v4/split.json"))
    final = commands.add_parser("seal")
    final.add_argument("--draft", type=Path, required=True)
    final.add_argument("--draft-sha256", required=True)
    final.add_argument("--data", type=Path, default=Path("data/ARC-AGI-2/data/training"))
    for command in (staging, final):
        command.add_argument(
            "--baseline", type=Path, default=Path("runs/v8/untrained-reference-baseline-01")
        )
        command.add_argument("--destination", type=Path, required=True)
        command.add_argument("--model-dir", type=Path, default=Path(".runtime/nanbeige/model-hf"))
    args = parser.parse_args()
    if args.command == "stage":
        result = stage(
            args.student_data,
            args.curriculum_audit,
            args.trainer_probe,
            args.split,
            args.baseline,
            args.destination,
            args.model_dir,
        )
        summary = {
            key: result[key]
            for key in (
                "kind",
                "examples",
                "schedule_draws",
                "baseline_gate_passed",
                "training_launched",
                "upload_performed",
            )
        }
        summary["draft_sha256"] = preflight.file_hash(args.destination / "draft-manifest.json")
    else:
        summary = seal(
            args.draft,
            args.draft_sha256,
            args.baseline,
            args.data,
            args.destination,
            args.model_dir,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
