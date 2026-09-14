# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.11.0", "transformers==4.57.6", "trl==0.25.1",
#   "peft==0.18.0", "accelerate==1.12.0", "datasets==4.4.1",
#   "trackio==0.11.0", "sentencepiece==0.2.1", "protobuf==4.25.8",
# ]
# ///
"""Local, read-only preflight for the future Nanbeige symbolic SFT job.

Validates a hash-bound evidence bundle and the actual native tokenizer. Never loads
model weights, uploads artifacts, grants permission or launches training. A passing
preflight is not a full-size GPU compatibility result or a trained student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

MODEL = "Nanbeige/Nanbeige4.2-3B"
REVISION = "3384e426066d1a49c3aea90a7190b81260a6533f"
TEACHER = "gpt-6-astra"
TOKENIZER_HASHES = {
    "tokenizer_config.json": "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518",
    "tokenizer.json": "1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f",
}
MODEL_CODE = {
    "config.json": "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19",
    "modeling_nanbeige.py": "547737f989f5cb741c1a568acaf85f83521fa67a8f7268e19f9a37e60127c0d5",
    "configuration_nanbeige.py": "c517227741f0fc007061bde1873138545ce15e67716a062513193c06300c7685",
}
CATEGORIES = {"grounding", "interpretation", "repair", "continuation", "reuse", "compaction"}
DATA_EVIDENCE_FILES = {
    "symbolic-sft.json",
    "nanbeige-symbolic-tokenized.json",
    "split.json",
    "curriculum-audit.json",
    "student-data-report.json",
    "trainer-probe.json",
    "source-dataset-report.json",
}
BUNDLE_EVIDENCE_FILES = DATA_EVIDENCE_FILES | {
    "baseline-report.json",
    "baseline-identity.json",
    "baseline-audit.json",
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_rows(examples, rows, training_ids):
    """Reject mismatched rows, non-training sources, missing labels and silent truncation."""
    if not examples or len(examples) != len(rows):
        raise ValueError("every admitted example must have one complete tokenized row")
    seen = set()
    for example, row in zip(examples, rows, strict=True):
        if example["source_task_id"] not in training_ids or example["category"] not in CATEGORIES:
            raise ValueError("invalid category or evaluation-derived training example")
        for key in ("source_task_id", "category", "record_ref"):
            if row[key] != example[key]:
                raise ValueError("row provenance differs from accepted symbolic example")
        if len(example["completion"]) != 1 or example["completion"][0]["role"] != "assistant":
            raise ValueError("require one admitted symbolic assistant completion")
        target = json.loads(example["completion"][0]["content"])
        expected = {"version", "entities", "relations"}
        if example["category"] != "grounding":
            expected |= {"definitions", "hypotheses", "unresolved", "proposed_transfer_skill"}
        if set(target) != expected or target["version"] != 1:
            raise ValueError("target must be explicit symbolic state, never code or final grids")
        key = digest({"prompt": example["prompt"], "completion": example["completion"]})
        if key in seen:
            raise ValueError(
                "duplicate admitted example; sampling duplicates belong in the schedule"
            )
        seen.add(key)
        ids, labels = row["input_ids"], row["labels"]
        prefix = row["prompt_tokens"]
        if not 0 < prefix < len(ids) <= 8192 or len(ids) != len(labels):
            raise ValueError("invalid or overlength row; never truncate")
        if not all(type(x) is int and 0 <= x < 166144 for x in ids):
            raise ValueError("token IDs differ from full Nanbeige vocabulary")
        if labels != [-100] * prefix + ids[prefix:]:
            raise ValueError("only the symbolic completion and terminator may be supervised")
        if row["attention_mask"] != [1] * len(ids):
            raise ValueError("unexpected masking before collation")
        if row["completion_tokens"] != len(ids) - prefix:
            raise ValueError("completion length changed")
    return {"rows": len(rows), "categories": dict(Counter(r["category"] for r in rows))}


def validate_native(examples, rows, tokenizer):
    for example, row in zip(examples, rows, strict=True):
        options = dict(
            enable_thinking=False, preserve_thinking=True, tool_call_format="xml", tools=[]
        )
        prefix = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True, **options
        )
        full = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tokenize=False,
            add_generation_prompt=False,
            **options,
        )
        if not full.startswith(prefix):
            raise ValueError("native completion boundary changed")
        if tokenizer.encode(full, add_special_tokens=False) != row["input_ids"]:
            raise ValueError("actual trainer tokenizer differs from prepared native tokens")
        if len(tokenizer.encode(prefix, add_special_tokens=False)) != row["prompt_tokens"]:
            raise ValueError("actual trainer prompt mask boundary changed")


def balanced_schedule(rows, *, epochs=2, seed=42):
    """Explicit equal-category draws, reproducible without a hidden sampler cursor.

    One logical epoch covers every admitted row and matches every category to the
    largest category. Rare categories are oversampled transparently; the expanded
    sequence is traversed once, not passed to a trainer for another two epochs.
    This is not a claim of additional independent examples or source-task diversity.
    """
    if not rows or not 1 <= epochs <= 2:
        raise ValueError("require nonempty data and one or two logical epochs")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["category"]].append(index)
    quota = max(len(indices) for indices in groups.values())
    rng, order = random.Random(seed), []
    for epoch in range(epochs):
        queues = {}
        for category, indices in sorted(groups.items()):
            queue = []
            while len(queue) < quota:
                sample = indices.copy()
                rng.shuffle(sample)
                queue.extend(sample)
            queues[category] = queue[:quota]
        for position in range(quota):
            categories = sorted(groups)
            rng.shuffle(categories)
            for category in categories:
                order.append({"row": queues[category][position], "logical_epoch": epoch})
    return order


def validate_baseline(report, identity):
    if (
        report["complete"] is not True
        or report["trained"] is not False
        or report["model"] != MODEL
        or identity["student"] != "untrained_nanbeige"
    ):
        raise ValueError("complete original Nanbeige baseline required")
    cfg = identity["config"]
    if cfg["seeds"] != [42, 43, 44] or cfg["modes"] != ["direct", "symbolic"]:
        raise ValueError("baseline must retain both modes and all three fixed seeds")
    if len(identity["cohort"]) != 20 or len(set(identity["cohort"])) != 20:
        raise ValueError("baseline must cover the frozen 20-task development cohort")
    keys = {f"{mode}:{seed}" for mode in cfg["modes"] for seed in cfg["seeds"]}
    if set(report["metrics"]) != keys:
        raise ValueError("missing baseline mode/seed")
    for metric in report["metrics"].values():
        if metric["processed_tasks"] != 20 or metric["test_outputs"] != 22:
            raise ValueError("incomplete baseline denominators")
        rate = metric["valid_response_rate"]
        if not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 0.95 <= rate <= 1:
            raise ValueError("baseline structured-response gate not passed")
    evidence = report["memory_and_runtime"]
    for key in (
        "memory_safe",
        "offline_enforced",
        "metal_verified",
        "prompt_cache_disabled_verified",
    ):
        if evidence.get(key) is not True:
            raise ValueError(f"baseline runtime gate not passed: {key}")
    elapsed = report["elapsed_seconds"]
    limit = cfg["global_seconds"]
    if (
        evidence.get("guard_error")
        or not math.isfinite(elapsed)
        or not math.isfinite(limit)
        or not 0 < elapsed <= limit
    ):
        raise ValueError("baseline runtime/safety failure")


def validate_data_evidence(root):
    """Data-only staging check. This never grants the baseline or training gates."""
    examples = read_json(root / "symbolic-sft.json")
    rows = read_json(root / "nanbeige-symbolic-tokenized.json")
    split = read_json(root / "split.json")
    partitions = [split["groups"][k] for k in ("training", "development", "lockbox")]
    groups = [set(partition) for partition in partitions]
    if [len(group) for group in groups] != [792, 103, 105] or any(
        len(group) != len(partition) for group, partition in zip(groups, partitions, strict=True)
    ):
        raise ValueError("require the complete, duplicate-free frozen task split")
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("task split groups overlap")
    validate_rows(examples, rows, groups[0])
    data = read_json(root / "student-data-report.json")
    if (
        data["loss_mask_data_gate_passed"] is not True
        or data["dataset_hash"] != digest(examples)
        or data["examples"] != len(examples)
        or data["truncation"] is not False
        or data["teacher_reasoning_exported"] is not False
        or data["tokenizer_hashes"] != TOKENIZER_HASHES
    ):
        raise ValueError("native data preparation gate failed")
    probe = read_json(root / "trainer-probe.json")
    if (
        probe["passed"] is not True
        or probe["real_data_hash"] != digest(examples)
        or probe["real_rows_checked"] != len(rows)
        or any(
            probe.get(key) is not True
            for key in (
                "actual_trainer_real_data_masks_verified",
                "collator_masks_verified",
                "adapter_save_reload_verified",
                "exact_adapter_resume_verified",
                "small_fixture_training_step_verified",
                "trainer_dataset_unchanged",
            )
        )
        or probe["unique_lora_modules"] != 154
        or probe["shared_loops"] != 2
        or probe["physical_layers"] != 22
        or probe["planned_trainable_parameters"] != 23969792
    ):
        raise ValueError("actual trainer mask integration gate failed")
    source = read_json(root / "source-dataset-report.json")
    audit = read_json(root / "curriculum-audit.json")
    if (
        audit["passed"] is not True
        or audit["dataset_hash"] != source["view_manifest"]["parent_dataset_hash"]
        or audit["examples"] != len(examples)
        or audit["new_teacher_repairs_after_holdout"] != 0
    ):
        raise ValueError("symbolic curriculum replay audit is missing or mismatched")
    if (
        source["view_manifest"]["dataset_hash"] != digest(examples)
        or source["dataset_hash"] != digest(examples)
        or source["teacher"] != TEACHER
        or source["student"] != MODEL
        or source["examples"] != len(examples)
        or source["view_manifest"]["target_unchanged"] is not True
        or source["view_manifest"]["split_hash"] != file_hash(root / "split.json")
    ):
        raise ValueError("compact view changed the accepted teacher target")
    return examples, rows, split


def validate_bundle(root, expected_manifest_hash, *, require_approval):
    if file_hash(root / "bundle-manifest.json") != expected_manifest_hash:
        raise ValueError("bundle manifest differs from the approved immutable artifact")
    manifest = read_json(root / "bundle-manifest.json")
    if (manifest["student"], manifest["student_revision"], manifest["teacher"]) != (
        MODEL,
        REVISION,
        TEACHER,
    ):
        raise ValueError("only the approved Astra-to-Nanbeige lineage is accepted")
    if set(manifest["files"]) != BUNDLE_EVIDENCE_FILES:
        raise ValueError("training bundle must contain exactly the declared evidence files")
    for name, expected in manifest["files"].items():
        if (root / name).is_symlink() or not (root / name).is_file():
            raise ValueError(f"bundle artifact must be a regular file: {name}")
        if file_hash(root / name) != expected:
            raise ValueError(f"bundle artifact changed: {name}")
    examples, rows, split = validate_data_evidence(root)
    baseline = read_json(root / "baseline-report.json")
    identity = read_json(root / "baseline-identity.json")
    validate_baseline(baseline, identity)
    if (
        identity["split_hash"] != file_hash(root / "split.json")
        or identity["cohort"] != split["groups"]["development"][:20]
    ):
        raise ValueError("baseline and training split identities differ")
    audit = read_json(root / "baseline-audit.json")
    if (
        audit.get("complete") is not True
        or audit.get("provisional_live_snapshot") is not False
        or audit.get("identity_hash") != digest(identity)
        or audit.get("cohort") != identity["cohort"]
        or audit.get("split_hash") != identity["split_hash"]
        or any(
            audit.get(key) is not True
            for key in (
                "cohort_verified",
                "source_archive_verified",
                "stored_evidence_verified",
            )
        )
        or set(audit.get("metrics", {})) != set(baseline["metrics"])
    ):
        raise ValueError("complete independent baseline audit is required")
    for key, metric in baseline["metrics"].items():
        for field in (
            "processed_tasks",
            "test_outputs",
            "correct_outputs",
            "dispatches",
            "generated_tokens",
            "prompt_tokens",
            "valid_response_rate",
        ):
            if field not in metric or audit["metrics"][key].get(field) != metric[field]:
                raise ValueError("baseline report differs from its independent audit")
    if require_approval and (
        not manifest.get("compute_approval_ref") or not manifest.get("output_use_permission_ref")
    ):
        raise ValueError(
            "explicit paid-compute approval and resolved output-use permission required"
        )
    return manifest, examples, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--require-approval", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, examples, rows = validate_bundle(
        args.data, args.manifest_sha256, require_approval=args.require_approval
    )
    for name, expected in {**MODEL_CODE, **TOKENIZER_HASHES}.items():
        if file_hash(args.model_dir / name) != expected:
            raise ValueError(f"pinned model/tokenizer changed: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    validate_native(examples, rows, tokenizer)
    schedule = balanced_schedule(rows)
    print(
        json.dumps(
            {
                "preflight_passed": True,
                "rows": len(rows),
                "weights_loaded": False,
                "schedule_draws": len(schedule),
                "per_row_exposure_counts": dict(Counter(draw["row"] for draw in schedule)),
                "approval_references_present": bool(
                    manifest.get("compute_approval_ref")
                    and manifest.get("output_use_permission_ref")
                ),
                "gpu_training_verified": False,
                "training_launched": False,
                "promotion_proven": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
