"""Pure preflight tests; synthetic evidence here never authorizes real training."""

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "training_preflight", Path(__file__).parents[1] / "scripts/validate_nanbeige_training.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def example_and_row():
    example = {
        "source_task_id": "train-0",
        "category": "grounding",
        "record_ref": "fixture-only",
        "prompt": [{"role": "user", "content": "Synthetic test input"}],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps({"version": 1, "entities": [], "relations": []}),
            }
        ],
    }
    row = {
        **{k: example[k] for k in ("source_task_id", "category", "record_ref")},
        "input_ids": [166100, 42, 51, 166101],
        "labels": [-100, -100, 51, 166101],
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "attention_mask": [1, 1, 1, 1],
    }
    return example, row


def baseline():
    identity = {
        "student": "untrained_nanbeige",
        "config": {
            "seeds": [42, 43, 44],
            "modes": ["direct", "symbolic"],
            "global_seconds": 39600,
        },
        "cohort": [f"dev-{i}" for i in range(20)],
    }
    report = {
        "complete": True,
        "trained": False,
        "model": preflight.MODEL,
        "elapsed_seconds": 100,
        "metrics": {
            f"{mode}:{seed}": {
                "processed_tasks": 20,
                "test_outputs": 22,
                "valid_response_rate": 1.0,
                "correct_outputs": 0,
                "dispatches": 20,
                "generated_tokens": 200,
                "prompt_tokens": 200,
            }
            for mode in ("direct", "symbolic")
            for seed in (42, 43, 44)
        },
        "memory_and_runtime": {
            "memory_safe": True,
            "offline_enforced": True,
            "metal_verified": True,
            "prompt_cache_disabled_verified": True,
            "guard_error": None,
        },
    }
    return report, identity


def test_valid_row_and_schedule():
    example, row = example_and_row()
    assert preflight.validate_rows([example], [row], {"train-0"})["rows"] == 1
    rows = (
        [{"category": "grounding"}] * 20
        + [{"category": "interpretation"}] * 18
        + [{"category": "repair"}]
    )
    schedule = preflight.balanced_schedule(rows)
    assert schedule == preflight.balanced_schedule(rows)
    assert schedule != preflight.balanced_schedule(rows, seed=43)
    assert len(schedule) == 120
    assert Counter(rows[s["row"]]["category"] for s in schedule) == {
        "grounding": 40,
        "interpretation": 40,
        "repair": 40,
    }
    assert Counter(s["logical_epoch"] for s in schedule) == {0: 60, 1: 60}
    for epoch in (0, 1):
        assert {s["row"] for s in schedule if s["logical_epoch"] == epoch} == set(range(39))


@pytest.mark.parametrize("fault", ["leak", "labels", "length", "provenance", "answer", "dup"])
def test_reject_bad_training_rows(fault):
    example, row = example_and_row()
    examples, rows = [example], [row]
    if fault == "leak":
        example["source_task_id"] = row["source_task_id"] = "dev-0"
    elif fault == "labels":
        row["labels"][0] = 166100
    elif fault == "length":
        row["input_ids"] = [1] * 8193
        row["labels"] = [-100, -100] + [1] * 8191
    elif fault == "provenance":
        row["record_ref"] = "another-example"
    elif fault == "answer":
        example["completion"][0]["content"] = '{"predictions":[[1,2]]}'
    else:
        examples, rows = [example, copy.deepcopy(example)], [row, copy.deepcopy(row)]
    with pytest.raises(ValueError):
        preflight.validate_rows(examples, rows, {"train-0"})


@pytest.mark.parametrize("epochs", [0, 3])
def test_schedule_epoch_limit(epochs):
    with pytest.raises(ValueError):
        preflight.balanced_schedule([{"category": "grounding"}], epochs=epochs)


@pytest.mark.parametrize(
    "fault", ["incomplete", "trained", "seed", "cohort", "unsafe", "nan", "timeout", "validity"]
)
def test_reject_bad_baseline(fault):
    report, identity = baseline()
    preflight.validate_baseline(report, identity)
    if fault == "incomplete":
        report["complete"] = False
    elif fault == "trained":
        report["trained"] = True
    elif fault == "seed":
        identity["config"]["seeds"] = [42]
    elif fault == "cohort":
        identity["cohort"][1] = identity["cohort"][0]
    elif fault == "unsafe":
        report["memory_and_runtime"]["memory_safe"] = False
    elif fault == "nan":
        report["elapsed_seconds"] = float("nan")
    elif fault == "timeout":
        report["elapsed_seconds"] = 39601
    else:
        report["metrics"]["symbolic:42"]["valid_response_rate"] = float("nan")
    with pytest.raises(ValueError):
        preflight.validate_baseline(report, identity)


@pytest.fixture
def bundle(tmp_path):
    example, row = example_and_row()
    dataset_hash = preflight.digest([example])
    report, identity = baseline()
    split = {
        "groups": {
            "training": [f"train-{i}" for i in range(792)],
            "development": [f"dev-{i}" for i in range(103)],
            "lockbox": [f"lock-{i}" for i in range(105)],
        }
    }
    (tmp_path / "split.json").write_text(json.dumps(split))
    split_hash = preflight.file_hash(tmp_path / "split.json")
    identity["split_hash"] = split_hash
    documents = {
        "symbolic-sft.json": [example],
        "nanbeige-symbolic-tokenized.json": [row],
        "split.json": split,
        "baseline-report.json": report,
        "baseline-identity.json": identity,
        "baseline-audit.json": {
            "complete": True,
            "provisional_live_snapshot": False,
            "cohort_verified": True,
            "source_archive_verified": True,
            "stored_evidence_verified": True,
            "cohort": identity["cohort"],
            "split_hash": split_hash,
            "identity_hash": preflight.digest(identity),
            "metrics": copy.deepcopy(report["metrics"]),
        },
        "curriculum-audit.json": {
            "passed": True,
            "dataset_hash": "synthetic-parent",
            "examples": 1,
            "new_teacher_repairs_after_holdout": 0,
        },
        "student-data-report.json": {
            "loss_mask_data_gate_passed": True,
            "dataset_hash": dataset_hash,
            "examples": 1,
            "truncation": False,
            "teacher_reasoning_exported": False,
            "tokenizer_hashes": preflight.TOKENIZER_HASHES,
        },
        "trainer-probe.json": {
            "passed": True,
            "real_data_hash": dataset_hash,
            "real_rows_checked": 1,
            "actual_trainer_real_data_masks_verified": True,
            "collator_masks_verified": True,
            "adapter_save_reload_verified": True,
            "exact_adapter_resume_verified": True,
            "small_fixture_training_step_verified": True,
            "trainer_dataset_unchanged": True,
            "unique_lora_modules": 154,
            "shared_loops": 2,
            "physical_layers": 22,
            "planned_trainable_parameters": 23969792,
        },
        "source-dataset-report.json": {
            "dataset_hash": dataset_hash,
            "examples": 1,
            "teacher": preflight.TEACHER,
            "student": preflight.MODEL,
            "view_manifest": {
                "dataset_hash": dataset_hash,
                "parent_dataset_hash": "synthetic-parent",
                "target_unchanged": True,
                "split_hash": split_hash,
            },
        },
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value))
    manifest = {
        "student": preflight.MODEL,
        "student_revision": preflight.REVISION,
        "teacher": preflight.TEACHER,
        "files": {name: preflight.file_hash(tmp_path / name) for name in documents},
    }
    (tmp_path / "bundle-manifest.json").write_text(json.dumps(manifest))
    return tmp_path, preflight.file_hash(tmp_path / "bundle-manifest.json")


def test_bundle_without_approval_does_not_authorize_training(bundle):
    root, manifest_hash = bundle
    assert len(preflight.validate_bundle(root, manifest_hash, require_approval=False)[2]) == 1
    with pytest.raises(ValueError, match="approval"):
        preflight.validate_bundle(root, manifest_hash, require_approval=True)


def test_tamper_after_manifest_rejected(bundle):
    root, manifest_hash = bundle
    (root / "symbolic-sft.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        preflight.validate_bundle(root, manifest_hash, require_approval=False)
    with pytest.raises(ValueError, match="manifest"):
        preflight.validate_bundle(root, "wrong", require_approval=False)


def test_native_tokens_and_prefix_are_revalidated():
    example, row = example_and_row()

    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "prefix" if add_generation_prompt else "prefix-completion"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [166100, 42] if text == "prefix" else [166100, 42, 51, 166101]

    preflight.validate_native([example], [row], Tokenizer())
    row["input_ids"][-1] = 1
    with pytest.raises(ValueError, match="tokens"):
        preflight.validate_native([example], [row], Tokenizer())
