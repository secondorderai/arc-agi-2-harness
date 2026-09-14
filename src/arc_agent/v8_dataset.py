"""Lossless compact input views for the verified symbolic SFT targets."""

import json
from collections import Counter
from pathlib import Path

from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import atomic_json
from arc_agent.v5_dataset import prepare_student
from arc_agent.v8_reasoner import compact_symbolic_prompt


def prepare_compact_student(source: Path, destination: Path, split_path: Path, tokenizer: Path):
    original = json.loads((source / "symbolic-sft.json").read_text())
    parent = json.loads((source / "dataset-report.json").read_text())
    split = json.loads(split_path.read_text())
    if content_hash(original) != parent["dataset_hash"]:
        raise ValueError("parent dataset changed; use a consistent verified checkpoint")
    if parent["teacher"] != "gpt-6-astra" or parent["student"] != "Nanbeige/Nanbeige4.2-3B":
        raise ValueError("incompatible teacher/student lineage")
    examples = []
    for example in original:
        if example["source_task_id"] not in split["groups"]["training"]:
            raise ValueError("development or lockbox record cannot become SFT data")
        examples.append({**example, "prompt": compact_symbolic_prompt(example["prompt"])})
    identity = {
        "parent_dataset_hash": content_hash(original),
        "dataset_hash": content_hash(examples),
        "split_hash": file_hash(split_path),
        "view": "lossless_digit_rows_v1",
        "target_unchanged": True,
    }
    if (destination / "view-manifest.json").exists() and json.loads(
        (destination / "view-manifest.json").read_text()
    ) != identity:
        raise ValueError("existing student view has different inputs; choose a new destination")
    atomic_json(destination / "view-manifest.json", identity)
    atomic_json(destination / "symbolic-sft.json", examples)
    atomic_json(
        destination / "dataset-report.json",
        {
            **parent,
            "dataset_hash": content_hash(examples),
            "view_manifest": identity,
            "category_counts": dict(Counter(e["category"] for e in examples)),
        },
    )
    report = prepare_student(destination, tokenizer)
    return {**report, "view_manifest": identity}
