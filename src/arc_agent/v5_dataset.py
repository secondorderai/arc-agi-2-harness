"""Local Nanbeige-native rendering and completion-only labels, without loading weights."""

import json
from pathlib import Path

from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_reasoner import NativeTemplate
from arc_agent.v4_state import atomic_json
from arc_agent.v5_symbolic import GroundedScene, SymbolicModel

TOKENIZER_HASHES = {
    "tokenizer_config.json": "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518",
    "tokenizer.json": "1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f",
}


def tokenize_example(example: dict, template, tokenizer, max_tokens: int) -> dict:
    prompt, completion = example["prompt"], example["completion"]
    if (
        len(completion) != 1
        or completion[0].get("role") != "assistant"
        or any(message.get("role") == "assistant" for message in prompt)
    ):
        raise ValueError("expected context-only prompt and one symbolic assistant completion")
    schema = GroundedScene if example["category"] == "grounding" else SymbolicModel
    schema.model_validate_json(completion[0]["content"])
    prefix = template.render(prompt, thinking=False)
    full = template.template.render(
        messages=prompt + completion,
        tools=[],
        preserve_thinking=True,
        enable_thinking=False,
        tool_call_format="xml",
        add_generation_prompt=False,
    )
    if not full.startswith(prefix):
        raise ValueError("native chat template changed the completion boundary")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    full_ids = tokenizer.encode(full, add_special_tokens=False).ids
    if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) <= len(prefix_ids):
        raise ValueError("tokenization does not preserve a nonempty completion boundary")
    if len(full_ids) > max_tokens:
        raise ValueError(f"overlength: {len(full_ids)} tokens; restructure, never truncate")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "source_task_id": example["source_task_id"],
        "category": example["category"],
        "record_ref": example["record_ref"],
        "prompt_tokens": len(prefix_ids),
        "completion_tokens": len(full_ids) - len(prefix_ids),
    }


def prepare_student(workspace: Path, tokenizer_dir: Path, max_tokens: int = 8192) -> dict:
    from tokenizers import Tokenizer

    for filename, expected in TOKENIZER_HASHES.items():
        if file_hash(tokenizer_dir / filename) != expected:
            raise ValueError("student tokenizer differs from the pinned Nanbeige revision")
    examples = json.loads((workspace / "symbolic-sft.json").read_text())
    dataset_report = json.loads((workspace / "dataset-report.json").read_text())
    if content_hash(examples) != dataset_report["dataset_hash"]:
        raise ValueError("symbolic dataset changed after verified export")
    template = NativeTemplate(tokenizer_dir / "tokenizer_config.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    rows, excluded = [], []
    for example in examples:
        try:
            rows.append(tokenize_example(example, template, tokenizer, max_tokens))
        except ValueError as exc:
            excluded.append({"record_ref": example["record_ref"], "error": str(exc)})
    atomic_json(workspace / "nanbeige-symbolic-tokenized.json", rows)
    report = {
        "examples": len(rows),
        "excluded": excluded,
        "max_tokens": max_tokens,
        "dataset_hash": content_hash(examples),
        "tokenizer_hashes": TOKENIZER_HASHES,
        "completion_only_labels": True,
        "teacher_reasoning_exported": False,
        "truncation": False,
        "training_launched": False,
        "loss_mask_data_gate_passed": bool(rows) and not excluded,
        "trainer_loss_mask_integration_verified": False,
        "max_observed_tokens": max((len(r["input_ids"]) for r in rows), default=0),
    }
    atomic_json(workspace / "student-data-report.json", report)
    return report
