"""Measure a lossless symbolic cell encoding locally; never infer, train, or upload.

This is a prototype, not a solver protocol or training-data exporter. Complete
states must round-trip exactly. Truncated replies remain explicitly incomplete;
their measurements cannot establish that a new generation would finish or solve.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import atomic_json
from arc_agent.v5_dataset import TOKENIZER_HASHES
from arc_agent.v5_symbolic import SymbolicModel

CODEC = "symbolic_cell_triples_v1_prototype"
COORD = r"(?:0|[1-9]|[12][0-9])"
CELL_TEXT = re.compile(rf'\{{"row":({COORD}),"column":({COORD}),"color":([0-9])\}}')


def packed_state(state: dict) -> dict:
    SymbolicModel.model_validate(state)
    result = copy.deepcopy(state)
    for entity in result["entities"]:
        entity["cells"] = [[c["row"], c["column"], c["color"]] for c in entity["cells"]]
    return result


def unpacked_state(state: dict) -> dict:
    result = copy.deepcopy(state)
    if not isinstance(result, dict) or not isinstance(result.get("entities"), list):
        raise ValueError("packed state requires entities")
    for entity in result["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("cells"), list):
            raise ValueError("packed entity requires cells")
        cells = []
        for cell in entity["cells"]:
            if (
                not isinstance(cell, list)
                or len(cell) != 3
                or any(type(v) is not int for v in cell)
            ):
                raise ValueError("packed cells require exactly three integers")
            cells.append(dict(zip(("row", "column", "color"), cell, strict=True)))
        entity["cells"] = cells
    SymbolicModel.model_validate(result)
    return result


def packed_schema(schema: dict) -> dict:
    """Transform only Cell records, preserving their per-grid scalar bounds."""
    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and set(node.get("properties", {})) == {
                "row",
                "column",
                "color",
            }:
                fields = node["properties"]
                node.clear()
                node.update(
                    type="array",
                    prefixItems=[fields[k] for k in ("row", "column", "color")],
                    minItems=3,
                    maxItems=3,
                )
            else:
                for value in node.values():
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def rewrite_complete_cell_objects(text: str) -> tuple[str, int]:
    """Prefix-only diagnostic: never closes, accepts or resumes an incomplete JSON reply."""
    return CELL_TEXT.subn(lambda match: "[" + ",".join(match.groups()) + "]", text)


def measure(workspace: Path, tokenizer) -> dict:
    complete, truncated = [], []
    for path in sorted((workspace / "artifacts").glob("*.json")):
        value = json.loads(path.read_text())
        if content_hash(value) != path.stem:
            raise ValueError(f"artifact hash mismatch: {path}")
        raw, final = value.get("raw", {}), value.get("final")
        if value.get("reasoning") != "" or not isinstance(final, str):
            continue
        record = {"artifact_ref": path.stem, "raw_generated_tokens": value["generated_tokens"]}
        try:
            state = json.loads(final)
            SymbolicModel.model_validate(state)
        except ValueError:
            if raw.get("stop_type") != "limit" or not final.startswith('{"entities":'):
                continue
            compact, cells = rewrite_complete_cell_objects(final)
            record.update(
                complete=False,
                round_trip_verified=False,
                complete_cell_objects_rewritten=cells,
                prompt_truncated=raw.get("truncated"),
                reached_definitions='"definitions":' in final,
            )
            original = final
            target = truncated
        else:
            packed = packed_state(state)
            if unpacked_state(packed) != state:
                raise ValueError("cell encoding changed the symbolic state")
            original = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
            compact = json.dumps(packed, separators=(",", ":"), ensure_ascii=False)
            record.update(
                complete=True,
                round_trip_verified=True,
                canonical_state_sha256=content_hash(state),
                packed_state_sha256=content_hash(packed),
                cell_count=sum(len(e["cells"]) for e in state["entities"]),
            )
            target = complete
        original_tokens = len(tokenizer.encode(original, add_special_tokens=False).ids)
        compact_tokens = len(tokenizer.encode(compact, add_special_tokens=False).ids)
        record.update(
            original_reencoded_tokens=original_tokens,
            packed_reencoded_tokens=compact_tokens,
            tokens_saved=original_tokens - compact_tokens,
            fraction_saved=(original_tokens - compact_tokens) / original_tokens,
        )
        target.append(record)
    return {
        "workspace": str(workspace),
        "complete_states": complete,
        "truncated_prefixes": truncated,
        "complete_state_count": len(complete),
        "truncated_prefix_count": len(truncated),
    }


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path(".runtime/nanbeige/model-hf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("choose a new probe output; preserve earlier evidence")
    for filename, expected in TOKENIZER_HASHES.items():
        if file_hash(args.tokenizer / filename) != expected:
            raise ValueError("tokenizer differs from the pinned Nanbeige revision")
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    report = {
        "codec": CODEC,
        "script_sha256": file_hash(Path(__file__)),
        "tokenizer_hashes": TOKENIZER_HASHES,
        **measure(args.workspace, tokenizer),
        "training_eligible": False,
        "training_data_exported": False,
        "solver_protocol_changed": False,
        "model_inference_performed": False,
        "instruction_overhead_included": False,
        "completion_or_accuracy_gain_proven": False,
        "limitations": (
            "Payload re-encoding counts only. Truncated prefixes are not repaired or completed. "
            "No inference, semantic correctness claim, training, data upload or paid job."
        ),
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"complete_states", "truncated_prefixes"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
