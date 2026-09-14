"""Local observation-catalog probe; no solver changes, model calls or training data.

Measure alternative deterministic segmentations independently. Catalog references
retain exact cell masks externally; they do not prove that a segmentation is useful.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

from arc_agent.models import ArcTask
from arc_agent.object_ir import extract_objects, infer_background
from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import ArtifactStore, atomic_json, blind_task
from arc_agent.v4_tools import task_grids
from arc_agent.v5_dataset import TOKENIZER_HASHES


def catalog_for_grid(grid, grid_id, *, connectivity=4, multicolor=False, background=None):
    background = infer_background(grid) if background is None else background
    objects = extract_objects(
        grid, background=background, connectivity=connectivity, multicolor=multicolor
    )
    rendered = [[background] * len(grid[0]) for _ in grid]
    records = []
    for obj in objects:
        cells = [[r, c, color] for (r, c), color in zip(obj.cells, obj.cell_colors, strict=True)]
        for row, column, color in cells:
            rendered[row][column] = color
        records.append(
            {
                "id": f"o{obj.object_id}",
                "bbox": list(obj.bbox.as_tuple),
                "area": obj.area,
                "colors": list(obj.colors),
                "cells": cells,
            }
        )
    if rendered != grid:
        raise ValueError("segmentation lost authoritative grid information")
    return {
        "kind": "observation_catalog_v1_probe",
        "grid_id": grid_id,
        "grid": copy.deepcopy(grid),
        "policy": {
            "background": background,
            "connectivity": connectivity,
            "multicolor": multicolor,
        },
        "objects": records,
    }


def checked_catalog(catalog, reference):
    if content_hash(catalog) != reference:
        raise ValueError("stale or changed observation catalog")


def catalog_page(catalog, reference, *, offset=0, size=24):
    checked_catalog(catalog, reference)
    count = len(catalog["objects"])
    if type(offset) is not int or type(size) is not int or not 1 <= size <= 24:
        raise ValueError("invalid catalog page")
    if not 0 <= offset < max(1, count):
        raise ValueError("catalog offset outside object list")
    stop = min(offset + size, count)
    return {
        "catalog_ref": reference,
        "grid_id": catalog["grid_id"],
        "policy": dict(catalog["policy"]),
        "object_count": count,
        "offset": offset,
        "objects": [
            {key: copy.deepcopy(value) for key, value in obj.items() if key != "cells"}
            for obj in catalog["objects"][offset:stop]
        ],
        "next_offset": stop if stop < count else None,
    }


def resolve_object(catalog, reference, object_id):
    checked_catalog(catalog, reference)
    for obj in catalog["objects"]:
        if obj["id"] == object_id:
            return copy.deepcopy(obj)
    raise ValueError("undefined catalog object")


def task_catalogs(task):
    return [
        catalog_for_grid(grid, name, connectivity=connectivity, multicolor=multicolor)
        for name, grid in task_grids(blind_task(task)).items()
        for connectivity in (4, 8)
        for multicolor in (False, True)
    ]


def measure(catalog, count_tokens):
    reference = content_hash(catalog)
    pages, offset, reconstructed = [], 0, []
    while True:
        page = catalog_page(catalog, reference, offset=offset)
        reconstructed.extend(
            resolve_object(catalog, reference, obj["id"]) for obj in page["objects"]
        )
        pages.append(count_tokens(json.dumps(page, separators=(",", ":"))))
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    if reconstructed != catalog["objects"]:
        raise ValueError("catalog pagination omitted or changed observations")
    return {
        "catalog_ref": reference,
        "grid_id": catalog["grid_id"],
        "policy": catalog["policy"],
        "object_count": len(reconstructed),
        "legacy_view_omitted_objects": max(0, len(reconstructed) - 24),
        "components_exceeding_legacy_128_cell_bound": sum(
            obj["area"] > 128 for obj in reconstructed
        ),
        "page_tokens": pages,
        "all_pages_tokens": sum(pages),
        "explicit_cell_payload_tokens": count_tokens(
            json.dumps(
                [{"id": obj["id"], "cells": obj["cells"]} for obj in reconstructed],
                separators=(",", ":"),
            )
        ),
        "exact_grid_reconstruction_verified": True,
        "all_objects_retrievable": True,
    }


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/ARC-AGI-2/data/training"))
    parser.add_argument("--split", type=Path, default=Path("runs/v4/split.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path(".runtime/nanbeige/model-hf"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.destination.is_symlink():
        raise ValueError("preserve previous evidence; choose a new destination")
    identity = json.loads((args.baseline / "identity.json").read_text())
    split = json.loads(args.split.read_text())
    if (
        not re.fullmatch(r"[0-9a-f]{8}", args.task_id)
        or args.task_id not in identity["cohort"]
        or args.task_id not in split["groups"]["development"]
        or identity["split_hash"] != file_hash(args.split)
    ):
        raise ValueError("probe requires a task in this frozen development cohort")
    raw = json.loads((args.data / f"{args.task_id}.json").read_text())
    task = ArcTask.model_validate({"task_id": args.task_id, **raw})
    if content_hash(task.model_dump(mode="json")) != split["task_hashes"][args.task_id]:
        raise ValueError("source task changed after split freeze")
    for name, expected in TOKENIZER_HASHES.items():
        if file_hash(args.tokenizer / name) != expected:
            raise ValueError("pinned Nanbeige tokenizer changed")
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    catalogs = task_catalogs(task)
    report = {
        "task_id": task.task_id,
        "blind_task_sha256": content_hash(blind_task(task).model_dump(mode="json")),
        "script_sha256": file_hash(Path(__file__)),
        "object_extractor_sha256": file_hash(Path("src/arc_agent/object_ir.py")),
        "tokenizer_hashes": TOKENIZER_HASHES,
        "catalogs": [
            measure(c, lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
            for c in catalogs
        ],
        "training_eligible": False,
        "model_inference_performed": False,
        "solver_protocol_changed": False,
        "accuracy_or_runtime_gain_proven": False,
        "limitations": (
            "One-task observation-interface probe; policies measured independently, "
            "not semantic truth. "
            "Catalog input/retrieval costs must be measured at solve time. "
            "No solver, training, upload or inference is performed."
        ),
    }
    required = 10 * 1024**3 + sum(len(json.dumps(value).encode()) for value in [report, *catalogs])
    if shutil.disk_usage(args.destination.parent).free < required:
        raise ValueError("preserve 10 GiB free disk after the probe")
    args.destination.mkdir()
    store = ArtifactStore(args.destination / "artifacts")
    for catalog in catalogs:
        reference = store.put(catalog)
        if store.get(reference) != catalog:
            raise ValueError("stored catalog changed")
    atomic_json(args.destination / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.destination / "report.json"),
                "catalogs": len(catalogs),
                "training_eligible": False,
                "solver_protocol_changed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
