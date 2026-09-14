"""Independent read-only audit of the fixed local score; never called by the solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tarfile
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_grid(grid):
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        return False
    if not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30:
        return False
    return all(
        isinstance(row, list)
        and len(row) == len(grid[0])
        and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
        for row in grid
    )


def audit(workspace: Path, split_path: Path, data_path: Path):
    identity = json.loads((workspace / "identity.json").read_text())
    split = json.loads(split_path.read_text())
    submission = json.loads((workspace / "submission.json").read_text())
    expected_ids = split["groups"]["development"][:20]
    if len(expected_ids) != 20 or identity["cohort_ids"] != expected_ids:
        raise ValueError("cohort differs from the original frozen 20 development tasks")
    if set(submission) != set(expected_ids) or digest(split) != identity["split_hash"]:
        raise ValueError("submission membership or frozen split changed")
    teacher = identity["config"]["teacher"]
    if teacher["model"] != "gpt-6-astra" or teacher["auth_mode"] != "chatgpt_subscription":
        raise ValueError("incorrect model or authentication lineage")
    archived_hashes = {}
    with tarfile.open(workspace / "source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (
                member.isfile()
                and path.parent.as_posix() == "src/arc_agent"
                and path.suffix == ".py"
            ):
                stream = archive.extractfile(member)
                archived_hashes[path.name] = hashlib.sha256(stream.read()).hexdigest()
    if digest(archived_hashes) != identity["source_hash"]:
        raise ValueError("archived source does not reproduce the recorded code identity")
    # Immutable reads are allowed only when no WAL exists, for a stopped/checkpointed DB.
    database = workspace / "run.sqlite3"
    suffix = (
        "?mode=ro"
        if database.with_name(database.name + "-wal").exists()
        else ("?mode=ro&immutable=1")
    )
    with sqlite3.connect(database.resolve().as_uri() + suffix, uri=True) as db:
        states = {
            key: json.loads(payload) for key, payload in db.execute("SELECT key,payload FROM work")
        }
    correct, total, strict, task_sum, completed, calls = 0, 0, 0, 0.0, 0, 0
    for task_id in expected_ids:
        task = json.loads((data_path / f"{task_id}.json").read_text())
        normalized = {
            "task_id": task_id,
            "train": [{"input": p["input"], "output": p.get("output")} for p in task["train"]],
            "test": [{"input": p["input"], "output": p.get("output")} for p in task["test"]],
        }
        if digest(normalized) != split["task_hashes"][task_id]:
            raise ValueError("task data differs from the frozen source")
        if len(submission[task_id]) != len(task["test"]):
            raise ValueError("missing or extra test input predictions")
        hits = []
        for pair, attempts in zip(task["test"], submission[task_id], strict=True):
            if set(attempts) != {"attempt_1", "attempt_2"}:
                raise ValueError("exactly two named predictions required")
            if not all(valid_grid(g) for g in attempts.values()):
                raise ValueError("invalid predicted grid")
            hits.append(any(grid == pair["output"] for grid in attempts.values()))
        total += len(hits)
        correct += sum(hits)
        strict += int(all(hits))
        task_sum += sum(hits) / len(hits)
        completed += bool(states.get(f"task:{task_id}", {}).get("done"))
    dispatched_tasks = set()
    for key, call in states.items():
        if not key.startswith("call:"):
            continue
        calls += 1
        dispatched_tasks.add(key.split(":")[1])
        if call["request"]["model"] != "gpt-6-astra":
            raise ValueError("non-Astra request in scored run")
        snapshot = call.get("snapshot")
        if snapshot and snapshot["body"].get("provider") != "chatgpt_subscription":
            raise ValueError("non-subscription response in scored run")
    if total != 22:
        raise ValueError("the original cohort must contain 22 test inputs")
    if not dispatched_tasks <= set(expected_ids):
        raise ValueError("requests outside the frozen cohort")
    result = {
        "cohort_verified": True,
        "source_archive_verified": True,
        "calls": calls,
        "dispatched_tasks": len(dispatched_tasks),
        "completed_tasks": completed,
        "task_status_counts": dict(Counter(
            states.get(f"task:{task_id}", {}).get("status", "not_completed")
            for task_id in expected_ids
        )),
        "unknown_usage_calls": sum(
            states.get(f"task:{task_id}", {}).get("unknown_usage_calls", 0)
            for task_id in expected_ids
        ),
        "correct_outputs": correct,
        "total_outputs": total,
        "exact_match": correct / total,
        "task_mean_exact_match": task_sum / 20,
        "strict_correct_tasks": strict,
        "strict_task_accuracy": strict / 20,
        "threshold_met": completed == 20 and len(dispatched_tasks) == 20
        and correct / total >= 0.7 and strict / 20 >= 0.7,
        "scope": "Astra subscription local harness; not offline Nanbeige",
        "process_termination_requires_separate_verification": True,
    }
    report = json.loads((workspace / "report.json").read_text())
    result["saved_report_matches"] = all(
        report[k] == result[k]
        for k in ("correct_outputs", "exact_match", "task_mean_exact_match", "strict_task_accuracy")
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("runs/v6/astra-local-eval-01"))
    parser.add_argument("--split", type=Path, default=Path("runs/v4/split.json"))
    parser.add_argument("--data", type=Path, default=Path("data/ARC-AGI-2/data/training"))
    args = parser.parse_args()
    print(json.dumps(audit(args.workspace, args.split, args.data), indent=2))
