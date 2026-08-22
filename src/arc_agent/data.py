from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from arc_agent.models import ArcPair, ArcTask, Attempt, Submission


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        for child in sorted(p for p in path.rglob("*.json") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()


def _task_from_payload(task_id: str, payload: dict[str, Any]) -> ArcTask:
    return ArcTask(
        task_id=task_id,
        train=[ArcPair.model_validate(pair) for pair in payload["train"]],
        test=[ArcPair.model_validate(pair) for pair in payload["test"]],
    )


def load_tasks(path: str | Path) -> list[ArcTask]:
    """Load an ARC directory, one task JSON, or Kaggle's challenges dictionary."""
    source = Path(path)
    if source.is_dir():
        return [
            _task_from_payload(file.stem, json.loads(file.read_text()))
            for file in sorted(source.glob("*.json"))
        ]
    payload = json.loads(source.read_text())
    if "train" in payload and "test" in payload:
        return [_task_from_payload(source.stem, payload)]
    return [_task_from_payload(task_id, task) for task_id, task in sorted(payload.items())]


def attach_solutions(tasks: Iterable[ArcTask], solutions_path: str | Path) -> list[ArcTask]:
    solutions = json.loads(Path(solutions_path).read_text())
    enriched: list[ArcTask] = []
    for task in tasks:
        outputs = solutions.get(task.task_id)
        if outputs is None or len(outputs) != len(task.test):
            raise ValueError(f"missing or invalid solutions for {task.task_id}")
        enriched.append(
            task.model_copy(
                update={
                    "test": [
                        ArcPair(input=pair.input, output=output)
                        for pair, output in zip(task.test, outputs, strict=True)
                    ]
                }
            )
        )
    return enriched


def deterministic_split(
    tasks: Iterable[ArcTask], *, validation_percent: int = 20
) -> tuple[list[ArcTask], list[ArcTask]]:
    learn: list[ArcTask] = []
    validate: list[ArcTask] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        bucket = int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16) % 100
        (validate if bucket < validation_percent else learn).append(task)
    return learn, validate


def submission_to_json(submission: Submission) -> dict[str, list[dict[str, Any]]]:
    return {
        task_id: [attempt.model_dump(mode="json") for attempt in attempts]
        for task_id, attempts in submission.items()
    }


def write_submission(
    submission: Submission,
    tasks: Iterable[ArcTask],
    output_path: str | Path,
) -> Path:
    expected = {task.task_id: len(task.test) for task in tasks}
    if set(submission) != set(expected):
        missing = sorted(set(expected) - set(submission))
        extra = sorted(set(submission) - set(expected))
        raise ValueError(f"submission task mismatch; missing={missing}, extra={extra}")
    for task_id, count in expected.items():
        if len(submission[task_id]) != count:
            raise ValueError(f"{task_id} needs {count} test outputs")
        for raw in submission[task_id]:
            Attempt.model_validate(raw)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(submission_to_json(submission), separators=(",", ":")))
    return target
