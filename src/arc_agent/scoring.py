from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from arc_agent.models import ArcTask, Grid, Submission


def grids_equal(left: Grid, right: Grid) -> bool:
    return left == right


@dataclass(frozen=True)
class Score:
    pass_at_2: float
    strict_task_accuracy: float
    correct_outputs: int
    total_outputs: int
    correct_tasks: int
    total_tasks: int


def score_submission(tasks: Iterable[ArcTask], submission: Submission) -> Score:
    correct_outputs = 0
    total_outputs = 0
    correct_tasks = 0
    total_tasks = 0
    for task in tasks:
        attempts = submission.get(task.task_id)
        if attempts is None or len(attempts) != len(task.test):
            raise ValueError(f"submission is incomplete for {task.task_id}")
        task_correct = True
        for pair, attempt in zip(task.test, attempts, strict=True):
            if pair.output is None:
                raise ValueError(f"{task.task_id} has no test ground truth")
            matched = grids_equal(attempt.attempt_1, pair.output) or grids_equal(
                attempt.attempt_2, pair.output
            )
            correct_outputs += int(matched)
            total_outputs += 1
            task_correct = task_correct and matched
        correct_tasks += int(task_correct)
        total_tasks += 1
    return Score(
        pass_at_2=correct_outputs / total_outputs if total_outputs else 0.0,
        strict_task_accuracy=correct_tasks / total_tasks if total_tasks else 0.0,
        correct_outputs=correct_outputs,
        total_outputs=total_outputs,
        correct_tasks=correct_tasks,
        total_tasks=total_tasks,
    )
