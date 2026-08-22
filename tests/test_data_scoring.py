from __future__ import annotations

import json

import pytest

from arc_agent.data import deterministic_split, load_tasks, write_submission
from arc_agent.models import ArcPair, ArcTask, Attempt
from arc_agent.scoring import score_submission


def test_load_directory_and_kaggle_dictionary(tmp_path):
    payload = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "abc.json").write_text(json.dumps(payload))
    assert load_tasks(task_dir)[0].task_id == "abc"
    challenge = tmp_path / "test_challenges.json"
    challenge.write_text(json.dumps({"xyz": {**payload, "test": [{"input": [[3]]}]}}))
    tasks = load_tasks(challenge)
    assert tasks[0].task_id == "xyz"
    assert tasks[0].test[0].output is None


def test_split_is_deterministic(identity_task):
    tasks = [identity_task.model_copy(update={"task_id": str(index)}) for index in range(100)]
    first = deterministic_split(tasks)
    second = deterministic_split(reversed(tasks))
    assert [task.task_id for task in first[0]] == [task.task_id for task in second[0]]
    assert 10 <= len(first[1]) <= 30


def test_official_pass_at_2_and_strict_task_accuracy(tmp_path):
    tasks = [
        ArcTask(
            task_id="a",
            train=[ArcPair(input=[[0]], output=[[1]])],
            test=[
                ArcPair(input=[[2]], output=[[3]]),
                ArcPair(input=[[4]], output=[[5]]),
            ],
        ),
        ArcTask(
            task_id="b",
            train=[ArcPair(input=[[0]], output=[[1]])],
            test=[ArcPair(input=[[6]], output=[[7]])],
        ),
    ]
    submission = {
        "a": [
            Attempt(attempt_1=[[0]], attempt_2=[[3]]),
            Attempt(attempt_1=[[0]], attempt_2=[[0]]),
        ],
        "b": [Attempt(attempt_1=[[7]], attempt_2=[[0]])],
    }
    score = score_submission(tasks, submission)
    assert score.pass_at_2 == pytest.approx(2 / 3)
    assert score.strict_task_accuracy == pytest.approx(1 / 2)
    output = write_submission(submission, tasks, tmp_path / "submission.json")
    assert set(json.loads(output.read_text())) == {"a", "b"}


def test_submission_requires_every_task(identity_task, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        write_submission({}, [identity_task], tmp_path / "submission.json")
