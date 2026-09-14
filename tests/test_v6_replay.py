from __future__ import annotations

import copy
import json

import pytest

from arc_agent.models import ArcTask
from arc_agent.v2_codex import _response_id, _snapshot
from arc_agent.v4_state import ArtifactStore, blind_task
from arc_agent.v6_eval import SolverResponse, evaluate_candidate, initial_prompt, select_predictions
from scripts.replay_v6_run import REPAIR_REQUEST, replay_task


@pytest.fixture
def saved(tmp_path):
    task = ArcTask.model_validate(
        {
            "task_id": "fixture",
            "train": [
                {"input": [[1, 0]], "output": [[0, 1]]},
                {"input": [[2, 0, 0]], "output": [[0, 0, 2]]},
            ],
            "test": [{"input": [[3, 0]], "output": [[0, 3]]}],
        }
    )
    response = SolverResponse(
        symbolic_model="Reflect each row and preserve colors.",
        python_source="def solve(train, grid):\n    return [list(reversed(row)) for row in grid]",
        direct_predictions=[["03"]],
        alternative_predictions=None,
    )
    result = evaluate_candidate(response, blind_task(task), 0)
    snapshot = _snapshot(
        response_id=_response_id("thread", "turn", "request"),
        status="completed",
        thread_id="thread",
        turn_id="turn",
        text=response.model_dump_json(),
    ).model_dump(mode="json")
    record = {
        "task_id": task.task_id,
        "split": "development",
        "training_export_allowed": False,
        "response": snapshot,
        "verification": result,
    }
    store = ArtifactStore(tmp_path / "artifacts")
    state = {
        "round": 1,
        "done": True,
        "status": "verified",
        "records": [store.put(record)],
        "candidates": copy.deepcopy(result["candidates"]),
        "feedback": result["feedback"],
        "previous_id": snapshot["response_id"],
    }
    calls = {
        "call:fixture:0": {
            "request": {
                "prompt": initial_prompt(task), "model": "gpt-6-astra", "request_key": "request"
            },
            "snapshot": snapshot,
        }
    }
    submission = select_predictions(blind_task(task), state["candidates"])
    return task, state, calls, store, submission


def run_replay(saved):
    task, state, calls, store, submission = saved
    return replay_task(task, state, calls, store.root, submission)


def test_replay_does_not_use_test_labels(saved):
    saved[0].test[0].output = [[9]]
    result = run_replay(saved)
    assert result["responses"] == result["valid_responses"] == result["verified_programs"] == 1


def test_replay_rejects_test_answers_in_prompt(saved):
    request = saved[2]["call:fixture:0"]["request"]
    prompt = json.loads(request["prompt"])
    prompt["test"][0]["output"] = ["03"]
    request["prompt"] = json.dumps(prompt)
    with pytest.raises(ValueError, match="label-blind"):
        run_replay(saved)


def test_replay_rejects_changed_candidates(saved):
    saved[1]["candidates"][0]["predictions"] = [[[8]]]
    with pytest.raises(ValueError, match="saved candidates differ"):
        run_replay(saved)


def test_replay_rejects_development_as_training(saved):
    state, store = saved[1], saved[3]
    record = store.get(state["records"][0])
    record["training_export_allowed"] = True
    state["records"] = [store.put(record)]
    with pytest.raises(ValueError, match="isolation"):
        run_replay(saved)


def test_replay_rejects_changed_submission(saved):
    saved[4][0]["attempt_1"] = [[8]]
    with pytest.raises(ValueError, match="submission differs"):
        run_replay(saved)


def test_replay_does_not_accept_unfinished_task(saved):
    saved[1]["done"] = False
    with pytest.raises(ValueError, match="unfinished"):
        run_replay(saved)


@pytest.mark.parametrize("cross_thread", [False, True])
def test_replay_checks_repair_feedback_and_conversation(saved, cross_thread):
    task, state, calls, store, submission = saved
    original = store.get(state["records"][0])
    final_response = SolverResponse.model_validate_json(
        original["response"]["body"]["output"][0]["content"][0]["text"]
    )
    first_response = final_response.model_copy(update={"python_source": None})
    first_result = evaluate_candidate(first_response, blind_task(task), 0)
    second_result = evaluate_candidate(final_response, blind_task(task), 1)
    candidates, references = [], []
    for index, (response, result) in enumerate(
        [(first_response, first_result), (final_response, second_result)]
    ):
        thread_id = "different-thread" if cross_thread and index == 1 else "thread"
        turn_id, request_key = f"turn-{index}", f"request-{index}"
        snapshot = _snapshot(
            response_id=_response_id(thread_id, turn_id, request_key),
            status="completed",
            thread_id=thread_id,
            turn_id=turn_id,
            text=response.model_dump_json(),
        ).model_dump(mode="json")
        prompt = (
            initial_prompt(task) if index == 0 else json.dumps(
                {
                    "demonstration_verifier_feedback": first_result["feedback"],
                    "request": REPAIR_REQUEST,
                }
            )
        )
        calls[f"call:fixture:{index}"] = {
            "request": {"prompt": prompt, "model": "gpt-6-astra", "request_key": request_key},
            "snapshot": snapshot,
        }
        references.append(store.put({**original, "response": snapshot, "verification": result}))
        candidates.extend(result["candidates"])
    state.update(
        round=2, records=references, candidates=candidates,
        previous_id=snapshot["response_id"], feedback=second_result["feedback"],
    )
    submission[:] = select_predictions(blind_task(task), candidates)
    if cross_thread:
        with pytest.raises(ValueError, match="continuity changed"):
            run_replay(saved)
    else:
        assert run_replay(saved)["responses"] == 2
