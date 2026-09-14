"""Replay saved V6 responses without LLM calls; complements the independent score audit."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from arc_agent.models import ArcTask
from arc_agent.v2_codex import _parse_response_id
from arc_agent.v2_openai import response_output_text
from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import blind_task
from arc_agent.v5_pipeline import source_identity
from arc_agent.v6_eval import (
    SolverResponse,
    evaluate_candidate,
    initial_prompt,
    select_predictions,
)

REPAIR_REQUEST = (
    "Revise the symbolic model and rule using the counterexamples. "
    "Recheck every demonstration; consider a different abstraction if needed. "
    "Return complete code and predictions, not a patch. Test labels are unavailable."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replay_task(task, state, calls, artifacts, submission):
    """Re-execute only the sandboxed witnesses; never expose test labels to verification."""
    require(state.get("done"), f"{task.task_id}: task is unfinished")
    require(
        state.get("status") in {"verified", "unresolved", "provider_failed", "timed_out"},
        f"{task.task_id}: unexpected terminal status",
    )
    task = blind_task(task)
    candidates, feedback, response_ids, thread_ids = [], None, [], set()
    valid_responses, verified_programs = 0, 0
    require(len(state.get("records", [])) == state["round"], "response record count differs")
    for index in range(state["round"]):
        call = calls[f"call:{task.task_id}:{index}"]
        expected = (
            initial_prompt(task)
            if index == 0
            else json.dumps(
                {"demonstration_verifier_feedback": feedback, "request": REPAIR_REQUEST}
            )
        )
        require(call["request"]["prompt"] == expected, "prompt is not the label-blind request")
        require(call["request"]["model"] == "gpt-6-astra", "non-Astra request")
        snapshot = call["snapshot"]
        require(snapshot["status"] == "completed", "recorded response is not completed")
        body = snapshot["body"]
        require(body.get("provider") == "chatgpt_subscription", "non-subscription response")
        thread_id, turn_id, request_key = _parse_response_id(snapshot["response_id"])
        require(
            body["codex_thread_id"] == thread_id
            and body["codex_turn_id"] == turn_id
            and call["request"]["request_key"] == request_key,
            "response identity differs from request",
        )
        thread_ids.add(thread_id)
        response_ids.append(snapshot["response_id"])
        try:
            response = SolverResponse.model_validate_json(response_output_text(body))
            result = evaluate_candidate(response, task, index)
            valid_responses += 1
        except ValueError as exc:
            result = {
                "candidates": [], "verified": False, "feedback": str(exc), "errors": [str(exc)]
            }
        verification = result.get("verification")
        if verification:
            require(verification["source_tests_total"] == 0, "test labels used in verification")
        verified_programs += bool(result["verified"])
        reference = state["records"][index]
        require(
            len(reference) == 64 and all(c in "0123456789abcdef" for c in reference),
            "invalid artifact reference",
        )
        record = json.loads((artifacts / f"{reference}.json").read_text())
        require(content_hash(record) == reference, "artifact hash mismatch")
        require(
            record["task_id"] == task.task_id
            and record["split"] == "development"
            and record["training_export_allowed"] is False,
            "development artifact isolation differs",
        )
        require(record["response"] == snapshot, "artifact response differs from checkpoint")
        for key in ("candidates", "verified", "feedback", "errors"):
            require(record["verification"][key] == result[key], f"replayed {key} differs")
        candidates.extend(result["candidates"])
        feedback = result["feedback"]
    require(len(thread_ids) <= 1, "task reasoning continuity changed conversations")
    require(len(set(response_ids)) == len(response_ids), "response reused across rounds")
    require(candidates == state["candidates"], "saved candidates differ from replay")
    if response_ids:
        require(state["previous_id"] == response_ids[-1], "continuation cursor differs")
        require(state["feedback"] == feedback, "saved verifier feedback differs")
    require(select_predictions(task, candidates) == submission, "submission differs from replay")
    return {
        "responses": len(response_ids),
        "valid_responses": valid_responses,
        "verified_programs": verified_programs,
        "thread_ids": thread_ids,
        "response_ids": response_ids,
    }


def replay(workspace):
    identity = json.loads((workspace / "identity.json").read_text())
    active = json.loads((workspace / "active.json").read_text())
    require(active == {"pid": None, "status": "completed"}, "run has not recorded completion")
    require(source_identity() == identity["source_hash"], "current source differs from run")
    teacher = identity["config"]["teacher"]
    require(teacher["model"] == "gpt-6-astra", "incorrect model")
    require(teacher["auth_mode"] == "chatgpt_subscription", "incorrect authentication mode")
    require(file_hash(Path(teacher["codex_cli"])) == identity["runtime_hash"], "runtime changed")
    database = workspace / "run.sqlite3"
    suffix = (
        "?mode=ro" if database.with_name(database.name + "-wal").exists()
        else "?mode=ro&immutable=1"
    )
    with sqlite3.connect(database.resolve().as_uri() + suffix, uri=True) as connection:
        states = {
            key: json.loads(payload)
            for key, payload in connection.execute("SELECT key,payload FROM work")
        }
    submission = json.loads((workspace / "submission.json").read_text())
    results, all_threads, all_responses = [], set(), set()
    for task_id in identity["cohort_ids"]:
        payload = json.loads((Path(identity["config"]["data"]) / f"{task_id}.json").read_text())
        task = ArcTask.model_validate({"task_id": task_id, **payload})
        result = replay_task(
            task, states[f"task:{task_id}"], states, workspace / "artifacts", submission[task_id]
        )
        require(not all_threads.intersection(result["thread_ids"]), "conversation reused by tasks")
        require(not all_responses.intersection(result["response_ids"]), "response reused by tasks")
        all_threads.update(result.pop("thread_ids"))
        all_responses.update(result.pop("response_ids"))
        results.append(result)
    return {
        "tasks_replayed": len(results),
        **{key: sum(result[key] for result in results) for key in results[0]},
        "source_unchanged": True,
        "runtime_unchanged": True,
        "label_blind_prompts": True,
        "development_artifact_flags_verified": True,
        "submission_reproduced": True,
        "scope": "deterministic replay; run the separate independent score audit too",
        "process_termination_requires_separate_verification": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("runs/v6/astra-local-eval-01"))
    print(json.dumps(replay(parser.parse_args().workspace), indent=2))
