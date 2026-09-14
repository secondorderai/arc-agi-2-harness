"""Replay the training prompt boundary before exporting symbolic-only targets."""

import json

from arc_agent.v2_openai import response_output_text
from arc_agent.v5_pipeline import export_examples, student_prompt
from arc_agent.v5_symbolic import teaching_task
from arc_agent.v7_symbolic import BridgedArtifact, PythonTrainingContract, verify_artifact


def audit_training_records(run, config, tasks, split):
    """Reject altered prompts, invented targets and holdout-driven extra repairs."""
    for task in tasks:
        state = run.get(f"task:{task.task_id}") or {}
        previous, feedback, previous_id, stopped = None, None, None, False
        for index, ref in enumerate(state.get("records", [])):
            if task.task_id not in split["groups"]["training"] or stopped:
                raise ValueError(
                    "training audit: split violation or repair after stopping boundary"
                )
            visible, heldout = teaching_task(task)
            record = run.artifacts.get(ref)
            call = run.get(f"call:{task.task_id}:{index}")
            messages = student_prompt(visible, previous, feedback)
            expected_prompt = (
                PythonTrainingContract.initial_prompt(messages)
                if index == 0
                else json.dumps(
                    {
                        "visible_verifier_feedback": feedback,
                        "request": "Revise the symbolic model and optional witness.",
                    }
                )
            )
            if (
                not call
                or call["student_prompt"] != messages
                or record["prompt"] != messages
                or call["request"]["prompt"] != expected_prompt
                or call["request"]["model"] != config.teacher.model
                or record["round"] != index
                or record["task_id"] != task.task_id
                or record["category"] != ("interpretation" if index == 0 else "repair")
            ):
                raise ValueError("training audit: prompt, source or category differs from replay")
            snapshot = call.get("snapshot") or {}
            if (
                snapshot.get("status") != "completed"
                or snapshot.get("response_id") != record["teacher_response_id"]
                or record["teacher_response_id"] == previous_id
            ):
                raise ValueError("training audit: missing or reused teacher response")
            raw = response_output_text(snapshot["body"])
            if run.artifacts.get(record["raw_response_ref"]) != {"output_text": raw}:
                raise ValueError("training audit: teacher text differs from response checkpoint")
            artifact = None
            try:
                artifact = BridgedArtifact.model_validate_json(raw)
                verification = verify_artifact(artifact, visible, heldout)
            except ValueError as exc:
                verification = {
                    "accepted": False,
                    "visible_verified": False,
                    "error": str(exc)[:1200],
                }
            if record["artifact"] != (
                artifact.model_dump(mode="json") if artifact else None
            ) or record["structured_valid"] != (artifact is not None):
                raise ValueError("training audit: symbolic target differs from teacher output")
            # Re-execution uses the same sealed candidate, not a new teacher attempt. No
            # withheld outcome can influence subsequent prompts or exported student inputs.
            for field in (
                "accepted",
                "grounding_valid",
                "visible_verified",
                "heldout_verified",
                "visible_feedback",
            ):
                if record["verification"].get(field) != verification.get(field):
                    raise ValueError(f"training audit: verification changed: {field}")
            previous_id = record["teacher_response_id"]
            if artifact:
                previous = artifact.symbolic.model_dump(mode="json")
            stopped = bool(
                verification.get("visible_verified")
                or (artifact and verification.get("grounding_valid") and artifact.witness is None)
            )
            if not stopped:
                feedback = verification.get("visible_feedback") or {
                    "error": verification.get("error", "No verified witness")
                }


def export_verified_examples(run, config, tasks, split):
    audit_training_records(run, config, tasks, split)
    return export_examples(run, config, tasks, split, contract=PythonTrainingContract())
