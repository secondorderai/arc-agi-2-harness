"""Label-blind timing milestones and explicit unresolved outcomes for local diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from statistics import median

from arc_agent.v4_config import content_hash


def hard_task_seconds(config):
    return (
        config.deadline_schedule_seconds[-1]
        if config.timing_mode == "diagnostic"
        else config.task_seconds
    )


def initialize_timing(state, config):
    state.setdefault(
        "timing",
        {
            "mode": config.timing_mode,
            "allowance_seconds": config.task_seconds,
            "hard_limit_seconds": hard_task_seconds(config),
            "deadline_extensions": [],
            "first_valid_response_seconds": None,
            "first_demo_verified_program_seconds": None,
            "stop_reason": None,
            "novel_valid_outputs": [],
            "rounds_without_novel_valid_output": 0,
            "rounds": [],
        },
    )


def advance_milestones(state, config):
    """An in-flight call may cross soft milestones; record them at its durable checkpoint."""
    if config.timing_mode != "diagnostic":
        return
    timing = state["timing"]
    for following in config.deadline_schedule_seconds[1:]:
        previous = timing["allowance_seconds"]
        if following > previous and state["elapsed_seconds"] >= previous:
            event = {
                "from_seconds": previous,
                "to_seconds": following,
                "crossed_at_solve_seconds": previous,
                "recorded_at_solve_seconds": state["elapsed_seconds"],
                "reason": "unresolved at timing milestone; retain history and continue",
            }
            timing["deadline_extensions"].append(event)
            timing["allowance_seconds"] = following
            print(
                json.dumps(
                    {"task": state["task_id"], "mode": state["mode"], "deadline_extension": event}
                ),
                flush=True,
            )


def record_round_timing(state, config, *, previous_valid, final, response=None):
    timing = state["timing"]
    elapsed = state["elapsed_seconds"]
    new_valid = state["valid_responses"] > previous_valid
    response = response or {}
    timing["rounds"].append(
        {
            "round": state["round"] - 1,
            "solve_seconds": elapsed,
            "valid": new_valid,
            "final_present": bool(final),
            "generation_stop_type": response.get("raw", {}).get("stop_type"),
            "generated_tokens": response.get("generated_tokens"),
            "prompt_tokens": response.get("prompt_tokens"),
            "context_tokens": response.get("context_tokens"),
        }
    )
    if new_valid and timing["first_valid_response_seconds"] is None:
        timing["first_valid_response_seconds"] = elapsed
    for candidate in state["candidates"]:
        candidate.setdefault("created_at_solve_seconds", elapsed)
    if (
        any(c.get("verified") for c in state["candidates"])
        and timing["first_demo_verified_program_seconds"] is None
    ):
        timing["first_demo_verified_program_seconds"] = elapsed
    # Do not treat new prose, token count, or model confidence as progress.
    fingerprint = content_hash(" ".join(final.split())) if new_valid else None
    if fingerprint and fingerprint not in timing["novel_valid_outputs"]:
        timing["novel_valid_outputs"].append(fingerprint)
        timing["rounds_without_novel_valid_output"] = 0
    else:
        timing["rounds_without_novel_valid_output"] += 1
    advance_milestones(state, config)


def terminal_reason(state, config):
    if not state["used_fallback"]:
        return "demonstrations_verified" if state["mode"] == "program" else "prediction_produced"
    if state["timing"]["stop_reason"]:
        return state["timing"]["stop_reason"]
    if state["elapsed_seconds"] >= hard_task_seconds(config):
        return "time_limit_unresolved"
    if state["round"] >= config.max_rounds:
        return "round_limit_unresolved"
    return "unresolved_error"


def timing_record(task, mode, state):
    """Post-result scorer only: labels never affect extension or stopping decisions."""
    record = {
        "task_id": task.task_id,
        "mode": mode,
        "status": state["status"] if state else "not_started",
        "elapsed_seconds": state["elapsed_seconds"] if state else 0,
        "first_valid_response_seconds": None,
        "first_demo_verified_program_seconds": None,
        "first_correct_predictions_seconds": None,
        "stop_reason": None,
        "deadline_extensions": [],
        "right_censored": None,
    }
    if not state:
        return record
    for key in (
        "first_valid_response_seconds",
        "first_demo_verified_program_seconds",
        "stop_reason",
        "deadline_extensions",
        "allowance_seconds",
        "hard_limit_seconds",
        "rounds",
    ):
        if key in state.get("timing", {}):
            record[key] = state["timing"][key]
    # Match the selected two-attempt submission, including different successful attempts
    # for different test inputs. A correct but unselected candidate is not a scored solve.
    correct_times = []
    for index, pair in enumerate(task.test):
        if pair.output not in state["attempts"][index].values() or state["used_fallback"]:
            break
        timestamps = [
            c["created_at_solve_seconds"]
            for c in state["candidates"]
            if c["predictions"][index] == pair.output and "created_at_solve_seconds" in c
        ]
        if not timestamps:
            break
        correct_times.append(min(timestamps))
    if len(correct_times) == len(task.test) and correct_times:
        record["first_correct_predictions_seconds"] = max(correct_times)
    # A grid answer or demo fit alone does not establish a correct held-out solution.
    record["right_censored"] = (
        state["status"] == "complete" and record["first_correct_predictions_seconds"] is None
    )
    record["timing_is_lower_bound"] = bool(state.get("unknown_usage_calls"))
    return record


def summarize_timings(records):
    summary = {}
    for mode in ("direct", "program"):
        selected = [r for r in records if r["mode"] == mode]
        complete = [r for r in selected if r["status"] == "complete"]
        correct = [
            r["first_correct_predictions_seconds"]
            for r in selected
            if r["first_correct_predictions_seconds"] is not None
        ]
        summary[mode] = {
            "completed_tasks": len(complete),
            "correct_tasks_observed": len(correct),
            "completed_without_correct_solution": sum(r["right_censored"] for r in complete),
            "median_seconds_among_correct_tasks_only": median(correct) if correct else None,
            "terminal_reasons": dict(Counter(r["stop_reason"] for r in complete)),
            "generation_limit_stops": sum(
                turn["generation_stop_type"] == "limit"
                for record in selected
                for turn in record.get("rounds", [])
            ),
        }
    return summary
