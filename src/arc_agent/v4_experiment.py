"""Sequential, label-blind Nanbeige pilot; score only after each solver result is durable."""

from __future__ import annotations

import json
import time

from arc_agent.models import ArcTask, validate_grid
from arc_agent.v4_config import V4Config, content_hash
from arc_agent.v4_reasoner import final_json, parse_tool_calls
from arc_agent.v4_state import ExperimentRun, SymbolicTaskState, atomic_json, blind_task
from arc_agent.v4_timing import (
    advance_milestones,
    hard_task_seconds,
    initialize_timing,
    record_round_timing,
    summarize_timings,
    terminal_reason,
    timing_record,
)
from arc_agent.v4_tools import invoke_tool, program_catalog, run_program, task_grids, tool_schema

MODES = ("direct", "program")


def decode_grid(value):
    if isinstance(value, list) and value and all(isinstance(row, str) for row in value):
        if any(not row or any(c not in "0123456789" for c in row) for row in value):
            raise ValueError("invalid compact grid")
        value = [[int(c) for c in row] for row in value]
    return validate_grid(value)


def predictions_from_json(value: dict, count: int) -> list[dict]:
    if set(value) != {"predictions"} or len(value["predictions"]) != count:
        raise ValueError("one prediction pair is required per test input")
    result = []
    for item in value["predictions"]:
        if set(item) != {"attempt_1", "attempt_2"}:
            raise ValueError("exactly two attempts are required")
        first = decode_grid(item["attempt_1"])
        second = first if item["attempt_2"] == "same_as_1" else decode_grid(item["attempt_2"])
        result.append({"attempt_1": first, "attempt_2": second})
    return result


def initial_messages(task: ArcTask, mode: str, run: ExperimentRun) -> list[dict]:
    grids = task_grids(task)
    artifact_ids = {name: run.artifacts.put(grid) for name, grid in grids.items()}
    state = SymbolicTaskState(task_id=task.task_id, artifact_refs=list(artifact_ids.values()))
    run.artifacts.put(state.model_dump(mode="json"))
    # Compact digit rows are lossless, not inferred descriptions. Labels never enter this function.
    payload = {name: ["".join(map(str, row)) for row in grid] for name, grid in grids.items()}
    common = (
        "Solve an ARC grid transformation from demonstrations. Infer one general rule. "
        "Rows are digit strings; colors are integers 0..9; coordinates are zero-based. "
        "Test outputs are unknown. Tools contain no oracle. Keep reasoning concise; "
        "reserve tokens for a complete final answer. Do not memorize demonstration grids. "
    )
    if mode == "direct":
        common += (
            'Return only JSON in the final answer: {"predictions":['
            '{"attempt_1":["012","345"],"attempt_2":["012","345"]}]} . '
            "Include one pair of attempts per test input in input order. "
            'Grids may also be integer matrices. To reuse attempt_1, set attempt_2 to "same_as_1". '
            "No tools in this mode."
        )
    else:
        common += (
            'Use run_program to test a DSL program. Final answer: {"program":'
            '{"op":"rotate","args":{"turns":1},"steps":[]}} . '
            'A composition is {"op":"compose","steps":[...]} . '
            "Only the following operators and argument names exist: "
            + json.dumps(program_catalog(), separators=(",", ":"))
            + ". flip axis is horizontal or vertical; select_component criterion is largest, "
            "smallest, widest or tallest. scale/tile rows and columns are positive integers. "
            "recolor mapping is an object of source color strings to target color integers."
        )
    return [
        {"role": "system", "content": common},
        {
            "role": "user",
            "content": json.dumps(
                {"task_id": task.task_id, "grids": payload}, separators=(",", ":")
            ),
        },
    ]


def fallback(task: ArcTask) -> list[dict]:
    return [{"attempt_1": pair.input, "attempt_2": pair.input} for pair in task.test]


def render_tool_result(result: dict) -> str:
    """Lossless grid serialization, not summarization or evidence deletion."""
    compact = dict(result)
    for key in ("train_predictions", "test_predictions"):
        if key in compact:
            compact[key] = [["".join(map(str, row)) for row in grid] for grid in compact[key]]
    return json.dumps(compact, separators=(",", ":"))


def _add_program(state: dict, payload: dict, result: dict, ancestry: str):
    if result["verified"]:
        candidate = {
            "predictions": result["test_predictions"],
            "program": payload,
            "ancestry": ancestry,
            "verified": True,
            "complexity": result["complexity"],
        }
        if not any(c["predictions"] == candidate["predictions"] for c in state["candidates"]):
            state["candidates"].append(candidate)


def _program_attempts(task: ArcTask, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return fallback(task)
    ordered = sorted(candidates, key=lambda c: (c["complexity"], c["ancestry"]))
    output = []
    for index in range(len(task.test)):
        distinct = []
        for candidate in ordered:
            grid = candidate["predictions"][index]
            if grid not in distinct:
                distinct.append(grid)
        output.append({"attempt_1": distinct[0], "attempt_2": distinct[min(1, len(distinct) - 1)]})
    return output


def solve_one(
    task: ArcTask,
    mode: str,
    config: V4Config,
    run: ExperimentRun,
    reasoner,
    check=lambda: None,
    on_checkpoint=lambda: None,
) -> dict:
    task = blind_task(task)
    key = f"{task.task_id}:{mode}"
    state = run.get(key)
    if state and state["status"] == "complete":
        return state
    if state is None:
        state = {
            "task_id": task.task_id,
            "mode": mode,
            "status": "running",
            "round": 0,
            "elapsed_seconds": 0.0,
            "messages": initial_messages(task, mode, run),
            "candidates": [],
            "attempts": fallback(task),
            "responses": [],
            "errors": [],
            "valid_responses": 0,
            "request_count": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "unknown_usage_calls": 0,
            "used_fallback": True,
        }
        run.save(key, state)
    initialize_timing(state, config)
    hard_limit = hard_task_seconds(config)
    while state["round"] < config.max_rounds and state["elapsed_seconds"] < hard_limit:
        check()
        advance_milestones(state, config)
        start = time.monotonic()
        call_key = f"{key}:{state['round']}"
        finished = False
        inference_pending = False
        final = ""
        response = None
        previous_valid = state["valid_responses"]
        state["request_count"] += 1
        try:
            if hasattr(reasoner, "set_deadline"):
                # Diagnostic milestones auto-extend; never kill a healthy call at a soft
                # boundary. The absolute per-task ceiling still includes context reloads.
                reasoner.set_deadline(start + hard_limit - state["elapsed_seconds"])
            saved = run.saved_request(call_key)
            prepared = (
                saved["prepared"]
                if saved
                else reasoner.prepare(
                    state["messages"],
                    [] if mode == "direct" else tool_schema(),
                    seed=config.seed + state["round"],
                    task=task,
                )
            )
            # No model call is repeated once its full response has been recorded.
            request = {"prepared": prepared, "model_revision": config.model_revision}
            response = run.request(call_key, request)
            if response is None:
                remaining = hard_limit - state["elapsed_seconds"] - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("task deadline reached before inference")
                inference_pending = True
                if config.timing_mode == "diagnostic":
                    remaining = min(remaining, config.diagnostic_request_seconds)
                    atomic_json(
                        run.root / "active-call.json",
                        {
                            "status": "running",
                            "task": key,
                            "round": state["round"],
                            "started_at": time.time(),
                            "completed_solve_seconds": state["elapsed_seconds"],
                            "soft_allowance_seconds": state["timing"]["allowance_seconds"],
                            "hard_limit_seconds": hard_limit,
                            "request_timeout_seconds": remaining,
                        },
                    )
                response = reasoner.complete(prepared, timeout=remaining)
                inference_pending = False
                response["charged_seconds"] = time.monotonic() - start
                run.response(call_key, response)
            else:
                # A replay pays the original inference time as well as recovery overhead.
                state["elapsed_seconds"] += response.get(
                    "charged_seconds", response["elapsed_seconds"]
                )
            state["prompt_tokens"] += response["prompt_tokens"]
            state["generated_tokens"] += response["generated_tokens"]
            state["responses"].append(run.artifacts.put(response))
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": response["final"],
                    "reasoning_content": response["reasoning"],
                }
            )
            check()
            final = response["final"]
            if mode == "program" and "<tool_call>" in final:
                calls = parse_tool_calls(final)
                valid = True
                for call in calls:
                    try:
                        remaining = (
                            hard_limit - state["elapsed_seconds"] - (time.monotonic() - start)
                        )
                        if remaining <= 0:
                            raise TimeoutError("tool deadline reached")
                        result = invoke_tool(task, call, timeout=min(3, remaining))
                        if call["name"] == "run_program":
                            _add_program(state, call["arguments"]["program"], result, call_key)
                    except Exception as error:
                        valid = False
                        result = {"error": f"{type(error).__name__}: {error}"}
                        state["errors"].append(result["error"])
                    run.artifacts.put(result)
                    state["messages"].append(
                        {"role": "tool", "content": render_tool_result(result)}
                    )
                state["valid_responses"] += int(valid)
            elif mode == "direct":
                state["attempts"] = predictions_from_json(final_json(final), len(task.test))
                state["valid_responses"] += 1
                state["used_fallback"] = False
                state["candidates"] = [
                    {
                        "predictions": [item[which] for item in state["attempts"]],
                        "ancestry": call_key,
                        "verified": False,
                    }
                    for which in ("attempt_1", "attempt_2")
                ]
                finished = True
            else:
                value = final_json(final)
                if set(value) != {"program"}:
                    raise ValueError("final answer must contain exactly program")
                remaining = hard_limit - state["elapsed_seconds"] - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("program deadline reached")
                result = run_program(task, value["program"], min(3, remaining))
                _add_program(state, value["program"], result, call_key)
                state["valid_responses"] += 1
                finished = bool(result["verified"])
                if not finished:
                    state["messages"].append(
                        {
                            "role": "user",
                            "content": "The rule fails demos. Revise it using this evidence: "
                            + render_tool_result(result),
                        }
                    )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if inference_pending or "interrupted in-flight" in message:
                state["unknown_usage_calls"] += 1
            state["errors"].append(message)
            run.call_error(call_key, message)
            state["messages"].append(
                {
                    "role": "user",
                    "content": "The response could not be used: "
                    + message[:700]
                    + ". Return a complete final answer using the required schema.",
                }
            )
            if "interrupted in-flight" in message or type(error).__name__ == "ContextOverflow":
                finished = True
                state["timing"]["stop_reason"] = (
                    "interrupted_inflight_unresolved"
                    if "interrupted in-flight" in message
                    else "context_limit_unresolved"
                )
            elif config.timing_mode == "fixed" and (
                not state["responses"] or isinstance(error, TimeoutError)
            ):
                finished = True
                state["timing"]["stop_reason"] = "request_error_unresolved"
        finally:
            state["elapsed_seconds"] += time.monotonic() - start
            if config.timing_mode == "diagnostic":
                atomic_json(
                    run.root / "active-call.json",
                    {
                        "status": "interrupted_or_failed" if inference_pending else "checkpointing",
                        "task": key,
                        "round": state["round"],
                        "completed_solve_seconds": state["elapsed_seconds"],
                    },
                )
        state["round"] += 1
        if mode == "program":
            state["attempts"] = _program_attempts(task, state["candidates"])
            state["used_fallback"] = not bool(state["candidates"])
            if state["candidates"]:
                finished = True
        record_round_timing(
            state, config, previous_valid=previous_valid, final=final, response=response
        )
        if (
            config.timing_mode == "diagnostic"
            and not finished
            and state["timing"]["rounds_without_novel_valid_output"]
            >= config.diagnostic_stall_rounds
        ):
            recent = state["timing"]["rounds"][-config.diagnostic_stall_rounds :]
            state["timing"]["stop_reason"] = (
                "repeated_generation_limit_unresolved"
                if all(r["generation_stop_type"] == "limit" and not r["valid"] for r in recent)
                else "stalled_unresolved"
            )
            finished = True
        run.save(key, state)
        on_checkpoint()
        if finished:
            break
    state["status"] = "complete"
    state["timing"]["stop_reason"] = terminal_reason(state, config)
    run.save(key, state)
    on_checkpoint()
    if config.timing_mode == "diagnostic":
        atomic_json(
            run.root / "active-call.json",
            {
                "status": "task_complete",
                "task": key,
                "completed_solve_seconds": state["elapsed_seconds"],
                "stop_reason": state["timing"]["stop_reason"],
            },
        )
    return state


def write_outputs(tasks: list[ArcTask], run: ExperimentRun, config: V4Config) -> dict:
    reports = {}
    timings = []
    pending = {row[0] for row in run.db.execute("SELECT key FROM calls WHERE response IS NULL")}
    for mode in MODES:
        submission = {}
        exact = total = strict = covered = completed = valid = requests = failures = 0
        elapsed = prompt_tokens = generated_tokens = unknown_usage_calls = 0
        for task in tasks:
            state = run.get(f"{task.task_id}:{mode}")
            attempts = state["attempts"] if state else fallback(task)
            submission[task.task_id] = attempts
            if state:
                completed += int(state["status"] == "complete")
                valid += state["valid_responses"]
                requests += state["request_count"]
                failures += int(bool(state["errors"]))
                elapsed += state["elapsed_seconds"]
                prompt_tokens += state["prompt_tokens"]
                generated_tokens += state["generated_tokens"]
                unknown_usage_calls += state.get("unknown_usage_calls", 0)
            # A hard interruption may occur before the per-task checkpoint records the call.
            prefix = f"{task.task_id}:{mode}:"
            unfinished = [
                key
                for key in pending
                if key.startswith(prefix)
                and int(key[len(prefix) :]) >= (state["round"] if state else 0)
            ]
            requests += len(unfinished)
            unknown_usage_calls += len(unfinished)
            scores = []
            for index, pair in enumerate(task.test):
                if pair.output is None:
                    raise ValueError("pilot scorer needs held-out labels; solver does not")
                hit = pair.output in attempts[index].values()
                total += 1
                exact += int(hit)
                scores.append(hit)
                covered += int(
                    bool(state)
                    and any(c["predictions"][index] == pair.output for c in state["candidates"])
                )
            strict += int(all(scores))
            timings.append(timing_record(task, mode, state))
        atomic_json(run.root / f"submission-{mode}.json", submission)
        reports[mode] = {
            "exact_match": exact / total if total else 0,
            "correct_outputs": exact,
            "test_outputs": total,
            "strict_task_accuracy": strict / len(tasks),
            "candidate_coverage": covered / total if total else 0,
            "completed_tasks": completed,
            "total_tasks": len(tasks),
            "valid_responses": valid,
            "response_attempts": requests,
            "valid_response_rate": valid / requests if requests else 0,
            "tasks_with_errors": failures,
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "unknown_usage_calls": unknown_usage_calls,
            "token_counts_are_lower_bounds": bool(unknown_usage_calls),
        }
    report = {
        "version": 1,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "config_hash": content_hash(config.model_dump(mode="json")),
        "modes": reports,
        "timing_mode": config.timing_mode,
        "timing_report": "timing.json",
        "phase0_gate": "pending_live_verification",
        "other_phases": "blocked_by_phase0",
    }
    atomic_json(
        run.root / "timing.json",
        {
            "mode": config.timing_mode,
            "deadline_schedule_seconds": list(config.deadline_schedule_seconds)
            or [config.task_seconds],
            "note": "Correctness is scored after predictions; labels never govern stopping. "
            "Unresolved outcomes are censored, not successful solve times. Soft timing "
            "milestones are persisted at response checkpoints without aborting in-flight work.",
            "tasks": timings,
            "summary": summarize_timings(timings),
        },
    )
    if config.timing_mode == "diagnostic":
        report["phase0_gate"] = "diagnostic_only_not_eligible_for_promotion"
        report["other_phases"] = "requires_fixed_allowance_validation"
    atomic_json(run.root / "report.json", report)
    return report
