"""Durable stage-by-stage student solving. Evaluation labels stay outside this module."""

from __future__ import annotations

import json
import time

import httpx

from arc_agent.v4_config import content_hash
from arc_agent.v4_experiment import fallback, predictions_from_json
from arc_agent.v4_reasoner import ContextOverflow, final_json
from arc_agent.v4_state import atomic_json, blind_task
from arc_agent.v4_tools import task_grids
from arc_agent.v5_symbolic import SymbolicModel, validate_symbols
from arc_agent.v8_codec import TRIPLES_FORMAT, unpack_symbolic_cells
from arc_agent.v8_eval import check_witness, select_predictions
from arc_agent.v8_reasoner import ProgramResponse, direct_prompt, symbolic_prompt, witness_prompt
from arc_agent.v8_reference_repair import apply_reference_patch


def grounding_feedback(symbolic, task, error):
    """Detailed visible-evidence compiler errors, never evaluation-label feedback."""
    grids = task_grids(blind_task(task))
    definitions = {d.id for d in symbolic.definitions}
    errors = []
    for entity in symbolic.entities:
        if entity.grid not in grids:
            errors.append({"entity": entity.id, "invalid_grid": entity.grid})
            continue
        grid = grids[entity.grid]
        for cell in entity.cells:
            if cell.row >= len(grid) or cell.column >= len(grid[0]):
                errors.append(
                    {
                        "entity": entity.id,
                        "outside_grid": cell.model_dump(),
                        "height": len(grid),
                        "width": len(grid[0]),
                    }
                )
            elif cell.color != grid[cell.row][cell.column]:
                errors.append(
                    {
                        "entity": entity.id,
                        "cell": cell.model_dump(),
                        "observed_color": grid[cell.row][cell.column],
                    }
                )
    for definition in symbolic.definitions:
        missing = sorted(set(definition.depends_on) - definitions)
        if missing:
            errors.append({"definition": definition.id, "undefined_dependencies": missing})
    for hypothesis in symbolic.hypotheses:
        missing = sorted(set(hypothesis.concepts + hypothesis.ordered_actions) - definitions)
        if missing:
            errors.append({"hypothesis": hypothesis.id, "undefined_concepts_or_actions": missing})
    return {
        "error": str(error),
        "diagnostics": errors,
        "available_definition_ids": sorted(definitions),
        "grid_shapes": {k: [len(g), len(g[0])] for k, g in grids.items()},
        "request": (
            "Correct invalid references/cells; ordered_actions must name defined operations."
        ),
    }


def state_key(task_id, mode, seed):
    return f"student:{seed}:{mode}:{task_id}"


def initial_state(task, mode, seed):
    return {
        "task_id": task.task_id,
        "mode": mode,
        "seed": seed,
        "stage": "direct" if mode == "direct" else "symbolic",
        "status": "running",
        "cycle": 0,
        "step": 0,
        "elapsed_seconds": 0.0,
        "symbolic": None,
        "previous_source": None,
        "feedback": None,
        "candidates": [],
        "attempts": fallback(task),
        "artifact_refs": [],
        "request_count": 0,
        "valid_responses": 0,
        "grounded_states": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "unknown_usage_calls": 0,
        "errors": [],
    }


def advance_stage(
    task, mode, seed, config, run, reasoner, *, check=lambda: None, global_deadline=None
):
    if mode not in {"direct", "symbolic"}:
        raise ValueError("unknown student mode")
    task = blind_task(task)
    key = state_key(task.task_id, mode, seed)
    state = run.get(key) or initial_state(task, mode, seed)
    if state["status"] != "running":
        return state
    started = time.monotonic()
    remaining = config.task_seconds - state["elapsed_seconds"]
    if global_deadline is not None:
        remaining = min(remaining, global_deadline - started)
    if remaining <= 0:
        state["status"] = "timed_out"
        run.save(key, state)
        return state
    deadline = started + remaining
    run.save(key, state)
    # Missing evidence is an integrity failure, not permission to regenerate completed work.
    for ref in state["artifact_refs"]:
        run.artifacts.get(ref)
    stage = state["stage"]
    if stage == "witness" and config.witness_repair_policy == "repair_static_errors":
        # A code-only retry may retain only a still-grounded, unchanged state. Fail
        # before dispatch on corrupt evidence; never repair it by inventing a new call.
        validate_symbols(SymbolicModel.model_validate(state["symbolic"]), task)
        feedback = state["feedback"] or {}
        if feedback.get("repair_scope") == "witness_only" and (
            feedback.get("symbolic_state_sha256") != content_hash(state["symbolic"])
        ):
            raise ValueError("witness-only repair symbolic state changed")
    messages = (
        direct_prompt(task)
        if stage == "direct"
        else symbolic_prompt(task, state["symbolic"], state["feedback"])
        if stage == "symbolic"
        else witness_prompt(task, state["symbolic"], state["previous_source"], state["feedback"])
    )
    call_key = f"{key}:{state['step']}"
    saved = run.get(call_key)
    if saved and saved["messages"] != messages:
        raise ValueError("student checkpoint prompt changed")
    if hasattr(reasoner, "set_deadline"):
        reasoner.set_deadline(deadline)
    response = None
    unknown_counted = False
    diagnostics = None
    try:
        check()
        if saved is None:
            prepared = reasoner.prepare_stage(
                messages, seed=seed + state["cycle"], task=task, stage=stage
            )
            saved = {
                "messages": messages,
                "prepared": prepared,
                "response": None,
                "error": None,
                "dispatched_at": None,
                "charged_seconds": 0.0,
            }
            run.save(call_key, saved)
        if saved["response"] is not None:
            response = saved["response"]
            state["elapsed_seconds"] += saved["charged_seconds"]
        elif saved["dispatched_at"] is not None:
            # An interrupted HTTP generation cannot be recovered from a local KV cache.
            # Charge the uncertain interval and move on; never duplicate this dispatch.
            charged = min(
                config.task_seconds - state["elapsed_seconds"],
                max(saved["charged_seconds"], time.time() - saved["dispatched_at"]),
            )
            state["elapsed_seconds"] += charged
            state["unknown_usage_calls"] += 1
            unknown_counted = True
            raise TimeoutError(saved["error"] or "interrupted local request; response unavailable")
        else:
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline reached before local dispatch")
            saved["dispatched_at"] = time.time()
            run.save(call_key, saved)
            atomic_json(
                run.root / "active-student.json",
                {
                    "key": key,
                    "stage": stage,
                    "step": state["step"],
                    "started_at": saved["dispatched_at"],
                    "call_key": call_key,
                },
            )
            response = reasoner.complete(saved["prepared"], timeout=deadline - time.monotonic())
            saved["response"] = response
            saved["charged_seconds"] = time.monotonic() - started
            run.save(call_key, saved)  # Persist response before any parser or verifier.
        state["request_count"] += 1
        state["prompt_tokens"] += response["prompt_tokens"]
        state["generated_tokens"] += response["generated_tokens"]
        state["artifact_refs"].append(run.artifacts.put(response))
        payload = final_json(response["final"])
        if stage == "direct":
            state["attempts"] = predictions_from_json(payload, len(task.test))
            state["valid_responses"] += 1
            state["status"] = "complete"
        elif stage == "symbolic":
            if saved["prepared"].get("_response_format") == "symbolic_reference_patch":
                base_hash = saved["prepared"]["_reference_base_sha256"]
                repaired = apply_reference_patch(state["symbolic"], payload, base_hash)
                state["artifact_refs"].append(
                    run.artifacts.put(
                        {
                            "kind": "symbolic_reference_patch",
                            "base_sha256": base_hash,
                            "patch": payload,
                            "result": repaired,
                        }
                    )
                )
                payload = repaired
            elif saved["prepared"].get("_response_format") == TRIPLES_FORMAT:
                canonical = unpack_symbolic_cells(payload)
                state["artifact_refs"].append(
                    run.artifacts.put(
                        {
                            "kind": "symbolic_cell_decode",
                            "encoding": "triples_v1",
                            "source_response_ref": content_hash(response),
                            "packed_sha256": content_hash(payload),
                            "canonical": canonical,
                        }
                    )
                )
                payload = canonical
            symbolic = SymbolicModel.model_validate(payload)
            state["valid_responses"] += 1
            state["symbolic"] = symbolic.model_dump(mode="json")
            try:
                validate_symbols(symbolic, task)
            except ValueError as exc:
                diagnostics = grounding_feedback(symbolic, task, exc)
                raise
            state["grounded_states"] += 1
            state["stage"] = "witness"
        else:
            ProgramResponse.model_validate(payload)
            state["valid_responses"] += 1
            state["previous_source"] = payload["python_source"]
            verification, feedback, candidate = check_witness(payload["python_source"], task)
            state["artifact_refs"].append(run.artifacts.put(verification))
            state["feedback"] = feedback
            if candidate and not any(
                c["ancestry"] == candidate["ancestry"] for c in state["candidates"]
            ):
                state["candidates"].append(candidate)
            state["attempts"] = select_predictions(task, state["candidates"])
            state["cycle"] += 1
            state["stage"] = "symbolic"
            if verification["accepted"]:
                state["status"] = "complete"
            elif (
                config.witness_repair_policy == "repair_static_errors"
                and verification["static_safe"] is False
                and verification["failures"]
                and all(f["case"] == "static" for f in verification["failures"])
            ):
                # A compiler rejection is not a transformation counterexample.
                # Charge the failed cycle, preserve its evidence and ask Nanbeige
                # to repair only the witness. Runtime/demo failures still revise
                # the symbolic hypothesis via the existing path above.
                state["stage"] = "witness"
                state["feedback"] = {
                    **feedback,
                    "repair_scope": "witness_only",
                    "symbolic_state_sha256": content_hash(state["symbolic"]),
                }
    except (ValueError, TimeoutError, httpx.HTTPError) as exc:
        if saved and saved["dispatched_at"] is not None and response is None:
            state["request_count"] += 1
            if not saved.get("error"):
                saved["error"] = f"{type(exc).__name__}: {exc}"
                saved["charged_seconds"] = time.monotonic() - started
                run.save(call_key, saved)
            if not unknown_counted:
                state["unknown_usage_calls"] += 1
        state["errors"].append({"stage": stage, "step": state["step"], "error": str(exc)[:2000]})
        state["feedback"] = diagnostics or {"error": str(exc)[:2000]}
        state["cycle"] += 1
        state["stage"] = "direct" if mode == "direct" else "symbolic"
        if isinstance(exc, (TimeoutError, httpx.TimeoutException, ContextOverflow)):
            state["status"] = (
                "timed_out" if not isinstance(exc, ContextOverflow) else "context_limit"
            )
    state["elapsed_seconds"] += time.monotonic() - started
    state["step"] += 1
    if state["status"] == "running" and state["cycle"] >= config.max_cycles:
        state["status"] = "exhausted"
    if state["status"] == "running" and state["elapsed_seconds"] >= config.task_seconds:
        state["status"] = "timed_out"
    run.save(key, state)
    check()  # Never swallow a memory/owned-process failure as an ordinary candidate error.
    print(
        json.dumps(
            {
                "student_task": task.task_id,
                "mode": mode,
                "seed": seed,
                "stage": stage,
                "status": state["status"],
                "seconds": round(state["elapsed_seconds"], 2),
            }
        ),
        flush=True,
    )
    return state
