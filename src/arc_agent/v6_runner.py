"""Coverage-first scored Astra run on the original frozen 20 development tasks."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import typer
import yaml

from arc_agent.data import load_tasks
from arc_agent.v2_models import ResponseSnapshot
from arc_agent.v2_openai import TransientAPIError, response_output_text
from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import ExperimentRun, atomic_json, blind_task, freeze_split
from arc_agent.v5_pipeline import source_identity
from arc_agent.v6_eval import (
    AstraSolver,
    EvalConfig,
    SolverResponse,
    evaluate_candidate,
    initial_prompt,
    write_outputs,
)

app = typer.Typer(no_args_is_help=True)
DEFAULT_CONFIG = Path("configs/v6-astra-local-eval.yaml")
DEFAULT_WORKSPACE = Path("runs/v6/astra-local-eval-01")


def fixed_cohort(config):
    if not config.split.exists():
        raise ValueError("the original frozen split is required")
    all_tasks = load_tasks(config.data)
    split = freeze_split(all_tasks, config.split)
    by_id = {t.task_id: t for t in all_tasks}
    ids = split["groups"]["development"][:20]
    if len(ids) != 20:
        raise ValueError("the scored pilot requires all 20 original development tasks")
    return [by_id[key] for key in ids], split


def run_round(task, cfg, run, solver, global_deadline):
    if shutil.disk_usage(run.root).free < 10 * 1024**3:
        raise RuntimeError("preserve at least 10 GiB free disk space")
    if any(pair.output is not None for pair in task.test):
        raise ValueError("solver must not receive test labels")
    key = f"task:{task.task_id}"
    state = run.get(key) or {
        "round": 0,
        "done": False,
        "candidates": [],
        "elapsed": 0,
        "previous_id": None,
        "feedback": None,
        "usage": {},
        "records": [],
    }
    if state["done"]:
        return
    index = state["round"]
    if state["elapsed"] >= cfg.task_seconds or time.time() >= global_deadline:
        state.update(done=True, status="timed_out")
        run.save(key, state)
        return
    call_key = f"call:{task.task_id}:{index}"
    call = run.get(call_key)
    if call is None:
        prompt = (
            initial_prompt(task)
            if state["previous_id"] is None
            else json.dumps(
                {
                    "demonstration_verifier_feedback": state["feedback"],
                    "request": "Revise the symbolic model and rule using the counterexamples. "
                    "Recheck every demonstration; consider a different abstraction if needed. "
                    "Return complete code and predictions, not a patch. "
                    "Test labels are unavailable.",
                }
            )
        )
        call = {
            "request": {
                "prompt": prompt,
                "model": cfg.teacher.model,
                "token_limit": cfg.output_token_hint,
                "request_key": content_hash({"run": str(run.root.resolve()), "call": call_key}),
            },
            "snapshot": None,
            "started_at": time.time(),
            "deadline": min(global_deadline, time.time() + cfg.task_seconds - state["elapsed"]),
        }
        run.save(call_key, call)
    run.save(key, state)
    atomic_json(
        run.root / "active.json",
        {
            "pid": os.getpid(),
            "task": task.task_id,
            "round": index,
            "status": "inference",
            "call_key": call_key,
            "deadline": call["deadline"],
            "model": cfg.teacher.model,
        },
    )

    def checkpoint(snapshot):
        call["snapshot"] = snapshot.model_dump(mode="json")
        run.save(call_key, call)

    snapshot = ResponseSnapshot.model_validate(call["snapshot"]) if call["snapshot"] else None
    retries = 0
    while snapshot is None or snapshot.status in {"queued", "in_progress"}:
        if time.time() >= call["deadline"]:
            if snapshot:
                solver.cancel(snapshot.response_id)
            state["unknown_usage_calls"] = state.get("unknown_usage_calls", 0) + 1
            state.update(done=True, status="timed_out")
            state["elapsed"] += time.time() - call["started_at"]
            run.save(key, state)
            return
        try:
            if snapshot is None:
                snapshot = solver.create(
                    prompt=call["request"]["prompt"],
                    task_id=task.task_id,
                    phase="development",
                    round_index=index,
                    max_output_tokens=cfg.output_token_hint,
                    previous_response_id=state["previous_id"],
                    request_key=call["request"]["request_key"],
                    model=cfg.teacher.model,
                    checkpoint=checkpoint,
                )
            else:
                time.sleep(min(cfg.poll_seconds, max(0, call["deadline"] - time.time())))
                snapshot = solver.retrieve(snapshot.response_id, request=call["request"])
            checkpoint(snapshot)
            retries = 0
        except TransientAPIError as exc:
            # Recover the exact provisional/started turn rather than dispatching a replacement.
            saved = run.get(call_key)
            if saved["snapshot"]:
                snapshot = ResponseSnapshot.model_validate(saved["snapshot"])
            retries += 1
            call["last_transient_error"] = str(exc)
            run.save(call_key, call)
            if retries >= 5:
                raise
            time.sleep(min(2**retries, 30, max(0, call["deadline"] - time.time())))
    if snapshot.status != "completed":
        state["unknown_usage_calls"] = state.get("unknown_usage_calls", 0) + 1
        state.update(
            done=True,
            status="provider_failed",
            provider_status=snapshot.status,
            provider_error=snapshot.body.get("error"),
        )
    else:
        try:
            response = SolverResponse.model_validate_json(response_output_text(snapshot.body))
            result = evaluate_candidate(response, task, index)
        except ValueError as exc:
            result = {
                "candidates": [],
                "verified": False,
                "feedback": str(exc),
                "errors": [str(exc)],
            }
        state["candidates"].extend(result["candidates"])
        state["feedback"] = result["feedback"]
        state["previous_id"] = snapshot.response_id
        record = {
            "task_id": task.task_id,
            "split": "development",
            "training_export_allowed": False,
            "response": snapshot.model_dump(mode="json"),
            "verification": result,
        }
        state["records"].append(run.artifacts.put(record))
        state["round"] += 1
        for name, amount in snapshot.usage.model_dump(mode="json").items():
            state["usage"][name] = state["usage"].get(name, 0) + amount
        state["done"] = result["verified"] or state["round"] >= cfg.max_rounds
        state["status"] = (
            "verified"
            if result["verified"]
            else ("unresolved" if state["done"] else "needs_repair")
        )
    state["elapsed"] += time.time() - call["started_at"]
    run.save(key, state)
    print(
        json.dumps(
            {
                "task": task.task_id,
                "round": state["round"],
                "status": state["status"],
                "seconds": round(state["elapsed"], 1),
            }
        ),
        flush=True,
    )


@app.command()
def run(config: Path = DEFAULT_CONFIG, workspace: Path = DEFAULT_WORKSPACE, resume: bool = False):
    cfg = EvalConfig.model_validate(yaml.safe_load(config.read_text()))
    tasks, split = fixed_cohort(cfg)
    identity = {
        "version": 6,
        "config": cfg.model_dump(mode="json"),
        "source_hash": source_identity(),
        "split_hash": content_hash(split),
        "cohort_ids": [t.task_id for t in tasks],
        "runtime_hash": file_hash(Path(cfg.teacher.codex_cli)),
        "test_outputs": sum(len(t.test) for t in tasks),
    }
    experiment = ExperimentRun(workspace, identity, resume=resume)
    solver = AstraSolver(settings=cfg.teacher, workspace=cfg.auth_workspace)
    try:
        atomic_json(workspace / "identity.json", identity)
        archive = workspace / "source.tar.gz"
        if not archive.exists():
            with tarfile.open(archive, "x:gz") as tar:
                tar.add(Path(__file__).parent, arcname="src/arc_agent")
                tar.add(config, arcname="config.yaml")
        write_outputs(experiment, tasks)  # complete predictions before any subscription request
        atomic_json(workspace / "preflight.json", solver.preflight())
        clock = experiment.get("run_clock") or {"started_at": time.time()}
        experiment.save("run_clock", clock)
        deadline = clock["started_at"] + cfg.global_seconds
        # Every task gets initial coverage before any task gets a repair round.
        for pass_index in range(cfg.max_rounds):
            for task in tasks:
                state = experiment.get(f"task:{task.task_id}") or {}
                if state.get("done") or state.get("round", 0) > pass_index:
                    continue
                run_round(blind_task(task), cfg, experiment, solver, deadline)
                report = write_outputs(experiment, tasks)
                print(
                    json.dumps(
                        {
                            "score": report["exact_match"],
                            "correct": report["correct_outputs"],
                            "total": report["test_outputs"],
                            "completed": report["completed_tasks"],
                            "attempted": report["attempted_tasks"],
                        }
                    ),
                    flush=True,
                )
        atomic_json(workspace / "active.json", {"pid": None, "status": "completed"})
    except BaseException as exc:
        atomic_json(
            workspace / "active.json",
            {"pid": None, "status": "paused", "error": f"{type(exc).__name__}: {exc}"},
        )
        write_outputs(experiment, tasks)
        raise
    finally:
        solver.close()
        experiment.close()


@app.command()
def status(workspace: Path = DEFAULT_WORKSPACE):
    report = json.loads((workspace / "report.json").read_text())
    active = json.loads((workspace / "active.json").read_text())
    typer.echo(json.dumps({"report": report, "active": active}, indent=2))


if __name__ == "__main__":
    app()
