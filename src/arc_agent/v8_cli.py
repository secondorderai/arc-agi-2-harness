"""Local Nanbeige comparison and symbolic workflow fixtures; no teacher/model API calls."""

import json
import tarfile
import time
from pathlib import Path

import typer

from arc_agent.data import load_tasks
from arc_agent.models import ArcTask
from arc_agent.v4_cli import runtime_evidence, source_hash
from arc_agent.v4_config import V4Config, content_hash, file_hash, load_v4_config, verify_manifest
from arc_agent.v4_experiment import predictions_from_json
from arc_agent.v4_reasoner import final_json
from arc_agent.v4_runtime import LocalServer
from arc_agent.v4_state import ExperimentRun, atomic_json, blind_task, freeze_split
from arc_agent.v8_config import load_config
from arc_agent.v8_dataset import prepare_compact_student
from arc_agent.v8_eval import score_submission
from arc_agent.v8_grammar import COMPACT_STYLES, converter_path, format_binding
from arc_agent.v8_reasoner import StudentReasoner, StudentSession
from arc_agent.v8_runner import advance_stage, initial_state, state_key

app = typer.Typer(no_args_is_help=True, help=__doc__)
DEFAULT_CONFIG = Path("configs/v8-nanbeige-baseline.yaml")


def long_context_fixture(run, reasoner, fixture, check):
    messages = [
        {
            "role": "user",
            "content": "Reference grid: 12/34.\n"
            * 500
            + '\nIgnore the repetitions. Return JSON only: {"predictions":'
            '[{"attempt_1":["5","6"],"attempt_2":"same_as_1"}]}. Keep reasoning brief.',
        }
    ]
    prepared = reasoner.prepare_stage(messages, seed=42, task=fixture, stage="direct")
    if not (
        prepared["_prompt_tokens"] > 4096
        and prepared["n_predict"] == reasoner.config.max_new_tokens
        and prepared["_prompt_tokens"] + prepared["n_predict"] + 8 <= reasoner.config.context_tokens
    ):
        raise ValueError("fixture must exercise attention beyond 4K with the full answer allowance")
    response = run.request("long-context-fixture", prepared)
    if response is None:
        response = reasoner.complete(prepared, timeout=reasoner.config.task_seconds)
        run.response("long-context-fixture", response)
    check()
    predictions = predictions_from_json(final_json(response["final"]), 1)
    return {
        "long_context_verified": predictions[0]["attempt_1"] == [[5], [6]],
        "long_context_prompt_tokens": prepared["_prompt_tokens"],
        "long_context_generated_tokens": response["generated_tokens"],
        "long_context_answer_allowance": prepared["n_predict"],
    }


def write_report(run, tasks, cfg, *, elapsed, evidence):
    metrics = {}
    for seed in cfg.seeds:
        for mode in cfg.modes:
            states = [
                run.get(state_key(t.task_id, mode, seed)) or initial_state(t, mode, seed)
                for t in tasks
            ]
            submission = {
                t.task_id: state["attempts"] for t, state in zip(tasks, states, strict=True)
            }
            atomic_json(run.root / f"submission-{mode}-{seed}.json", submission)
            score = score_submission(tasks, submission)
            calls_by_task = [
                [
                    run.get(f"{state_key(t.task_id, mode, seed)}:{step}")
                    for step in range(state["step"] + 1)
                ]
                for t, state in zip(tasks, states, strict=True)
            ]
            dispatched = [
                [c for c in calls if c and c.get("dispatched_at") is not None]
                for calls in calls_by_task
            ]
            dispatches = sum(len(calls) for calls in dispatched)
            metrics[f"{mode}:{seed}"] = {
                **score,
                "processed_tasks": sum(s["status"] != "running" for s in states),
                "dispatched_tasks": sum(bool(calls) for calls in dispatched),
                "dispatches": dispatches,
                "completed_responses": sum(
                    c.get("response") is not None for calls in dispatched for c in calls
                ),
                "valid_response_rate": sum(s["valid_responses"] for s in states) / dispatches
                if dispatches
                else None,
                "grounded_states": sum(s["grounded_states"] for s in states),
                "generated_tokens": sum(s["generated_tokens"] for s in states),
                "prompt_tokens": sum(s["prompt_tokens"] for s in states),
                "unknown_usage_calls": sum(
                    c.get("response") is None for calls in dispatched for c in calls
                ),
                "task_seconds": sum(s["elapsed_seconds"] for s in states),
                "failures": [
                    {"task_id": s["task_id"], "status": s["status"], "errors": s["errors"]}
                    for s in states
                    if s["errors"] or s["status"] not in {"complete", "running"}
                ],
            }
    report = {
        "model": "Nanbeige/Nanbeige4.2-3B",
        "trained": False,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "memory_and_runtime": evidence,
        "complete": all(m["processed_tasks"] == len(tasks) for m in metrics.values()),
        "promotion_proven": False,
    }
    atomic_json(run.root / "report.json", report)
    return report


def archive_source(workspace, config, runtime_config, *, grammar_converter=None):
    if (workspace / "source.tar.gz").exists():
        return
    with tarfile.open(workspace / "source.tar.gz", "w:gz") as archive:
        for path in sorted(Path("src/arc_agent").glob("*.py")):
            archive.add(path, arcname=str(path))
        for path in (
            config,
            runtime_config,
            Path("scripts/v4-loopback.sb"),
            Path("tests/test_v4.py"),
        ):
            archive.add(path, arcname=str(path))
        if grammar_converter is not None:
            archive.add(grammar_converter, arcname="runtime-tools/json_schema_to_grammar.py")


def experiment_binding(cfg, runtime_cfg, manifest):
    return {
        "source_hash": source_hash(),
        "manifest_hash": file_hash(runtime_cfg.manifest),
        "witness_repair_policy": cfg.witness_repair_policy,
        "symbolic_cell_encoding": cfg.symbolic_cell_encoding,
        "max_new_tokens": runtime_cfg.max_new_tokens,
        # Smoke contexts differ by design; all other settings must match, including
        # generation, reasoning, deadlines, safety, and the two-stage fixture policy.
        "runtime_policy_hash": content_hash(
            runtime_cfg.model_dump(mode="json", exclude={"context_tokens"})
        ),
        "experiment_policy_hash": content_hash(cfg.model_dump(mode="json")),
        **format_binding(cfg.structured_style, manifest),
    }


def matching_smoke(report, binding, context):
    return (
        report.get("passed") is True
        and report.get("context_tokens") == context
        and all(report.get(key) == value for key, value in binding.items())
    )


def smoke_contexts(cfg):
    return (8192, 12288) if cfg.smoke_contexts == "8k_then_12k" else (4096, 8192)


def require_pilot_smoke(report, binding, cfg, runtime_cfg):
    _, final_context = smoke_contexts(cfg)
    if (
        runtime_cfg.context_tokens != final_context
        or not matching_smoke(report, binding, final_context)
        or not report.get("long_context_verified")
        or report.get("long_context_answer_allowance") != runtime_cfg.max_new_tokens
        or (runtime_cfg.prompt_cache_mib == 0 and not report.get("prompt_cache_disabled_verified"))
    ):
        raise ValueError(
            f"pass current-source {final_context}-token symbolic smoke with matching "
            "runtime/output policy before the baseline"
        )


@app.command()
def smoke(
    config: Path = DEFAULT_CONFIG,
    workspace: Path = Path("runs/v8/smoke-4k"),
    context: int = 4096,
    resume: bool = False,
    baseline_smoke: Path | None = None,
):
    cfg = load_config(config)
    first_context, final_context = smoke_contexts(cfg)
    if context not in {first_context, final_context}:
        raise ValueError(f"smoke contexts are {first_context} then {final_context} tokens")
    runtime_cfg = V4Config.model_validate(
        {**load_v4_config(cfg.runtime_config).model_dump(), "context_tokens": context}
    )
    manifest = verify_manifest(runtime_cfg.manifest, runtime_cfg)
    binding = experiment_binding(cfg, runtime_cfg, manifest)
    if context == final_context:
        previous = json.loads(baseline_smoke.read_text()) if baseline_smoke else {}
        if not matching_smoke(previous, binding, first_context):
            raise ValueError(
                f"{final_context} requires the current-source {first_context}-token "
                "symbolic workflow fixture with matching runtime/output policy"
            )
    fixture = ArcTask.model_validate(
        {
            "task_id": "student_identity_fixture",
            "train": [
                {"input": [[1, 0]], "output": [[1, 0]]},
                {"input": [[2], [0]], "output": [[2], [0]]},
            ],
            "test": [{"input": [[3, 0]], "output": [[3, 0]]}],
        }
    )
    run = ExperimentRun(
        workspace, {**binding, "config": runtime_cfg.model_dump(mode="json"), "v": 8}, resume=resume
    )
    archive_source(
        workspace,
        config,
        cfg.runtime_config,
        grammar_converter=converter_path(manifest)
        if cfg.structured_style in COMPACT_STYLES
        else None,
    )
    server = LocalServer(runtime_cfg, manifest, workspace)
    evidence = {**binding, "context_tokens": context, "passed": False}
    try:
        with server:
            reasoner = StudentReasoner(
                runtime_cfg,
                manifest,
                structured_style=cfg.structured_style,
                symbolic_cell_encoding=cfg.symbolic_cell_encoding,
            )
            try:
                for _ in range(cfg.max_cycles * 2):
                    for mode in cfg.modes:
                        advance_stage(fixture, mode, 42, cfg, run, reasoner, check=server.check)
                states = [run.get(state_key(fixture.task_id, mode, 42)) for mode in cfg.modes]
                evidence["valid_response_rates"] = {
                    s["mode"]: s["valid_responses"] / s["request_count"]
                    if s["request_count"]
                    else 0.0
                    for s in states
                }
                if context == final_context:
                    evidence.update(long_context_fixture(run, reasoner, fixture, server.check))
                before = [s["request_count"] for s in states]
                run.close()
                run = ExperimentRun(
                    workspace,
                    {**binding, "config": runtime_cfg.model_dump(mode="json"), "v": 8},
                    resume=True,
                )
                replays = [
                    advance_stage(fixture, mode, 42, cfg, run, reasoner, check=server.check)
                    for mode in cfg.modes
                ]
                evidence.update(runtime_evidence(server, workspace))
                evidence.update(
                    {
                        "resume_verified": before == [s["request_count"] for s in replays]
                        and states == replays,
                        "stages": states,
                        "passed": all(
                            s["status"] == "complete" and s["attempts"][0]["attempt_1"] == [[3, 0]]
                            for s in states
                        ),
                    }
                )
                evidence["passed"] &= (
                    evidence["resume_verified"]
                    and evidence["memory_safe"]
                    and evidence["offline_enforced"]
                    and evidence["metal_verified"]
                    and all(rate >= 0.95 for rate in evidence["valid_response_rates"].values())
                    and (
                        runtime_cfg.prompt_cache_mib != 0
                        or evidence["prompt_cache_disabled_verified"]
                    )
                )
                if context == final_context:
                    evidence["passed"] &= evidence.get("long_context_verified") is True
            finally:
                reasoner.close()
    except (Exception, KeyboardInterrupt) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["passed"] = False
    finally:
        evidence.update(runtime_evidence(server, workspace))
        evidence["passed"] &= evidence["memory_safe"]
        atomic_json(workspace / "smoke.json", evidence)
        run.close()
    typer.echo(json.dumps({k: v for k, v in evidence.items() if k != "stages"}, indent=2))
    raise typer.Exit(0 if evidence["passed"] else 1)


@app.command()
def pilot(
    config: Path = DEFAULT_CONFIG,
    workspace: Path = Path("runs/v8/untrained-baseline-01"),
    smoke_report: Path = Path("runs/v8/smoke-8k/smoke.json"),
    resume: bool = False,
    stop_after: int | None = typer.Option(None, min=1, help="Pause after this many new stages."),
):
    cfg = load_config(config)
    runtime_cfg = load_v4_config(cfg.runtime_config)
    manifest = verify_manifest(runtime_cfg.manifest, runtime_cfg)
    binding = experiment_binding(cfg, runtime_cfg, manifest)
    smoke_result = json.loads(smoke_report.read_text())
    require_pilot_smoke(smoke_result, binding, cfg, runtime_cfg)
    tasks = load_tasks(cfg.data)
    if not cfg.split.is_file():
        raise ValueError("frozen split is required")
    split = freeze_split(tasks, cfg.split)
    lookup = {t.task_id: t for t in tasks}
    tasks = [lookup[key] for key in split["groups"]["development"][:20]]
    identity = {
        "version": 8,
        "config": cfg.model_dump(mode="json"),
        "runtime_config": runtime_cfg.model_dump(mode="json"),
        **binding,
        "split_hash": file_hash(cfg.split),
        "cohort": [t.task_id for t in tasks],
        "student": "untrained_nanbeige",
    }
    run = ExperimentRun(workspace, identity, resume=resume)
    atomic_json(workspace / "identity.json", identity)
    archive_source(
        workspace,
        config,
        cfg.runtime_config,
        grammar_converter=converter_path(manifest)
        if cfg.structured_style in COMPACT_STYLES
        else None,
    )
    timer = run.get("timer") or {"elapsed_seconds": 0.0}
    start, count = time.monotonic(), 0
    remaining = cfg.global_seconds - timer["elapsed_seconds"]
    runtime = StudentSession(
        runtime_cfg,
        manifest,
        workspace,
        evidence_fn=runtime_evidence,
        structured_style=cfg.structured_style,
        symbolic_cell_encoding=cfg.symbolic_cell_encoding,
    )
    try:
        write_report(run, tasks, cfg, elapsed=timer["elapsed_seconds"], evidence={})
        if remaining > 0:
            with runtime:
                for _ in range(cfg.max_cycles * 2):
                    for seed in cfg.seeds:
                        for task in tasks:
                            for mode in cfg.modes:
                                state = run.get(state_key(task.task_id, mode, seed))
                                if state and state["status"] != "running":
                                    continue
                                if time.monotonic() - start >= remaining:
                                    return
                                advance_stage(
                                    blind_task(task),
                                    mode,
                                    seed,
                                    cfg,
                                    run,
                                    runtime,
                                    check=runtime.check,
                                    global_deadline=start + remaining,
                                )
                                elapsed = timer["elapsed_seconds"] + time.monotonic() - start
                                run.save("timer", {"elapsed_seconds": elapsed})
                                write_report(
                                    run, tasks, cfg, elapsed=elapsed, evidence=runtime.evidence()
                                )
                                count += 1
                                if stop_after is not None and count >= stop_after:
                                    return
    finally:
        elapsed = timer["elapsed_seconds"] + time.monotonic() - start
        run.save("timer", {"elapsed_seconds": elapsed})
        report = write_report(run, tasks, cfg, elapsed=elapsed, evidence=runtime.evidence())
        run.close()
        typer.echo(
            json.dumps(
                {"complete": report["complete"], "elapsed_seconds": elapsed, "new_stages": count},
                indent=2,
            )
        )


@app.command("prepare-student")
def prepare(
    source: Path,
    destination: Path,
    split: Path = Path("runs/v4/split.json"),
    tokenizer: Path = Path(".runtime/nanbeige/model-hf"),
):
    typer.echo(json.dumps(prepare_compact_student(source, destination, split, tokenizer), indent=2))


if __name__ == "__main__":
    app()
