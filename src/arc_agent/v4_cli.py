"""Isolated local-first V4 commands. No cloud submission or legacy-provider imports."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import typer

from arc_agent.data import load_tasks
from arc_agent.models import ArcTask
from arc_agent.v4_adaptive import AdaptiveSession
from arc_agent.v4_config import content_hash, file_hash, load_v4_config, verify_manifest
from arc_agent.v4_experiment import MODES, solve_one, write_outputs
from arc_agent.v4_reasoner import LocalReasoner, parse_tool_calls
from arc_agent.v4_runtime import (
    LocalServer,
    external_network_denied,
    memory_sample,
    pressure_allows_start,
)
from arc_agent.v4_state import ExperimentRun, atomic_json, freeze_split
from arc_agent.v4_tools import invoke_tool, tool_schema

app = typer.Typer(no_args_is_help=True, help=__doc__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/v4-nanbeige-local.yaml")
DEFAULT_DATA = Path("data/ARC-AGI-2/data/training")
DEFAULT_SPLIT = Path("runs/v4/split.json")
DEFAULT_CHECKS = Path("runs/v4/correctness.json")


def source_hash() -> str:
    paths = sorted((ROOT / "src/arc_agent").glob("*.py"))
    paths += [ROOT / "tests/test_v4.py", ROOT / "scripts/v4-loopback.sb"]
    return content_hash({str(p.relative_to(ROOT)): file_hash(p) for p in paths})


def binding(config) -> dict:
    return {"source_hash": source_hash(), "manifest_hash": file_hash(config.manifest)}


def read_checks(path: Path) -> dict:
    checks = json.loads(path.read_text())
    if checks.get("source_hash") != source_hash() or not checks.get("passed"):
        raise ValueError("correctness tests must pass for this exact source revision")
    return checks


def gate_decision(report: dict, evidence: dict, checks: dict) -> dict:
    failures = []
    if report.get("timing_mode") == "diagnostic":
        failures.append("extended-deadline diagnostic is not a fixed-allowance Phase 0 gate run")
    if not checks.get("passed"):
        failures.append("harness correctness tests have not passed")
    for name in (
        "offline_enforced",
        "metal_verified",
        "memory_safe",
        "smoke_passed",
        "resume_verified",
    ):
        if evidence.get(name) is not True:
            failures.append(name + " is not verified")
    for mode in MODES:
        result = report["modes"][mode]
        if result["completed_tasks"] != 20 or result["total_tasks"] != 20:
            failures.append(mode + " does not contain the full 20-task pilot")
        if result["valid_response_rate"] < 0.95:
            failures.append(mode + " structured-response rate is below 95%")
    return {
        "passed": not failures,
        "failures": failures,
        "unlocks": ["phase1_portability"] if not failures else [],
        "accuracy_is_separate": True,
        "cloud_training_requires_approval": True,
    }


@app.command("freeze-split")
def freeze(data: Path = DEFAULT_DATA, output: Path = DEFAULT_SPLIT):
    """Freeze source task membership; derived examples must use this same membership."""
    result = freeze_split(load_tasks(data), output)
    typer.echo(
        json.dumps(
            {
                "split": str(output.resolve()),
                "counts": {k: len(v) for k, v in result["groups"].items()},
            },
            indent=2,
        )
    )


@app.command("preflight")
def preflight(config: Path = DEFAULT_CONFIG, output: Path = Path("runs/v4/preflight.json")):
    """Check local resource readiness without loading any model."""
    cfg = load_v4_config(config)
    sample = memory_sample()
    free = shutil.disk_usage(ROOT).free
    network_denied = external_network_denied()
    ready = (
        pressure_allows_start(sample["pressure_level"], cfg)
        and sample["swap_used_bytes"] is not None
        and sample["server_rss_bytes"] is not None
        and network_denied
        and free >= cfg.min_free_disk_gib * 1024**3
    )
    report = {
        "ready_for_smoke": ready,
        "phase0_passed": False,
        "memory": sample,
        "free_disk_bytes": free,
        "external_network_denied": network_denied,
        "warning_pressure_policy": cfg.warning_pressure_policy,
    }
    atomic_json(output, report)
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(0 if ready else 1)


@app.command("verify")
def verify(output: Path = DEFAULT_CHECKS):
    """Run V4 correctness tests and bind their result to the exact source files."""
    output.parent.mkdir(parents=True, exist_ok=True)
    xml_path = output.with_suffix(".xml")
    initial = source_hash()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_v4.py",
            "-q",
            f"--junitxml={xml_path.resolve()}",
        ],
        cwd=ROOT,
        check=False,
    )
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    if xml_path.exists():
        for suite in ET.parse(xml_path).getroot().iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.attrib.get(key, 0))
    passed = (
        result.returncode == 0
        and initial == source_hash()
        and counts["tests"] > 0
        and not any(counts[k] for k in ("failures", "errors", "skipped"))
    )
    atomic_json(
        output,
        {
            "source_hash": initial,
            "passed": passed,
            "counts": counts,
            "created_at": time.time(),
            "junit": str(xml_path.resolve()),
        },
    )
    raise typer.Exit(0 if passed else 1)


def runtime_evidence(server: LocalServer, workspace: Path) -> dict:
    samples = server.samples
    baseline = (server.baseline or {}).get("swap_used_bytes") or 0
    path = workspace / "server.log"
    log = (
        path.read_bytes()[server.log_start_offset :].decode(errors="replace")
        if path.exists()
        else ""
    )
    offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    return {
        "offline_enforced": server.offline_verified,
        "metal_verified": "ggml_metal_init" in log
        and any(int(a) == int(b) > 0 for a, b in offload),
        "memory_safe": not server.guard_error and bool(samples),
        "prompt_cache_mib": server.config.prompt_cache_mib,
        "prompt_cache_disabled_verified": server.config.prompt_cache_mib == 0
        and "prompt cache is disabled" in log,
        "warning_pressure_policy": server.config.warning_pressure_policy,
        "warning_pressure_samples": sum(s["pressure_level"] == 2 for s in samples),
        "guard_error": server.guard_error,
        "peak_server_rss_bytes": max((s["server_rss_bytes"] or 0 for s in samples), default=0),
        "swap_growth_bytes": max(
            ((s["swap_used_bytes"] or baseline) - baseline for s in samples), default=0
        ),
        "peak_pressure_level": max(
            (s["pressure_level"] or 0 for s in samples),
            default=(server.baseline or {}).get("pressure_level") or 0,
        ),
        "initial_pressure_level": (server.baseline or {}).get("pressure_level"),
        "initial_swap_used_bytes": (server.baseline or {}).get("swap_used_bytes"),
        "sample_count": len(samples),
    }


@app.command("smoke")
def smoke(
    config: Path = DEFAULT_CONFIG,
    workspace: Path = Path("runs/v4/smoke"),
    resume: bool = False,
    baseline_smoke: Path | None = None,
):
    """Live Metal/native-format/tool/restart fixtures, with external networking denied."""
    cfg = load_v4_config(config)
    if cfg.context_tokens != 4096:
        if baseline_smoke is None:
            raise ValueError("first local compatibility check must use 4K context")
        prior_smoke = json.loads(baseline_smoke.read_text())
        if (
            not prior_smoke.get("passed")
            or prior_smoke.get("context_tokens") != 4096
            or prior_smoke.get("cache_type") != cfg.cache_type
            or any(prior_smoke.get(k) != v for k, v in binding(cfg).items())
        ):
            raise ValueError("larger context requires a current-source, matching-cache 4K smoke")
    manifest = verify_manifest(cfg.manifest, cfg)
    identity = {**binding(cfg), "config": cfg.model_dump(mode="json"), "kind": "smoke"}
    run = ExperimentRun(workspace, identity, resume=resume)
    fixture = ArcTask.model_validate(
        {
            "task_id": "v4_identity_fixture",
            "train": [
                {"input": [[1, 0], [0, 2]], "output": [[1, 0], [0, 2]]},
                {"input": [[3, 4]], "output": [[3, 4]]},
            ],
            "test": [{"input": [[5], [6]], "output": [[5], [6]]}],
        }
    )
    evidence = {
        **binding(cfg),
        "passed": False,
        "cache_type": cfg.cache_type,
        "context_tokens": cfg.context_tokens,
    }
    server = LocalServer(cfg, manifest, workspace)
    try:
        with server:
            reasoner = LocalReasoner(cfg, manifest)
            try:
                state = solve_one(fixture, "direct", cfg, run, reasoner, server.check)
                server.check()
                messages = [
                    {
                        "role": "user",
                        "content": 'Call run_program now with program {"op":"identity"}. '
                        "Use the native tool-call format; do not answer with prose.",
                    }
                ]
                prepared = reasoner.prepare(messages, tool_schema(), seed=cfg.seed, task=fixture)
                response = run.request("native-tool-fixture", prepared)
                if response is None:
                    response = reasoner.complete(prepared, timeout=cfg.task_seconds)
                    run.response("native-tool-fixture", response)
                calls = parse_tool_calls(response["final"])
                tool_ok = any(
                    call["name"] == "run_program" and invoke_tool(fixture, call).get("verified")
                    for call in calls
                )
                if cfg.context_tokens > 4096:
                    # A deterministic long-context fixture, never a training trajectory.
                    # Exercise attention beyond 4K rather than just allocating an 8K cache.
                    padding = "Reference grid: 12/34.\n" * 550
                    long_messages = [
                        {
                            "role": "user",
                            "content": padding
                            + (
                                "\nIgnore the reference repetitions. Return JSON only: "
                                '{"predictions":[{"attempt_1":["5","6"],'
                                '"attempt_2":"same_as_1"}]} . Keep reasoning brief.'
                            ),
                        }
                    ]
                    prepared_long = reasoner.prepare(long_messages, [], seed=cfg.seed, task=fixture)
                    if not 4096 < prepared_long["_prompt_tokens"] < cfg.context_tokens - 128:
                        raise ValueError(
                            "long-context fixture does not exercise the requested window"
                        )
                    long_response = run.request("long-context-fixture", prepared_long)
                    if long_response is None:
                        long_response = reasoner.complete(prepared_long, timeout=cfg.task_seconds)
                        run.response("long-context-fixture", long_response)
                    from arc_agent.v4_experiment import predictions_from_json
                    from arc_agent.v4_reasoner import final_json

                    long_predictions = predictions_from_json(final_json(long_response["final"]), 1)
                    evidence["long_context_verified"] = long_predictions[0]["attempt_1"] == [
                        [5],
                        [6],
                    ]
                    evidence["long_context_prompt_tokens"] = prepared_long["_prompt_tokens"]
                # Reopen the durable database; completed work must not invoke the model.
                before = run.db.execute("SELECT count(*) FROM calls").fetchone()[0]
                run.close()
                run = ExperimentRun(workspace, identity, resume=True)
                replay = solve_one(fixture, "direct", cfg, run, reasoner, server.check)
                after = run.db.execute("SELECT count(*) FROM calls").fetchone()[0]
                server.check()
                evidence.update(runtime_evidence(server, workspace))
                evidence.update(
                    {
                        "direct_valid": not state["used_fallback"],
                        "direct_correct": state["attempts"][0]["attempt_1"] == [[5], [6]],
                        "native_tool_verified": tool_ok,
                        "reasoning_separated": bool(response["reasoning"])
                        and bool(response["final"]),
                        "resume_verified": before == after and replay == state,
                    }
                )
                evidence["passed"] = all(
                    evidence.get(k) is True
                    for k in (
                        "offline_enforced",
                        "metal_verified",
                        "memory_safe",
                        "direct_valid",
                        "direct_correct",
                        "native_tool_verified",
                        "reasoning_separated",
                        "resume_verified",
                    )
                )
                if cfg.context_tokens > 4096:
                    evidence["passed"] &= evidence.get("long_context_verified") is True
            finally:
                reasoner.close()
    except (Exception, KeyboardInterrupt) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
    finally:
        evidence.update(runtime_evidence(server, workspace))
        if not evidence["memory_safe"]:
            evidence["passed"] = False
        atomic_json(workspace / "smoke.json", evidence)
        run.close()
    typer.echo(json.dumps(evidence, indent=2))
    raise typer.Exit(0 if evidence["passed"] else 1)


@app.command("pilot")
def pilot(
    config: Path = DEFAULT_CONFIG,
    data: Path = DEFAULT_DATA,
    split: Path = DEFAULT_SPLIT,
    workspace: Path = Path("runs/v4/local-pilot"),
    checks: Path = DEFAULT_CHECKS,
    smoke_report: Path = Path("runs/v4/smoke/smoke.json"),
    resume: bool = False,
    adaptive_context: bool = False,
):
    """Run the fixed local development pilot. Labels enter only the post-result scorer."""
    cfg = load_v4_config(config)
    if cfg.context_tokens not in {4096, 8192}:
        raise ValueError("Phase 0 accepts 4K or memory-checked 8K contexts only")
    if adaptive_context and cfg.context_tokens != 8192:
        raise ValueError("adaptive context pilot must start at 8K")
    if cfg.timing_mode == "diagnostic" and not adaptive_context:
        raise ValueError("extended timing diagnostic requires the adaptive-context runtime")
    correctness = read_checks(checks)
    manifest = verify_manifest(cfg.manifest, cfg)
    smoke_result = json.loads(smoke_report.read_text())
    if not smoke_result.get("passed") or any(
        smoke_result.get(k) != v for k, v in binding(cfg).items()
    ):
        raise ValueError(
            "pass the live 4K smoke check with these exact artifacts and sources first"
        )
    if smoke_result.get("cache_type") != cfg.cache_type or smoke_result.get("context_tokens") != (
        8192 if adaptive_context else 4096
    ):
        raise ValueError("pass the 4K smoke check with the same KV-cache format first")
    if adaptive_context and not smoke_result.get("long_context_verified"):
        raise ValueError("adaptive pilot requires the live beyond-4K prompt fixture")
    tasks = load_tasks(data)
    if not split.exists():
        raise ValueError("freeze the split explicitly before running a pilot")
    frozen = freeze_split(tasks, split)
    lookup = {task.task_id: task for task in tasks}
    selected = [lookup[key] for key in frozen["groups"]["development"][: cfg.pilot_tasks]]
    if len(selected) != cfg.pilot_tasks:
        raise ValueError("insufficient development tasks")
    identity = {
        **binding(cfg),
        "config": cfg.model_dump(mode="json"),
        "split_hash": file_hash(split),
        "task_ids": [t.task_id for t in selected],
    }
    if adaptive_context:
        identity["context_policy"] = {
            "start": 8192,
            "step": 4096,
            "ceiling": 262144,
            "trigger": "full per-call answer allowance no longer fits",
            "preserve_history": True,
            "reset_task_deadline": False,
        }
    run = ExperimentRun(workspace, identity, resume=resume)
    started = time.monotonic()
    evidence = {
        **binding(cfg),
        "smoke_passed": True,
        "resume_verified": smoke_result["resume_verified"],
    }
    try:
        write_outputs(selected, run, cfg)  # Complete ordered fallbacks before model startup.
        runtime = (
            AdaptiveSession(cfg, manifest, workspace, evidence_fn=runtime_evidence)
            if adaptive_context
            else LocalServer(cfg, manifest, workspace)
        )
        with runtime as server:
            reasoner = server if adaptive_context else LocalReasoner(cfg, manifest)
            try:

                def checkpoint_report():
                    partial = write_outputs(selected, run, cfg)
                    partial["evidence"] = server.evidence()
                    atomic_json(workspace / "report.json", partial)

                for task in selected:
                    for mode in MODES:
                        result = solve_one(
                            task,
                            mode,
                            cfg,
                            run,
                            reasoner,
                            server.check,
                            on_checkpoint=checkpoint_report
                            if cfg.timing_mode == "diagnostic"
                            else lambda: None,
                        )
                        report = write_outputs(selected, run, cfg)
                        if adaptive_context:
                            report["evidence"] = server.evidence()
                            atomic_json(workspace / "report.json", report)
                        typer.echo(
                            json.dumps(
                                {
                                    "task": task.task_id,
                                    "mode": mode,
                                    "seconds": round(result["elapsed_seconds"], 2),
                                    "valid": result["valid_responses"],
                                    "errors": result["errors"],
                                    "completed": report["modes"][mode]["completed_tasks"],
                                    "stop_reason": result.get("timing", {}).get("stop_reason"),
                                    "time_allowance": result.get("timing", {}).get(
                                        "allowance_seconds"
                                    ),
                                }
                            ),
                            color=False,
                        )
                        server.check()
            finally:
                reasoner.close()
                evidence.update(
                    server.evidence() if adaptive_context else runtime_evidence(server, workspace)
                )
    except (Exception, KeyboardInterrupt) as error:
        evidence.update({"interrupted": True, "error": f"{type(error).__name__}: {error}"})
    finally:
        evidence["session_wall_seconds"] = time.monotonic() - started
        report = write_outputs(selected, run, cfg)
        decision = gate_decision(report, evidence, correctness)
        if evidence.get("interrupted"):
            decision.update({"passed": False, "unlocks": []})
            decision["failures"].append(evidence["error"])
        diagnostic = cfg.timing_mode == "diagnostic"
        report.update(
            {
                "evidence": evidence,
                "phase0_gate": decision,
                "other_phases": "requires_fixed_allowance_validation"
                if diagnostic
                else ("phase1_unlocked" if decision["passed"] else "blocked_by_phase0"),
            }
        )
        atomic_json(workspace / "report.json", report)
        atomic_json(
            workspace / "gate.json",
            {**binding(cfg), **decision, "report_sha256": file_hash(workspace / "report.json")},
        )
        run.close()
    typer.echo(json.dumps(report, indent=2))
    raise typer.Exit(0 if decision["passed"] else 1)


if __name__ == "__main__":
    app()
