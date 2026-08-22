from __future__ import annotations

import json
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer

from arc_agent.config import SolverConfig, load_config
from arc_agent.data import (
    attach_solutions,
    deterministic_split,
    load_tasks,
    sha256_path,
    write_submission,
)
from arc_agent.llm import OpenAICompatibleAdapter
from arc_agent.report import make_summary, write_run_artifacts
from arc_agent.scoring import score_submission
from arc_agent.skills import (
    assert_no_eval_provenance,
    learn_skills,
    load_cards,
)
from arc_agent.solver import ArcSolver

app = typer.Typer(no_args_is_help=True, help="Verifier-guided ARC-AGI-2 solver")


def _adapter(config: SolverConfig):
    if config.mlx_ttt.enabled:
        from arc_agent.ttt import MlxArcTTTAdapter

        settings = config.mlx_ttt
        return MlxArcTTTAdapter(
            model_path=settings.model_path,
            rank=settings.rank,
            scale=settings.scale,
            learning_rate=settings.learning_rate,
            epochs=settings.epochs,
            min_epochs=settings.min_epochs,
            early_stop_median_loss=settings.early_stop_median_loss,
            early_stop_max_loss=settings.early_stop_max_loss,
            train_token_layers=settings.train_token_layers,
            augmentation_count=settings.augmentation_count,
            inference_augmentations=settings.inference_augmentations,
            scoring_augmentations=settings.scoring_augmentations,
            samples=settings.samples,
            include_greedy=settings.include_greedy,
            temperature=settings.temperature,
            top_p=settings.top_p,
            dfs_min_probability=settings.dfs_min_probability,
            dfs_max_candidates=settings.dfs_max_candidates,
            dfs_max_nodes=settings.dfs_max_nodes,
            max_tokens=settings.max_tokens,
            max_grid_side=settings.max_grid_side,
            seed=config.seed,
        )
    if not config.model.enabled:
        return None
    return OpenAICompatibleAdapter(
        base_url=config.model.base_url,
        model=config.model.model,
        api_key=config.model.api_key,
        max_tokens=config.model.max_tokens,
        thinking_budget_tokens=config.model.thinking_budget_tokens,
        temperature=config.model.temperature,
        timeout_seconds=config.model.timeout_seconds,
        vision_enabled=config.model.vision_enabled,
        structural_summary_enabled=config.model.structural_summary_enabled,
    )


def _run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _cards(path: Path | None):
    return load_cards(path) if path and path.exists() else []


@app.command()
def learn(
    data: Annotated[Path, typer.Option(exists=True, readable=True, help="Training task directory")],
    output: Annotated[Path, typer.Option(help="Generated skill directory")] = Path(
        "skills/generated"
    ),
    validation_percent: Annotated[
        int,
        typer.Option(min=0, max=50, help="Internal validation percentage; 0 learns all tasks"),
    ] = 20,
) -> None:
    """Mine verified programs and build provenance-bearing Markdown skills."""
    tasks = load_tasks(data)
    if validation_percent:
        learn_tasks, validation_tasks = deterministic_split(
            tasks, validation_percent=validation_percent
        )
    else:
        learn_tasks, validation_tasks = tasks, []
    cards = learn_skills(learn_tasks, output, source_path=data)
    split_manifest = {
        "validation_percent": validation_percent,
        "learn_task_ids": [task.task_id for task in learn_tasks],
        "validation_task_ids": [task.task_id for task in validation_tasks],
    }
    (output / "development_split.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True)
    )
    typer.echo(
        f"Wrote {len(cards)} skill cards from {len(learn_tasks)} tasks; "
        f"held out {len(validation_tasks)} tasks at {output}"
    )


def _solve(
    *,
    data: Path,
    config_path: Path,
    skills: Path | None,
    output: Path,
    solutions: Path | None,
    score: bool,
) -> None:
    config = load_config(config_path)
    tasks = load_tasks(data)
    if solutions:
        tasks = attach_solutions(tasks, solutions)
    cards = _cards(skills)
    solver = ArcSolver(config, cards=cards, adapter=_adapter(config))
    run_id = _run_id(config.name)
    run_dir = output / run_id
    runs, submission = solver.solve_tasks(tasks)
    submission_path = write_submission(submission, tasks, run_dir / "submission.json")
    measured = score_submission(tasks, submission) if score else None
    summary = make_summary(
        run_id=run_id,
        dataset_sha256=sha256_path(data),
        config_sha256=config.sha256(),
        runs=runs,
        score=measured,
    )
    report_path = write_run_artifacts(run_dir, summary, runs)
    typer.echo(f"Submission: {submission_path}")
    typer.echo(f"Report: {report_path}")
    if measured:
        typer.echo(
            f"pass@2={measured.pass_at_2 * 100:.2f}% "
            f"strict_task_accuracy={measured.strict_task_accuracy * 100:.2f}%"
        )


@app.command()
def solve(
    data: Annotated[Path, typer.Option(exists=True, readable=True)],
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/local-mlx.yaml"
    ),
    skills: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    output: Annotated[Path, typer.Option()] = Path("runs"),
    solutions: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
) -> None:
    """Solve unlabelled tasks and create submission.json."""
    _solve(
        data=data,
        config_path=config,
        skills=skills,
        output=output,
        solutions=solutions,
        score=False,
    )


@app.command()
def evaluate(
    data: Annotated[Path, typer.Option(exists=True, readable=True)],
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/deterministic.yaml"
    ),
    skills: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    output: Annotated[Path, typer.Option()] = Path("runs"),
    public_eval: Annotated[
        Path | None,
        typer.Option(
            exists=True, readable=True, help="Check skill provenance against this eval set"
        ),
    ] = None,
) -> None:
    """Solve labelled tasks and report official pass@2 and strict accuracy."""
    tasks = load_tasks(data)
    if any(pair.output is None for task in tasks for pair in task.test):
        raise typer.BadParameter("evaluate requires test outputs")
    if skills and public_eval:
        assert_no_eval_provenance(skills, load_tasks(public_eval))
    _solve(
        data=data,
        config_path=config,
        skills=skills,
        output=output,
        solutions=None,
        score=True,
    )


@app.command("validate-submission")
def validate_submission_command(
    data: Annotated[Path, typer.Option(exists=True, readable=True)],
    submission: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Validate Kaggle task coverage, attempt count, grid shapes, and cell values."""
    from arc_agent.models import Attempt

    tasks = load_tasks(data)
    payload = json.loads(submission.read_text())
    parsed = {
        task_id: [Attempt.model_validate(attempt) for attempt in attempts]
        for task_id, attempts in payload.items()
    }
    temporary = submission.with_suffix(".validated.json")
    write_submission(parsed, tasks, temporary)
    temporary.unlink()
    typer.echo(
        f"Valid submission: {len(tasks)} tasks, {sum(len(task.test) for task in tasks)} outputs"
    )


@app.command("benchmark-model")
def benchmark_model(
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/local-mlx.yaml"
    ),
    output: Annotated[Path, typer.Option()] = Path("runs/model-benchmark.json"),
) -> None:
    """Measure local server health, response latency, and reported token throughput."""
    settings = load_config(config)
    url = f"{settings.model.base_url.rstrip('/')}/chat/completions"
    started = time.monotonic()
    with httpx.Client(timeout=settings.model.timeout_seconds) as client:
        response = client.post(
            url,
            headers={"Authorization": f"Bearer {settings.model.api_key}"},
            json={
                "model": settings.model.model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return only this JSON object: {"status":"ok"}',
                    }
                ],
                "temperature": 0,
                "max_tokens": min(640, settings.model.max_tokens),
                "thinking_budget_tokens": min(32, settings.model.thinking_budget_tokens),
            },
        )
        response.raise_for_status()
    elapsed = time.monotonic() - started
    body = response.json()
    message = body["choices"][0]["message"]
    usage = body.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens", 0))
    result = {
        "model": settings.model.model,
        "base_url": settings.model.base_url,
        "elapsed_seconds": elapsed,
        "reported_usage": usage,
        "completion_tokens_per_second": completion_tokens / elapsed if elapsed else None,
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("v2-build-bank")
def v2_build_bank(
    data: Annotated[
        Path, typer.Option(exists=True, readable=True, help="Labelled 1,000-task training set")
    ] = Path("data/ARC-AGI-2/data/training"),
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/v2-luna-xhigh.yaml"
    ),
    workspace: Annotated[Path, typer.Option(help="Durable V2 run workspace")] = Path(
        "runs/v2-main"
    ),
    resume: Annotated[
        bool, typer.Option(help="Require an existing run instead of creating one")
    ] = False,
    restart_task: Annotated[
        bool,
        typer.Option(help="Start a fresh Luna chain for the active task without skipping it"),
    ] = False,
) -> None:
    """Sequentially synthesize, verify, checkpoint, and freeze the V2 program bank."""
    from arc_agent.v2_config import load_v2_config
    from arc_agent.v2_experiment import V2Orchestrator
    from arc_agent.v2_state import StateMismatch, V2State, WorkspaceBusy

    tasks = load_tasks(data)
    if any(pair.output is None for task in tasks for pair in task.test):
        raise typer.BadParameter("V2 bank building requires labelled test outputs")
    try:
        with V2State(workspace) as state:
            state.set_meta("training_data_path", str(data.resolve()))
            state.set_meta("training_config_path", str(config.resolve()))
            orchestrator = V2Orchestrator(load_v2_config(config), state)
            try:
                result = orchestrator.build_bank(
                    tasks,
                    dataset_hash=sha256_path(data),
                    resume_only=resume,
                    restart_task=restart_task,
                )
            finally:
                orchestrator.close()
    except (StateMismatch, WorkspaceBusy) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result.model_dump_json(indent=2))
    if result.status == "paused_quota":
        raise typer.Exit(75)
    if result.status == "paused_configuration":
        raise typer.Exit(78)


@app.command("v2-evaluate")
def v2_evaluate(
    data: Annotated[
        Path, typer.Option(exists=True, readable=True, help="Labelled 120-task public eval set")
    ] = Path("data/ARC-AGI-2/data/evaluation"),
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/v2-luna-xhigh.yaml"
    ),
    workspace: Annotated[Path, typer.Option(help="Completed V2 bank workspace")] = Path(
        "runs/v2-main"
    ),
    resume: Annotated[bool, typer.Option(help="Require an existing evaluation run")] = False,
    restart_task: Annotated[
        bool,
        typer.Option(help="Start a fresh Luna chain for the active task without skipping it"),
    ] = False,
) -> None:
    """Run or resume the frozen, label-blind V2 public evaluation."""
    from arc_agent.v2_config import load_v2_config
    from arc_agent.v2_experiment import V2Orchestrator
    from arc_agent.v2_state import StateMismatch, V2State, WorkspaceBusy

    tasks = load_tasks(data)
    if any(pair.output is None for task in tasks for pair in task.test):
        raise typer.BadParameter("V2 evaluation scoring requires labelled test outputs")
    try:
        with V2State(workspace) as state:
            state.set_meta("evaluation_data_path", str(data.resolve()))
            state.set_meta("evaluation_config_path", str(config.resolve()))
            orchestrator = V2Orchestrator(load_v2_config(config), state)
            try:
                result = orchestrator.evaluate(
                    tasks,
                    dataset_hash=sha256_path(data),
                    resume_only=resume,
                    restart_task=restart_task,
                )
            finally:
                orchestrator.close()
    except (StateMismatch, WorkspaceBusy) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result.model_dump_json(indent=2))
    if result.status == "paused_quota":
        raise typer.Exit(75)
    if result.status == "paused_configuration":
        raise typer.Exit(78)


@app.command("v2-status")
def v2_status(
    workspace: Annotated[Path, typer.Option(exists=True, readable=True)] = Path("runs/v2-main"),
    phase: Annotated[str, typer.Option(help="training or evaluation")] = "training",
) -> None:
    """Show checkpoint, quota-pause, usage, and exact V2 resume information."""
    from arc_agent.v2_state import V2State

    if phase not in {"training", "evaluation"}:
        raise typer.BadParameter("phase must be training or evaluation")
    with V2State(workspace, read_only=True) as state:
        summary = state.summary(phase)
        data_path = state.get_meta(f"{phase}_data_path")
        config_path = state.get_meta(f"{phase}_config_path")
    command = "v2-build-bank" if phase == "training" else "v2-evaluate"
    parts = ["uv run arc-agent", command]
    if data_path:
        parts.extend(["--data", shlex.quote(str(data_path))])
    if config_path:
        parts.extend(["--config", shlex.quote(str(config_path))])
    parts.extend(["--workspace", shlex.quote(str(workspace)), "--resume"])
    summary["resume_command"] = " ".join(parts)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("v2-auth-login")
def v2_auth_login(
    workspace: Annotated[Path, typer.Option(help="Durable V2 run workspace")] = Path(
        "runs/v2-main"
    ),
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/v2-luna-xhigh.yaml"
    ),
) -> None:
    """Sign this V2 workspace into Codex with a ChatGPT subscription."""
    from arc_agent.v2_codex import (
        codex_auth_command,
        restart_subscription_daemon,
        subscription_home,
    )
    from arc_agent.v2_config import load_v2_config

    settings = load_v2_config(config).openai
    if settings.auth_mode != "chatgpt_subscription":
        raise typer.BadParameter("config openai.auth_mode must be chatgpt_subscription")
    command, environment = codex_auth_command(settings, workspace)
    typer.echo(f"Codex subscription home: {subscription_home(workspace).resolve()}")
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)
    restart_subscription_daemon(settings, workspace)


@app.command("v2-auth-status")
def v2_auth_status(
    workspace: Annotated[Path, typer.Option(help="Durable V2 run workspace")] = Path(
        "runs/v2-main"
    ),
    config: Annotated[Path, typer.Option(exists=True, readable=True)] = Path(
        "configs/v2-luna-xhigh.yaml"
    ),
) -> None:
    """Show the ChatGPT subscription login used by this V2 workspace."""
    from arc_agent.v2_codex import codex_auth_command
    from arc_agent.v2_config import load_v2_config

    settings = load_v2_config(config).openai
    if settings.auth_mode != "chatgpt_subscription":
        raise typer.BadParameter("config openai.auth_mode must be chatgpt_subscription")
    command, environment = codex_auth_command(settings, workspace)
    completed = subprocess.run([*command, "status"], env=environment, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


if __name__ == "__main__":
    app()
