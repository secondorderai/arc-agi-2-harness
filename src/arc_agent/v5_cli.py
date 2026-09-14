"""Astra subscription teacher -> verified symbolic artifacts -> Nanbeige SFT dataset."""

import json
import sqlite3
from pathlib import Path

import typer

from arc_agent.v4_state import ExperimentRun, atomic_json
from arc_agent.v5_config import load_config
from arc_agent.v5_dataset import prepare_student
from arc_agent.v5_pipeline import collect_task, export_examples, run_identity, select_tasks
from arc_agent.v5_teacher import AstraTeacher

app = typer.Typer(no_args_is_help=True, help=__doc__)
DEFAULT_CONFIG = Path("configs/v5-astra-symbolic.yaml")
DEFAULT_WORKSPACE = Path("runs/v5/astra-symbolic-pilot-ready")


@app.command()
def status(workspace: Path = DEFAULT_WORKSPACE):
    """Read checkpoints without loading either model or acquiring a writer lock."""
    with sqlite3.connect(
        (workspace / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        states = [
            json.loads(row[0])
            for row in db.execute("SELECT payload FROM work WHERE key LIKE 'task:%' ORDER BY key")
        ]
    typer.echo(
        json.dumps(
            {
                "tasks": [
                    {k: s.get(k) for k in ("task_id", "status", "round", "started_at", "error")}
                    for s in states
                ],
                "competition_score": None,
                "training_launched": False,
            },
            indent=2,
        )
    )


@app.command("prepare-student")
def prepare(
    workspace: Path = DEFAULT_WORKSPACE,
    tokenizer_dir: Path = Path(".runtime/nanbeige/model-hf"),
    config: Path = DEFAULT_CONFIG,
):
    """Prepare native Nanbeige tokens/labels locally, without loading model weights."""
    report = prepare_student(workspace, tokenizer_dir, load_config(config).student_sequence_tokens)
    typer.echo(json.dumps(report, indent=2))


@app.command()
def preflight(config: Path = DEFAULT_CONFIG):
    """Verify exact subscription model access without sending task data."""
    cfg = load_config(config)
    teacher = AstraTeacher(settings=cfg.teacher, workspace=cfg.auth_workspace)
    try:
        typer.echo(json.dumps(teacher.preflight(), indent=2))
    finally:
        teacher.close()


@app.command()
def collect(
    config: Path = DEFAULT_CONFIG,
    workspace: Path = DEFAULT_WORKSPACE,
    resume: bool = False,
):
    """Collect training-only symbolic trajectories; never launch GPU training."""
    cfg = load_config(config)
    tasks, split = select_tasks(cfg)
    identity = run_identity(cfg, split)
    run = ExperimentRun(workspace, identity, resume=resume)
    teacher = AstraTeacher(settings=cfg.teacher, workspace=cfg.auth_workspace)
    try:
        atomic_json(workspace / "identity.json", identity)
        atomic_json(workspace / "preflight.json", teacher.preflight())
        for task in tasks:
            collect_task(task, cfg, run, teacher)
            report = export_examples(run, cfg, tasks, split)
            atomic_json(
                workspace / "status.json",
                {
                    "tasks": [run.get(f"task:{t.task_id}") for t in tasks],
                    "dataset": report,
                    "competition_score": None,
                    "is_training_data_pilot": True,
                },
            )
        typer.echo(json.dumps(report, indent=2))
    finally:
        teacher.close()
        run.close()


if __name__ == "__main__":
    app()
