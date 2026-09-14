"""V7 training-only curriculum pilot. No student training or paid job launch."""

import json
import tarfile
from collections import Counter
from pathlib import Path

import typer

from arc_agent.v4_state import ExperimentRun, atomic_json
from arc_agent.v5_config import load_config
from arc_agent.v5_dataset import prepare_student
from arc_agent.v5_pipeline import collect_task, run_identity, select_tasks
from arc_agent.v7_dataset import export_verified_examples
from arc_agent.v7_symbolic import PythonTrainingContract
from arc_agent.v7_teacher import BridgedAstraTeacher

app = typer.Typer(no_args_is_help=True, help=__doc__)
DEFAULT_CONFIG = Path("configs/v7-astra-symbolic-bridge.yaml")
DEFAULT_WORKSPACE = Path("runs/v7/astra-symbolic-bridge-01")


def pilot_report(run, tasks, dataset):
    states = [run.get(f"task:{t.task_id}") for t in tasks]
    records = [run.artifacts.get(ref) for s in states if s for ref in s.get("records", [])]
    calls = [
        run.get(f"call:{t.task_id}:{i}")
        for t in tasks
        for i in range((run.get(f"task:{t.task_id}") or {}).get("round", 0) + 1)
    ]
    dispatches = sum(bool(c and c.get("snapshot")) for c in calls)
    categories = Counter(r["category"] for r in records if r["verification"].get("accepted"))
    valid = sum(bool(r["structured_valid"]) for r in records)
    statuses = Counter(s["status"] if s else "not_started" for s in states)
    complete = all(s and s["status"] not in {"running", "provider_stopped"} for s in states)
    accepted_tasks = len({r["task_id"] for r in records if r["verification"].get("accepted")})
    return {
        "cohort": [t.task_id for t in tasks],
        "split": "training",
        "mode": "structured_symbolic_with_bounded_python_witness",
        "tasks": states,
        "status_counts": dict(statuses),
        "dispatched_calls": dispatches,
        "completed_responses": len(records),
        "valid_structured_responses": valid,
        "structured_valid_rate_per_dispatch": valid / dispatches if dispatches else None,
        "accepted_categories": dict(categories),
        "full_symbolic_source_tasks": accepted_tasks,
        "rejections": [
            {"task_id": r["task_id"], "round": r["round"], "verification": r["verification"]}
            for r in records
            if not r["verification"].get("accepted")
        ],
        "complete": complete,
        "data_quality_gate_passed": bool(
            complete
            and dispatches
            and valid / dispatches >= 0.95
            and accepted_tasks >= 2
            and categories["interpretation"]
            and categories["repair"]
        ),
        "dataset": dataset,
        "student_promotion_gate_passed": False,
        "student_training_launched": False,
        "competition_score": None,
    }


@app.command()
def collect(
    config: Path = DEFAULT_CONFIG,
    workspace: Path = DEFAULT_WORKSPACE,
    resume: bool = False,
    stop_after: int | None = typer.Option(
        None, min=1, help="Checkpoint after this many new tasks."
    ),
):
    cfg = load_config(config)
    tasks, split = select_tasks(cfg)
    identity = run_identity(cfg, split)
    identity.update(version=7, contract="symbolic_python_witness_v1")
    run = ExperimentRun(workspace, identity, resume=resume)
    teacher = BridgedAstraTeacher(settings=cfg.teacher, workspace=cfg.auth_workspace)
    contract = PythonTrainingContract()
    try:
        atomic_json(workspace / "identity.json", identity)
        if not (workspace / "source.tar.gz").exists():
            # Archive only source/config, never authentication, data labels or old traces.
            with tarfile.open(workspace / "source.tar.gz", "w:gz") as archive:
                for path in sorted(Path("src/arc_agent").glob("*.py")):
                    archive.add(path, arcname=str(path))
                archive.add(config, arcname=f"configs/{config.name}")
        atomic_json(workspace / "preflight.json", teacher.preflight())
        count = 0
        for task in tasks:
            before = run.get(f"task:{task.task_id}")
            collect_task(task, cfg, run, teacher, contract=contract)
            dataset = export_verified_examples(run, cfg, tasks, split)
            report = pilot_report(run, tasks, dataset)
            atomic_json(workspace / "pilot-report.json", report)
            if not before or before["status"] == "running":
                count += 1
            if stop_after is not None and count >= stop_after:
                break
        typer.echo(
            json.dumps(
                {k: v for k, v in report.items() if k not in {"tasks", "rejections", "cohort"}},
                indent=2,
            )
        )
    finally:
        teacher.close()
        run.close()


@app.command("prepare-student")
def prepare(
    workspace: Path = DEFAULT_WORKSPACE,
    config: Path = DEFAULT_CONFIG,
    tokenizer_dir: Path = Path(".runtime/nanbeige/model-hf"),
):
    report = prepare_student(workspace, tokenizer_dir, load_config(config).student_sequence_tokens)
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
