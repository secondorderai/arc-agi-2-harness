#!/usr/bin/env python3
r"""Run a leakage-resistant benchmark for the object-centric hybrid experiment.

The runner deliberately lives outside :mod:`arc_agent.cli`: it is an experiment
driver, not part of the competition submission path.  It makes the data flow
explicit:

``development tasks -> deterministic learn/holdout split -> neural training``
``                                      \-> holdout solver comparison``

An optional public/evaluation directory is inspected for a manifest only.  Its
tasks are never used to construct examples, fit a model, choose a checkpoint,
or score the developmental result.  This makes it safe to point the runner at
the public evaluation set while iterating locally.

The world-model imports are intentionally lazy.  This keeps ``--help`` and the
split utility usable if the experimental model is temporarily unavailable or
if an optional numerical backend has not been installed yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSplit:
    """A task-level split with the IDs retained for auditability."""

    learn: tuple[Any, ...]
    holdout: tuple[Any, ...]
    seed: int
    holdout_percent: int

    @property
    def learn_ids(self) -> tuple[str, ...]:
        return tuple(str(task.task_id) for task in self.learn)

    @property
    def holdout_ids(self) -> tuple[str, ...]:
        return tuple(str(task.task_id) for task in self.holdout)


def deterministic_task_split(
    tasks: Iterable[Any], *, holdout_percent: int = 20, seed: int = 0
) -> ExperimentSplit:
    """Split whole ARC tasks deterministically, never individual examples.

    A seeded SHA-256 ordering avoids Python's process-randomised ``hash`` and
    makes the manifest reproducible across machines.  For a non-empty input and
    a non-zero holdout percentage, at least one complete task is held out.  The
    latter matters for small smoke datasets where ``20%`` would otherwise round
    to zero.
    """

    if not 0 <= holdout_percent <= 100:
        raise ValueError("holdout_percent must be between 0 and 100")
    material = list(tasks)
    if len({str(task.task_id) for task in material}) != len(material):
        raise ValueError("task IDs must be unique before splitting")
    ordered = sorted(
        material,
        key=lambda task: hashlib.sha256(
            f"{seed}:{task.task_id}".encode()
        ).hexdigest(),
    )
    if not ordered or holdout_percent == 0:
        holdout_count = 0
    elif holdout_percent == 100:
        holdout_count = len(ordered)
    else:
        holdout_count = max(1, math.ceil(len(ordered) * holdout_percent / 100.0))
        holdout_count = min(holdout_count, len(ordered) - 1) if len(ordered) > 1 else 1
    holdout = tuple(ordered[:holdout_count])
    learn = tuple(ordered[holdout_count:])
    return ExperimentSplit(
        learn=learn,
        holdout=holdout,
        seed=seed,
        holdout_percent=holdout_percent,
    )


def _bounded_tasks(tasks: Sequence[Any], cap: int | None) -> tuple[Any, ...]:
    if cap is not None and cap < 1:
        raise ValueError("task caps must be positive when supplied")
    return tuple(tasks if cap is None else tasks[:cap])


def _score_to_dict(score: Any) -> dict[str, Any]:
    """Convert the repository's ``Score`` without importing it at module load."""

    names = (
        "pass_at_2",
        "strict_task_accuracy",
        "correct_outputs",
        "total_outputs",
        "correct_tasks",
        "total_tasks",
    )
    return {name: getattr(score, name) for name in names if hasattr(score, name)}


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _finite(value) for key, value in values.items()}


def _load_arc_tasks(path: str | Path) -> list[Any]:
    # This import is delayed so importing this script does not force the full
    # application dependency graph (useful when world_model is being edited).
    from arc_agent.data import load_tasks

    return load_tasks(path)


def _build_neural_config(args: argparse.Namespace) -> Any:
    from arc_agent.world_model import WorldModelConfig

    return WorldModelConfig(
        max_grid_size=args.max_grid_size,
        hidden_size=args.hidden_size,
        num_slots=args.num_slots,
        recurrent_steps=args.recurrent_steps,
        max_objects=args.max_objects,
        seed=args.seed,
        program_loss_weight=args.program_loss_weight,
    )


def _train_and_evaluate_world_model(
    *,
    learn_tasks: Sequence[Any],
    holdout_tasks: Sequence[Any],
    args: argparse.Namespace,
    checkpoint_path: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    """Train/evaluate one local backend using only development tasks."""

    from arc_agent.world_model_data import build_world_model_examples

    common = {
        "augmentation_count": args.augmentation_count,
        "seed": args.seed,
        "max_examples": args.max_examples,
    }
    learn_examples = build_world_model_examples(learn_tasks, **common)
    holdout_examples = build_world_model_examples(holdout_tasks, **common)
    if not learn_examples:
        raise ValueError("the learn split produced no labelled world-model examples")

    config = _build_neural_config(args)
    started = time.monotonic()
    metadata = {
        "experiment": "hybrid-world-model",
        "backend": args.backend,
        "seed": args.seed,
        "learn_task_ids": [str(task.task_id) for task in learn_tasks],
        "holdout_task_ids": [str(task.task_id) for task in holdout_tasks],
        "training_task_count": len(learn_tasks),
        "public_evaluation_used": False,
    }
    history_rows: list[dict[str, Any]] = []
    if args.backend == "mlx":
        from arc_agent.world_model import (
            MlxHybridWorldModel,
            evaluate_mlx_world_model,
            train_mlx_world_model,
        )

        model = MlxHybridWorldModel(config, learning_rate=args.learning_rate)
        if args.train_neural:
            history_rows = train_mlx_world_model(
                model,
                learn_examples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
            )
            if checkpoint_path is not None:
                checkpoint_path = model.save_checkpoint(checkpoint_path, metadata=metadata)
        elif checkpoint_path is not None and checkpoint_path.is_file():
            model = MlxHybridWorldModel.load_checkpoint(checkpoint_path)
        evaluate = evaluate_mlx_world_model
    else:
        from arc_agent.world_model import (
            HybridWorldModel,
            evaluate_world_model,
            train_world_model,
        )

        model = HybridWorldModel(config)
        if args.train_neural:
            history = train_world_model(
                model,
                learn_examples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                shuffle=True,
                seed=args.seed,
            )
            history_rows = [asdict(row) for row in history.epochs]
            if checkpoint_path is not None:
                checkpoint_path = model.save_checkpoint(checkpoint_path, metadata=metadata)
        elif checkpoint_path is not None and checkpoint_path.is_file():
            model = HybridWorldModel.load_checkpoint(checkpoint_path)
        evaluate = evaluate_world_model

    learn_metrics = evaluate(model, learn_examples, batch_size=args.batch_size)
    holdout_metrics = (
        evaluate(model, holdout_examples, batch_size=args.batch_size)
        if holdout_examples
        else {}
    )
    result = {
        "available": True,
        "backend": args.backend,
        "trained": bool(args.train_neural),
        "learn_tasks": len(learn_tasks),
        "holdout_tasks": len(holdout_tasks),
        "learn_examples": len(learn_examples),
        "holdout_examples": len(holdout_examples),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "training_seconds": time.monotonic() - started,
        "history": history_rows,
        "learn_metrics": _finite_mapping(learn_metrics),
        "holdout_metrics": _finite_mapping(holdout_metrics),
        "public_evaluation_used_for_training": False,
    }
    return result, checkpoint_path


def _solver_config_pair(
    config_path: str | Path,
    checkpoint_path: Path | None,
    *,
    backend: str,
) -> tuple[Any, Any]:
    from arc_agent.config import load_config

    supplied = load_config(config_path)
    disabled_hybrid = supplied.hybrid.model_copy(update={"enabled": False})
    baseline = supplied.model_copy(update={"hybrid": disabled_hybrid})
    hybrid_updates: dict[str, Any] = {
        "enabled": True,
        "neural_enabled": checkpoint_path is not None,
        "neural_backend": backend,
        # Do not silently reuse a checkpoint named in the supplied config when
        # this run did not train/load an explicitly approved checkpoint.
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    hybrid_settings = supplied.hybrid.model_copy(update=hybrid_updates)
    hybrid = supplied.model_copy(update={"hybrid": hybrid_settings})
    return baseline, hybrid


def _adapter_for_config(config: Any) -> Any:
    """Construct the configured adapter only when the supplied config enables one."""

    if not config.model.enabled and not config.mlx_ttt.enabled:
        return None
    # Reuse the CLI's tested adapter construction, but keep it out of import time.
    from arc_agent.cli import _adapter

    return _adapter(config)


def _run_solver(
    tasks: Sequence[Any], config: Any, *, solver_factory: Callable[..., Any] | None = None
) -> dict[str, Any]:
    from arc_agent.scoring import score_submission
    from arc_agent.solver import ArcSolver

    factory = solver_factory or ArcSolver
    started = time.monotonic()
    try:
        solver = factory(config, adapter=_adapter_for_config(config))
        runs, submission = solver.solve_tasks(tasks)
        score = score_submission(tasks, submission)
        task_by_id = {str(task.task_id): task for task in tasks}
        diagnostics = []
        for run in runs:
            task = task_by_id[run.task_id]
            attempts = submission[run.task_id]
            output_hits = [
                bool(
                    pair.output is not None
                    and (attempt.attempt_1 == pair.output or attempt.attempt_2 == pair.output)
                )
                for pair, attempt in zip(task.test, attempts, strict=True)
            ]
            diagnostics.append(
                {
                    "task_id": run.task_id,
                    "seconds": run.elapsed_seconds,
                    "output_hits": output_hits,
                    "verified_by_source": {
                        source: sum(
                            int(candidate.verified and candidate.source == source)
                            for candidate in run.candidates
                        )
                        for source in sorted({candidate.source for candidate in run.candidates})
                    },
                    "hybrid_programs": [
                        candidate.symbolic_program
                        for candidate in run.candidates
                        if candidate.source == "hybrid_symbolic" and candidate.verified
                    ],
                }
            )
        return {
            "available": True,
            "error": None,
            "seconds": time.monotonic() - started,
            "score": _score_to_dict(score),
            "tasks": len(tasks),
            "verified_candidates": sum(
                int(candidate.verified) for run in runs for candidate in run.candidates
            ),
            "hybrid_world_candidates": sum(
                int(candidate.source == "hybrid_world_model")
                for run in runs
                for candidate in run.candidates
            ),
            "task_diagnostics": diagnostics,
        }
    except Exception as exc:  # Keep benchmark output machine-readable on optional failures.
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.monotonic() - started,
            "score": {},
            "tasks": len(tasks),
        }


def _delta(baseline: dict[str, Any], hybrid: dict[str, Any], key: str) -> float | None:
    left = baseline.get("score", {}).get(key)
    right = hybrid.get("score", {}).get(key)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(right) - float(left)


def _public_manifest(path: str | Path | None, development_tasks: Sequence[Any]) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "label": "evaluation-only",
            "used_for_training": False,
            "used_for_scoring": False,
        }
    public_tasks = _load_arc_tasks(path)
    development_ids = {str(task.task_id) for task in development_tasks}
    public_ids = {str(task.task_id) for task in public_tasks}
    overlap = sorted(development_ids & public_ids)
    if overlap:
        raise ValueError(
            "development/public task ID overlap; refusing to risk leakage: "
            + ", ".join(overlap[:10])
        )
    return {
        "provided": True,
        "path": str(path),
        "label": "evaluation-only",
        "tasks": len(public_tasks),
        "task_ids_sha256": hashlib.sha256(
            "\n".join(sorted(public_ids)).encode("utf-8")
        ).hexdigest(),
        "used_for_training": False,
        "used_for_scoring": False,
    }


def run_experiment(
    *,
    data: str | Path,
    solver_config: str | Path,
    output: str | Path,
    public_eval_data: str | Path | None = None,
    seed: int = 0,
    holdout_percent: int = 20,
    max_learn_tasks: int | None = None,
    max_holdout_tasks: int | None = None,
    train_neural: bool = True,
    checkpoint: str | Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-2,
    augmentation_count: int = 1,
    max_examples: int | None = None,
    max_objects: int = 16,
    max_grid_size: int = 30,
    hidden_size: int = 48,
    num_slots: int = 8,
    recurrent_steps: int = 4,
    program_loss_weight: float = 0.5,
    backend: str = "numpy",
    run_solver: bool = True,
    solver_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the complete developmental hybrid benchmark and write two reports."""

    if backend not in {"numpy", "mlx"}:
        raise ValueError("backend must be 'numpy' or 'mlx'")
    if epochs < 1 or batch_size < 1 or augmentation_count < 1:
        raise ValueError("epochs, batch_size and augmentation_count must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= program_loss_weight <= 1:
        raise ValueError("program_loss_weight must be between 0 and 1")
    all_tasks = _load_arc_tasks(data)
    if not all_tasks:
        raise ValueError("the development dataset contains no tasks")
    public_manifest = _public_manifest(public_eval_data, all_tasks)
    split = deterministic_task_split(
        all_tasks, holdout_percent=holdout_percent, seed=seed
    )
    learn_tasks = _bounded_tasks(split.learn, max_learn_tasks)
    holdout_tasks = _bounded_tasks(split.holdout, max_holdout_tasks)
    if not learn_tasks:
        raise ValueError("task caps removed the entire learn split")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_checkpoint = "world_model.safetensors" if backend == "mlx" else "world_model.npz"
    checkpoint_path = (
        Path(checkpoint) if checkpoint is not None else output_dir / default_checkpoint
    )
    args = argparse.Namespace(
        backend=backend,
        seed=seed,
        train_neural=train_neural,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        augmentation_count=augmentation_count,
        max_examples=max_examples,
        max_objects=max_objects,
        max_grid_size=max_grid_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        recurrent_steps=recurrent_steps,
        program_loss_weight=program_loss_weight,
    )
    neural: dict[str, Any]
    checkpoint_result: Path | None = None
    if not train_neural and checkpoint is None:
        neural = {
            "available": False,
            "trained": False,
            "error": "neural training skipped and no checkpoint was supplied",
            "public_evaluation_used_for_training": False,
        }
    else:
        try:
            neural, checkpoint_result = _train_and_evaluate_world_model(
                learn_tasks=learn_tasks,
                holdout_tasks=holdout_tasks,
                args=args,
                checkpoint_path=checkpoint_path,
            )
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            # The runner remains useful for split/solver smoke tests if NumPy/MLX
            # experimental code is temporarily unavailable.
            neural = {
                "available": False,
                "trained": False,
                "error": f"{type(exc).__name__}: {exc}",
                "public_evaluation_used_for_training": False,
            }
            checkpoint_result = checkpoint_path if checkpoint_path.is_file() else None

    solver_results: dict[str, Any] = {
        "available": False,
        "skipped": not run_solver,
        "baseline": {},
        "hybrid": {},
        "delta": {},
    }
    if run_solver and holdout_tasks:
        baseline_config, hybrid_config = _solver_config_pair(
            solver_config,
            checkpoint_result,
            backend=backend,
        )
        baseline = _run_solver(holdout_tasks, baseline_config, solver_factory=solver_factory)
        hybrid = _run_solver(holdout_tasks, hybrid_config, solver_factory=solver_factory)
        solver_results = {
            "available": bool(baseline.get("available") and hybrid.get("available")),
            "skipped": False,
            "baseline": baseline,
            "hybrid": hybrid,
            "delta": {
                "pass_at_2": _delta(baseline, hybrid, "pass_at_2"),
                "strict_task_accuracy": _delta(
                    baseline, hybrid, "strict_task_accuracy"
                ),
            },
        }

    result: dict[str, Any] = {
        "experiment": "hybrid-object-centric-world-model",
        "format_version": 1,
        "backend": backend,
        "seed": seed,
        "dataset": {
            "path": str(data),
            "tasks": len(all_tasks),
            "task_ids_sha256": hashlib.sha256(
                "\n".join(sorted(str(task.task_id) for task in all_tasks)).encode("utf-8")
            ).hexdigest(),
        },
        "split": {
            "seed": split.seed,
            "holdout_percent": split.holdout_percent,
            "learn_task_ids": [str(task.task_id) for task in learn_tasks],
            "holdout_task_ids": [str(task.task_id) for task in holdout_tasks],
            "learn_task_count": len(learn_tasks),
            "holdout_task_count": len(holdout_tasks),
            "caps": {
                "max_learn_tasks": max_learn_tasks,
                "max_holdout_tasks": max_holdout_tasks,
                "max_examples": max_examples,
            },
        },
        "public_evaluation": public_manifest,
        "neural": neural,
        "solver": solver_results,
        "leakage_guard": {
            "training_task_ids": [str(task.task_id) for task in learn_tasks],
            "holdout_task_ids": [str(task.task_id) for task in holdout_tasks],
            "public_tasks_used_for_training": False,
            "public_tasks_used_for_scoring": False,
        },
    }
    result_path = output_dir / "hybrid_experiment.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(result)
    markdown_path = output_dir / "hybrid_experiment.md"
    markdown_path.write_text(markdown)
    result["artifacts"] = {"json": str(result_path), "markdown": str(markdown_path)}
    # Rewrite JSON after adding artifact paths so both files cross-reference.
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _percent(value: Any) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{float(value) * 100:.2f}%"


def render_markdown(result: dict[str, Any]) -> str:
    """Render the concise human-readable companion to the JSON manifest."""

    split = result["split"]
    solver = result["solver"]
    baseline = solver.get("baseline", {}).get("score", {})
    hybrid = solver.get("hybrid", {}).get("score", {})
    delta = solver.get("delta", {})
    neural = result["neural"]
    lines = [
        "# Hybrid object-centric world-model experiment",
        "",
        f"- Seed: `{result['seed']}`",
        f"- Development tasks: **{result['dataset']['tasks']}**",
        f"- Learn / holdout: **{split['learn_task_count']} / {split['holdout_task_count']}**",
        "- Public evaluation: **evaluation-only; never used for training or scoring**",
        "",
        "## Neural episodic metrics",
        "",
        f"- Available: `{neural.get('available', False)}`",
        "- Learn exact / cell: "
        f"**{_percent(neural.get('learn_metrics', {}).get('exact_accuracy'))}** / "
        f"**{_percent(neural.get('learn_metrics', {}).get('cell_accuracy'))}**",
        "- Holdout exact / cell: "
        f"**{_percent(neural.get('holdout_metrics', {}).get('exact_accuracy'))}** / "
        f"**{_percent(neural.get('holdout_metrics', {}).get('cell_accuracy'))}**",
        "",
        "## Solver comparison on holdout",
        "",
        "| Metric | Baseline | Hybrid | Delta |",
        "|---|---:|---:|---:|",
        "| pass@2 | "
        f"{_percent(baseline.get('pass_at_2'))} | "
        f"{_percent(hybrid.get('pass_at_2'))} | {_percent(delta.get('pass_at_2'))} |",
        "| strict task accuracy | "
        f"{_percent(baseline.get('strict_task_accuracy'))} | "
        f"{_percent(hybrid.get('strict_task_accuracy'))} | "
        f"{_percent(delta.get('strict_task_accuracy'))} |",
        "",
        "## Leakage audit",
        "",
        "- Public tasks used for training: "
        f"**{result['leakage_guard']['public_tasks_used_for_training']}**",
        "- Public tasks used for scoring: "
        f"**{result['leakage_guard']['public_tasks_used_for_scoring']}**",
        f"- Learn IDs: `{', '.join(split['learn_task_ids'])}`",
        f"- Holdout IDs: `{', '.join(split['holdout_task_ids'])}`",
        "",
    ]
    if neural.get("error"):
        lines.extend([f"Neural status: `{neural['error']}`", ""])
    if solver.get("skipped"):
        lines.extend(["Solver comparison was skipped.", ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--train-data", dest="data", required=True)
    parser.add_argument("--solver-config", default="configs/deterministic.yaml")
    parser.add_argument("--output", default="runs/hybrid-experiment")
    parser.add_argument("--public-eval-data", "--public-eval", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-percent", type=int, default=20)
    parser.add_argument("--max-learn-tasks", type=int, default=None)
    parser.add_argument("--max-holdout-tasks", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--augmentation-count", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--program-loss-weight", type=float, default=0.5)
    parser.add_argument("--backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--skip-neural", dest="train_neural", action="store_false", help="Skip neural training"
    )
    parser.add_argument(
        "--skip-solver", dest="run_solver", action="store_false", help="Skip ArcSolver comparison"
    )
    parser.set_defaults(train_neural=True, run_solver=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply small deterministic caps unless explicitly supplied",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.smoke:
        args.max_learn_tasks = args.max_learn_tasks or 4
        args.max_holdout_tasks = args.max_holdout_tasks or 2
        args.max_examples = args.max_examples or 32
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 4)
    try:
        result = run_experiment(
            data=args.data,
            solver_config=args.solver_config,
            output=args.output,
            public_eval_data=args.public_eval_data,
            seed=args.seed,
            holdout_percent=args.holdout_percent,
            max_learn_tasks=args.max_learn_tasks,
            max_holdout_tasks=args.max_holdout_tasks,
            train_neural=args.train_neural,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            augmentation_count=args.augmentation_count,
            max_examples=args.max_examples,
            max_objects=args.max_objects,
            max_grid_size=args.max_grid_size,
            hidden_size=args.hidden_size,
            num_slots=args.num_slots,
            recurrent_steps=args.recurrent_steps,
            program_loss_weight=args.program_loss_weight,
            backend=args.backend,
            checkpoint=args.checkpoint,
            run_solver=args.run_solver,
        )
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"json": result["artifacts"]["json"], "markdown": result["artifacts"]["markdown"]}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    sys.exit(main())
