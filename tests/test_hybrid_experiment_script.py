from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_hybrid_experiment import deterministic_task_split, run_experiment


def _task_payload(task_id: str, *, value: int = 1) -> dict[str, object]:
    grid = [[0, value], [value, 0]]
    return {
        "id": task_id,
        "train": [{"input": grid, "output": grid}],
        "test": [{"input": grid, "output": grid}],
    }


def _write_tasks(directory: Path, task_ids: list[str]) -> Path:
    directory.mkdir()
    for index, task_id in enumerate(task_ids):
        payload = _task_payload(task_id, value=(index % 8) + 1)
        task_json = {key: value for key, value in payload.items() if key != "id"}
        (directory / f"{task_id}.json").write_text(json.dumps(task_json))
    return directory


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "hybrid-script-test",
                "model": {"enabled": False},
                "mlx_ttt": {"enabled": False},
                "hybrid": {"enabled": False},
                "budget": {"total_seconds": 30.0},
                "seed": 17,
            }
        )
    )
    return path


def test_split_is_deterministic_and_never_splits_a_task(tmp_path: Path) -> None:
    data = _write_tasks(tmp_path / "development", ["a", "b", "c", "d", "e"])
    from arc_agent.data import load_tasks

    tasks = load_tasks(data)
    first = deterministic_task_split(tasks, holdout_percent=40, seed=23)
    second = deterministic_task_split(tasks, holdout_percent=40, seed=23)
    assert first.learn_ids == second.learn_ids
    assert first.holdout_ids == second.holdout_ids
    assert set(first.learn_ids).isdisjoint(first.holdout_ids)
    assert set(first.learn_ids) | set(first.holdout_ids) == {"a", "b", "c", "d", "e"}


def test_smoke_runner_writes_auditable_reports_and_excludes_public_tasks(tmp_path: Path) -> None:
    data = _write_tasks(tmp_path / "development", ["dev-a", "dev-b", "dev-c", "dev-d"])
    public = _write_tasks(tmp_path / "public", ["eval-only"])
    config = _write_config(tmp_path / "solver.yaml")
    output = tmp_path / "result"

    result = run_experiment(
        data=data,
        solver_config=config,
        output=output,
        public_eval_data=public,
        seed=7,
        holdout_percent=25,
        max_learn_tasks=2,
        max_holdout_tasks=1,
        epochs=1,
        batch_size=1,
        hidden_size=8,
        num_slots=2,
        recurrent_steps=1,
        max_objects=4,
        max_grid_size=4,
    )

    assert (output / "hybrid_experiment.json").is_file()
    assert (output / "hybrid_experiment.md").is_file()
    assert result["public_evaluation"]["label"] == "evaluation-only"
    assert result["public_evaluation"]["used_for_training"] is False
    assert result["public_evaluation"]["used_for_scoring"] is False
    assert result["leakage_guard"]["public_tasks_used_for_training"] is False
    assert result["neural"]["available"] is True
    assert result["solver"]["baseline"]["available"] is True
    assert result["solver"]["hybrid"]["available"] is True
    assert set(result["split"]["learn_task_ids"]).isdisjoint(
        result["split"]["holdout_task_ids"]
    )
    persisted = json.loads((output / "hybrid_experiment.json").read_text())
    assert persisted["artifacts"]["markdown"].endswith("hybrid_experiment.md")


def test_public_task_overlap_is_rejected_before_training(tmp_path: Path) -> None:
    data = _write_tasks(tmp_path / "development", ["same-id", "other"])
    public = _write_tasks(tmp_path / "public", ["same-id"])
    config = _write_config(tmp_path / "solver.yaml")
    with pytest.raises(ValueError, match="overlap"):
        run_experiment(
            data=data,
            solver_config=config,
            output=tmp_path / "result",
            public_eval_data=public,
            train_neural=False,
            run_solver=False,
        )


def test_neural_and_solver_can_be_skipped_for_split_only_smoke(tmp_path: Path) -> None:
    data = _write_tasks(tmp_path / "development", ["a", "b"])
    config = _write_config(tmp_path / "solver.yaml")
    result = run_experiment(
        data=data,
        solver_config=config,
        output=tmp_path / "result",
        holdout_percent=50,
        train_neural=False,
        run_solver=False,
    )
    assert result["neural"]["available"] is False
    assert result["solver"]["skipped"] is True
