from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from arc_agent.models import ArcTask
from arc_agent.v2_config import RankerConfig
from arc_agent.v2_models import BankProgram
from arc_agent.v2_prompt import episodic_features

EPISODIC_FEATURE_KEYS = (
    "dimension_change_ratio",
    "horizontal_symmetry_ratio",
    "mean_area_ratio",
    "mean_components",
    "mean_components_8",
    "mean_input_cells",
    "mean_input_colors",
    "mean_output_cells",
    "mean_output_colors",
    "mean_separator_lines",
    "output_uses_new_color_ratio",
    "rotational_symmetry_ratio",
    "test_pairs",
    "train_pairs",
    "vertical_symmetry_ratio",
)


def compatibility_features(task: ArcTask, program: BankProgram) -> list[float]:
    target = episodic_features(task)
    source = program.features
    values: list[float] = []
    for key in EPISODIC_FEATURE_KEYS:
        values.append(float(target.get(key, 0.0)))
    for key in EPISODIC_FEATURE_KEYS:
        values.append(abs(float(target.get(key, 0.0)) - float(source.get(key, 0.0))))
    values.extend(
        [
            min(1.0, program.complexity / 1_000.0),
            min(1.0, len(program.strategy_tags) / 12.0),
            min(1.0, len(program.source_task_ids) / 20.0),
            min(1.0, len(program.python_source) / 20_000.0),
        ]
    )
    return values


def heuristic_rank(task: ArcTask, programs: list[BankProgram]) -> list[BankProgram]:
    target = episodic_features(task)

    def distance(program: BankProgram) -> tuple[float, int, str]:
        score = 0.0
        for key in EPISODIC_FEATURE_KEYS:
            left = float(target.get(key, 0.0))
            right = float(program.features.get(key, 0.0))
            score += abs(left - right) / max(1.0, abs(left), abs(right))
        return score, program.complexity, program.program_hash

    return sorted(programs, key=distance)


def _imports() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("V2 ranker dependencies are missing; run `uv sync --extra v2`") from exc
    return np, SGDClassifier, (make_pipeline, StandardScaler)


def _fit_pipeline(x: Any, y: Any, settings: RankerConfig) -> Any:
    _, classifier, helpers = _imports()
    make_pipeline, scaler = helpers
    return make_pipeline(
        scaler(),
        classifier(
            loss="log_loss",
            penalty="l2",
            alpha=settings.alpha,
            class_weight="balanced",
            max_iter=settings.epochs,
            tol=None,
            random_state=settings.seed,
        ),
    ).fit(x, y)


def train_ranker(
    rows: list[dict[str, Any]],
    *,
    settings: RankerConfig,
    output_path: str | Path,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("compatibility matrix is empty")
    np, _, _ = _imports()
    x = np.asarray([json.loads(row["features_json"]) for row in rows], dtype=np.float64)
    y = np.asarray([int(row["exact"]) for row in rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise ValueError("compatibility matrix needs both positive and negative examples")
    task_ids = [str(row["task_id"]) for row in rows]
    program_hashes = [str(row["program_hash"]) for row in rows]

    predictions = np.zeros(len(rows), dtype=np.float64)
    fold_by_task = {
        task_id: int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % settings.folds
        for task_id in set(task_ids)
    }
    for fold in range(settings.folds):
        train_indices = [
            index for index, task_id in enumerate(task_ids) if fold_by_task[task_id] != fold
        ]
        test_indices = [
            index for index, task_id in enumerate(task_ids) if fold_by_task[task_id] == fold
        ]
        if not train_indices or not test_indices or len(set(y[train_indices].tolist())) < 2:
            continue
        model = _fit_pipeline(x[train_indices], y[train_indices], settings)
        predictions[test_indices] = model.predict_proba(x[test_indices])[:, 1]

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        grouped[task_id].append(index)
    recall: dict[str, float] = {}
    for cutoff in (1, 8, 32):
        hits = 0
        for indices in grouped.values():
            ranked = sorted(indices, key=lambda index: (-predictions[index], program_hashes[index]))
            hits += int(any(y[index] for index in ranked[:cutoff]))
        recall[f"top_{cutoff}_recall"] = hits / len(grouped)

    fitted = _fit_pipeline(x, y, settings)
    artifact = {
        "schema_version": 1,
        "feature_keys": list(EPISODIC_FEATURE_KEYS),
        "feature_count": int(x.shape[1]),
        "settings": settings.model_dump(mode="json"),
        "model": fitted,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(temporary, target)
    return {
        "examples": len(rows),
        "tasks": len(grouped),
        "positive_examples": int(y.sum()),
        **recall,
    }


def load_ranker(path: str | Path) -> dict[str, Any]:
    artifact = pickle.loads(Path(path).read_bytes())
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported V2 ranker artifact")
    return artifact


def rank_programs(
    task: ArcTask,
    programs: list[BankProgram],
    *,
    ranker_path: str | Path | None,
) -> list[BankProgram]:
    if not programs:
        return []
    if ranker_path is None or not Path(ranker_path).is_file():
        return heuristic_rank(task, programs)
    np, _, _ = _imports()
    artifact = load_ranker(ranker_path)
    x = np.asarray(
        [compatibility_features(task, program) for program in programs], dtype=np.float64
    )
    probabilities = artifact["model"].predict_proba(x)[:, 1]
    return [
        program
        for _, program in sorted(
            zip(probabilities, programs, strict=True),
            key=lambda item: (-float(item[0]), item[1].program_hash),
        )
    ]
