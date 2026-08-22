from __future__ import annotations

import json
from pathlib import Path

from arc_agent.models import ArcPair, ArcTask
from arc_agent.v2_config import RankerConfig
from arc_agent.v2_models import BankProgram, InductionVerification
from arc_agent.v2_prompt import episodic_features
from arc_agent.v2_ranker import (
    compatibility_features,
    load_ranker,
    rank_programs,
    train_ranker,
)


def _task(task_id: str, value: int) -> ArcTask:
    return ArcTask(
        task_id=task_id,
        train=[ArcPair(input=[[value]], output=[[value]])],
        test=[ArcPair(input=[[value, value]], output=[[value, value]])],
    )


def _program(task: ArcTask, suffix: str) -> BankProgram:
    source = "def solve(train, grid):\n    return [row[:] for row in grid]"
    return BankProgram(
        program_hash=f"program-{suffix}",
        source_task_ids=[task.task_id],
        hypothesis="identity",
        strategy_tags=["identity"],
        invariants=["preserve cells"],
        python_source=source,
        canonical_ast=source,
        features=episodic_features(task),
        verification=InductionVerification(accepted=True),
        complexity=10,
    )


def test_ranker_trains_serializes_and_ranks(tmp_path: Path) -> None:
    tasks = [_task("a", 1), _task("b", 2)]
    programs = [_program(tasks[0], "a"), _program(tasks[1], "b")]
    rows = []
    for task in tasks:
        for program in programs:
            rows.append(
                {
                    "task_id": task.task_id,
                    "program_hash": program.program_hash,
                    "features_json": json.dumps(compatibility_features(task, program)),
                    "exact": int(task.task_id in program.source_task_ids),
                }
            )
    output = tmp_path / "ranker.pkl"
    metrics = train_ranker(
        rows,
        settings=RankerConfig(folds=2, epochs=20),
        output_path=output,
    )
    assert metrics["examples"] == 4
    assert metrics["positive_examples"] == 2
    assert load_ranker(output)["feature_count"] == len(
        compatibility_features(tasks[0], programs[0])
    )
    ranked = rank_programs(tasks[0], programs, ranker_path=output)
    assert {program.program_hash for program in ranked} == {"program-a", "program-b"}
