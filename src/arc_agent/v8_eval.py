"""Deterministic, label-blind candidate checks and separately called scoring."""

from arc_agent.models import validate_grid
from arc_agent.v2_config import GuardConfig
from arc_agent.v2_sandbox import canonicalize_source
from arc_agent.v2_verifier import verify_induction_program
from arc_agent.v4_config import content_hash


def check_witness(source, task):
    if any(p.output is not None for p in task.test):
        raise ValueError("witness verifier refuses test labels")
    result = verify_induction_program(
        source,
        task,
        include_test_labels=False,
        run_transformations=False,
        guards=GuardConfig(require_leave_one_out=True, d4_transforms=False, color_permutations=0),
    )
    feedback = {
        "verified": result.accepted,
        "exact_cases": result.exact_cases,
        "total_cases": result.total_cases,
        "failures": [f.model_dump(mode="json") for f in result.failures],
    }
    candidate = None
    if result.static_safe and len(result.predictions) == len(task.test):
        base_fit = not any(f.case.startswith("demo_full_") for f in result.failures)
        _, complexity = canonicalize_source(source)
        candidate = {
            "predictions": result.predictions,
            "rank": [int(base_fit), int(result.accepted), -complexity],
            "ancestry": content_hash(source),
        }
    return result.model_dump(mode="json"), feedback, candidate


def select_predictions(task, candidates):
    result = []
    for index, pair in enumerate(task.test):
        unique = []
        for candidate in sorted(candidates, key=lambda c: c["rank"], reverse=True):
            grid = validate_grid(candidate["predictions"][index])
            if grid not in unique:
                unique.append(grid)
        unique = unique[:2] or [pair.input]
        result.append({"attempt_1": unique[0], "attempt_2": unique[-1]})
    return result


def score_submission(tasks, submission):
    """Post-hoc only; never call this from a prompt, verifier, ranker or scheduler."""
    if set(submission) != {t.task_id for t in tasks}:
        raise ValueError("submission must cover the entire fixed cohort")
    correct, total, strict = 0, 0, 0
    for task in tasks:
        attempts = submission[task.task_id]
        if len(attempts) != len(task.test):
            raise ValueError("wrong number of ordered test predictions")
        hits = []
        for pair, guess in zip(task.test, attempts, strict=True):
            if pair.output is None or set(guess) != {"attempt_1", "attempt_2"}:
                raise ValueError("scoring requires labels and exactly two predictions")
            validate_grid(guess["attempt_1"])
            validate_grid(guess["attempt_2"])
            hits.append(pair.output in guess.values())
        correct += sum(hits)
        total += len(hits)
        strict += all(hits)
    return {
        "correct_outputs": correct,
        "test_outputs": total,
        "exact_match": correct / total,
        "strict_correct_tasks": strict,
        "tasks": len(tasks),
        "strict_accuracy": strict / len(tasks),
    }
