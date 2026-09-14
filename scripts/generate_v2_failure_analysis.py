#!/usr/bin/env python3
"""Generate a reproducible, post-freeze failure analysis for V2 public evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Grid = list[list[int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_accuracy(predicted: Grid, expected: Grid) -> float:
    if not predicted or not expected or len(predicted) != len(expected):
        return 0.0
    width = len(expected[0])
    if (
        not width
        or any(len(row) != width for row in expected)
        or any(len(row) != width for row in predicted)
    ):
        return 0.0
    correct = sum(
        predicted[row][column] == expected[row][column]
        for row in range(len(expected))
        for column in range(width)
    )
    return correct / (len(expected) * width)


def _load_tasks(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted(directory.glob("*.json"))
    }


def _json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _diagnosis(
    *,
    strict_correct: bool,
    selected_guarded: int,
    timed_out: int,
    quota_failed: int,
    completed: int,
    synthesis_candidates: int,
    synthesis_guarded: int,
) -> tuple[str, str]:
    if strict_correct and selected_guarded:
        return "success_guarded", "Correct: guard-passing program"
    if strict_correct:
        return "success_unguarded", "Correct: unguarded fallback/retrieval"
    if selected_guarded:
        return "guard_generalization_gap", "Guard-passing program failed hidden output"
    if timed_out:
        return "model_timeout_fallback", "Model timed out; unguarded fallback failed"
    if quota_failed:
        return "quota_failure_fallback", "Quota failure; unguarded fallback failed"
    if completed and synthesis_candidates and not synthesis_guarded:
        return "synthesis_rejected_fallback", "Synthesis returned but failed guards"
    return "retrieval_only_failure", "No guard-passing program; retrieval/fallback failed"


def analyze(data_dir: Path, workspace: Path) -> dict[str, Any]:
    report_path = workspace / "evaluation_report.json"
    submission_path = workspace / "submission.json"
    database_path = workspace / "state.sqlite3"
    frozen_report = json.loads(report_path.read_text())
    submission = json.loads(submission_path.read_text())
    tasks = _load_tasks(data_dir)

    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in connection.execute(
        "SELECT * FROM candidates WHERE phase='evaluation' ORDER BY task_id, score DESC"
    ):
        row = dict(raw)
        row["verification"] = _json(row.get("verification_json"), {})
        candidates_by_task[str(row["task_id"])].append(row)

    requests_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in connection.execute(
        "SELECT * FROM requests WHERE phase='evaluation' ORDER BY task_id, round_index"
    ):
        row = dict(raw)
        row["body"] = _json(row.get("body_json"), {})
        requests_by_task[str(row["task_id"])].append(row)

    provenance_by_task = {
        str(row["task_id"]): _json(row["provenance_json"], {})
        for row in connection.execute(
            "SELECT task_id, provenance_json FROM evaluation_outputs ORDER BY task_id"
        )
    }
    phase_by_task = {
        str(row["task_id"]): dict(row)
        for row in connection.execute(
            "SELECT * FROM phase_tasks WHERE phase='evaluation' ORDER BY position"
        )
    }
    connection.close()

    task_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    source_slot_counts: Counter[str] = Counter()
    request_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    selection_stage_counts: Counter[str] = Counter()
    direct_candidates_total = 0
    direct_guarded_total = 0
    synthesis_candidates_total = 0
    synthesis_guarded_total = 0
    oracle_exact_outputs = 0
    selected_exact_outputs = 0
    oracle_strict_tasks = 0
    selection_or_guard_miss_tasks = 0

    for task_id in sorted(tasks):
        payload = tasks[task_id]
        expected_outputs = [item["output"] for item in payload["test"]]
        attempts = submission[task_id]
        candidates = candidates_by_task.get(task_id, [])
        requests = requests_by_task.get(task_id, [])
        provenance = provenance_by_task.get(task_id, {})
        phase = phase_by_task.get(task_id, {})
        candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
        selected_ids = [
            provenance.get("attempt_1_candidate"),
            provenance.get("attempt_2_candidate"),
        ]
        selected_rows = [candidate_by_id.get(str(value)) if value else None for value in selected_ids]
        selected_sources = [row["source_kind"] if row else "fallback" for row in selected_rows]
        source_slot_counts.update(selected_sources)
        selection_stage = str(provenance.get("selection_stage", "unknown"))
        selection_stage_counts[selection_stage] += 1

        attempt_1_exact = 0
        attempt_2_exact = 0
        correct_outputs = 0
        same_attempt_outputs = 0
        shape_matched_outputs = 0
        selected_accuracy_sum = 0.0
        for expected, attempt in zip(expected_outputs, attempts, strict=True):
            first = attempt["attempt_1"]
            second = attempt["attempt_2"]
            first_exact = first == expected
            second_exact = second == expected
            attempt_1_exact += int(first_exact)
            attempt_2_exact += int(second_exact)
            correct_outputs += int(first_exact or second_exact)
            same_attempt_outputs += int(first == second)
            shape_match = any(
                len(predicted) == len(expected)
                and bool(predicted)
                and len(predicted[0]) == len(expected[0])
                for predicted in (first, second)
            )
            shape_matched_outputs += int(shape_match)
            selected_accuracy_sum += max(
                _cell_accuracy(first, expected), _cell_accuracy(second, expected)
            )

        strict_correct = correct_outputs == len(expected_outputs)
        selected_exact_outputs += correct_outputs
        direct = [row for row in candidates if row["source_kind"] == "direct"]
        synthesis = [row for row in candidates if row["source_kind"] != "direct"]
        direct_guarded = sum(bool(row["accepted"]) for row in direct)
        synthesis_guarded = sum(bool(row["accepted"]) for row in synthesis)
        selected_guarded = sum(bool(row and row["accepted"]) for row in selected_rows)
        direct_candidates_total += len(direct)
        direct_guarded_total += direct_guarded
        synthesis_candidates_total += len(synthesis)
        synthesis_guarded_total += synthesis_guarded

        status_counts = Counter(str(row["status"]) for row in requests)
        request_counts.update(status_counts)
        model_counts.update(str(row.get("model") or "unknown") for row in requests)
        quota_failed = sum(
            row["status"] == "failed"
            and (
                _json(row.get("body_json"), {}).get("error", {}).get("code")
                in {"usageLimitExceeded", "insufficient_quota"}
            )
            for row in requests
        )

        full_correct_candidates: list[str] = []
        guarded_full_correct_candidates: list[str] = []
        candidate_output_oracle = [False] * len(expected_outputs)
        for row in candidates:
            predictions = row["verification"].get("predictions") or []
            if len(predictions) != len(expected_outputs):
                continue
            exact_vector = [
                predicted == expected
                for predicted, expected in zip(predictions, expected_outputs, strict=True)
            ]
            for index, exact in enumerate(exact_vector):
                candidate_output_oracle[index] |= exact
            if all(exact_vector):
                full_correct_candidates.append(str(row["candidate_id"]))
                if bool(row["accepted"]):
                    guarded_full_correct_candidates.append(str(row["candidate_id"]))
        candidate_oracle_outputs = sum(candidate_output_oracle)
        oracle_exact_outputs += candidate_oracle_outputs
        candidate_oracle_strict = candidate_oracle_outputs == len(expected_outputs)
        oracle_strict_tasks += int(candidate_oracle_strict)
        selected_set = {str(value) for value in selected_ids if value}
        missed_full_correct = bool(set(full_correct_candidates) - selected_set)
        selection_or_guard_miss_tasks += int(not strict_correct and missed_full_correct)

        diagnosis_code, diagnosis_label = _diagnosis(
            strict_correct=strict_correct,
            selected_guarded=selected_guarded,
            timed_out=status_counts["timed_out"],
            quota_failed=quota_failed,
            completed=status_counts["completed"],
            synthesis_candidates=len(synthesis),
            synthesis_guarded=synthesis_guarded,
        )
        failure_counts[diagnosis_code] += 1
        task_rows.append(
            {
                "task_id": task_id,
                "result": "correct" if strict_correct else "failed",
                "diagnosis_code": diagnosis_code,
                "diagnosis": diagnosis_label,
                "test_outputs": len(expected_outputs),
                "correct_outputs": correct_outputs,
                "attempt_1_exact": attempt_1_exact,
                "attempt_2_exact": attempt_2_exact,
                "same_attempt_outputs": same_attempt_outputs,
                "shape_matched_outputs": shape_matched_outputs,
                "best_selected_cell_accuracy": round(
                    selected_accuracy_sum / len(expected_outputs), 6
                ),
                "direct_candidates": len(direct),
                "direct_guarded": direct_guarded,
                "synthesis_candidates": len(synthesis),
                "synthesis_guarded": synthesis_guarded,
                "selected_guarded": selected_guarded,
                "request_count": len(requests),
                "request_completed": status_counts["completed"],
                "request_timed_out": status_counts["timed_out"],
                "request_failed": status_counts["failed"],
                "quota_failed": quota_failed,
                "models": ", ".join(sorted({str(row.get("model") or "unknown") for row in requests}))
                or "none",
                "selection_stage": selection_stage,
                "attempt_1_source": selected_sources[0],
                "attempt_2_source": selected_sources[1],
                "attempt_1_candidate": selected_ids[0],
                "attempt_2_candidate": selected_ids[1],
                "candidate_oracle_outputs": candidate_oracle_outputs,
                "candidate_oracle_strict": candidate_oracle_strict,
                "full_correct_candidate_count": len(full_correct_candidates),
                "guarded_full_correct_candidate_count": len(guarded_full_correct_candidates),
                "missed_full_correct_candidate": missed_full_correct,
                "best_verifier_score": round(float(phase.get("best_score", 0.0)), 6),
                "refinement_rounds": int(phase.get("round_index", 0)),
            }
        )

    total_tasks = len(task_rows)
    total_outputs = sum(row["test_outputs"] for row in task_rows)
    correct_tasks = sum(row["result"] == "correct" for row in task_rows)
    request_total = sum(request_counts.values())
    tasks_with_guarded = sum(
        row["direct_guarded"] + row["synthesis_guarded"] > 0 for row in task_rows
    )
    tasks_with_fallback = sum(
        "fallback" in {row["attempt_1_source"], row["attempt_2_source"]}
        for row in task_rows
    )
    tasks_with_timeout = sum(row["request_timed_out"] > 0 for row in task_rows)
    tasks_with_completed_model = sum(row["request_completed"] > 0 for row in task_rows)
    guarded_correct_tasks = sum(
        row["result"] == "correct" and row["selected_guarded"] > 0 for row in task_rows
    )
    unguarded_correct_tasks = correct_tasks - guarded_correct_tasks

    category_labels = {
        "success_guarded": "Correct: guard-passing program",
        "success_unguarded": "Correct: unguarded fallback/retrieval",
        "guard_generalization_gap": "Guard-passing program failed hidden output",
        "model_timeout_fallback": "Model timed out; fallback failed",
        "quota_failure_fallback": "Quota failure; fallback failed",
        "synthesis_rejected_fallback": "Synthesis failed guards; fallback failed",
        "retrieval_only_failure": "Retrieval/fallback failed",
    }
    failure_categories = [
        {
            "code": code,
            "label": category_labels[code],
            "tasks": failure_counts.get(code, 0),
            "share": round(failure_counts.get(code, 0) / total_tasks, 6),
        }
        for code in category_labels
        if failure_counts.get(code, 0)
    ]

    summary = {
        "tasks": total_tasks,
        "test_outputs": total_outputs,
        "correct_tasks": correct_tasks,
        "correct_outputs": selected_exact_outputs,
        "strict_task_accuracy": correct_tasks / total_tasks,
        "pass_at_2": selected_exact_outputs / total_outputs,
        "direct_candidates_tested": direct_candidates_total,
        "direct_guarded_candidates": direct_guarded_total,
        "tasks_with_guarded_candidate": tasks_with_guarded,
        "guarded_task_coverage": tasks_with_guarded / total_tasks,
        "synthesis_candidates": synthesis_candidates_total,
        "synthesis_guarded_candidates": synthesis_guarded_total,
        "model_requests": request_total,
        "model_requests_completed": request_counts["completed"],
        "model_requests_timed_out": request_counts["timed_out"],
        "model_requests_failed": request_counts["failed"],
        "model_timeout_rate": request_counts["timed_out"] / request_total,
        "tasks_with_timeout": tasks_with_timeout,
        "tasks_with_completed_model": tasks_with_completed_model,
        "tasks_with_fallback_attempt": tasks_with_fallback,
        "guarded_correct_tasks": guarded_correct_tasks,
        "unguarded_correct_tasks": unguarded_correct_tasks,
        "candidate_oracle_correct_outputs": oracle_exact_outputs,
        "candidate_oracle_pass_at_2_upper_bound": oracle_exact_outputs / total_outputs,
        "candidate_oracle_strict_tasks": oracle_strict_tasks,
        "selection_or_guard_miss_tasks": selection_or_guard_miss_tasks,
        "active_seconds": frozen_report["active_seconds"],
        "paused_seconds": frozen_report["paused_seconds"],
        "quota_pause_count": frozen_report["quota_pause_count"],
    }

    return {
        "title": "V2 public evaluation was dominated by model timeouts and fallback selection",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": str(data_dir.resolve()),
            "workspace": str(workspace.resolve()),
            "evaluation_report": str(report_path.resolve()),
            "submission": str(submission_path.resolve()),
            "database": str(database_path.resolve()),
            "evaluation_report_sha256": _sha256(report_path),
            "submission_sha256": _sha256(submission_path),
            "database_sha256": _sha256(database_path),
            "frozen_submission_sha256": frozen_report["submission_sha256"],
        },
        "metric_definitions": {
            "strict_task_accuracy": "A task is correct only when at least one of the two submitted attempts exactly matches every labelled test output for that task.",
            "pass_at_2": "The share of labelled test outputs for which attempt 1 or attempt 2 exactly matches the expected grid.",
            "guarded_candidate": "A candidate that passed the configured static-safety, leave-one-demonstration-out, D4 transform, and colour-permutation checks.",
            "candidate_oracle": "A post-hoc diagnostic upper bound that uses labels to ask whether any stored candidate had the correct output. It is not a deployable selection policy.",
        },
        "summary": summary,
        "failure_categories": failure_categories,
        "request_statuses": [
            {"status": status, "requests": count, "share": count / request_total}
            for status, count in sorted(request_counts.items())
        ],
        "selected_source_slots": [
            {"source": source, "attempt_slots": count, "share": count / (2 * total_tasks)}
            for source, count in sorted(source_slot_counts.items())
        ],
        "selection_stages": [
            {"stage": stage, "tasks": count, "share": count / total_tasks}
            for stage, count in sorted(selection_stage_counts.items())
        ],
        "models": [
            {"model": model, "requests": count}
            for model, count in sorted(model_counts.items())
        ],
        "tasks": task_rows,
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    categories = analysis["failure_categories"]
    rows = analysis["tasks"]
    lines = [
        "# V2 public evaluation was dominated by model timeouts and fallback selection",
        "",
        f"Generated `{analysis['generated_at']}` from the frozen evaluation workspace. This is a post-hoc diagnostic: labels were used only after submission freeze.",
        "",
        "## Executive finding",
        "",
        f"V2 solved **{summary['correct_tasks']}/{summary['tasks']} tasks ({_percent(summary['strict_task_accuracy'])})** and **{summary['correct_outputs']}/{summary['test_outputs']} outputs ({_percent(summary['pass_at_2'])} Pass@2)**. The run did not meaningfully test the intended resynthesis strategy across the full set: **{summary['model_requests_timed_out']}/{summary['model_requests']} model requests ({_percent(summary['model_timeout_rate'])}) timed out**, while only **{summary['tasks_with_guarded_candidate']}/{summary['tasks']} tasks** obtained any guard-passing candidate.",
        "",
        f"The three solved tasks were {'all' if summary['guarded_correct_tasks'] == summary['correct_tasks'] else 'not all'} backed by selected guard-passing programs ({summary['guarded_correct_tasks']} guarded; {summary['unguarded_correct_tasks']} unguarded). The other {summary['tasks'] - summary['correct_tasks']} tasks failed at exact-match scoring.",
        "",
        "## Where the pipeline failed",
        "",
        "| Outcome / primary operational diagnosis | Tasks | Share |",
        "|---|---:|---:|",
    ]
    for item in categories:
        lines.append(f"| {item['label']} | {item['tasks']} | {_percent(item['share'])} |")
    lines.extend(
        [
            "",
            "The diagnosis is assigned from the observable pipeline state, not inferred ARC semantics. A timeout diagnosis means the task had no selected guard-passing program and its model turn exceeded the configured 360-second limit before a usable response was ingested.",
            "",
            "## Evidence by stage",
            "",
            f"- Retrieval evaluated **{summary['direct_candidates_tested']:,}** bank programs, but only **{summary['direct_guarded_candidates']}** passed every evaluation demonstration guard.",
            f"- Luna produced only **{summary['synthesis_candidates']} parsed candidates** across **{summary['tasks_with_completed_model']} tasks**; **{summary['synthesis_guarded_candidates']}** passed all guards.",
            f"- The scheduler recorded **{summary['model_requests_completed']} completed**, **{summary['model_requests_timed_out']} timed-out**, and **{summary['model_requests_failed']} failed** requests.",
            f"- At least one fallback attempt was used on **{summary['tasks_with_fallback_attempt']} tasks**.",
            f"- The post-hoc candidate oracle finds **{summary['candidate_oracle_correct_outputs']}/{summary['test_outputs']} outputs** and **{summary['candidate_oracle_strict_tasks']}/{summary['tasks']} tasks** represented somewhere in the stored candidate pool. This is an analysis-only upper bound because it uses labels.",
            f"- **{summary['selection_or_guard_miss_tasks']} failed tasks** had a fully correct stored candidate that was not selected. These are concrete selection-or-guard-ranking opportunities; all other failures require better candidate generation or completed model responses.",
            "",
            "## Interpretation",
            "",
            "The strongest supported conclusion is operational: V2's 2.50% score is primarily a **coverage failure**. The six-minute model-turn ceiling combined with two LLM slots and a six-hour first-model stage left nearly every task without a completed resynthesis. The direct ranker then supplied high-scoring partial programs, but demonstration guards admitted only one of them. Consequently, the submitted second attempt was often an input-copy or zero-grid fallback rather than an independently synthesized solution.",
            "",
            "This result does not establish that xhigh Luna resynthesis is intrinsically ineffective on ARC-AGI-2. It establishes that this harness configuration rarely captured a completed Luna result within its per-turn and stage budgets.",
            "",
            "## Recommended V3 gates",
            "",
            "1. Run a 10-task held-out smoke test and require at least 9 model responses to be ingested before the turn deadline. Shorten prompts/output limits or reduce reasoning effort first; increasing the timeout without increasing throughput would reduce 120-task coverage.",
            "2. Treat timed-out background work as resumable instead of terminal: preserve the response/thread identifier, poll it later, and ingest late completion exactly once when the global budget permits.",
            "3. Rework retrieval around task-family and behavior features. Training top-32 recall did not translate into public demonstration compatibility: only 1 of 120 tasks found a guard-passing direct program in the scanned pool.",
            "4. Use the post-hoc oracle only to test candidate selection on held-out training tasks. Never use public labels in the deployable selector.",
            "5. Repeat the frozen 12-hour protocol in a new workspace after the smoke gates pass; preserve this run as the V2 baseline.",
            "",
            "## Per-task audit (all 120 tasks)",
            "",
            "`Oracle` is the number of labelled outputs matched by any stored candidate, whether or not that candidate passed guards. `Cell` is the mean best cell accuracy across the two submitted attempts.",
            "",
            "| Task | Result | Primary diagnosis | Outputs | Correct | Direct guarded | Synth guarded | Requests C/T/F | Sources | Cell | Oracle |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        sources = f"{row['attempt_1_source']} / {row['attempt_2_source']}"
        requests = f"{row['request_completed']}/{row['request_timed_out']}/{row['request_failed']}"
        lines.append(
            f"| `{row['task_id']}` | {row['result']} | {row['diagnosis']} | "
            f"{row['test_outputs']} | {row['correct_outputs']} | {row['direct_guarded']} | "
            f"{row['synthesis_guarded']} | {requests} | {sources} | "
            f"{_percent(row['best_selected_cell_accuracy'])} | {row['candidate_oracle_outputs']} |"
        )
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "The report joins the frozen submission and labelled public-evaluation files to read-only rows from `state.sqlite3`. Candidate verifier predictions were compared with labels after freeze to compute the candidate-oracle diagnostic. No candidate was rerun and neither frozen workspace nor submission was modified.",
            "",
            "Primary diagnosis is intentionally operational. It cannot tell whether a timed-out model would eventually have returned a correct program, and exact public labels cannot be used to choose future competition outputs without creating evaluation leakage.",
            "",
            "## Source integrity",
            "",
            f"- Evaluation report SHA-256: `{analysis['scope']['evaluation_report_sha256']}`",
            f"- Frozen submission SHA-256: `{analysis['scope']['submission_sha256']}`",
            f"- State database SHA-256 at analysis time: `{analysis['scope']['database_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def app_snapshot(analysis: dict[str, Any]) -> dict[str, Any]:
    scope = analysis["scope"]
    definitions = analysis["metric_definitions"]
    common_source = {
        "label": "Frozen V2 public-evaluation artifacts",
        "sourceFiles": [scope["evaluation_report"], scope["submission"], scope["database"]],
        "filters": [
            "Phase: evaluation",
            "Dataset: 120 ARC-AGI-2 public evaluation tasks / 167 labelled test outputs",
            "Submission was frozen before labels were compared for this report",
        ],
        "assumptions": [
            "Candidate-oracle values are post-hoc diagnostics and must not be used by a deployable selector.",
            "Primary diagnoses describe observed pipeline state; they do not infer the latent ARC transformation family.",
        ],
    }
    return {
        "surface": "report",
        "title": analysis["title"],
        "generatedAt": analysis["generated_at"],
        "status": "reviewed",
        "queries": {
            "evaluation_summary": {
                "rows": [analysis["summary"]],
                "source": {
                    **common_source,
                    "metricDefinitions": [
                        {
                            "label": "Strict task accuracy",
                            "definition": definitions["strict_task_accuracy"],
                            "componentIds": ["report-summary", "metric-strict"],
                            "numerator": {"field": "correct_tasks", "label": "Correct tasks"},
                            "denominator": {"field": "tasks", "label": "Public evaluation tasks"},
                            "sourceLineage": [{"files": [scope["evaluation_report"], scope["submission"]]}],
                        },
                        {
                            "label": "Pass@2",
                            "definition": definitions["pass_at_2"],
                            "componentIds": ["report-summary", "metric-pass-at-2"],
                            "numerator": {"field": "correct_outputs", "label": "Correct test outputs"},
                            "denominator": {"field": "test_outputs", "label": "Labelled test outputs"},
                            "sourceLineage": [{"files": [scope["evaluation_report"], scope["submission"]]}],
                        },
                        {
                            "label": "Model timeout rate",
                            "definition": "Timed-out evaluation model requests divided by all evaluation model requests.",
                            "componentIds": ["report-summary", "metric-timeout", "pipeline-funnel", "report-methods"],
                            "numerator": {"field": "model_requests_timed_out", "label": "Timed-out requests"},
                            "denominator": {"field": "model_requests", "label": "Model requests"},
                            "sourceLineage": [{"files": [scope["database"]]}],
                        },
                        {
                            "label": "Guarded task coverage",
                            "definition": "Tasks with at least one direct or synthesized candidate passing all configured evaluation guards, divided by all tasks.",
                            "componentIds": ["report-summary", "metric-coverage", "pipeline-funnel", "report-methods"],
                            "numerator": {"field": "tasks_with_guarded_candidate", "label": "Tasks with guarded candidate"},
                            "denominator": {"field": "tasks", "label": "Public evaluation tasks"},
                            "sourceLineage": [{"files": [scope["database"]]}],
                        },
                    ],
                },
            },
            "failure_categories": {
                "rows": analysis["failure_categories"],
                "source": {
                    **common_source,
                    "metricDefinitions": [
                        {
                            "label": "Tasks by primary diagnosis",
                            "definition": "Each of the 120 tasks is assigned exactly one operational outcome/diagnosis from frozen request, candidate, guard, selection, and exact-score evidence.",
                            "componentIds": ["failure-breakdown", "failure-interpretation"],
                            "sourceLineage": [{"files": [scope["database"], scope["submission"]]}],
                        }
                    ],
                },
            },
            "request_statuses": {
                "rows": analysis["request_statuses"],
                "source": {
                    **common_source,
                    "metricDefinitions": [
                        {
                            "label": "Request status",
                            "definition": "Final durable status of each evaluation request row. Timed-out means the 360-second model-turn ceiling elapsed before a usable response was ingested.",
                            "componentIds": ["request-breakdown", "request-interpretation", "pipeline-funnel", "report-methods"],
                            "sourceLineage": [{"files": [scope["database"]]}],
                        }
                    ],
                },
            },
            "selected_source_slots": {
                "rows": analysis["selected_source_slots"],
                "source": {
                    **common_source,
                    "metricDefinitions": [
                        {
                            "label": "Selected attempt source",
                            "definition": "Candidate source selected for each of two task-level attempt slots. A source applies to every test output in its task slot.",
                            "componentIds": ["source-breakdown"],
                            "sourceLineage": [{"files": [scope["database"]]}],
                        }
                    ],
                },
            },
            "task_audit": {
                "rows": analysis["tasks"],
                "source": {
                    **common_source,
                    "metricDefinitions": [
                        {
                            "label": "Per-task exact result",
                            "definition": definitions["strict_task_accuracy"],
                            "componentIds": ["task-audit", "task-audit-intro", "report-methods"],
                            "sourceLineage": [{"files": [scope["submission"]]}],
                        },
                        {
                            "label": "Candidate oracle",
                            "definition": definitions["candidate_oracle"],
                            "componentIds": ["task-audit", "task-audit-intro", "report-methods"],
                            "sourceLineage": [{"files": [scope["database"]]}],
                        },
                    ],
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--app-snapshot-output", type=Path)
    args = parser.parse_args()
    analysis = analyze(args.data, args.workspace)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(analysis))
    if args.app_snapshot_output:
        args.app_snapshot_output.parent.mkdir(parents=True, exist_ok=True)
        args.app_snapshot_output.write_text(json.dumps(app_snapshot(analysis), indent=2) + "\n")


if __name__ == "__main__":
    main()
