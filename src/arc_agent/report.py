from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from arc_agent.models import RunSummary, TaskRun, Usage
from arc_agent.scoring import Score


def make_summary(
    *,
    run_id: str,
    dataset_sha256: str,
    config_sha256: str,
    runs: Iterable[TaskRun],
    score: Score | None = None,
) -> RunSummary:
    run_list = list(runs)
    usage = Usage()
    for run in run_list:
        usage.add(run.usage)
    return RunSummary(
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        tasks=len(run_list),
        test_outputs=sum(len(run.attempts) for run in run_list),
        pass_at_2=score.pass_at_2 if score else None,
        strict_task_accuracy=score.strict_task_accuracy if score else None,
        elapsed_seconds=sum(run.elapsed_seconds for run in run_list),
        usage=usage,
        level_counts=dict(Counter(run.final_level for run in run_list)),
    )


def write_run_artifacts(
    output_dir: str | Path,
    summary: RunSummary,
    runs: Iterable[TaskRun],
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_list = list(runs)
    (destination / "summary.json").write_text(summary.model_dump_json(indent=2))
    with (destination / "tasks.jsonl").open("w") as stream:
        for run in run_list:
            stream.write(run.model_dump_json() + "\n")
    score_text = "not scored"
    if summary.pass_at_2 is not None:
        score_text = f"{summary.pass_at_2 * 100:.2f}%"
    strict_text = "not scored"
    if summary.strict_task_accuracy is not None:
        strict_text = f"{summary.strict_task_accuracy * 100:.2f}%"
    markdown = f"""# ARC-AGI-2 Run {summary.run_id}

- Official pass@2 output accuracy: **{score_text}**
- Strict whole-task accuracy: **{strict_text}**
- Tasks: {summary.tasks}
- Test outputs: {summary.test_outputs}
- Elapsed solver time: {summary.elapsed_seconds:.1f}s
- Model calls: {summary.usage.calls}
- Prompt tokens: {summary.usage.prompt_tokens}
- Completion tokens: {summary.usage.completion_tokens}
- Final level counts: {json.dumps(summary.level_counts, sort_keys=True)}
- Dataset SHA-256: `{summary.dataset_sha256}`
- Config SHA-256: `{summary.config_sha256}`

## Task results

| Task | Initial | Final | Verified candidates | Seconds | Calls | Timed out |
|---|---:|---:|---:|---:|---:|---|
"""
    for run in run_list:
        verified = sum(candidate.verified for candidate in run.candidates)
        markdown += (
            f"| `{run.task_id}` | {run.initial_level} | {run.final_level} | {verified} | "
            f"{run.elapsed_seconds:.2f} | {run.usage.calls} | {run.timed_out} |\n"
        )
    report_path = destination / "report.md"
    report_path.write_text(markdown)
    (destination / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>ARC-AGI-2 report</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font:15px system-ui;"
        "white-space:pre-wrap}</style>"
        f"<body>{html.escape(markdown)}</body>"
    )
    return report_path
