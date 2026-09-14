# V6 — fixed-cohort Astra score experiment

Goal: reach at least 70% local games score without changing the original evaluation cohort.
The previous V5 turn made implementation/protocol progress, but its one training task did
not establish ARC accuracy. V6 supplies the missing scored experiment.

## Verified result — 2026-09-08

**The >=70% local-score goal is met on the original frozen pilot.** The process exited with
code 0 after all 20 tasks were dispatched and processed. Independent scoring and deterministic
replay of all 19 completed Astra responses passed; replay also reproduced the timeout fallbacks.

| Measurement | Final result |
| --- | --- |
| Output exact match, either of two predictions | 20/22 — **90.91%** |
| First-prediction output exact match | 20/22 — **90.91%** |
| Strict whole-task accuracy | 19/20 — **95%** |
| Mean per-task output exact match | **95%** |
| Dispatched / processed tasks | 20 / 20 |
| Valid structured responses / dispatched requests | 19/20 — **95%** |
| Demonstration-verified programs | 19 |
| Timeouts | 1 task, containing 2 test inputs |
| Recorded run wall time | 3,916.4 seconds — **65 minutes 16 seconds** |

`f18ec8cc` reached its 30-minute deadline without a final response. Both fallback predictions
were incorrect and remain in the denominator. The 19 successful tasks averaged 111.4 seconds
(median 101.5, maximum 202.1). There were no repair rounds in this run. The known usage totals
are 124,927 input tokens and 62,921 output tokens, including 41,747 reasoning tokens; the timed-out
call's usage is unavailable, so these are lower bounds. Opaque reasoning is not training data.

The source, runtime, frozen task content and cohort checks passed, and all saved predictions
were reproduced without new LLM requests. The regression suite passes **324 tests**. The original
report's attempted-task telemetry caveat is documented below; actual dispatch coverage is 20/20.
Artifacts are in `runs/v6/astra-local-eval-01/`, with regression evidence in
`runs/v6-final-regression.xml`.

This is **subscription-backed Astra through a local harness**, on 20 public development tasks
drawn from the official training corpus. It is not an offline Nanbeige score, a Kaggle submission,
or evidence of private-benchmark/generalisation performance. The lockbox was not evaluated, and
these development traces are excluded from student training. Training-only symbolic trajectory
collection and Nanbeige fine-tuning remain separate work.

## Measurement and isolation

The cohort is the first **20 development tasks / 22 test inputs**, in the original frozen
V4 hash order. Membership, task content, source files, runtime binary and configuration are
bound into `identity.json`. There is no easy-task filter or removal of timeouts/failures.
The lockbox stays sealed. `training_export_allowed: false` is attached to development records.

Report both output-level exact match (either of two ordered predictions matches) and the
mean per-task output score, plus strict whole-task accuracy. The runner's threshold flag
requires all 20 tasks to be processed, >=70% output exact match and >=70% strict task accuracy.
The complete saved submission must still be independently rescored before declaring the goal
achieved. A high score on an incomplete prefix is not completion. Pretraining contamination
cannot be ruled out for the public tasks; this is not a claim of private-competition performance.

Only the post-hoc reporting function receives test labels. Model prompts, program verification,
candidate ranking and the pass schedule operate on blinded tasks. Reports do not control which
tasks get a repair. Direct predictions and program-generated predictions are deduplicated;
matching samples are not treated as independent votes. Training export remains in the separate
V5 training-only pipeline, which rejects these development records.

## Scored solver

Exact model: **`gpt-6-astra` / `xhigh` / ChatGPT subscription**, using the pinned desktop
Codex executable and its effective permission-profile checks. No API-key backend or fallback.
The provider retains the same task conversation across repairs. Only demonstration-verifier
feedback is returned. Explicit symbolic models remain separate from opaque provider reasoning.

The output contract contains a concise symbolic working model, an optional general
`solve(train, grid)` Python witness, and direct/alternative predictions encoded as lossless
digit rows. This broader witness removes the fixed DSL's expressiveness bottleneck; it does
not change the symbolic-only student training objective. The witness is not executed by the
model. Existing deterministic AST/builtin restrictions and isolated subprocess execution are
used locally: no imports, IO, dunder access or arbitrary external calls, 2 CPU seconds and a
3-second wall timeout per program case. The existing 256 MiB address-space limit is applied
where supported; it is not claimed as a portable macOS memory guarantee.

Programs are checked on all demonstrations and leave-one-demonstration-out calls. A fitting
program is preferred to an unverified direct grid, which is preferred to a program with
demonstration failures. Test answers never rank candidates. Full demonstration agreement and
leave-one-out agreement are useful diagnostics, not proof of generalisation.

One teacher request at a time. Four coverage-first passes, up to 30 minutes cumulative wall
time per task, and at most 11 hours for the run. These are the active Astra experiment settings,
not the former five-minute Nanbeige baseline. Completed response IDs, exact requests, candidates,
usage and task cursors are durable. Transient failures reconcile the saved provider turn before
any new dispatch. Quota failures pause rather than changing models, purchasing usage or using
a reset. Keep at least 10 GiB disk space free.

The metric is explicitly **online Astra through a local harness**. It is not an offline
Nanbeige/Kaggle result. Fine-tuning the student and demonstrating its own improvement remain
separate work. No paid training, public release or competition submission is launched here.

## Run and recovery

```bash
.runtime/nanbeige/venv/bin/python -m arc_agent.v6_runner run
.runtime/nanbeige/venv/bin/python -m arc_agent.v6_runner status
.runtime/nanbeige/venv/bin/python -m arc_agent.v6_runner run --resume
.runtime/nanbeige/venv/bin/python scripts/audit_v6_score.py
.runtime/nanbeige/venv/bin/python scripts/replay_v6_run.py
```

Default run: `runs/v6/astra-local-eval-01/`. `source.tar.gz` preserves its exact source;
source/config changes require a new run directory. `submission.json` is complete from startup
and replaced atomically. `report.json` keeps the full denominator; `active.json` locates the
current task, but a PID or state file alone is not proof a process is alive. Verify the process
or its live execution session before treating it as running or restarting it.

The independent audit reads the submission directly, reconstructs scores without importing
the solver scorer, checks frozen task content and cohort membership, verifies the source archive
and model/auth lineage, and compares the saved report. During an active run, separately updated
files can briefly disagree; the final audit must be performed again after verified termination.

After termination, `replay_v6_run.py` complements that independent score calculation by
re-executing the saved witnesses in the existing sandbox, without LLM calls. It checks exact
label-blind prompts, demonstration-only repair feedback, task-conversation continuity, immutable
response artifacts, development/training isolation, and reproduction of the submitted predictions.
It refuses unfinished tasks and requires the recorded source and runtime. This replay intentionally
uses the solver's deterministic verifier and ranker; it is not a second independent accuracy metric.
Both commands still require separate confirmation that the original process has actually exited.

Telemetry caveat in the archived runner: `report.json`'s `attempted_tasks` counts tasks with a
completed model response, so a timeout before the first response is not included in that field.
The independent audit reports the actual `dispatched_tasks` from durable call records, terminal
status counts, and `unknown_usage_calls`. It requires dispatch to all 20 tasks before certification.
Timeouts remain in every score denominator. When usage is unavailable for a timed-out request,
the original token totals are a lower bound, not complete usage. The archived report is preserved
unchanged rather than retroactively rewriting this experiment's telemetry.

Prelaunch verification: **316 tests passed**, including 20 new checks covering the frozen
cohort, label isolation, two-attempt scoring, unsafe-program rejection, complete fallbacks,
model/auth restrictions and in-flight/completed response reuse. Scoped Ruff and diff checks pass.

OpenAI Docs informed the structured-output and continuity implementation:
[App Server](https://learn.chatgpt.com/docs/app-server) and
[Astra guidance](https://developers.openai.com/api/docs/guides/latest-model).
