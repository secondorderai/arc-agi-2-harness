# V7 — Astra symbolic-to-witness training-data bridge

Active roadmap: [Astra teacher / Nanbeige student](ASTRA_NANBEIGE_PLAN.md).

The V5 collector now accepts an injected witness contract; its default remains the original
DSL. V7 supplies a separate bounded Python witness while leaving Nanbeige's symbolic target
schema unchanged. V1–V6 artifacts and source archives are preserved. Source changes require
a new experiment workspace; do not resume a historical run against changed source hashes.

## Boundaries

- Exact `gpt-6-astra` / `xhigh` / ChatGPT subscription teacher. No alternative model or API fallback.
- First 20 training tasks in the existing frozen split; no development or lockbox SFT targets.
- Withhold the last demonstration before generation. Only visible pairs enter `solve(train, grid)`,
  including when evaluating the sealed input. Original test labels never enter the teacher prompt.
- Stop at the first visible-demo fit and check the sealed example. A sealed failure is rejection,
  never feedback for another teacher attempt. Deterministic export audits replay that same fixed
  candidate without sending any new teacher request or exposing the sealed outcome to repair.
- Each symbolic ordered action links to a statically reachable witness function. This is a
  checked reference, **not** proof of prose semantics, dynamic action execution or action order.
  Those limitations are explicit in every verification record.
- Witness code executes through the existing isolated, restricted Python runner with time limits.
  V7 additionally rejects module-scope statements, duplicate/nested helpers, decorators, defaults
  and annotations. No host tools, network, imports or model-side execution.
- Export replays prompts, checks raw teacher response checkpoints and target identity, re-runs
  grounding/witness checks, and rejects tampering or repair beyond the stopping boundary.
- SFT completions contain symbolic state only, never witness bodies, binding tables, final
  prediction grids or opaque reasoning. Failed attempts and visible feedback are masked context.

## Commands

```sh
.runtime/nanbeige/venv/bin/python -m arc_agent.v7_cli collect --stop-after 1
PYTHONPATH=runs/v7/astra-symbolic-bridge-01/frozen/src .runtime/nanbeige/venv/bin/python -m arc_agent.v7_cli collect --resume
.runtime/nanbeige/venv/bin/python -m arc_agent.v7_cli prepare-student
```

Configuration: [v7-astra-symbolic-bridge.yaml](../configs/v7-astra-symbolic-bridge.yaml).
Workspace: `runs/v7/astra-symbolic-bridge-01`.
`--stop-after` pauses between tasks, not within a request; resume skips completed tasks.
Task/round/request checkpoints and source archive are durable. No command launches GPU training.
The archived `frozen/src` tree was extracted and its hash verified against the run identity before
the subsequent local memory-policy changes. It keeps this teacher pilot exactly resumable without
reverting current student code or weakening source checks. Do not edit the frozen source tree.

Read `pilot-report.json` for dispatch-based structured validity (timeouts remain in denominator),
all task statuses, accepted categories, rejection reasons and full-symbolic source-task count.
Grounding-only exports remain separate. The data-quality gate requires the complete cohort,
at least 95% structured validity, multiple accepted source tasks and both interpretation and
verified repair examples. Passing this gate would still not establish student improvement.

## First live checkpoint — 8 September 2026

**Later completed result:** all 20 training tasks processed; 21/21 responses structurally valid;
19 accepted full-symbolic examples (18 interpretations and one repair), plus 20 grounding examples.
One sealed-example failure remains rejected and was not retried using its outcome. The full
data-quality gate and [post-completion replay audit](../runs/v7/astra-symbolic-bridge-01/final-curriculum-audit.json)
passed. All 39 losslessly encoded student examples fit native 8K formatting with no truncation;
actual TRL data-loader masks and the small architecture probe passed separately in
[V8](V8_STUDENT_BASELINE.md). Official-weight student training and promotion remain unproven.

The following checkpoints preserve the historical progression:

- Full regression: **352 tests passed**, zero failures/errors/skips; two existing SWIG warnings.
- First task `7e0986d6`: one valid structured response, grounded symbols and a visible-demo-fitting
  Python witness; **sealed demonstration failed**. One grounding-only export, zero full-symbolic.
- No heldout-driven repair was sent. The same frozen pilot continues with the remaining tasks.
- Fresh local student preflight: warning-level Mac memory pressure, so no Nanbeige process loaded.
  External networking was correctly denied by the student sandbox; free disk exceeded 10 GiB.
- No student training, paid GPU job, upload, Kaggle deployment or student promotion has occurred.

The user subsequently approved advisory warning-level memory pressure for local student tests.
New `v7-nanbeige-warning-*-local.yaml` configurations record warnings while preserving critical
pressure, telemetry-loss, swap-growth and disk stops. A fresh preflight after the user's cleanup
returned normal pressure. Original V4 configurations and the teacher pilot remain unchanged.

Subsequent local checks passed at both 4K and 8K, including the 6,122-token long-context fixture,
native tools, reasoning/final separation and restart/resume. Both observed normal pressure and
zero additional swap; the 8K peak server RSS was 3,673,882,624 bytes. This is compatibility
evidence, not a completed ARC baseline or trained student score.

At the four-task teacher checkpoint, three full-symbolic examples were accepted across three
sources (two interpretations and one verified repair). All five returned responses were
structured-valid; the first sealed failure remains rejected. Four grounding plus three symbolic
examples passed native tokenization and completion-only masking, with no truncation or exclusions;
the longest was 7,996 tokens. The remaining 16 tasks and the full curriculum gate were still pending.

Local check evidence: [4K](../runs/v7/nanbeige-warning-smoke-4k/smoke.json),
[8K](../runs/v7/nanbeige-warning-smoke-8k/smoke.json),
[native student data](../runs/v7/astra-symbolic-bridge-01/student-data-report.json).
The final warning-policy regression suite passed **369 tests**, zero failures/errors/skips:
[final regression XML](../runs/v7-warning-regression-final.xml).

Evidence: [regression XML](../runs/v7-regression.xml),
[pilot report](../runs/v7/astra-symbolic-bridge-01/pilot-report.json),
[student preflight](../runs/v7/nanbeige-preflight.json).

The OpenAI Docs skill informed the exact subscription and per-turn structured-output contract.
The Hugging Face LLM Trainer skill informed validation-before-training and completion-only labels.
Actual TRL collation and small-fixture adapter checks subsequently passed in V8; official-weight
GPU training compatibility remains a separate, unpassed gate.
