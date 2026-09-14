# V5 — Astra symbolic teacher, Nanbeige student

> Active consolidated roadmap: [Astra–Nanbeige plan](ASTRA_NANBEIGE_PLAN.md).
> The implementation details and first pilot below are preserved; their phase schedule is
> superseded by that roadmap. The separate [V6 Astra baseline](V6_LOCAL_SCORE.md) is complete
> at 20/22 outputs correct. This does not change V5's zero accepted full-symbolic examples
> or establish Nanbeige training/generalisation.

Active decision: September 8, 2026. This supersedes the prior ban on frontier teachers,
not the requirement to preserve Nanbeige as the small-model student or retain old artifacts.
The research hypothesis is **distilling representations and revisions improves generalisation
more effectively than training only on final grid answers**. It is not yet an observed gain.

## Fixed boundaries

- Teacher and online research reasoning: exact `gpt-6-astra`, `xhigh`, ChatGPT subscription.
  No API-key backend, fallback model, quota bypass, credit redemption or automatic purchases.
- Student: `Nanbeige/Nanbeige4.2-3B`, revision
  `3384e426066d1a49c3aea90a7190b81260a6533f`. No other student models.
- Active local config pins the desktop app's Codex binary (verified version `0.153.4`).
  The PATH binary (`0.149.0`) did not expose Astra. Each run records the binary SHA-256,
  reported version, code hash, config, split hash and requested teacher alias. The subscription
  does not currently expose an immutable underlying Astra snapshot in these records.
- Existing isolated subscription login lives under `runs/v3-pilot/.codex-subscription`.
  Only authentication/transport is reused: no V3 teacher prompts, task conversations, scores,
  candidates, skill bank or cached predictions are imported. Each V5 task starts a new
  provider conversation. Only that conversation resumes during its repairs.
- No Nanbeige server is needed for teacher collection or tokenizer preparation. Preserve
  existing weights/runtimes and at least 10 GiB free disk. No system-wide installation changes.
- October 20 remains the submission-ready target. The intended Kaggle artifact uses only
  the Nanbeige student plus deterministic components; subscription teacher calls are online
  research/preparation, **not an offline submission dependency**. Recheck the actual event's
  current rules, permitted training data, licenses and output-use terms before scaling or
  submitting. This implementation is not a competition-eligibility or legal determination.
- No dollar ceilings. Separately billed GPU training needs approval before launch. No training,
  upload, public release, Kaggle execution or competition submission occurs in these commands.

## What the student learns

The first symbolic schema contains exact grounded entities, provisional object roles,
task-local concept definitions with acyclic dependencies, checked relationships, competing
hypotheses, ordered concept actions, counterevidence, uncertainty and a proposed transfer skill.
It is an authored working representation, **not hidden chain of thought** and not a complete
programming language. No encrypted provider reasoning is exported or claimed transferable.

An optional executable witness is separate: it names a hypothesis and supplies a bounded
existing DSL program. The interpreter never executes free-form concept meanings. Undefined
concepts, false grounded cells/relations and invalid programs fail validation. No generated
Python, shell, file inspection, browsing, MCP or external tools are allowed for the teacher.

Two different validation levels are retained:

| Training category | Target | Admission check |
| --- | --- | --- |
| Grounding | Exact cells, entity membership and checked relations; no proposed roles | Deterministic grid/relationship checks |
| Interpretation or repair | Full symbolic working model, with meanings/roles/skills explicitly provisional | Grounding plus an executable witness fitting visible and withheld demonstrations |

A passing witness does not establish that its free-form explanation is correct, nor that
the task generalises. Reports explicitly set `free_form_semantics_verified: false`.
Unverified interpretations remain research artifacts and are not full-symbolic SFT targets.
Even a failed/unsupported witness can yield a separately checked grounding-only example.
Entity membership is verified, not the claim that a selected group is the best segmentation.

## Phases and gates

### 0. Subscription route and protocol pilot — September 8–15

Implemented: fail-closed Astra config, live subscription availability check, exact output
schema, restricted teacher execution, durable requests/responses, source-bound resume, and
local Nanbeige-native tokenization. Collect one task first, not the entire dataset.
The default has four rounds, 30 minutes total wall time per task, one request at a time.
This is a data-quality pilot, not the old fixed-time ARC scoring pilot. App Server does not
enforce `output_token_hint`; record actual usage and enforce time instead.

Gate: a live schema-valid artifact, deterministic grounding checks, recoverable provider
conversation, complete provenance, correct student token boundaries and completion-only masks.
Model availability alone is not a successful generation or a training-quality gate.

### 1. Establish the symbolic teaching curriculum — September 16–25

Use the same frozen 792/103/105 task-level split. The current collector accepts training IDs
only, in frozen task-hash order. It withholds the last demonstration entirely, presents the
remaining demonstrations plus unlabelled original test inputs, and never sends test labels.
At the first visible-demo witness success it stops, tests the withheld demonstration locally,
and does not feed back its result or labels to optimise the witness against that holdout.

Implemented curriculum categories: grounding, interpretation, repair. Next: separately tested
checkpoint continuation, concept reuse and counterfactual state revision. Compare fixed scene
descriptions with revisable symbolic models; expand the verifier when the current 15-operation
DSL cannot express a correct rule. Do not confuse a DSL coverage failure with an Astra failure.

Provider conversation continuity is retained through response/thread IDs. Provider-managed
compaction may occur, but is not a controlled compaction experiment. Explicit symbolic
compaction, five-compaction integrity checks, and retrieval timing remain separate future work.

Gate: audit the first 20 training tasks; report schema validity, grounded coverage, full-symbolic
acceptance, withheld-demo success, failure categories, tokens and time. Grounding-only acceptance
must never inflate full-symbolic or ARC accuracy. Preserve rejected attempts as context only.

### 2. Dataset and Nanbeige SFT — September 26–October 7

Start with up to 3,000 verified episodes. Balance available categories at training time; keep
derived examples with the source task and reserve synthetic generator families for evaluation.
No legacy model traces, evaluation tasks or lockbox-derived examples enter training. Retain
source hashes, actual Astra response IDs, validation results and sample lineage.

The exporter produces conversational prompt/completion JSON. Targets exclude executable
witnesses and final output grids. Student inputs may include demonstrations and verifier
counterexamples. Local preparation uses pinned Nanbeige tokenizer JSON and its native chat
template, with an empty thinking span rather than fabricated Astra reasoning. Only the final
symbolic completion and terminator get labels; prompt tokens are `-100`. Overlength examples
are excluded with explicit diagnostics for restructuring, never silently truncated.

Training is **not implemented or launched yet**. Retain the prior first LoRA trial settings:
rank 16, learning rate `1e-4`, at most two epochs, 8K sequences, microbatch one, effective batch
16. Validate shared-layer adapter attachment, native tokenizer, trainer/collator loss masking,
save/reload, optimizer/scheduler/RNG/data-cursor resume and local regression fixtures first.
Use private durable checkpoints and Trackio when an approved HF job is ready. A tokenized
dataset passing its data check does not establish trainer compatibility or model improvement.

Gate: compare the student against untrained Nanbeige on development data at equal time.
Promote for +3 percentage points mean exact-match across three fixed seeds, or 20% less runtime
without mean accuracy loss, plus integrity checks. These are engineering thresholds, not
statistical proof. Keep the original Nanbeige checkpoint on regression.

### 3. Generalisation and strategy integration — October 8–14

Test new task families, color/object renamings and valid symmetries with correctly transformed
labels. Separate representation skill from solving accuracy and avoid treating several
correlated teacher samples as independent evidence. Compare trained vs untrained Nanbeige,
symbolic vs direct-grid baselines, and deterministic ranking at equal runtime. Any output-grid
supervision control would be a separately declared ablation, not mixed into this dataset.

Gate: outperform the strongest validated Nanbeige strategy without sacrificing coverage.

### 4. Freeze and offline rehearsal — October 15–20

Use the sealed lockbox only once for the frozen comparison, then fall back on regression
without tuning against it. Test final student artifacts locally before Kaggle deployment.
Rehearse approved Kaggle hardware offline: target 10 hours active work, complete within
11 hours, never beyond the stated 12-hour limit. Produce two ordered predictions per test
input, atomically checkpoint complete fallbacks, and audit student/teacher provenance.
Submission and public release still require separate confirmation.

## Commands

From the repository root, using the existing isolated Python runtime:

```bash
.runtime/nanbeige/venv/bin/python -m arc_agent.v5_cli preflight
.runtime/nanbeige/venv/bin/python -m arc_agent.v5_cli collect
.runtime/nanbeige/venv/bin/python -m arc_agent.v5_cli status
.runtime/nanbeige/venv/bin/python -m arc_agent.v5_cli collect --resume
.runtime/nanbeige/venv/bin/python -m arc_agent.v5_cli prepare-student
```

The completed pilot's exact source is in
`runs/v5/astra-symbolic-pilot-ready/source-pilot.tar.gz`. Live `--resume` was tested against
that source and reused the saved answer. Subsequent permission-guard hardening/lint cleanup
changed the source hash; use a **new** workspace for subsequent collection, for example
`collect --workspace runs/v5/astra-symbolic-next`. Do not mutate an old identity to force resume.

Do **not** run the teacher inside `scripts/v4-loopback.sb`; the subscription transport requires
network access. The teacher itself remains a no-tools structured-output model. Source/config
changes require a new collection workspace; completed calls are never repeated on matching
resume. If interrupted after starting a provider turn, resume reconciles its persisted turn ID.
The time cap includes downtime after interruption. Timed-out pending turns may need explicit
reconciliation before another experiment; do not start a replacement on uncertain dispatch.

Default outputs under `runs/v5/astra-symbolic-pilot-ready/`: `identity.json`, `preflight.json`,
SQLite checkpoints, immutable `artifacts/`, `status.json`, `dataset-report.json`,
`symbolic-sft.json`; then `student-data-report.json` and
`nanbeige-symbolic-tokenized.json` after local preparation. This collector does not calculate
competition accuracy or implement the final Kaggle solver. No trained student score exists yet.

The first protocol attempt under `runs/v5/astra-symbolic-pilot/` was rejected before inference:
the documented `readOnly.access` field is no longer accepted in Codex `0.153.4`. The corrected
runner uses a thread-local `arc-symbolic-teacher` permission profile: minimal platform reads,
read-only access to its empty model sandbox, no command networking, and no approvals. It checks
the server's returned model, provider, active profile, sandbox and instruction sources before
dispatch. No global configuration changed. Generated runtime protocol schemas are preserved
under `runs/v5/protocol-0.153.4*` for reproducibility.

## First live result — September 8

Task `7e0986d6`, the first frozen training ID: one Astra turn, **101.02 seconds**,
7,917 input tokens and 3,188 output tokens (1,552 reported reasoning tokens, not exported).
The output contained seven grounded entities and a provisional rectangle-denoising rule:
infer support/disturbance colors, recover support-component bounding boxes, restore disturbance
inside boxes to support color and remove disturbance outside them. It also described a local
adjacency alternative and counterexamples. Those textual claims are not mechanically proved.

- Schema and deterministic grounding checks passed.
- The witness was null: the existing DSL lacks the needed component-mask/replacement rule.
  No full-symbolic training example was accepted; no withheld-demo success was claimed.
- One grounding-only example was exported and prepared locally with the pinned tokenizer:
  **1,886 total tokens**, no truncation, correct completion-only masks.
- Completed-run resume reused the provider response without another generation.
- The pilot has finished. No Nanbeige training job or competition-scoring evaluation ran.

The next substantive experiment is to implement and test a **generic**, task-independent
component-box/mask interpreter primitive, then validate the proposed rule on unseen examples.
Do not add a hard-coded answer for this task or mark the provisional skill verified merely
because the explanation looks plausible.

## Source references

The OpenAI Docs skill guided exact-model routing and subscription/API separation:
[Codex authentication](https://learn.chatgpt.com/docs/auth),
[App Server protocol](https://learn.chatgpt.com/docs/app-server),
[GPT-6 Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra).
The Hugging Face LLM Trainer skill guided SFT dataset shape and validation-before-training:
[TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats).
