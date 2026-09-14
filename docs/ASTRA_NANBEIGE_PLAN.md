# ARC-AGI-2 — Astra teacher, Nanbeige symbolic student

Updated: **9 September 2026**. Submission-ready target: **20 October 2026**.
This is the active, consolidated roadmap. It supersedes the Nanbeige-only teacher restriction
and the earlier phase schedules, while preserving their artifacts, local safety requirements,
evaluation isolation and offline submission target. Dates are targets; failed gates are not waived.

## Decisions and model roles

The objective is to transfer **symbolic knowledge, abstraction, revision and reusable skills**
from Astra into Nanbeige, rather than train the student to reproduce puzzle answers. Improved
generalisation is the hypothesis to test, not an established outcome of the teacher's score.

| Role | Active choice | Boundary |
| --- | --- | --- |
| Online research solver and reference baseline | `gpt-6-astra`, `xhigh`, ChatGPT subscription | Local harness, networked provider; not an offline solver |
| Symbolic teacher, curriculum author and repair demonstrator | The same Astra route | Explicit symbolic artifacts from training tasks only; deterministic admission checks |
| Fine-tuned student and eventual competition LLM | `Nanbeige/Nanbeige4.2-3B` | No alternative student or teacher-at-solve-time dependency |
| Execution, verification and candidate ranking | Deterministic tools and non-LLM ranker | No external model judge or test-answer oracle |

- Preserve exact Astra routing. No Qwen, Bonsai, other frontier teachers or alternative-model
  fallbacks; no API-key billing fallback, quota bypass, automatic purchases or usage resets.
  On subscription unavailability, save state and pause. Codex supports subscription sign-in
  separately from API-key access; this project deliberately uses the former.
  [OpenAI authentication](https://learn.chatgpt.com/docs/auth)
- Keep the validated `xhigh` setting and record the effective model, requested alias, Codex
  version/binary hash, prompts, source/config hashes and response IDs. The existing records do
  not expose an immutable underlying Astra weight snapshot; do not claim one is pinned.
  [Exact Astra model and supported effort levels](https://developers.openai.com/api/docs/models/gpt-6-astra)
- Nanbeige stays pinned to official revision `3384e426066d1a49c3aea90a7190b81260a6533f`;
  retain the authors' `nanbeige42` llama.cpp revision
  `c6640a1c0cf7b38df342b67021a3900b04d092e7`, native tokenizer and local Q4_K_M artifacts.
- Preserve V1–V6 runs, old models and source archives. Reuse infrastructure/authentication,
  not old model predictions, conversations, teacher judgments or skill-bank contents.
  V4's original Nanbeige-only checks stay intact for baseline reproduction.
- **No dollar ceilings, phase spending allocations, reserves or cost-based stopping rules.**
  Technical deadlines, quality gates, memory safeguards and provenance remain. Separately
  billed GPU jobs still need approval before launch; subscription access is not HF GPU credit.
  [HF Jobs billing](https://huggingface.co/docs/hub/en/jobs-overview)
- Keep local-first student validation on the 16 GB M1 Pro. One owned Nanbeige process and one
  request at a time; preserve at least 10 GiB free disk, stop on critical pressure/swap growth,
  and do not close user apps, delete models or replace system-wide runtimes.
- **September 8 user-approved local exception:** warning-level pressure may be advisory under
  an explicit `warning_pressure_policy: record` configuration. Record every warning; retain
  immediate critical/unrecognized-pressure stops, telemetry-loss/OOM stops and the 1 GiB
  additional-swap limit. Historical configurations retain the original warning-stop default.
  Baseline/candidate comparisons must use the same declared policy.
- Teacher collection does not require a local Nanbeige inference process and may proceed
  while student runtime work is pending. It does **not** waive the student's pre-training or
  pre-Kaggle local compatibility gates.
- The submission path is offline Nanbeige plus deterministic components. Keep the supplied
  12-hour ceiling, targeting 10 hours active solving and completion within 11 hours. Verify
  actual competition rules, GPU allocation, training-data permissions, licenses and output-use
  terms before scaling/deployment. L4/L4×4 are planning targets, not verified allocations.
  Submission and public release require separate confirmation.
- **Output-use review is still open.** The September 9 review fetched the governing
  individual, business and service-specific terms, including the competing-model provision
  and business exceptions. Coverage of this subscription-based generative student is not
  established. Preserve existing data; do not scale collection, upload it or train the real
  student until applicability/permission is documented. The local untrained baseline can
  continue. [Findings and unsent clarification request](ASTRA_OUTPUT_USE_REVIEW.md).

## Current evidence and what remains unproven

**Latest student status — September 9, 09:31 Brisbane:** the approved 4K-output
baseline is now stopped incomplete after its recovered run exceeded the 1 GiB
additional-swap safeguard (observed 1.099 GiB). The original pre-model recovery
exception was preserved and does not cover this actual runtime failure. The
stopped score/evidence audit passed; the safety audit rejected continuation and
automatic follow-up is paused. All 68 completed responses, the one interrupted
unknown-usage request and complete fallback submissions are preserved. A new
local attempt or GPU-baseline plan change needs user direction; no student
training or promotion has occurred. [Safety-stop record](../runs/v8/untrained-output4k-baseline-01/swap-stop-01.md).

| Workstream | Verified status | What it does not establish |
| --- | --- | --- |
| Astra solver pilot, V6 | **20/22 outputs correct (90.91%); 19/20 tasks fully correct (95%)**. All 20 dispatched and processed; 65m16s wall time; one 30-minute timeout counted as two wrong outputs. Independent scoring and response replay passed. | Private-competition performance, student accuracy, controlled compaction gains or generalisation beyond this public pilot |
| Astra symbolic-data pilot, V5 | One training task; one grounding-only example prepared in Nanbeige's native format; **zero full-symbolic examples** accepted | A training-ready curriculum or verified symbolic explanation |
| Nanbeige local baseline, V4 | Pinned model/Metal setup and 4K/8K smoke fixtures passed. Larger-context timing pilot stopped on memory pressure; original runs preserved. | A completed, useful ARC baseline or L4 throughput |
| Symbolic witness bridge, V7 | **20-task pilot and post-completion replay audit passed**: 21/21 valid responses, 19 accepted full-symbolic examples (18 interpretations, one repair), 20 grounding examples. One sealed failure rejected. All 39 compact native-format examples fit 8K without truncation. | A large/diverse curriculum, validated compaction/transfer skills or a trained student |
| Current local student check, V7 | 4K and 8K smoke/tool/resume checks passed after cleanup; normal pressure and zero swap growth. Warning-tolerant policy explicitly recorded. | A completed ARC baseline or trained student score |
| Student symbolic baseline harness, V8 | Cache-disabled **4K/8K fixtures passed**. The development pilot was then stopped after 45m31s because symbolic seed 42 could no longer reach the configured 95% validity gate. Partial direct snapshot: **1/22 outputs correct**, not a completed baseline. Memory remained safe with zero swap growth; all artifacts preserved. | Completed Nanbeige ARC accuracy or a passed full student gate |
| Local trainer compatibility probe | Actual TRL data-loader masks checked for all 39 examples; 154 unique LoRA modules across 22 shared physical layers verified on meta device. Small random Nanbeige architecture fixture passed numerical updates, adapter save/reload and exact resumed-adapter equality. | Training the official checkpoint, full-size/GPU memory, GPU optimizer-resume compatibility or student promotion |
| Harness regression | **749 tests passed**, including the prior 718 plus 31 expanded output-budget checks covering native requests, smoke policy isolation and context reservation; zero failures/skips. V6's original 324-test record stays frozen | Full-size training compatibility, GPU portability or five actual model compactions |
| Independent student score audit | Live and stopped V8 audits matched saved scores and submissions; frozen source, cohort, task hashes, response artifacts, native parsing and usage ledgers checked. Interrupted dispatches remain counted. Live unresolved replies may still succeed; stopped lost replies remain failures. The direct-mode bound additionally caps valid replies at one per task, detecting impossible gates without assuming unused retries can all succeed. | A completed baseline, full inference replay, trained-model provenance or promotion |
| Replacement-format 4K fixture | Direct reply and four symbolic replies were **100% structurally valid**, but the symbolic path exhausted its repairs with zero grounded states. Memory stayed normal with zero additional swap; the owned server exited. Overall fixture **failed**; no new 8K/baseline run started. | A solved symbolic workflow, useful ARC accuracy or readiness for real student training |
| Revised reference-patch 4K fixture | **Passed**: a 62-token Nanbeige-authored patch resolved the initial undefined references, then the state/witness passed verification and produced the correct fixture output. 100% structured validity, normal pressure, zero additional swap and replay verified. | Development ARC accuracy, a complete baseline or a trained-student gain |
| Revised 8K fixture and replacement baseline | **8K fixture passed**, but `runs/v8/untrained-reference-baseline-01` was stopped incomplete after **1h57m41s**: two confirmed invalid direct seed-43 replies capped possible validity at **90.91%**, below 95%. Stopped evidence audit passed; normal pressure, zero additional swap, all fallbacks/checkpoints preserved. | A completed three-seed development baseline, trained model or promotion |
| Bounded-whitespace correction | New opt-in `reference_repair_bounded_ws` protocol bounds structural JSON whitespace without changing string contents, reasoning budgets or symbolic targets. All 627 tests passed. Fresh **4K fixture failed correctness** despite 7/7 structurally valid replies: direct was correct, but both grounded-state witnesses used forbidden `Raise` syntax and the final symbolic revision asserted a false relation. Normal pressure, zero swap growth and replay passed; server exited. | A passed model fixture, solved development tasks or a trained-student gain |
| Stage-specific witness repair | New opt-in `witness_repair_policy: repair_static_errors` retains a hash-checked, revalidated symbolic state on compiler rejection and asks Nanbeige to replace only the witness code. **4K and 8K fixtures passed**: code-only repair succeeded without regenerating the grounded state; both modes correct, all replies valid, normal pressure, zero swap growth and replay passed. 8K also passed the 5,571-token prompt fixture. | Resolved symbolic-output truncation, a completed development baseline or a trained-student gain |
| Lossless compact-cell trial | Prototype round-tripped all 26 complete saved states and reduced their aggregate payload token count by **16.7%**. Re-encoding 11 incomplete prefixes saved **36.1%**; those replies remain incomplete. All 668 tests passed. **Fresh 4K and 8K fixtures passed**, including long-context/recovery checks, correct outputs, all replies valid, normal pressure and zero swap growth. | Reduced development truncation, general end-to-end speedup, changed teacher targets or a trained-student gain |
| Compact-cell development baseline | **Stopped after 40 stages and 57m08s**. Symbolic seed 42 has eight invalid replies, capping possible final validity at **94.29%**, below 95%. All nine malformed direct/symbolic replies hit the 2K output cap, not context truncation. Normal pressure, zero swap growth; stopped audit and 40-stage replay passed, owned server exited, artifacts preserved. Do not resume. | A completed baseline, passed training gate, trained student or promotion |
| Approved 4K-output protocol | New opt-in configs preserve old 2K runs. All 749 tests and fresh offline **8K/12K fixtures passed**, including correct outputs, 100% validity, verified recovery, the 5,571-token long prompt, normal pressure and zero swap growth. New `untrained-output4k-baseline-01` is running with a first 40-stage checkpoint and gate-audited task follow-up. Both untrained and eventual trained comparisons must use the same allowance. | A completed ARC baseline, paid-launch approval, output-use clearance or trained-student gain |
| Observation-catalog feasibility probe | Standalone deterministic probe retained exact raw grids/cell masks for 28 segmentation views of one development task. All objects were retrievable through hash-bound pages; 28 focused tests passed. Whole catalogs can increase prompt size (one policy totaled 9,927 input tokens); pages were at most 805 tokens. Active solver source is unchanged. | A model-validated representation, solved truncation, correct semantic segmentation, training examples or a student gain |
| Saved student state-transition replay | Read-only replay reproduced 31 completed stages from a live snapshot, including malformed replies and grounding failures, without new model calls. Exact later prompts, artifacts, symbolic state and predictions are checked; 20 new tests and the 698-test full suite passed. Active solver/source identity is unchanged. | A complete 120-state replay, timing/unknown-response recovery, inclusive runtime savings, trained checkpoint or promotion |
| Full-size training runner | Approval-gated single-CUDA compatibility and SFT paths implemented, with native labels, shared-layer checks, private synchronous checkpoints, pinned adapter lineage and exact-resume comparison. Completed-checkpoint recovery now restores the adapter without extra updates; pinned PEFT/TrainerState API fixture passed. | A launched job, actual full-size GPU compatibility, a trained student or a promotion |
| Local training-data draft | **Draft 03 staged and natively revalidated** with all 39 unchanged symbolic targets, frozen split/provenance, masks and copied tool hashes. It is bound to the now-failed compact-cell baseline and cannot be sealed or rebound. Drafts 01/02 also remain preserved. No output-use/compute approval or new draft has been created. | A passed baseline gate, sealed training bundle, upload, paid-compute approval or trained student |
| Trained-checkpoint provenance audit | Read-only local cross-check of the approved bundle, pinned tooling, compatibility/SFT reports, complete schedule, checkpoint files and immutable durable reference implemented; 34 synthetic-fixture tests passed. No optimizer/pickle deserialization or model load. | Real trained checkpoint validation, tensor contents, remote availability, conversion, candidate inference or promotion |
| Adapter payload preflight | Added provenance-first, CPU-sliced tensor validation. Official expected inventory matches pinned Nanbeige/PEFT meta-device model: 308 tensors, 22 physical layers, two loops, 23,969,792 parameters. Twenty new tests and all 718 regressions pass. No base weights loaded. | A real trained adapter, nonzero effective matrix update, converted/loaded GGUF, student accuracy or promotion |
| Student training / Kaggle / lockbox | **Not completed**; no training job or competition submission launched | No student promotion or competition-readiness claim is allowed |

Evidence: [V6 result](V6_LOCAL_SCORE.md),
[final audit](../runs/v6/astra-local-eval-01/final-audit.json),
[V5 dataset report](../runs/v5/astra-symbolic-pilot-ready/dataset-report.json),
[student tokenization report](../runs/v5/astra-symbolic-pilot-ready/student-data-report.json),
[preserved V4 status](V4_IMPLEMENTATION_STATUS.md).

Current student work: [V8 protocol and compatibility status](V8_STUDENT_BASELINE.md).

The teacher's 70% local-score goal is complete. The next bottleneck is **verified symbolic
training data and student transfer**, not another run to re-establish that teacher score.
The 20-task development set remains evaluation-only. A clean harness split cannot exclude
pretraining contamination of public ARC tasks.

## The symbolic training contract

Keep a compact, versioned working model between free prose and a complete program:
grounded entities and cells; roles; task-local definitions; relations and constraints;
competing hypotheses; evidence and counterexamples; ordered actions; unresolved choices;
and explicitly scoped reusable concepts. Segmentation stays revisable and raw grids remain
authoritative, available through immutable artifact references.

Three things must remain distinct:

1. **Symbolic target:** the explicit representation, useful revision or checkpoint continuation
   Nanbeige is trained to produce. Hypotheses and uncertain roles retain their status.
2. **Executable witness:** a separate bounded DSL or sandboxed general program used to check
   consequences. Undefined symbolic names and free-form prose never execute.
3. **Puzzle result:** output grids and exact-match scores used for validation/evaluation,
   not assistant SFT targets. Demonstration grids can still be input evidence.

Do not export private/opaque Astra reasoning, encrypted provider state, final-answer grids or
executable witness bodies into this symbolic SFT target. Direct-grid and program-generation
strategies may still be evaluated at inference time; that is not answer-grid supervision.
An answer-supervised control would require a separately agreed experiment, not a silent mix-in.

Admission levels remain separate:

- **Grounding-only:** deterministic cell/membership/relationship checks; exclude unsupported
  interpretations and proposed roles from the target.
- **Full symbolic interpretation/repair:** grounding plus a witness fitting the visible and
  withheld demonstrations. Passing code supports its tested consequences, not every sentence
  of the explanation; uncheckable meanings stay provisional and are not labelled proven.
- **Transfer/continuation:** additional held-out-family or state-integrity checks appropriate
  to the claim. A renamed concept or a fluent summary is not evidence of a reusable skill.

Every model-authored supervision target comes from Astra. Nanbeige failures may become
training-task context for an Astra revision, but are not self-certified teacher labels.
No LLM confidence score or external LLM judge admits examples.

## Phased roadmap

### Phase 0 — Astra route and reference baseline

**September 8–15 window; solver baseline completed September 8.**

Keep the audited V6 run frozen: exact cohort, original deadlines, complete fallbacks, raw
responses, source archive and independent audit. The V5 subscription/protocol and local
tokenization pilot is also complete, but its full-symbolic data gate is not passed.
Do not describe these two different successes as student fine-tuning or offline validation.

**Exit evidence:** the V6 final audit, completed process, preserved two-prediction submission,
19 valid returned responses out of 20 dispatches, and the V5 grounding/tokenization reports.

### Phase 1 — Symbolic representation, witness bridge and local student baseline

**September 9–15; next implementation work.**

- Bridge the V5 structured symbolic schema to a sufficiently expressive deterministic verifier.
  Reuse V6's tested sandbox/verifier infrastructure where useful, but never its development
  answers or traces. Prefer generic component/mask/geometry operations; permit a separate
  bounded Python witness when the fixed DSL lacks coverage. No task-ID lookup or hard-coded grid.
- **Implemented as V7:** the collector accepts a separate bounded Python witness with static
  symbolic-action/function links and prompt/response replay auditing. V5's default DSL contract
  remains intact. Dynamic action-order validation and full prose semantics are not claimed.
- Use the first 20 **training** tasks in frozen order for the data-quality pilot. Withhold a
  demonstration before teacher generation; stop at visible-demo success and validate the
  withheld demonstration once without returning its answer or outcome as repair feedback.
  The witness receives only visible training pairs when checking the withheld input; its
  withheld output belongs to the checker, never the witness's `train` argument.
- Require grounded definitions, checkable links between claims/actions and witnesses, explicit
  uncertainty, and rejected-example reasons. A DSL coverage failure is not evidence that
  Astra cannot solve a task. Do not scale a grounding-only dataset as if it taught reasoning.
- Revalidate local Nanbeige compatibility and complete an untrained direct-grid/symbolic
  baseline on the frozen development cohort, with all failures/timeouts included. Start from
  validated 4K/8K settings; expand only when evidence does not fit and memory checks pass.
  Do not transfer Astra's context capacity or timing to the 16 GB Mac.
- **Current correction:** preserve `untrained-baseline-01` as an incomplete failed-formatting
  pilot, not a comparison baseline. Direct replies included non-rectangular grids; symbolic
  replies often reached the 2K output cap. Enforce rectangular grid syntax and test compact
  symbolic serialization under a new experiment identity, without changing the answer cap
  or supplying test answers. Re-pass local fixtures and complete the replacement baseline
  before real student training. Do not compare a trained model against this partial run.
  The opt-in formatting correction is implemented as
  `configs/v8-nanbeige-format-baseline.yaml`; converter hash/style/source isolation and
  native grammar checks pass. Its first 4K model fixture returned valid JSON but failed
  symbolic grounding/reference checks. Preserve it and correct identifier/reference
  generation before another fixture; do not bypass that gate or lower the memory safeguards.
  The corrective `compact_identifiers` configuration restricts the spelling of names and
  reference slots, not available operations or meanings. It does not invent or resolve
  definitions. Admitted teacher targets already satisfy its lexical contract and remain
  unchanged. A fresh model fixture is required; regression success alone does not qualify it.
  That corrective fixture also repeated undefined references and failed. The subsequent
  `reference_repair` runtime lets Nanbeige emit a hash-bound patch over only broken reference
  fields, choosing from its existing definitions. It preserves meanings and evidence, saves
  the explicit delta, and reruns all semantic checks. This is a revised inference interface,
  not an automatic correction, changed teacher target or reduced quality gate.
  Its 4K and matching 8K model fixtures passed, but the replacement development baseline
  `runs/v8/untrained-reference-baseline-01` was stopped when its direct seed-43 validity
  gate became impossible. Three invalid direct replies exhausted their post-reasoning
  output allowance on whitespace; they were not context or memory failures.
  The new `configs/v8-nanbeige-bounded-baseline.yaml` bounds structural JSON whitespace
  for direct/witness replies and retains the symbolic reference-repair interface. Require
  fresh 4K/8K model fixtures and investigate the separate symbolic output-cap failures
  before another long baseline. The first bounded-whitespace 4K fixture was structurally
  valid but failed witness/grounding correctness. Preserve it and diagnose stage-specific
  repair before another fixture; do not relax sandbox or semantic admission rules.
  Preserve the stopped baseline; never resume it under new code.
  The next configuration, `configs/v8-nanbeige-witness-baseline.yaml`, opts into code-only
  repair after static rejection. It preserves grounded state and compiler evidence, retains
  the same four-cycle allowance, and falls back to symbolic revision for actual verifier
  counterexamples. Its policy is bound into smoke/run identities and checked independently.
  Its 4K and 8K fixtures passed; they do not resolve oversized symbolic responses that
  exhaust the 2K allowance by transcribing cells or accumulating prose.
  The opt-in `configs/v8-nanbeige-triples-baseline.yaml` tests lossless `[row,column,color]`
  cell encoding at the same 2K limit, with canonical decoding before every semantic check.
  Raw replies and decoded-state evidence are preserved. Teacher targets, raw grids and
  symbolic meanings remain unchanged. The codec needs fresh 4K/8K model fixtures and a
  measured development run; byte/token savings alone do not prove successful generation.

**Gate:** correctness/isolation tests pass; at least 95% valid structured responses in each
declared pilot mode; examples from multiple training tasks demonstrate full-symbolic interpretation
and verified repair, not only grounding. Report every acceptance denominator and failure.
The student local gate additionally requires no OOM, critical pressure or swap-safety failure and
complete baseline outputs; accuracy is reported separately, not required to match Astra.
Record tolerated warning pressure separately; do not call a warning-tolerant run warning-free.
If local compatibility remains broken after one corrective retry, revise the runtime before
student cloud training or Kaggle. Teacher-only data work can continue independently.

### Phase 2 — Astra curriculum, continuity and compaction

**September 16–25.**

- Scale through audited training-only batches toward **up to 3,000 verified episodes** for
  the initial SFT run. Cover grounding, interpretation, counterexample repair, checkpoint
  continuation, compositional concept reuse and uncertainty-preserving compaction. Balance
  available verified categories; never fabricate successful examples to fill a category.
- Keep the existing **792 training / 103 development / 105 lockbox** split. All derived views,
  teacher conversations and repaired trajectories follow their source task. Reserve complete
  synthetic generator families for evaluation. Use deterministic generators and executable
  checks; exclude legacy model traces and every development/lockbox-derived training target.
- Save exact teacher request/response IDs, task and generator lineage, schema version, all
  raw artifacts and admission evidence. Keep failed attempts as masked context; supervise
  only useful admitted symbolic turns.
- Preserve teacher conversation identity during revisions and reconcile uncertain dispatch
  before retrying. App Server resumes stored conversations by ID; do not assume every
  Responses API reasoning/compaction parameter is exposed through the subscription route.
  [App Server continuity](https://learn.chatgpt.com/docs/app-server)
- The 30-minute V6 timeout motivates testing shorter **explicit state checkpoints between
  turns**. Any changed prompt/deadline policy gets a new experiment identity. Do not restart a
  live request or silently extend the frozen benchmark. Diagnose long-running tasks separately
  with predeclared adaptive deadlines, cumulative time accounting and unchanged safety caps.
- Compare direct programs, fixed scene descriptions and revisable working models. Separately
  compare reasoning removed, reasoning retained, prose compaction and structured compaction.
  Inspect rendered student prompts; `preserve_thinking` alone is not an ablation.
- For the GPU student experiments, retain a 16K context cap, compare 8K/12K compaction triggers
  and target about 1K summary tokens. On the Mac, use only memory-validated contexts. Never
  truncate required evidence; count retrieval, tokenization and compaction in solve time.
- Test five successive actual compactions, missing/corrupt artifacts, undefined concepts,
  context overflow and restart. Preserve uncertainty and evidence links. Provider-internal
  persistence, explicit text retention, compaction and prompt caching are distinct; opaque
  Astra state is not transferable into Nanbeige.

**Gate:** training corpus passes lineage, grounding, leakage, native-format and target-mask
checks. Admit full-symbolic records only at their declared validation level. Promote each
representation/retention/compaction change independently for **+3 percentage points mean
development exact match across three fixed seeds**, or **20% less runtime without mean
accuracy loss**, plus integrity tests. These are engineering thresholds, not statistical proof;
teacher improvements do not automatically promote the student configuration.

### Phase 3 — Nanbeige symbolic SFT

**September 26–October 7; not launched.**

- Train one multitask LoRA adapter on symbolic targets, not final grids or witness programs.
  Retain rank 16, learning rate `1e-4`, maximum two epochs, 8K sequences, microbatch one and
  effective batch 16 as the initial experiment. Full pretraining, online RL, tokenizer
  replacement and alternative student models remain outside this iteration.
- Preserve Nanbeige's pinned tokenizer and native conversation template. Use conversational
  prompt/completion data with completion-only loss and verify the actual trainer/collator
  labels: prompt, observations,
  failed attempts, grids and witness evidence are masked; only the admitted symbolic completion
  and terminator are supervised. Restructure or explicitly exclude overlength examples,
  never silently truncate necessary evidence.
  [TRL formats](https://huggingface.co/docs/trl/dataset_formats) ·
  [SFT loss configuration](https://huggingface.co/docs/trl/sft_trainer)
- Before a full job, verify architecture/shared-layer adapter attachment, a small training
  step, loss masking, save/reload and exact optimizer/scheduler/RNG/data-cursor resume.
  Local data preparation passing is not proof of GPU trainer compatibility. Finish the
  local student gate first; obtain approval before any separately billed compatibility job.
- Use approved HF Jobs with validated private datasets, pinned dependencies, Trackio and
  durable private checkpoints. Measure a short compatibility run to choose hardware and
  timeout with startup/save margin. Do not assume generic trainer or accelerator support
  for Nanbeige's shared-layer architecture, and do not assume the Mac can train 8K sequences.
- The local read-only training preflight now cross-checks hashes, split isolation, native
  token IDs/masks, curriculum and trainer evidence, baseline completeness and the matching
  independent stopped-run audit. Its ten evidence files include `baseline-audit.json`;
  a self-reported score or provisional live audit cannot substitute for it. It never
  uploads data, loads model weights or launches training. Approval-reference presence is
  not proof of legal permission. The full-size GPU runner is now implemented separately,
  but remains unvalidated on real GPU hardware and has not been launched.
- `scripts/prepare_nanbeige_bundle.py` separates local **data staging** from **evidence
  sealing**. The preserved native-verified draft is `runs/v8/training-bundle-draft-03`, bound
  to the now-failed `runs/v8/untrained-triples-baseline-01`; it cannot be sealed or rebound.
  A replacement protocol requires new fixtures, a new baseline identity and a new draft.
  Seal only after the replacement baseline completes and passes its gates and independent
  audit, holding its reader lock through audit/copy. Use new destinations,
  preserve original targets and runs, and re-stage if the copied tooling changes. A sealed
  evidence bundle still has no compute/output-use approval and cannot launch a job.
  Draft 01 remains preserved; its older runner hash is deliberately rejected by current
  tooling. Draft 02 contains the completed-checkpoint recovery correction but remains
  tied to the failed reference baseline. Draft 03 keeps identical teacher targets/native
  rows and binds the failed compact-cell inference comparison. Staging changed neither
  the then-running solver nor its source/config identity.
- For the 39-example pilot, equal-category sampling with complete row coverage means 60
  draws per logical epoch: 20 grounding, 20 interpretation and 20 repair. Two logical epochs
  produce 120 draws, including 40 repetitions of the sole repair example. Record each row's
  exposure; do not describe the schedule as 120 independent examples or claim broad repair
  generalisation. Traverse this expanded schedule once, never for another two trainer epochs.
- The GPU runner's `compatibility` mode uses original weights and admitted rows for two
  optimizer updates, with 8K padded tensors and ignored padding labels. It verifies adapter
  changes, save/reload, and an exact checkpoint-1-to-step-2 replay. Its updates are probe
  artifacts, not the promoted student. `train` requires the matching successful GPU report
  before consuming the full two-logical-epoch schedule. Real CUDA execution is still unproven.
- A resume from the final checkpoint must skip the training loop, restore/verify the saved
  adapter and reporting cursor, and reuse its original immutable checkpoint reference.
  Report zero new updates and do not label report-only recovery a fresh GPU-training or
  optimizer/RNG-resume test. A local single-linear-layer API fixture verified this path;
  full-size compatibility and trained-student evaluation are still required.
- **HF access preflight, refreshed September 9:** the connected `kinwo` account reports no Pro
  entitlement and OAuth scopes without repository-write access. Organization eligibility
  and a private-repository write credential remain unresolved. No account upgrade, private
  upload or paid job is authorized by this observation; select/configure an eligible account
  and obtain launch approval first. Never paste access tokens into task messages.
- Convert each candidate to the validated GGUF format and run its local regression fixtures
  before Kaggle deployment. Record base, adapter, tokenizer, converter and runtime lineage.
- **Candidate evaluation is not implemented yet.** The current V8 entry point only admits
  the original checkpoint and writes `trained: false`; the independent run auditor also
  rejects trained identities. Before evaluating a real adapter, add a separate, fail-closed
  candidate path with verified training/conversion lineage and matching solver protocol.
  Keep V4's original-checkpoint guard and the frozen baseline unchanged. Do not make a
  candidate appear supported by clearing its adapter metadata or relabelling a baseline
  report. Complete both the accuracy and inclusive-runtime promotion checks; the existing
  arithmetic helper alone cannot prove either full promotion route. See the
  [candidate handoff requirements](V8_STUDENT_BASELINE.md#trained-candidate-handoff--not-implemented).
- The separate `scripts/audit_nanbeige_trained_checkpoint.py` now checks stored training
  provenance before future conversion. It accepts report-only completed-checkpoint recovery
  only with zero new updates and an unchanged final checkpoint reference. Its success is
  explicitly not tensor validation, remote verification, inference readiness or promotion;
  no real trained artifact exists to audit yet.

**Gate:** the trained Nanbeige beats the untrained Nanbeige baseline under the Phase 2 promotion
rule, at equal wall-clock allowance, with integrity and recovery checks passing. Otherwise
retain the original/best validated Nanbeige checkpoint. Astra's 90.91% is not this gate.

### Phase 4 — Local-first GPU portability, generalisation and strategy integration

**October 8–14.**

- After local validation, package model/tokenizer/runtime/dependencies for **offline single-L4**
  tests using the same llama.cpp source with CUDA. Re-run the same fixtures and pilot;
  formatting/tool semantics must agree, but sampled completions need not be byte-identical.
- Compare validated quantized GGUF with BF16; treat the authors' vLLM branch as an optional
  backend requiring compatibility and accuracy checks. Test 8K then 16K, selecting development
  accuracy at equal time and breaking ties by runtime. Keep GGUF if acceleration fails.
- Measure student skill transfer on held-out generator families, new compositions, color/object
  renamings and valid geometric transformations with transformed labels. Report symbolic
  validity and actual ARC exact match separately; neither a fluent state nor training loss
  proves generalisation. Keep the sealed lockbox untouched until the frozen final comparison.
- Combine Nanbeige direct-grid, symbolic-program and deterministic-search candidates. Deduplicate
  outputs and track hypothesis ancestry; correlated samples are not independent votes.
  Fit the non-LLM ranker on training-only out-of-fold records, never test answers.
- If approved L4×4 resources are available, compare four-grid, two-grid/two-symbolic and
  one-grid/three-symbolic workers. Give every task initial coverage before refinement;
  use bounded queues, cumulative/global deadlines and durable state. Write complete fallback
  predictions early and update atomically. No Astra call is permitted in the offline solver.

**Gate:** reproducible offline operation, measured memory/runtime, complete outputs and
student generalisation evidence. Keep an ensemble only if it beats the strongest individual
Nanbeige strategy at equal time; otherwise ship that strategy. Any hardware change requires
remeasurement, not extrapolation from the Astra or M1 pilot.

### Phase 5 — Freeze and full offline rehearsal

**October 15–20.**

- Freeze model/adapter, harness, prompts, ranker, data lineage and runtime before evaluating
  baseline and candidate once on the sealed lockbox. Fall back on regression; do not tune
  against revealed lockbox results. Repeat local fixtures on every final student artifact.
- Package only approved Nanbeige-derived inference artifacts and deterministic components.
  No teacher weights, provider credentials, remote-call dependency or evaluation-answer cache.
  Audit Astra-derived training provenance without bundling teacher access at solve time.
- Verify the event's actual rules/resources and rehearse the complete Kaggle bundle with
  networking disabled, including startup, loading, retrieval, compaction, restart and writing.
  Target **10 hours active solving / completion within 11 hours / 12-hour hard maximum**.
- Verify exactly two correctly ordered predictions for every test input, atomic completeness,
  failure recovery, reproducible instructions, licenses and artifact hashes.

**Gate:** valid complete offline submission artifact, approved teacher/student lineage,
independent scoring/serialization checks and runtime compliance. Publishing and submitting
are separately confirmed actions. If a gate fails, report it and retain the strongest
validated Nanbeige configuration; do not substitute online Astra to claim offline compliance.

## Interfaces and reporting

- `SymbolicTaskState`: versioned grounded objects, definitions, hypotheses, counterevidence,
  uncertainty, ordered actions and authoritative artifact references.
- `AstraTeacher` / online reasoner: exact subscription model, provider continuity, explicit
  structured artifacts and separate witness/results; research and training runs kept distinct.
- `LocalReasoner`: Nanbeige-only student inference, state revision, approved tools and
  candidates; separate retained reasoning text from final symbolic content.
- `SymbolicExample`: training-task/family lineage, target category, validation level, witness
  references, source/teacher identities and loss-mask evidence; reject evaluation-derived data.
- `ExperimentRun`: model/runtime/data hashes, checkpoints, phase gates, telemetry and complete
  submission artifacts. Report dispatches separately from completed responses, censored
  timeouts and unknown token usage explicitly. No monetary budget enforcement.

Primary solver metric: exact-match with two ordered predictions per test input; also report
mean per-task output accuracy, strict whole-task accuracy, candidate coverage, structured
validity, tokens, runtime, memory/swap and failures. Keep separate scoreboards for Astra,
untrained Nanbeige and trained Nanbeige. Preserve current run identities and use matching
`--resume` only for that experiment; changed sources/configurations require a new workspace.

## Immediate next work and document map

1. Preserve the implemented bridge, audited 20-training-task curriculum and all 39 admitted
   symbolic examples. Do not use development failures as training targets.
2. Preserve all failed runs, including `runs/v8/untrained-triples-baseline-01`: its stopped
   40-stage audit and replay passed integrity, but its 95% validity gate is impossible.
   Do not resume it. On September 9, 2026, the user approved **4K generated tokens per
   call** for a new matched untrained/trained comparison. Revalidate the new output4k
   configuration at 8K then 12K context, binding generation/runtime policy into smoke
   eligibility; old 2K smoke results cannot qualify it. Start a fresh 12K baseline only
   if those checks pass. Direct/witness reasoning caps, cycle/task limits, split and
   memory safeguards remain unchanged. Measure truncation/validity with failures included.
3. Preserve drafts 01/02/03, all bound to failed runs; none may be rebound or sealed as a
   passed baseline. Stage a new draft against a new frozen identity and seal only after
   complete qualifying evidence. Local mask/architecture probes do not waive these gates.
4. Resolve output-use applicability and HF private-write/job eligibility, then request
   approval for the exact separately billed compatibility launch. Do not upload or train
   before those gates pass; a trained candidate must then satisfy the Phase 2 promotion rule.

The output-cap approval authorizes this local comparison change, not paid GPU launches,
uploads, expanded teacher collection or output-use clearance. The
original documentation update authorized no job, model call, upload or deployment. The
subsequent explicit student-promotion goal authorizes implementation and subscription teacher
collection. Separately billed GPU jobs, public release and submission still require approval.

- This file: active roadmap and decisions.
- [V5 implementation record](V5_SYMBOLIC_DISTILLATION.md): current collector, data formats,
  commands and first training-only pilot; its old phase schedule is superseded here.
- [V6 evidence](V6_LOCAL_SCORE.md): frozen, completed Astra evaluation and audit commands.
- [V7 bridge](V7_SYMBOLIC_BRIDGE.md): implemented training witness contract, audits and pilot.
- [Output-use review](ASTRA_OUTPUT_USE_REVIEW.md): fetched governing terms, unresolved
  applicability and a clarification draft; no clearance or message sent.
- [V4 local-first baseline](V4_LOCAL_FIRST.md): preserved Nanbeige setup, safeguards and diagnostics.

The **OpenAI Docs** skill informed exact-model/subscription boundaries and provider continuity.
The **Hugging Face LLM Trainer** skill informed symbolic SFT data validation, trainer-specific
loss checks, Trackio and durable checkpoints. This is a plan update, not a training launch.
