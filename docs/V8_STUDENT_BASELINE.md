# V8 — Nanbeige student baseline and training compatibility

## Purpose and current boundary

Measure **untrained Nanbeige**, then compare the eventual symbolic-SFT student using the
same solver protocol, cohort, seeds, hardware and time allowance. Astra's development
score is a separate teacher reference, never a student baseline or training corpus.

The V4 15-operation DSL cannot express every V7 verified witness. V8 adds a student
path that produces an explicit symbolic working model, then a separate bounded Python
witness. A direct-grid mode is the control. The witness is an inference/verification
artifact, not a supervised training target. No alternative LLM is permitted.

## Local protocol

### Approved recovery of the September 9 pre-model startup failure

**Latest status, September 9 at 09:31 Brisbane:** the corrected restart subsequently
hit a real inference-time safety stop at **1.099 GiB additional swap**, above the
unchanged 1 GiB limit. Both owned processes exited; automatic follow-up is paused.
There are 68 completed responses plus one interrupted call with unknown usage,
and 2h04m49s measured cumulative runtime. The stopped score/evidence audit passed,
but the pinned supplemental recovery auditor correctly rejected this new failed
execution segment. The startup-only exception does not cover it. Do not resume
this run under the existing recovery approval or repeat the interrupted call.
See [the preserved safety-stop report](../runs/v8/untrained-output4k-baseline-01/swap-stop-01.md).

The first output4k segment stopped normally after 40 stages. A subsequent launch
omitted the required sandbox wrapper and was rejected before model startup, without
new calls. The user explicitly approved audited recovery. The frozen solver source,
all 40 completed call records, all 20 terminal task states and the failed preflight
record remain unchanged. The original incident report is preserved and hash-bound.

`scripts/recover_nanbeige_preflight.py` is a separate launcher and supplemental audit,
not a solver change. It accepts only this run and this exact empty pre-model segment.
It verifies the before/after audit and exact 40-stage replay, pins completed records,
reconstructs executed-segment memory/Metal/cache evidence, preserves the cumulative
swap baseline, and rejects any other missing telemetry or failed runtime. Its resume
mode checks offline enforcement before touching the run and executes the original
frozen pilot with `resume=True` and 40 additional stages. Focused recovery, score-audit
and replay tests: **116 passed**; Ruff checks passed before sealing.

The recovery receipt is
`runs/v8/untrained-output4k-baseline-01/approved-preflight-recovery-01.json`, SHA-256
`001041fd9132a91f02f9efa62e8e35d3264328c9200c30717bf8f60b6c49b940`.
The sealed recovery-script SHA-256 is
`5cf29e699a7bdf26b0935e8bcfc1f0bfe083bf26382805641c96707932ff191f`.
Do not edit either artifact or silently overwrite prior audit files.

**Raw aggregate safety flags remain false** because they retain the failed preflight.
Use the separate recovery audit's `executed_segments_safe` result alongside the
original evidence, never present it as an unchanged/raw passing aggregate. This
approved classification does not establish a complete baseline, training readiness
or promotion; downstream audits must explicitly account for it, not blindly trust a
false raw flag or silently replace it. Any additional failed segment is not covered.

Resume from the project root only, with a new unused output filename each time:

```sh
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige/venv/bin/python -u scripts/recover_nanbeige_preflight.py resume --receipt-sha256 001041fd9132a91f02f9efa62e8e35d3264328c9200c30717bf8f60b6c49b940 --output runs/v8/untrained-output4k-baseline-01/recovery-before-resume-NEXT.json
```

The first corrected restart was verified on September 9: one model process, 12K
context, Metal offload and prompt-cache disablement confirmed, normal pressure and
no additional swap. Its first request was `student:42:symbolic:91714a58:1`, continuing
the stored state rather than repeating step zero. All existing deadlines, memory
limits, complete fallbacks and Nanbeige-only constraints remain in force.

### Unchanged solver/runtime policy

- Exact pinned Nanbeige Q4_K_M, native chat template and authors' llama.cpp/Metal build.
- One owned model process and one inference request at a time; external networking denied.
- Warning pressure is advisory only under the user-approved `record` configuration.
  Critical/unknown pressure, lost telemetry, OOM, more than 1 GiB additional swap and
  less than 10 GiB free disk remain stop conditions. Do not close user applications.
- The new `v8-nanbeige-local.yaml` sets `prompt_cache_mib: 0`, passed as `--cache-ram 0`.
  The authors' server otherwise permits an additional 8 GiB host-RAM prompt cache,
  independent of the current context's KV cache. Requests already disable prompt reuse.
  Disabling this auxiliary cache does not remove durable symbolic state or evidence.
  Earlier cached-runtime fixture results are preserved. Fresh cache-disabled 4K/8K checks
  and server-log verification passed before launching the long baseline.
- Historical 2K-output configurations use 4K then 8K context fixtures. On September 9,
  2026, the user approved a new **4K generated-token allowance per call**, equally for
  untrained/trained comparisons. `v8-nanbeige-output4k-baseline.yaml` uses 8K then 12K
  context fixtures and a 12K baseline start: the long fixture plus a full 4K answer
  cannot fit in 8K. Expand in measured 4K steps only when the full allowance does not fit.
  Never silently reduce the answer allowance or discard required evidence.
- Symbolic JSON is generated at the native non-thinking boundary, matching symbolic
  SFT targets. Direct predictions and witnesses use separate native thinking/final text.
- Digit-row strings encode every grid losslessly. This is serialization, not reasoning
  compaction. No claim of opaque provider-state transfer or five verified compactions.
- Typed symbolic decoding constrains entity grid names and coordinate bounds using only
  visible evidence. A separate verifier checks colors, relationships and defined actions.
  Compiler diagnostics identify offending cells/references; no test labels enter feedback.

## Frozen comparison design

`configs/v8-nanbeige-baseline.yaml` selected the first 20 development tasks from the existing
task-hash split: 22 test inputs, seeds 42/43/44, direct and symbolic modes. Each task/mode/seed
has a cumulative **900-second** allowance, at most four symbolic/repair cycles and at most
2K generated tokens per call. This is a new declared comparison, not a modification of the
preserved V4 adaptive timing experiments. The run has an 11-hour global ceiling; reaching
it leaves unfinished slots explicitly incomplete rather than claiming a complete baseline.

Round-robin stage scheduling provides initial coverage before deeper refinement. Each of
the six seed/mode submissions contains two predictions for every input from the beginning;
copy-input fallbacks are explicitly retained when no better candidate exists. Scores always
use the full cohort. Exact-match accuracy, strict task accuracy, dispatches, response validity,
grounding, tokens, elapsed time, memory and failures are reported separately.

Prompts, exact prepared requests, raw responses, verification artifacts and state are durable.
Completed responses replay without another model call. An interrupted local dispatch with no
recoverable response is charged and marked unavailable, not silently repeated. Missing or
changed evidence fails closed. Source/config/model/split identities and source archives are
saved before baseline generation; resumes must use the frozen source.

### Replacement formatting protocol

`configs/v8-nanbeige-format-baseline.yaml` keeps the same checkpoint, cohort, six groups,
900-second task allowance, four cycles and 2K generated-token cap. Its explicit
`structured_style: compact_rectangular` changes final-output formatting only:

- Direct decoding chooses a width from 1–30 for each grid, then enforces that width for
  every row. Both guesses remain free to choose different shapes and colors. All 1–30
  row/column dimensions are allowed. No predicted values are supplied by the grammar.
- Symbolic decoding compiles the unchanged, visible-evidence-bounded schema with the
  authors' pinned Python schema converter and changes only its structural `space` rule
  to the empty string. Whitespace inside JSON strings is preserved. Upstream required-first
  key ordering remains; canonical parsing and grounding verification remain authoritative.
  The upstream converter supports a subset of JSON Schema, not a semantic proof.
- The converter is checked against SHA-256
  `ee451dc460aa31185226e58988626f64e75ab735169fa3e484fcf16889475ae3` before its verified bytes
  execute; external schema references are forbidden. Its hash and the style are bound to
  smoke/run identities, and its source is included in each new source archive.
- Legacy decoding is still the default. Old fixtures cannot qualify a new style or source
  revision. No failed-run response is rewritten or reused as a fresh experiment result.

This can reduce syntax failures; it does not establish better ARC reasoning or a trained
student improvement. New 4K/8K fixtures and a full replacement baseline must pass separately.

The corrective `configs/v8-nanbeige-identifiers-baseline.yaml` opts into
`structured_style: compact_identifiers`. In addition to the preceding constraints, names
and reference fields must match `[A-Za-z_][A-Za-z0-9_.:-]{0,63}`. A generic prompt example
clarifies that a definition named `d1` is referenced as `d1`, not a sentence or a renamed
action. Model-generated definitions and meanings remain unrestricted by any operation list.
This is a lexical restriction only: the verifier still rejects well-spelled undefined names,
duplicate observations and invalid relationships. There is no automatic reference repair.

All admitted teacher targets retain their original bytes; a read-only compatibility test
checks every name/reference in all 39 examples against the lexical restriction. The native
GBNF tests cover valid names, prose rejection and undefined-but-well-spelled reference
rejection by the separate verifier. Earlier styles and failed source archives remain intact.

### Runtime revision: explicit reference patches

The identifier-only corrective model fixture still failed. The separate
`configs/v8-nanbeige-reference-baseline.yaml` therefore revises the repair interface rather
than continuing prompt-only retries. Initial symbolic generation uses the identifier protocol.
When a structurally valid state has undefined hypothesis references, the next Nanbeige call
may return a small `hypotheses` patch containing only the broken `concepts`/`ordered_actions`
fields. Its grammar lists exactly the already-declared definition IDs as choices.

Nanbeige chooses those references by reading the retained definitions, hypotheses and raw
evidence. Deterministic code does not choose the mapping, invent definitions, rewrite meanings
or drop observations. Already-valid fields are omitted from the patch. The immutable base is
hash-bound; stale bases, duplicate/missing hypothesis entries, invented references and extra
fields fail closed. Both the patch and resulting state are saved as auditable artifacts.
The independent response auditor checks the same patch contract and its authoritative input.

All existing semantic checks run after applying the patch. A linked state may still express
a wrong transformation, and any witness must still pass the visible demonstrations. Other
error types still use full state revision. Calls, failed attempts, generation and repair time
remain charged to the unchanged four-cycle/900-second protocol. Native non-thinking symbolic
boundaries and the 2K generated-token cap are unchanged. This is an inference repair role,
not a new teacher label or a change to the admitted SFT targets.

## Verification status — September 8–9

- **718 regression tests passed**, including 20 read-only training-preflight tests,
  46 GPU-runner/recovery safety tests, 51 independent-baseline-audit tests,
  34 trained-checkpoint provenance tests, 56 formatting/identifier tests,
  24 reference-patch tests, 11 witness-repair tests, 22 cell-codec/protocol tests and
  16 bundle/audit-gate tests, 10 observation-catalog tests, 20 state-replay tests and
  20 adapter-payload tests;
  native JSON/Python parsing, isolation,
  response replay, unknown-dispatch accounting, complete fallbacks and lossless encoding
  covered. This does not establish live student accuracy.
  The formatting tests execute the pinned runtime's native GBNF validator without loading
  a model: rectangular boundaries, malformed grids, exact guess/input counts, compact JSON,
  unchanged string contents, converter pinning, native boundaries and identity isolation.
  [Full regression record](../runs/v8-adapter-payload-full-regression-01.xml)
- Initial 4K symbolic smoke and one prompt-only corrective retry both returned 100% valid
  structured responses but failed grounding. Both stayed at normal pressure with no swap
  growth. Failures are preserved in `runs/v8/smoke-4k` and `smoke-4k-retry-01`.
- The local interface was subsequently revised with bounded entity decoding, detailed
  compiler diagnostics and up to four cycles. The revised **4K fixture passed**, including
  an undefined-concept repair, a Python syntax repair and completed-state replay. Both modes
  returned 100% valid structured responses; normal pressure, no swap growth, peak server RSS
  3,866,378,240 bytes. The matching **8K fixture also passed**, including a 5,571-token prompt,
  normal pressure and zero swap growth, peaking at 4,420,124,672 bytes RSS.
  These used the original auxiliary cache.
  [4K evidence](../runs/v8/typed-smoke-4k-01/smoke.json)
  · [8K evidence](../runs/v8/typed-smoke-8k-01/smoke.json)
- Fresh **cache-disabled 4K and 8K checks passed** with 100% validity, normal pressure,
  zero additional swap, and restart/replay verification. Peak RSS was 3,155,017,728 bytes
  at 4K and 3,552,346,112 bytes at 8K; the latter passed the 5,571-token prompt fixture.
  These are compatibility results, not ARC accuracy.
  [4K evidence](../runs/v8/no-cache-smoke-4k-01/smoke.json)
  · [8K evidence](../runs/v8/no-cache-smoke-8k-01/smoke.json)
- The frozen pilot in `runs/v8/untrained-baseline-01` was **stopped incomplete** after
  2,730.65 seconds. At the decision point, nine invalid symbolic replies made its best
  possible final seed-42 validity 93.88%, below the configured 95% gate. Both owned processes
  were verified terminal. Source/config archives, fallbacks and checkpoints are preserved.
  The direct snapshot is 1/22 outputs correct (4.55%), with only 12 task states terminal;
  it must not be presented as a completed student benchmark or used for promotion.
  [Decision record](../runs/v8/untrained-baseline-01/failure-decision.json)
  · [Stopped audit](../runs/v8/untrained-baseline-01/stopped-audit-01.json)
- The replacement-format **4K fixture finished but did not pass**. Direct decoding returned
  one valid response and completed; all four symbolic replies were valid JSON, but none
  passed grounding (one duplicate-cell error, then three undefined-concept errors). No
  executable witness was reached. Normal pressure, zero warning samples, zero additional
  swap, peak RSS 3,160,342,528 bytes, offline Metal/cache-disabled checks and completed-state
  replay passed. These results establish formatting compatibility on this fixture only,
  not ARC accuracy. Do not advance this source to 8K or the replacement baseline yet.
  [Failed formatting fixture](../runs/v8/format-smoke-4k-01/smoke.json)
- The identifier-only **4K corrective fixture also failed**: four structurally valid symbolic
  replies repeated undefined references, with zero grounded states. Direct decoding completed;
  normal pressure, zero additional swap, peak RSS 3,144,368,128 bytes and replay verification
  passed. This motivated the explicit patch interface; it does not qualify for an 8K run.
  [Failed identifier fixture](../runs/v8/identifiers-smoke-4k-01/smoke.json)
- The revised **reference-patch 4K fixture passed** end to end. Following one undefined-name
  failure, Nanbeige emitted a 62-token patch, the retained state passed grounding, and its
  subsequent witness produced the correct fixture output. Direct and symbolic modes both
  returned 100% structurally valid responses. Normal pressure, zero additional swap, peak
  RSS 3,146,448,896 bytes, offline Metal/cache-disabled checks and restart/replay passed.
  Symbolic solving took 98.90 seconds; direct solving took 46.91 seconds. This is a toy
  correctness fixture, not development accuracy or a trained-student promotion.
  [Passed reference fixture](../runs/v8/reference-smoke-4k-01/smoke.json)
- The matching **reference-patch 8K fixture passed**, including the 5,571-token prompt
  fixture (1,049 generated tokens), symbolic reference repair, witness execution and
  restart/replay. Both modes had 100% structured validity, normal pressure and zero swap
  growth; peak RSS was 3,538,419,712 bytes. The source hash for both passed checks is
  `1dbf1b2a8e3668e5ed2d3de5e2e1f517fca5a230731b5e458ae9475bcc5fb5b8`.
  [Passed 8K evidence](../runs/v8/reference-smoke-8k-01/smoke.json)
- The replacement baseline `runs/v8/untrained-reference-baseline-01` was **stopped incomplete**
  after 7,060.61 seconds (1h57m41s), with 75 completed stages. Two confirmed invalid direct
  seed-43 replies made its maximum possible validity 20/22 = **90.91%**, below the 95% gate.
  Partial direct accuracy was 1/22 outputs for seed 42 and 2/22 for seed 43; seed 44 had not
  started. These are incomplete snapshots, not a qualifying baseline or training comparison.
  All six fallback exports, source/config archives and checkpoints remain intact. The owned
  server exited; normal pressure, zero additional swap and peak RSS 3,531,882,496 bytes were
  recorded. Interrupted dispatch `student:43:symbolic:8a371977:0` stays unavailable and must
  not be reissued as though it never happened. The stopped audit is complete as an audit,
  but explicitly reports the benchmark incomplete and its baseline gate failed.
  [Stopped evidence audit](../runs/v8/untrained-reference-baseline-01/audit-stopped-gate-failure-01.json)
- The opt-in `reference_repair_bounded_ws` correction passed 627 tests, including the native
  grammar validator, but its **4K model fixture failed correctness**. Direct returned the
  correct output in 46.61 seconds. All six symbolic/witness replies were structurally valid;
  two states passed grounding, but both witnesses failed static admission with
  `disallowed syntax: Raise`. A subsequent full symbolic revision asserted a false
  `same_colors` relation and exhausted the four-cycle allowance after 276.22 seconds.
  Normal pressure, zero warning samples, zero additional swap, peak RSS 3,145,383,936 bytes,
  offline Metal/cache-disabled checks and restart/replay passed. The owned server exited.
  No 8K fixture or new baseline was launched. This is a semantic/code-admission failure,
  not memory pressure or context overflow. The separate development symbolic 2K-output
  truncation remains unresolved.
  [Failed bounded-whitespace fixture](../runs/v8/bounded-smoke-4k-01/smoke.json)
- Stage-specific witness repair **passed the fresh offline 4K fixture**. Nanbeige's second
  code-only retry removed the forbidden exception self-check; the original grounded
  symbolic state was retained. Direct completed in 46.66 seconds, symbolic in 178.76 seconds;
  all six replies were structurally valid and both outputs correct. Normal pressure, zero
  warning samples, zero additional swap, peak RSS 3,144,941,568 bytes and replay passed.
  [Passed witness-repair 4K fixture](../runs/v8/witness-smoke-4k-01/smoke.json).
  The matching **8K fixture also passed**, including the 5,571-token prompt fixture with
  1,059 generated tokens. Both modes returned correct outputs with 100% structured
  validity, normal pressure, zero swap growth, peak RSS 3,539,271,680 bytes and verified
  replay. Its owned server exited. No new development baseline has started.
  [Passed witness-repair 8K fixture](../runs/v8/witness-smoke-8k-01/smoke.json).
- The cell-triples protocol **passed the fresh offline 4K fixture**. Direct completed in
  46.62 seconds; symbolic completed in 57.44 seconds using three replies (initial state,
  reference repair and successful witness), one grounded state and 1,043 generated tokens.
  Both outputs were correct, all four replies were structurally valid, and normal pressure,
  zero warnings/swap growth, peak RSS 3,103,244,288 bytes and replay passed. Matching **8K
  also passed**, including the 5,571-token prompt/1,059-token response fixture, replay,
  normal pressure, zero swap growth and peak RSS 3,496,951,808 bytes.
  [Passed compact-cell 8K fixture](../runs/v8/triples-smoke-8k-01/smoke.json).
  This one toy task does not demonstrate reduced
  development truncation or the Phase 2 runtime improvement. Old configurations and all
  teacher targets remain unchanged.
  [Passed compact-cell 4K fixture](../runs/v8/triples-smoke-4k-01/smoke.json).
- The compact-cell development baseline `runs/v8/untrained-triples-baseline-01` has
  **stopped incomplete after 40 stages**. Its symbolic validity upper bound is 94.29%, so
  it cannot pass the 95% gate and must not resume. All six fallback exports and the source
  identity are preserved; draft 03 cannot be sealed against this failed experiment.
  The full comparison still requires both modes, all three seeds and all 20 tasks/22
  test inputs in a qualifying new run. See the stopped audit and diagnosis below.
- The completed V7 pilot supplied **39 examples** fitting native 8K inputs/targets without
  truncation; maximum length 5,524 tokens and maximum completion 1,312 tokens. Full symbolic
  and grounding instructions remain
  distinct. Compact views change input encoding, never the accepted Astra target.
- No trained student, full Nanbeige development score or promotion result exists yet.

## Training compatibility, separately gated

`configs/nanbeige-trainer-requirements.txt` and `.runtime/nanbeige-trainer` isolate the
experimental training dependencies from the proven inference environment. The model card
recommends Transformers 4.45.1; the 4.57.6/TRL 0.25.1 stack must earn compatibility through
tests, not version assumptions. The pinned model architecture imports locally.
[Nanbeige model card](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/3384e426066d1a49c3aea90a7190b81260a6533f/README.md)
· [TRL 0.25.1](https://huggingface.co/docs/trl/v0.25.1/sft_trainer)

`scripts/check_nanbeige_trainer.py` checks actual TRL collation of admitted symbolic data,
meta-device adapter attachment to 22 unique physical layers reused across two loops, and
a tiny **random-initialized Nanbeige architecture fixture** for numerical training and
checkpoint recovery. It does not load or train the official weights; it cannot establish
GPU compatibility, student accuracy or promotion. Fixture data never enters the real corpus.

The final local probe **passed**: all 39 real rows retained their completion-only masks through
the actual Trainer data loader; 154 unique LoRA modules represented 23,969,792 planned trainable
parameters across 22 physical layers. The random fixture's adapter updates, save/reload and
resumed final parameters matched exactly. These are local architecture/collator checks, not a
trained 3B student or a full-size GPU proof. [Probe report](../runs/v8/trainer-probe-final/report.json)

The actual GPU job still requires the completed local student/data gates, verified private
durable checkpoints, licenses/output-use review, measured hardware/timeout selection and
explicit approval for separately billed compute. The Hugging Face LLM Trainer skill informed
the completion-only mask checks, isolated compatibility tests and checkpoint requirements.

`scripts/validate_nanbeige_training.py` is a read-only preflight, **not a GPU trainer**.
It checks a hash-bound ten-file evidence bundle, the 792/103/105 split, row provenance,
symbolic-only targets, native token IDs/loss boundaries, the curriculum and local trainer
audits, and all six completed baseline mode/seed groups. The mandatory `baseline-audit.json`
must be complete and non-live, verify source/cohort/stored evidence, and match baseline
identity, split and every group's score/usage/validity counts. `--require-approval` additionally
requires approval references; it does not establish their legal sufficiency or launch a job.
The unfinished live baseline was correctly rejected. All 39 real symbolic rows passed its
separate native-token validation; passing that subset is not passing the whole preflight.

The proposed balanced schedule covers every row in each logical epoch. With 20 grounding,
18 interpretation and one repair example, two logical epochs produce 120 draws, including
40 uses of the same repair example. This is explicit oversampling, not extra independent
supervision. Traverse the expanded schedule once. More verified repair and continuation
examples remain necessary before claiming those skills generalise.

## Local training bundle — staged, not sealed or approved

`scripts/prepare_nanbeige_bundle.py` has two local-only commands. Neither loads official
weights, uploads data, grants permission or launches training:

- **`stage`:** copy exactly seven data/provenance files and three preparation/training scripts
  into a new directory; verify hashes, task isolation, target bytes and the pinned native
  tokenizer's actual token IDs/completion boundaries. Bind the draft to the untrained
  baseline identity. Write `draft-manifest.json` only after these checks pass.
- **`seal`:** verify the immutable draft and unchanged copied tooling; require the stopped,
  complete baseline and an independent audit using its extracted frozen source. Hold a
  shared baseline lock throughout audit/copy. Add report, identity and audit as the three
  remaining evidence files, recheck native data, validate the full bundle and write a final
  `assembly-report.json` with status `evidence_sealed_pending_approval`. Manifest presence
  alone is not a success receipt; incomplete artifacts are preserved if a check fails.

Both require a new destination and a 10 GiB free-disk reserve. Refuse symlinks, changed
evidence and overwrites. A draft is intentionally not accepted by the training preflight;
a sealed bundle still has empty approval references and no authorized GPU launches.

Actual September 9 result, under the external-network-denied sandbox:

- Initial preserved draft: [training-bundle-draft-01](../runs/v8/training-bundle-draft-01/draft-manifest.json).
  All 39 real examples passed native token/mask revalidation; zero official weights loaded.
  Categories remain 20 grounding, 18 interpretation and one repair; 120 scheduled draws
  do not increase that example count. No teacher targets changed.
- Draft manifest SHA-256: `6a86d82febff7d4b10b1b91428d096b4fb70c6ef207befca74b67e6f860615ea`.
- Attempted sealing against the actual live baseline was correctly rejected with
  `baseline is still owned; never seal a live run`. The proposed sealed destination was
  not created. This is a negative gate test, not a failed training job.
- The contemporaneous [independent live audit](../runs/v8/untrained-reference-baseline-01/audit-bundle-stage-01.json)
  passed stored-evidence/source checks but remains provisional and incomplete. It cannot
  be used to seal a training bundle. At this snapshot memory was normal with zero swap
  growth; grounding errors and truncated symbolic replies remained recorded failures.

The completed-checkpoint recovery correction changed the runner hash after this initial
staging. The old draft now correctly fails the unchanged-tooling check, without creating
a sealed directory. The preserved [draft 02](../runs/v8/training-bundle-draft-02/draft-manifest.json)
passed fresh native validation under the same network-denied sandbox, with all 39 unchanged
targets/rows and 120 scheduled draws. Its manifest SHA-256 is
`604b2f86348ddba343d6e86bf0a7be50df2b768194b97611f7b82e1c71b63932`.
Neither draft is a passed baseline gate or an approved training launch. **Both are now
bound to the stopped, failed reference baseline and cannot be sealed into a qualifying
training bundle.** Preserve them unchanged. A replacement protocol/baseline identity needs
a newly staged draft; do not retroactively rebind draft 02 or use its hash to imply approval.
Seal that new draft only after the replacement baseline finishes and passes its audit/gates.

The preserved **[draft 03](../runs/v8/training-bundle-draft-03/draft-manifest.json)** passed
native tokenization/data validation under the external-network-denied local sandbox. It
contains the same 39 symbolic examples and 120 explicit draws, including 40 uses of the
single repair example—not 120 independent training episodes. Dataset content hash remains
`ade805b5c99c6e904254bb1aaa50948b9258c219e2b97e03667f5f9cbc16266d`.
No weights were loaded, targets changed, data uploaded or training launched.

- Draft manifest SHA-256:
  `64e882cc17849c9c68605ee508e776ebc807b8c1ac8a98f1033612142ae4f809`.
- Bound baseline: `runs/v8/untrained-triples-baseline-01`; identity hash
  `c577853979a043e3b3f8d809e96ccc4cb6f8cea7cb17bd3a73b4b7886af4a3d3`.
- `verify_draft` rechecked the complete file/tool allowlist and data evidence successfully.
  `baseline_gate_passed` remains false, and both approval references remain null.
- The baseline source archive was extracted into its new `frozen/src` directory without
  changing the archive or running source. Its first frozen-source audit passed source,
  cohort and stored-evidence checks but is explicitly live/provisional/incomplete.
  [Frozen-source snapshot](../runs/v8/untrained-triples-baseline-01/audit-frozen-source-01.json).
- The development run has now stopped after 40 stages with an impossible 95% validity
  gate. **Draft 03 cannot be sealed or rebound.** Preserve it and the failed baseline.
  A changed protocol requires a new source/config identity, local fixtures and draft.

If preparation or training scripts change, stage a new draft first; never rewrite the old
manifest. To stage a fresh version, use `stage --destination` with a new path. The defaults
point to the frozen split, final curriculum audit, final student data and local trainer probe.
The Hugging Face LLM Trainer skill informed native validation and separating data readiness
from baseline evidence, private persistence and paid-launch approval.

## Full-size GPU runner — implemented, not yet GPU-validated

`scripts/train_nanbeige_symbolic.py` has separate `compatibility` and `train` modes.
Both reject non-CUDA execution before Hub access, require a hash-bound approved private
bundle, and load only the pinned official Nanbeige weights, shard index, tokenizer and
custom model code. Full-size GPU execution has **not** happened; tests use synthetic
metadata/checkpoint fixtures and fake publication APIs, not trained solver artifacts.

- Compatibility: 32 draws from admitted rows, including the longest row; microbatch one,
  effective batch 16, two updates. Padding to 8K exercises allocated sequence length while
  attention/padding labels remain masked. This is not a claim of 8K unpadded training data.
- Verify every loader row's order and labels, the 154 physical LoRA attachments, actual
  weight updates, exact adapter reload, and exact resumed step-2 parameters from checkpoint 1.
- SFT: require the matching full-size compatibility report, then traverse the 120-draw
  two-logical-epoch schedule once. It must not reuse the compatibility adapter as the base.
- Every checkpoint includes adapter, optimizer, scheduler, RNG, trainer state, explicit data
  cursor, run identity and file hashes. Publication is synchronous to a pre-existing private
  repository under a unique run prefix; parent-commit checks prevent concurrent overwrites.
  An immutable remote marker is read back before more training proceeds. Resumes verify
  every downloaded checkpoint member before deserializing optimizer/RNG state.
- Trackio Space and metrics dataset must already be private. The script does not create
  public repositories or infer compute/output-use approval. Future job submission must use
  the exact approved hardware and timeout and pass the script inline with a write credential.

### Finished-checkpoint recovery

A job may stop after its final durable checkpoint but before its result report is published.
The runner previously entered `train()` even when the saved data cursor had consumed the
whole schedule, then expected a new checkpoint in the new job directory. This could rerun
work or fail to find the already-finished adapter.

`execute_schedule` now validates the full checkpoint and path/cursor agreement first. For
a completed cursor, it restores the adapter through the pinned PEFT API and reads the actual
Transformers state without calling `train()`. Partial resumes still use the trainer for the
remaining draws. Every path requires a complete final checkpoint before reporting success.
The original immutable remote checkpoint reference survives recovery and is exported as
`final_checkpoint_ref`; no finished weights are republished or overwritten.

Reports include `optimizer_updates_this_invocation` and `recovered_completed_checkpoint`.
Report-only recovery records zero new updates and `gpu_training_verified: false`; it does
not claim to have exercised optimizer/RNG restoration. The full-size compatibility mode
still requires actual updates and exact interrupted-training replay.

Eleven additional unit tests cover finished/partial/fresh execution, exact adapter loading,
tampered optimizer hashes, cursor/path mismatch, empty schedules and missing durable output.
The separate [installed-library probe](../runs/v8/completed-resume-api-probe-01/report.json)
passed under the network-denied sandbox using a **single synthetic linear projection**:
PEFT 0.18.0 and Transformers 4.57.6 restored exact adapter tensors and step 2, with no
checkpoint change, optimizer update, language model, official weight load or training job.
This supports the API correction, not full-size Nanbeige/GPU compatibility or promotion.
Reproduce in a new output directory:

```sh
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige-trainer/bin/python scripts/check_nanbeige_completed_resume.py --output runs/v8/completed-resume-api-probe-02
```

September 9 access refresh: connected HF account `kinwo` reports no Pro entitlement and no
repository-write OAuth scope. An eligible account/organization and write credential remain
to be configured. The local baseline and output-use review must also pass before launch.
The governing OpenAI terms and their competing-model provision/business exceptions were
also fetched on September 9. Applicability to this exact subscription route and generative
student remains unresolved; the technical SFT example is not a permission grant. See the
[source-backed review and unsent clarification draft](ASTRA_OUTPUT_USE_REVIEW.md).
No paid job, account upgrade, expanded teacher collection or private upload was launched.

The runner's keyword arguments were checked against the installed pinned TRL, PEFT and
collator interfaces without loading weights. This is not an end-to-end GPU trainer test.
The Hugging Face LLM Trainer skill informed the job separation, Trackio configuration and
durable checkpoint requirements. Paid compute, uploads and public release remain unlaunched.

## Trained candidate handoff — not implemented

Source inspection on September 9 confirmed that a successful training job would **not yet**
be evaluable through the current baseline command:

- `v4_config.verify_manifest` rejects non-null adapters and is intentionally the original
  checkpoint guard. A merged model must not bypass this by claiming `adapter: null`.
- `v8_cli.pilot` fixes the identity to `untrained_nanbeige`; its report writer fixes
  `trained: false`. Neither is a trained-candidate entry point.
- `audit_nanbeige_baseline.audit` rejects any other student identity. Its paired arithmetic
  helper does not verify trained lineage, hardware, recovery or inclusive runtime, and
  deliberately never returns a proven promotion.

After an approved real training run, the remaining implementation and verification are:

1. Export into new artifact paths with a hash-bound chain from the pinned official base,
   accepted training bundle and durable trained checkpoint through merge/conversion to
   the actual GGUF. Record adapter provenance even when its weights are merged. Keep the
   native tokenizer, model configuration and runtime checks; preserve the original files.
2. Add a separate candidate entry point and model identity, without weakening V4 or editing
   a baseline report. Share or verify the frozen solver logic: prompts, schemas, parser,
   symbolic tools, scheduling, context policy, seeds, time allowances and cohort. Any
   substantive inference change needs a matching new untrained baseline, not attribution
   of the harness gain to training.
3. Re-run local model fixtures and restart/recovery checks on the actual trained artifact,
   then all six development mode/seed groups. Record hardware and inclusive elapsed time,
   including startup, retrieval, verification, context changes and recovery; per-task
   duration sums alone cannot establish the 20%-runtime route.
4. Independently audit the complete candidate and baseline, including artifact lineage,
   safety, outputs, failures and state integrity. Only then apply the three-seed Phase 2
   accuracy/runtime rule. A changed `trained` flag, mock artifact or arithmetic-only result
   is not evidence of a trained-student improvement.

The initial inspection changed documentation only. The subsequent stored-checkpoint audit
below is one handoff component, not candidate inference support. The reference baseline is
preserved but failed; a qualifying replacement plus the outstanding output-use/compute
prerequisites remain required. No official weights were loaded, converted or trained by
this provenance-audit work.

### Stored-checkpoint provenance component

`scripts/audit_nanbeige_trained_checkpoint.py` is now a local, read-only audit. It requires
the approved sealed bundle hash, exact training/compatibility report hashes and the final
checkpoint directory. It reuses repository-owned preflight/checkpoint validation code,
verifies the copied training tools against that code, and never executes code from the
supplied bundle or deserializes optimizer, RNG or model tensors.

It checks the task-isolated data and completed baseline evidence, pinned Nanbeige/model
revision, training environment and schedule, per-launch approval references, two-update
compatibility report, SFT versus probe identity, complete cursor, checkpoint file hashes,
adapter configuration, and matching immutable durable reference. Missing/changed files,
symlinks and unaccounted checkpoint members fail. A report-only final-checkpoint recovery
must report zero new updates and no new GPU training; the completed checkpoint is still
required. Optional `--output` creates a new audit file and refuses overwriting.

Its result is deliberately limited to `stored_training_evidence_verified`. It returns false
for remote availability verification, GPU replay, tensor payload validation, inference
readiness and promotion. Hashes and recorded approval references are not legal permission,
provider attestations or independent evidence that a GPU executed the reported work.
Future conversion and inference must validate actual tensor contents and runtime behavior.

All **34 synthetic-fixture tests passed**, including hash/identity/reference faults,
compatibility-versus-training distinctions, completed-checkpoint recovery, missing permission,
and read-only behavior. Their placeholder tensor bytes are intentionally not real adapters;
passing these tests does not validate a trained model. Full regression: **609 passed**, two
dependency deprecation warnings, zero failures/skips. At that historical checkpoint, the
solver source hash and the three scripts in draft 02 were unchanged, so the provenance
audit alone did not require restaging. Draft 03 now binds the later compact-cell baseline.

The Hugging Face LLM Trainer skill informed the durable-checkpoint handoff and separation
of stored provenance from actual model validation. No real trained checkpoint is available
yet. Inspect the local-only CLI without starting any work:

```sh
.runtime/nanbeige/venv/bin/python scripts/audit_nanbeige_trained_checkpoint.py --help
```

## Independent score and comparison audit

`scripts/audit_nanbeige_baseline.py` independently recomputes exact-match and strict-task
scores from the saved state and frozen task files. It checks source-archive hashes, database
identity, immutable response/verifier references, raw-versus-parsed native responses, token
usage, structured-validity counts, test-input ordering and all six complete fallback exports.
It does not execute model calls or feed evaluation outcomes into training.

A run with an actual lock owner requires `--allow-running` and is always labelled provisional.
Stopped-run report/submission mismatches fail the audit. Live exports may trail the atomic
SQLite snapshot by a stage and are explicitly reported as such. `--output` may create a new
audit snapshot, but never overwrites an existing report or changes experiment records.

The live optimistic validity bound must not treat an unresolved request as a fixed failure:
it may still return a valid reply. The audit keeps its dispatch in observed counts but allows
that reply to succeed in the **upper-bound calculation only**. Already-accounted malformed,
timed-out and lost responses remain failures. For a stopped run, unreconciled lost requests
also remain failures; they cannot be made free by resuming or repeating them.

September 9 correction: the [live snapshot](../runs/v8/untrained-reference-baseline-01/audit-live-bound-fix-01.json)
had 18 direct-mode dispatches, one fixed failure and one unresolved request. The old bound
would have been 93.75%; allowing the unresolved reply to succeed gives the proper optimistic
bound of 96.875%. Thus the run must not be stopped as mathematically hopeless on that evidence.
This does not raise its measured accuracy or structured-validity rate, change its protocol,
or pass the training gate. Both modes were still potentially eligible in this snapshot.

The initial corrected audits included `auditor_sha256`, policy `live_unreconciled_may_succeed_v1`,
`potential_pending_successes_for_bound` and `fixed_failures_for_bound`. Preserve older audit
snapshots, but do not use their pessimistic live bounds to make early-stop decisions. Nine
new tests cover the threshold boundary, fixed failures, stopped unknowns, invalid counts and
actual live/stopped audit integration. No solver source or staged training-runner hash changed.

The subsequent `live_unreconciled_may_succeed_v2_direct_task_cap` policy tightens only the
optimistic direct-mode bound. A valid direct reply immediately completes its task, so the
20-task cohort can produce at most **20 valid direct replies**, not 22 (the number of test
inputs) and not one success for every unused retry. With `F` fixed invalid calls, its validity
cannot exceed `20 / (20 + F)`. Take the smaller of this cap and the request-count bound.
Thus one fixed failure still permits 95.24%, but two permit at most 90.91% and make the 95%
gate impossible. This does not change measured validity, the threshold or any solve limits.

Live pending replies are still allowed to succeed in the bound; never treat a second
in-flight request as a second fixed failure. Symbolic mode can produce several valid
stage replies per task and does **not** receive the direct-task cap. Reports expose both
`request_count_validity_bound` and `direct_task_validity_bound`, with their minimum in
`optimistic_final_validity_bound`. Older request-only bounds may be looser, not optimistic
evidence that a direct group with two confirmed failures can qualify.

The [first capped live snapshot](../runs/v8/untrained-reference-baseline-01/audit-direct-success-cap-01.json)
had one confirmed invalid call in each started direct group. Both retained a 95.24% upper
bound, so the baseline continued. Nine additional tests cover the cap, pending/stopped
distinction, invalid counts and unchanged symbolic behavior; actual audit integration is
also checked. Full regression: **618 passed**, two dependency deprecation warnings, no
failures/skips. The frozen solver source and staged training tools remain unchanged.

The paired arithmetic helper compares three-seed mean output exact match against the
strongest untrained mode, rejecting Astra scores, incomplete/live runs, missing seeds, altered
cohorts or unequal declared inference settings. A +3-point numeric gain is **not** a promotion
by itself: trained-artifact lineage, hardware/protocol identity, full run audits and recovery
checks are still required. The 20%-runtime route requires separate inclusive end-to-end timing;
the helper does not treat sums of per-task durations as that evidence.

V8 failures include non-rectangular direct grids and symbolic replies cut off at the
2K generated-token allowance. The inspected runtime responses report no prompt truncation;
these failures must not be attributed to the Mac's memory pressure or context capacity.
One interrupted symbolic dispatch (`794b24be`, seed 42, step 0) remains explicitly
unavailable and must never be reissued as if it had not happened. Stopped-run auditing
counts it in the denominator even before any resume reconciles its stage. No model is
currently running for this failed pilot. Its frozen source is extracted under
`runs/v8/untrained-baseline-01/frozen`; use that `src` on `PYTHONPATH` when auditing it with
the independent auditor. The separate formatting-correctness configuration is implemented
and regression-tested; fresh model fixtures and a replacement baseline are separate gates.
The first new fixtures exposed symbolic-reference errors despite valid syntax. The subsequent
reference-patch runtime passed fresh grounded-state/witness fixtures at 4K and 8K, but its
development baseline subsequently failed the direct-response validity gate. Do not weaken
the memory guards, drop failed calls or count a harness change as a trained-student gain.

### Bounded-whitespace protocol correction

All three invalid direct replies in the stopped reference baseline ended reasoning normally
and then emitted only whitespace until the 2,048-token output limit. The grammar's
`ws ::= [ \\t\\n\\r]*` allowed that unbounded formatting path. Each response reported no
prompt truncation; the correctly named native reasoning budget was present. Neither larger
context nor ignoring memory warnings addresses this particular failure.

The new opt-in style `reference_repair_bounded_ws` uses at most one whitespace character at
each structural JSON boundary in direct/witness output. It preserves code-string contents,
the native lazy-grammar reasoning boundary, 1,024/384 direct/witness reasoning budgets,
the 2,048 generated-token allowance and all prior reference-patch checks. Historical styles
are unchanged. Source hash:
`7f023fbcc5d3ebe0378e9160b16a888841451da89f5db10bb5c584f4ba0a94aa`.
Symbolic grammar already has zero structural whitespace; its separate truncation failures
remain unresolved. Passing a toy fixture alone will not justify another long baseline.

The first bounded-whitespace fixture generated both witnesses with a forbidden `raise`
statement. Static verification rejected them even though that branch need not execute on
the visible demonstrations; the sandbox restriction remains authoritative. The historical
retry path revised the full symbolic state after every rejected witness, including code-only
admission errors. Do not silently remove unsafe code, weaken the sandbox or turn this
fixture into a teacher/training example.

### Stage-specific witness repair

`configs/v8-nanbeige-witness-baseline.yaml` retains bounded-whitespace/reference-patch
formatting and opts into `witness_repair_policy: repair_static_errors`. The legacy default
is `revise_symbolic`; existing configurations are not silently opted in. The policy is
included in smoke/run identities and independently checked against the run configuration.
Current source hash:
`5e09ca1c2997db91e895fd5479d286d7d6a006459b59768dacfad8b4f216dc67`.

- A static rejection retains the unchanged provisional symbolic state and asks Nanbeige
  for a complete corrected witness, using the actual compiler feedback and prior code.
  The failed program stays in the evidence store; the harness never edits it into acceptance.
- Before dispatching a code-only retry, revalidate grounding and the saved symbolic hash.
  Corruption is fatal, not permission to regenerate evidence. Successful grounding is
  not proof of the hypothesis; executable checks remain mandatory.
- Execution/demo failures still trigger the original symbolic revision path. Code-only
  failures still consume one of four cycles; all calls, errors, tokens and time remain charged.
- Tests cover unchanged default behavior, visible counterexamples, bounded repeated failure,
  mutated state rejection, exact saved-response replay and unknown-dispatch accounting.
  All **644 tests passed**, with no failures/errors/skips and two dependency warnings.
  Fresh offline 4K and matching 8K fixtures passed. These are compatibility checks, not
  a completed ARC baseline or trained-student gain.

The stopped reference baseline contains 11 symbolic replies reaching the 2,048-token limit
without prompt truncation. They emit 47–157 individual cell records each; six stop before
reaching `definitions`. This supports investigating a lossless compact observation encoding
or explicitly versioned output policy, not attributing those failures to context or memory.
Do not drop raw-grid evidence, silently truncate targets, or change the comparison protocol
inside an existing run. This issue remains open before another long baseline.

### Lossless cell-triples trial

The standalone [codec probe](../scripts/probe_nanbeige_cell_codec.py) uses the pinned local
tokenizer, not model weights or inference. It checks stored artifact hashes and measures
only serialization changes; it never completes a truncated response, exports training
data or changes a solver protocol. The frozen
[measurement report](../runs/v8/cell-codec-probe-01/report.json) records:

| Saved response group | Payload tokens before | After re-encoding | Aggregate saving |
| --- | ---: | ---: | ---: |
| 26 complete, structurally valid states; every round trip exact | 34,270 | 28,554 | 16.7% |
| 11 still-incomplete prefixes; complete cell records rewritten only | 20,567 | 13,138 | 36.1% |

These are payload re-tokenization counts, excluding instruction overhead; no generation
completion, grounding, runtime or accuracy improvement follows from these numbers alone.
The diagnostic records are evaluation-derived and explicitly not eligible for training.

The opt-in `symbolic_cell_encoding: triples_v1` in
`configs/v8-nanbeige-triples-baseline.yaml` now tests this encoding at inference time.
`src/arc_agent/v8_codec.py` preserves the prototype's exact mappings. Named fields remain
the default. New protocol source hash:
`89c884e8ee62c123ca5eb87fa6b2036bccf013b96140212db91aedbf41266952`.

- Nanbeige emits cells as exactly `[row,column,color]`, retaining all scalar/collection
  bounds and other symbolic fields. No observations, relationships or uncertainties are
  removed. Raw grids remain authoritative. This is serialization, not learned compaction.
- Decode to the existing canonical state before grounding, reference repair or witnessing.
  Keep raw replies plus a hash-linked decoding artifact. Mixed encodings, booleans, wrong
  tuple lengths and invalid bounds fail validation; incorrect observations still fail
  grounding. The harness never repairs an incomplete JSON response into acceptance.
- Reference-patch calls remain unchanged and operate on canonical states. Context expansion
  retains the declared encoding. Smoke/run identities isolate the policy, and the independent
  audit checks response encoding and required decoding evidence against the frozen protocol.
- All 39 admitted teacher targets, student-data drafts and loss masks remain unchanged.
  The future trained model must be tested under this same declared inference protocol;
  a grammar-forced wire format is not evidence that training improved symbolic knowledge.
- All 668 tests passed, including native tuple grammar, round-trip identity, canonical
  witnessing, actual request/response formatting, saved-response recovery, artifact tampering
  and encoding-isolation checks. Fresh 4K and matching 8K model fixtures passed.
  A completed development baseline, real student training and a paired candidate evaluation
  remain required. Do not treat this fixture's speedup as a trained-student promotion.

The independent auditor was also rerun against the **frozen source** of the old failed
reference baseline after codec support was added. Source/cohort/stored-evidence checks
still passed, with `complete: false` and no live owner; the new protocol did not rewrite
or rehabilitate that experiment.
[Backward-compatible stopped audit](../runs/v8/untrained-reference-baseline-01/audit-after-cell-codec-01.json).

### Observation-catalog feasibility — not an active solver protocol

The first compact-cell development replies still transcribed large cell lists. While the
frozen baseline was running, [a standalone probe](../scripts/probe_nanbeige_scene_catalog.py)
measured the existing deterministic `object_ir.extract_objects` implementation on task
`91714a58`, a member of the frozen development cohort. It inspected only demonstrations
and test inputs; evaluation labels never enter catalogs. No LLM call, solver modification,
teacher target, training example or upload was produced.

The [current probe report](../runs/v8/scene-catalog-probe-02/report.json) stores all 28
catalogs as hash-addressed artifacts: seven visible grids, independently measured with
4/8 connectivity and mono-/multicolor grouping. Exact grid reconstruction and retrieval
of every object across 24-object pages passed. Changing a grid or segmentation changes
the catalog hash; short object IDs are valid only within that catalog. These are physical
observations, not proof that any segmentation expresses the task's intended objects.

- The current `inspect_scene` tool would truncate 12 of these views at its declared
  24-object summary limit; the largest view has 83 objects. It reports truncation, but
  lacks persistent object references and pagination. Raw grids remain accessible.
- Complete 4-connected monochrome catalog pages across the task consume 9,927 input
  tokens, versus 5,682 tokens for explicit object masks. Automatically adding a complete
  catalog to every prompt would therefore worsen this case's context problem.
- The 8-connected multicolor view totals 2,755 catalog-page tokens, versus 3,342 mask
  tokens, but this one example does not justify choosing that segmentation universally.
- Every individual page is at most 805 tokens in this probe. An on-demand, revisable
  interface is a candidate for a future experiment; retrieval and prompt overhead must
  remain charged. Background inference is a heuristic, not semantic ground truth.
- No component in this sample exceeds the existing 128-cell canonical-entity bound;
  that does not establish compatibility for other tasks or authorize silent splitting.

The 10 new probe tests plus 18 existing object-IR tests passed (28 total, no failures or
skips), covering exact retrieval, pagination, stale references, uniform grids, label
isolation and mutation safety. [Focused test record](../runs/v8-scene-catalog-probe-tests.xml).
The last full-suite record remains 668 tests. Probe 01 is preserved; probe 02 records the
same measurements after a source-formatting-only correction. The frozen solver source hash
remains `89c884e8ee62c123ca5eb87fa6b2036bccf013b96140212db91aedbf41266952`.
The prototype alone does not qualify a replacement solver protocol.

### Saved-response state-transition replay

[`scripts/replay_nanbeige_baseline.py`](../scripts/replay_nanbeige_baseline.py) now
reconstructs student state transitions from saved responses without tokenization or new
model calls. It checks the frozen source and independent evidence audit first, then takes
a read-only database snapshot. All replay state writes stay in memory. Recorded prompts,
canonical symbolic states, grounding feedback, reference patches, decoding artifacts,
sandboxed witness results, candidates, predictions and counters must reproduce exactly.
The in-memory store preserves the real database's sorted-JSON round trip, including nested
key order used when rendering subsequent prompts. Terminal states must remain unchanged
when advanced again. No test labels are passed into prompt generation or verification.

The [third live replay snapshot](../runs/v8/untrained-triples-baseline-01/replay-progress-03.json)
reproduced 31 completed stages across 32 initialized states, including malformed replies
and grounding failures. One initialized state had a pending call. This is explicitly
provisional, not a complete replay of all 120 mode/seed/task states. Timing/context-limit
and missing-response transitions are reported as unsupported, never reconstructed with
invented timing, counted as successes or sent to the model again. Replay does not verify
inclusive runtime, weight lineage, full inference, or promotion.

All 20 new replay tests passed, including compact-cell and reference-patch reconstruction,
witness-only repair, terminal no-op recovery, changed prompts/states, corrupted artifacts,
stage mismatches and refusal to generate missing responses. The full suite now has
698 passing tests with no failures or skips and two dependency deprecation warnings.
[Full regression record](../runs/v8-baseline-replay-full-regression-01.xml).
The solver source/config and current baseline identity remain unchanged; no training
bundle was sealed and no student was trained by this work.
Read-only diagnostics and regression tests ran concurrently with this development pilot.
Its wall time is descriptive, not a controlled speed-comparison result: the 20% runtime
promotion branch still requires paired measurements under comparable background load,
with startup, recovery, retrieval and serialization included.

### Compact-cell baseline stopped at its planned 40-stage pause

`runs/v8/untrained-triples-baseline-01` exited normally after 40 stages and
3,428.49 seconds (57m08s). The owned server PID 11080 was separately verified absent.
The [stopped audit](../runs/v8/untrained-triples-baseline-01/audit-paused-40-stages-01.json)
passed source/cohort/stored-evidence checks but proves the validity gate is now impossible:

- Direct seed 42: 20 dispatches, 19 valid, one invalid; 19/20 task states terminal.
  The partial submission has 1/22 correct outputs, not a completed baseline score.
- Symbolic seed 42: 20 dispatches, 12 valid, eight invalid, no terminal tasks.
  The optimistic final validity bound is **94.29%**, below the required 95%.
- All nine malformed replies reached exactly 2,048 generated tokens with `stop_type: limit`.
  None reported prompt truncation. Several symbolic replies transcribed cells; others
  exhausted the cap in relations, definitions or hypotheses. The direct failure lacked
  its final JSON closing delimiters. Never close these replies by hand or count them valid.
- No OOM, warning pressure, swap growth, context expansion or safety error occurred.
  The run stayed at 8K; peak server RSS was 3,534,733,312 bytes.
- All six complete fallback exports remain intact. Seeds 43/44 were not started.
  [Stopped replay](../runs/v8/untrained-triples-baseline-01/replay-paused-40-stages-01.json)
  reproduced all 40 recorded stages without new calls; it does not establish completion.

Do not resume this failed attempt. Draft 03 remains preserved and unsealed against this
failed identity; it cannot qualify for training or be rebound to a new run. No replacement
baseline had been launched at this stopped-run checkpoint. The user subsequently approved
raising the per-call allowance to 4K, equally for untrained and eventual trained comparisons.
This is distinct from context expansion and does not authorize paid compute, teacher-data
scaling/upload, or resolve output-use applicability. The new output4k configurations preserve
all old 2K runs/configurations and require fresh 8K/12K fixtures. Smoke eligibility now binds
the complete generation/runtime and experiment policies, not source/model hashes alone;
2K evidence cannot qualify a 4K-output pilot. Direct/witness reasoning caps remain 1024/384,
symbolic generation remains non-thinking, and task/cycle/safety limits are unchanged.
Implementation and local validation are in progress; this is not a passed baseline.

### Approved 4K-output comparison (September 9, 2026)

The new configs are `configs/v8-nanbeige-output4k-local.yaml` and
`configs/v8-nanbeige-output4k-baseline.yaml`. Historical 2K configs remain unchanged.
The full regression suite passed **749 tests**, with no failures/skips and two dependency
warnings: [test evidence](../runs/v8-output4k-full-regression-02.xml). This includes full-cap
request checks for direct, symbolic, witness and reference-repair paths; stale/config-only
smoke rejection; explicit long-prompt answer reservation; and single-process context
expansion preserving evidence and deadline. Earlier unsuccessful test invocations remain
preserved (one old output-cap assertion; one standalone pytest import-path error).

The [first real 8K fixture](../runs/v8/output4k-smoke-8k-01/smoke.json) **passed**:
direct and symbolic outputs correct, 100% structured validity, grounded-state witness
verification and restart/resume passed. Nanbeige authored a reference repair before its
working witness; this was not a manual correction. Normal pressure, zero swap growth,
peak server RSS 3,482,861,568 bytes. All requests reserve 4K generated tokens.
The [matching 12K fixture](../runs/v8/output4k-smoke-12k-01/smoke.json) also **passed**:
correct outputs, 100% structured validity, recovery verified, normal pressure, zero swap
growth and peak server RSS 3,889,741,824 bytes. The 5,571-token long prompt produced the
correct reply in 1,059 tokens while reserving the full 4,096-token output allowance.

The fresh `runs/v8/untrained-output4k-baseline-01` is now running, with a bounded first
40 stages and an independent gate audit before continuation. Source hash:
`40a79e960b8d06d3778882e9763777cfef257f7028474af9fd6b597e6eeaa1cc`.
Source/configs are archived and extracted under its `frozen/` directory; use that source
for later audits/resumes if the checkout changes. Fixture success is not ARC accuracy.
The task's `nanbeige-baseline-follow-up` heartbeat checks every 15 minutes, remains quiet
on non-actionable progress and resumes bounded chunks only after stopped-run safety and
validity checks. It pauses on completion, an impossible gate or required user action.
No training, upload or paid job is authorized by this follow-up.

```sh
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige/venv/bin/python -m arc_agent.v8_cli smoke --config configs/v8-nanbeige-output4k-baseline.yaml --context 8192 --workspace runs/v8/output4k-smoke-8k-01
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige/venv/bin/python -m arc_agent.v8_cli smoke --config configs/v8-nanbeige-output4k-baseline.yaml --context 12288 --workspace runs/v8/output4k-smoke-12k-01 --baseline-smoke runs/v8/output4k-smoke-8k-01/smoke.json
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige/venv/bin/python -m arc_agent.v8_cli pilot --config configs/v8-nanbeige-output4k-baseline.yaml --workspace runs/v8/untrained-output4k-baseline-01 --smoke-report runs/v8/output4k-smoke-12k-01/smoke.json --stop-after 40
```

These commands identify this experiment, not permission to overwrite/restart a used
workspace. Preserve source/config identities and require `--resume` for a deliberate
continuation of a stopped, still-qualifying run. Both sides of any future Phase 2 comparison
must use this same 4K generation and context policy. No paid launch or output-use clearance
is implied by the output-budget approval.

### Adapter payload validation before future local export

[`scripts/validate_nanbeige_adapter.py`](../scripts/validate_nanbeige_adapter.py) reruns the
existing training-provenance/approval audit before inspecting tensors. It requires the
official pinned base configuration and exactly 308 LoRA A/B tensors: seven target modules
across 22 physical layers, rank 16, totaling 23,969,792 parameters. The inventory was checked
against the actual pinned Nanbeige/PEFT meta-device model with two shared loops and no
base-weight allocation. Tiny numerical fixtures are not mistaken for official adapters.

The checker uses CPU-only [sliced Safetensors reads](https://huggingface.co/docs/safetensors/en/index)
to check shapes, floating-point dtypes, finite values, file hashes and potentially nonzero
A/B pairs. It rejects unsupported inference variants such as DoRA or rank-stabilized
scaling. It never deserializes optimizer/pickle files. Nonzero factors do **not** prove a
nonzero effective matrix product or a useful learned update; inference remains required.
The command reports `inference_ready: false` and `promotion_proven: false` even when all
payload checks pass. There is no real trained checkpoint to inspect yet.

All 20 new tests passed; the full suite is 718 tests, no failures/skips, two dependency
deprecation warnings. [Regression record](../runs/v8-adapter-payload-full-regression-01.xml).
The Hugging Face LLM Trainer skill's conversion guidance informed the local-export
preflight. Inspection of the pinned authors' converter also found an adapter-only GGUF
path using local base configuration without full base weights. The Nanbeige runtime shares
physical layer weights across loops and uses LoRA-aware projections. This is source-level
feasibility evidence only: no adapter GGUF was converted, loaded or tested, and no cloud
job, upload, output-use permission or trained-student promotion was produced.

## Commands

Historical reference-fixture commands are recorded below for reproducibility, **not for
restarting the stopped baseline or overwriting these workspaces**. Changed solver sources
require new workspaces and matching source-bound fixtures. Run from the repository root
with external networking denied by the local sandbox:

```sh
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige/venv/bin/python -m arc_agent.v8_cli smoke --config configs/v8-nanbeige-reference-baseline.yaml --workspace runs/v8/reference-smoke-4k-01
```

Only after the preceding current-source 4K report passes:

```sh
/usr/bin/sandbox-exec -f scripts/v4-loopback.sb .runtime/nanbeige/venv/bin/python -m arc_agent.v8_cli smoke --config configs/v8-nanbeige-reference-baseline.yaml --context 8192 --workspace runs/v8/reference-smoke-8k-01 --baseline-smoke runs/v8/reference-smoke-4k-01/smoke.json
```

For a qualifying future protocol only: after its current-source 8K report passes and known
output-cap failures have been addressed, launch a new baseline with the matching
configuration, a new unused workspace and matching smoke report. Do not reuse
`runs/v8/untrained-reference-baseline-01`. Use `--stop-after` for a checkpointed
initial subset of stages and `--resume` only once the preceding process is terminal.

To rebuild the native grammar test tool without modifying the pinned model/runtime libraries:

```sh
mkdir -p .runtime/nanbeige/grammar-tools/bin
clang++ -std=c++17 -O2 -I .runtime/nanbeige/llama.cpp/include -I .runtime/nanbeige/llama.cpp/ggml/include .runtime/nanbeige/llama.cpp/tests/test-gbnf-validator.cpp -L .runtime/nanbeige/llama.cpp/build/bin -lllama -lggml-base -Wl,-rpath,/Users/henry/workspace/arc-agi-2/.runtime/nanbeige/llama.cpp/build/bin -o .runtime/nanbeige/grammar-tools/bin/test-gbnf-validator
```

The upstream validator returns exit zero even for rejected text; the tests check its explicit
acceptance message. Its path is project-local, and it does not load model weights.

The Phase 2 promotion rule remains +3 percentage points mean exact match over three fixed
seeds, or 20% less runtime without mean accuracy loss, with equal allowances and integrity
checks. The present runner accepts the original checkpoint only; trained-artifact provenance
and the paired promotion audit remain required work before any promotion claim.
