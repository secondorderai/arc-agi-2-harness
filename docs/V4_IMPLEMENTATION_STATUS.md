# V4 implementation status — 8 September 2026

> V4 is now a preserved baseline. The active plan is
> [Astra teacher, Nanbeige symbolic student roadmap](ASTRA_NANBEIGE_PLAN.md).
> The final V4 timing run stopped on sustained memory pressure at 12,288 context tokens:
> two direct tasks and one program task completed, with no correct completed predictions.
> The second program task was interrupted. Checkpoints remain in
> `runs/v4/local-timing-diagnostic`; matching pre-switch source is archived in
> `runs/v4/source-f844979-before-astra-switch.tar.gz`. No V4 model was restarted.

**Phase 0 is incomplete. Kaggle, cloud training and later phases remain blocked.**
This is an implemented local-first foundation, not a claimed finished competition solver.

## Delivered

- A standalone Nanbeige-only V4 configuration and runner; legacy models/runs are preserved
  and are not used by V4. No monetary budget enforcement exists in V4.
- Official pinned weights downloaded locally, checksum verified, converted through BF16
  to Q4_K_M. The Q4 GGUF is approximately 2.4 GiB. Original weights and intermediate remain.
- The authors' pinned llama.cpp branch built locally with Metal. Tokenizer, architecture,
  runtime binaries and shared libraries have a hash/provenance manifest.
- Native chat rendering, explicit reasoning retention/removal, bounded deterministic tools,
  structured final-output grammars and a versioned symbolic-state foundation.
- Frozen 792/103/105 task split, fixed-hash 20-task development pilot, label-blind solver inputs,
  time/token accounting, memory/disk safeguards, exclusive run ownership and durable recovery.
- Atomic complete fallbacks for all 20 pilot tasks / 22 test inputs, two grids per input,
  including when the pilot is interrupted.
- **93 V4 correctness tests and 260 total regression tests passed**, with no failures or skips;
  formatting/lint checks passed. This includes native grammar validation; the separate live
  smoke result below validates the revised memory/cache configuration on the local fixtures.
  Current-source results are under `runs/v4/correctness-timing.json` and
  `runs/v4/full-regression-timing.xml`; the earlier reports are preserved separately.

## What actually ran

| Run | Observed outcome |
|---|---|
| Initial local smoke | Started the server, then hit sustained system memory pressure. Its shutdown timeout and telemetry shortcomings were corrected. |
| F16-cache smoke after closing apps | Correct grid, native tool call, separated reasoning and restart recovery worked. Metal logs show all 45/45 layers offloaded. |
| Bounded-reasoning F16 smoke | Passed live checks; about 3.4 GiB peak process RSS, normal pressure and no swap growth. Approximately 21 tokens/s in these fixtures—not L4 throughput. |
| First fixed 4K ARC pilot | One task completed in each mode without a usable structured response. Repairs exhausted context. The run was paused, and its memory guard also recorded sustained warning pressure. Nineteen tasks were not completed in either mode. |
| Revised structured-output / Q8-cache configuration, initial attempt | Native grammar and correctness tests passed. Live startup was refused because the Mac was already under warning-level memory pressure. |
| Revised configuration after the user freed memory | **Live 4K smoke passed** in `runs/v4/smoke-ready`: correct grid, native tool execution, separated reasoning, durable restart recovery, Metal and enforced offline operation. Peak server RSS 3,212,673,024 bytes (about 2.99 GiB), zero swap growth, pressure normal throughout. |
| Revised fixed 4K pilot | Paused at the user's request: 5/20 direct tasks and 4/20 program tasks completed; zero correct outputs so far. All four completed program runs hit context limits. Peak server RSS about 4.88 GiB, normal pressure and no swap growth. One interrupted in-flight call has unknown usage, explicitly reported. |
| Adaptive-context implementation | Starts at 8K, grows by 4K only when retained history plus answer allowance does not fit, and preserves task deadlines and cumulative memory safeguards. Unit/regression tests pass. |
| Adaptive-context revision's 4K and 8K smoke | **Both passed.** The 8K test answered correctly after a 6,122-token prompt, with native tools, reasoning separation, restart recovery, Metal and enforced offline operation verified. Peak RSS 3,670,360,064 bytes (about 3.42 GiB), pressure normal, zero swap growth. This validates 8K fixtures, not higher contexts or ARC accuracy. |
| Five-minute adaptive pilot | Paused at the user's request before implementing extended deadlines. Two direct tasks and one program task completed, with zero correct outputs so far. First-task direct and program attempts reached five-minute deadlines. Two direct responses also hit the 2,048-token generation cap without a final answer. Saved reports include interrupted calls with unknown usage. |
| Extended-deadline diagnostic implementation | Explicit 15 → 30 → 60-minute timing schedule, 128-round allowance and five-round stagnation stop, with unchanged memory/token safeguards. Records first valid response, demonstration verification, post-hoc correctness, generation limits and unresolved outcomes. It cannot unlock Phase 1. |
| Timing revision's live compatibility checks | **4K and 8K passed.** The 8K check used a 6,122-token prompt; correct output, native tools, reasoning separation, restart recovery, Metal and offline execution passed. Peak RSS 3,679,223,808 bytes (about 3.43 GiB), normal memory pressure, zero swap growth. |
| Time-to-resolution diagnostic | Started in `runs/v4/local-timing-diagnostic`. `timing.json` reports per-task milestones, unresolved outcomes and generation-token stops; `active-call.json` identifies the ongoing request. Extensions are recorded at response checkpoints. No diagnostic solve-time distribution or accuracy improvement is established yet. |

There is **no completed 20-task ARC accuracy result**. The early fallback scores in the
interrupted pilot are not a model accuracy estimate. Failures were retained, not excluded.
No full five-compaction experiment, fine-tuning, GPU portability test, ensemble experiment,
lockbox evaluation, Kaggle submission or public release has occurred.

## Latest resource check and remaining gate

After the user closed programs, the model-free preflight reported memory pressure level
**1 (normal)**, 1,522,270,208 bytes of system swap (about **1.42 GiB**) and 20,828,004,352 bytes
of free disk (about **19.40 GiB**). External networking was correctly denied. The subsequent
revised smoke passed without pressure warnings or swap growth.

The immediate startup memory blocker is cleared. The remaining Phase 0 gate requires a complete
fixed pilot, at least 95% usable structured responses in each mode, and all runtime safeguards.
A passing smoke is not a passing ARC pilot. The immediate observed program-mode bottleneck
was the configured 4K context. The user authorized larger contexts, so the next experiment
uses an adaptive window rather than discarding history. This does not guarantee correct
solutions. The harness did not close other applications or relax memory safeguards. No Kaggle
or separately billed work has started.

## Resume path

Follow the adaptive-context command sequence in the [local-first guide](V4_LOCAL_FIRST.md).
For an interrupted run of the **same** experiment, add `--resume` only after its model process
has stopped. The paused 4K run uses the previous source identity, archived in
`runs/v4/source-f89bb5f-before-context-expansion.tar.gz`. It is not silently converted into the
new adaptive run. Completed responses remain durable and are never silently called again on
same-experiment resume.

The latest requested run is the separate **time-to-resolution diagnostic** in that guide.
The preceding five-minute adaptive run and its source snapshot
`runs/v4/source-bbfb7ef-before-adaptive-deadlines.tar.gz` are preserved; the diagnostic does not
inherit its cached answers or erase its failures.

## Evidence locations

- Artifact manifest: `.runtime/nanbeige/manifest.json`
- Frozen split: `runs/v4/split.json`
- Successful earlier configuration: `runs/v4/smoke-bounded/smoke.json`
- Successful revised configuration: `runs/v4/smoke-ready/smoke.json`
- Revised pilot: `runs/v4/local-pilot-structured/report.json` and `run.sqlite3`
- Adaptive implementation verification: `runs/v4/correctness-adaptive.json`
- Current-source baseline smoke: `runs/v4/smoke-adaptive-base4k/smoke.json`
- Extended-context smoke: `runs/v4/smoke-adaptive-8k/smoke.json`
- Adaptive pilot target: `runs/v4/local-pilot-adaptive/`
- Extended-time correctness: `runs/v4/correctness-timing.json`
- Extended-time smoke: `runs/v4/smoke-timing-base4k/smoke.json`, `runs/v4/smoke-timing-8k/smoke.json`
- Time-to-resolution diagnostic target: `runs/v4/local-timing-diagnostic/`
- Interrupted pilot: `runs/v4/local-pilot/report.json`, `gate.json`, and `run.sqlite3`
- Revised configuration's failed startup: `runs/v4/smoke-structured/smoke.json`
- Model-free resource check: `runs/v4/preflight.json`
- Full phased requirements and commands: [V4 local-first guide](V4_LOCAL_FIRST.md)

The provided October 20 target and technical promotion gates remain. No alternative LLM
fallback has been introduced. No separately billed job has been launched or authorized.
