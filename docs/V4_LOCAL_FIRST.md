# V4: Nanbeige-only, local-first implementation

> Superseded as the active plan on September 8 by
> [Astra teacher, Nanbeige symbolic student roadmap](ASTRA_NANBEIGE_PLAN.md).
> [V5](V5_SYMBOLIC_DISTILLATION.md) records the initial teacher/student implementation.
> V4's original lineage checks remain unchanged for baseline reproduction.

Status: V4 foundation implemented; Phase 0 is **not passed**. The revised Q8-cache / structured-output
configuration passed the local smoke test after the user freed memory. The 4K pilot was paused
at the user's request because program histories filled the context window. An adaptive-context
runner starts at 8K and grows in 4K steps. The user subsequently authorized an explicitly
diagnostic timing run with 15/30/60-minute allowances; memory safeguards remain unchanged. See
[implementation status and evidence](V4_IMPLEMENTATION_STATUS.md). Later phases are gated
research work, not completed experiments. Submission-ready target: **20 October 2026**.

## Non-negotiable boundaries

- Every solver LLM role and every model-generated training trajectory uses
  `Nanbeige/Nanbeige4.2-3B`. V4 has no remote-provider adapter, alternate-model fallback,
  inherited legacy defaults, external teacher or LLM judge.
- Legacy V1–V3 code, models, adapters, cached predictions and runs remain untouched. They are
  **not** V4 training data, initialization, predictions or fallback sources.
- Deterministic grid tools, a bounded executable DSL and non-LLM ranking are permitted.
- There are no financial ceilings, phase spending allowances, financial reserves or
  cost-based stopping rules. Time, token, memory, disk, correctness and quality limits remain.
- Local Mac verification precedes Kaggle or cloud training. Cloud training requires separate
  approval. Neither public release nor a competition submission is authorized by running V4.

## Pinned artifacts and reproducibility

| Artifact | Pin / specification |
|---|---|
| Official model | `3384e426066d1a49c3aea90a7190b81260a6533f` |
| Authors' runtime | `Nanbeige/llama.cpp`, branch `nanbeige42` |
| Runtime revision | `c6640a1c0cf7b38df342b67021a3900b04d092e7` |
| Local model | Official safetensors → BF16 GGUF → Q4_K_M GGUF |
| Architecture | Nanbeige; 22 shared layers, two loops; complete 166,144-token vocabulary |
| Local build | Apple Metal enabled; OpenMP, CUDA, web UI and OpenSSL disabled |
| Data checkout inspected | `f3283f727488ad98fe575ea6a5ac981e4a188e49` |

`scripts/setup_nanbeige.py` verifies both official weight checksums and the runtime commit,
checks the GGUF architecture, and hashes the final model, tokenizer files, launcher and shared
libraries. Conversion does not execute the model repository's custom Python inference code.
The manifest also records the installed environment and license source files. These records
do not by themselves establish competition eligibility; that audit belongs to Phase 1.

Everything large lives under `.runtime/nanbeige/`. The installer uses a project-local cache
and isolated environment, never deletes existing models, and refuses an operation that would
consume the last 10 GiB of free disk space. Partial conversions are preserved for inspection,
not silently overwritten. The initial OpenMP linker failure was corrected by disabling
OpenMP and unnecessary UI assets; both native build targets then completed successfully.

Fresh setup, without changing the project's existing Python environment:

```sh
uv --cache-dir .runtime/nanbeige/uv-cache run --no-project --with-editable '.[v4]' \
  python scripts/setup_nanbeige.py --stage all
```

Repeat a specific stage using the isolated environment:

```sh
.runtime/nanbeige/venv/bin/python scripts/setup_nanbeige.py --stage manifest
```

Stages are `setup`, `build`, `convert`, and `manifest`. A moved branch, changed tracked runtime
source, incompatible adapter or artifact hash mismatch is an error, never a fallback trigger.

## Phase 0 commands

All commands below run from the repository root. The standalone `arc-v4` entry point is
separate from the legacy `arc-agent` command. The module form also works without reinstalling
the console launcher.

```sh
.runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli freeze-split
.runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli verify

sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli preflight

sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli smoke \
  --workspace runs/v4/smoke-ready

sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli pilot \
  --smoke-report runs/v4/smoke-ready/smoke.json \
  --workspace runs/v4/local-pilot-structured
```

Use `--resume` with the **same** workspace, config, source, manifest and split. A changed
experiment needs a new workspace; do not edit old identities or delete checkpoints to force
reuse. The retained failed compatibility runs are separate from the bounded-reasoning run.

The 1,000 official training tasks freeze into **792 training / 103 development / 105 lockbox**
tasks. Membership and order depend on a stable task-ID hash, with source-content hashes checked
on every pilot. The first 20 development IDs are selected in that order. No accuracy-based task
selection occurs. Derived examples must follow their source task into the same split. The
public evaluation set is not used by these commands. Existing model pretraining contamination
is not ruled out by a clean harness split.

The initial configuration uses one server, one slot, one request at a time, 4K context,
at most 2,048 generated tokens per call and 300 seconds per task **per mode**. Up to eight
sequential calls fit within that task deadline. Native reasoning is limited to the smaller
of 1,024 tokens or half the available generation allowance, reserving room for the final answer.
This is a token-allocation control, not a monetary budget. A native sampler is initialized
from the actual `<think>` generation prefix; otherwise a reasoning cap may not activate.

The latest runtime settings use Q8_0 keys/values in the inference cache, explicit flash
attention, batch 128 and microbatch 64. Model weights remain Q4_K_M. This lower-memory revision
passed the live 4K smoke test in `runs/v4/smoke-ready`: correct grid output, native tool execution,
reasoning separation, restart recovery, Metal offload and enforced offline operation. Peak server
RSS was 3,212,673,024 bytes (about 2.99 GiB), with normal pressure and zero swap growth. This
certifies those local fixtures, not the complete ARC pilot or L4 throughput.

Final-output grammars activate after `</think>` to prevent an empty final answer from ending
an otherwise unfinished response. They constrain only formatting, grid color/dimension bounds,
approved tools and bounded programs—not puzzle answers. Reasoning remains unconstrained until
its token allowance is reached. The pinned native grammar validator tests acceptance/rejection.
Large program-result grids use the same lossless digit-row encoding as the input; complete
tool results are also preserved as artifacts. No model-generated evidence is silently removed.

Direct-grid and program-generation modes use the exact same checkpoint. Direct responses may
use lossless digit rows and `"attempt_2":"same_as_1"`; the writer always expands these to two
ordinary integer grids. Program mode exposes `read_grid`, revisable `inspect_scene`, and
`run_program`. Only bounded, allowlisted DSL programs execute, in a timed child process—never
arbitrary generated Python, shell commands, or undefined symbolic concepts.

The network policy permits loopback only. A literal external-IP connection must fail with an
OS permission error before the model starts; an ordinary DNS failure or timeout is insufficient
proof. There is no model download or external call during inference. The server binds to
`127.0.0.1`, uses one slot, restricts browser origins, and has context shifting disabled.

Memory checks use the native macOS process API because the privileged `ps` executable cannot
run inside the network sandbox. Initial warning/critical pressure prevents startup. Three
successive warning/critical samples, loss of telemetry, more than 1 GiB swap growth, or less
than 10 GiB free disk stops only the owned server. Graceful shutdown has a two-second limit,
followed by termination of that same child process. The harness never closes other apps.

After a successful 4K memory check, an explicitly separate 8K local configuration may be
evaluated. Local token rates are **not** estimates of L4 throughput.

### User-authorized adaptive context experiment

The user subsequently authorized increasing context beyond 8K when it runs out of room.
`configs/v4-nanbeige-adaptive-local.yaml` starts at 8,192 tokens. With `--adaptive-context`,
the runner adds 4,096 tokens when a complete retained prompt plus the full 2,048-token answer
allowance and eight-token margin no longer fits. Only one Nanbeige process runs at a time.
Reloads, tokenization and retrieval remain inside the same 300-second task/mode allowance;
expansion never resets the clock, round allowance or cumulative swap-growth baseline.

History, tool results and raw artifacts are unchanged. This is **not compaction**. Each context
transition is recorded in `adaptive-state.json`, and each runtime has separate memory and
server logs under `runtime-segments/`. Completed responses are recovered from their exact
saved requests, even if the active context has since grown. No old 4K prediction is imported
into the new adaptive experiment. No label or accuracy result controls context expansion.

The pinned model metadata's 262,144-token ceiling is an absolute upper bound, **not a claim
that such a context fits a 16 GB Mac**. Real memory pressure, swap growth, disk reserve and
task deadlines can stop expansion much earlier. The GPU phase's existing context and runtime
constraints are unchanged. If a task still cannot proceed, its failure and fallback remain
in the results; the controller does not increase context in response to wrong answers alone.

The new implementation requires current-source correctness and 4K smoke checks, then an 8K
smoke including a prompt longer than 4K. Run in order, only after each prerequisite passes:

```sh
.runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli verify \
  --output runs/v4/correctness-adaptive.json

sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli smoke \
  --workspace runs/v4/smoke-adaptive-base4k

sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli smoke \
  --config configs/v4-nanbeige-adaptive-local.yaml \
  --baseline-smoke runs/v4/smoke-adaptive-base4k/smoke.json \
  --workspace runs/v4/smoke-adaptive-8k

sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli pilot \
  --config configs/v4-nanbeige-adaptive-local.yaml \
  --checks runs/v4/correctness-adaptive.json \
  --smoke-report runs/v4/smoke-adaptive-8k/smoke.json \
  --workspace runs/v4/local-pilot-adaptive --adaptive-context
```

Add `--resume` for an existing run with identical source, configuration and artifacts; never
launch alongside an existing model process. Older sources were preserved before this change
in `runs/v4/source-f89bb5f-before-context-expansion.tar.gz`. Their original runs remain separate
and cannot be resumed under the changed source identity without restoring that source.

## User-authorized time-to-resolution diagnostic

The user requested longer, automatically extending deadlines to learn how long tasks take.
`configs/v4-nanbeige-timing-local.yaml` is a **separate diagnostic**, not an equal-runtime
baseline or a Kaggle-compliance result. Its schedule per task and per solving mode is
**15 → 30 → 60 minutes**. An in-flight request may cross a soft timing milestone; the extension
and its crossing time are persisted at the next response checkpoint, without restarting that
request. The absolute one-hour task ceiling remains enforced, including context reloads.
There is also a 15-minute maximum per request to prevent a stalled transport from waiting
indefinitely. These are technical time limits, not spending limits.

Context still grows in 4K increments when needed. The diagnostic allows up to 128 rounds
instead of eight, but stops after five consecutive rounds without a novel usable output.
Changing reasoning prose or model confidence is not counted as progress. Completed responses
and their original charged time survive restart; a time extension never resets elapsed time.
Memory pressure, cumulative swap growth, disk reserve, model lineage, and loopback-only
networking are unchanged. The 2,048-token per-response cap is unchanged and reported separately
from task deadlines. Prior 8K direct failures reached that token cap without a final answer;
more wall-clock time alone does not remove that bottleneck.

`timing.json` records each task/mode's elapsed solve time, allowance extensions, round token
counts and generation stop types, plus these distinct milestones:

- First usable response (not necessarily a correct grid or a program).
- First program verified against **all supplied demonstrations** (not an oracle for test outputs).
- First correctly selected two-attempt predictions covering every test input, scored afterward.
- Explicit unresolved reasons: time cap, stalled output, repeated generation-token caps,
  context ceiling, request failure, interruption or round cap.

Unresolved observations remain in the report as censored outcomes. The median successful time
is labeled as conditional on observed correct tasks; it does not estimate a solve time for
unsolved tasks. No test label drives continuation, deadline growth, context expansion or stop
decisions. Direct mode still stops after its first usable prediction; it does not receive a
hidden correctness signal to retry a wrong answer. Program mode stops on demonstration fit.
One-off model startup is included in the run wall time, not assigned to an individual puzzle.

Reports refresh at every response checkpoint. `active-call.json` identifies the in-flight
round, its start time, prior elapsed solve time and hard/request limits. `gate.json` always
rejects promotion for a diagnostic run, even if every task happens to be correct. A later
frozen, fixed-allowance evaluation and full offline rehearsal are still required.

After current-source correctness, 4K smoke, and extended-prompt 8K smoke pass:

```sh
sandbox-exec -f scripts/v4-loopback.sb \
  .runtime/nanbeige/venv/bin/python -m arc_agent.v4_cli pilot \
  --config configs/v4-nanbeige-timing-local.yaml \
  --checks runs/v4/correctness-timing.json \
  --smoke-report runs/v4/smoke-timing-8k/smoke.json \
  --workspace runs/v4/local-timing-diagnostic --adaptive-context
```

Use `--resume` only for this same diagnostic identity after its process has stopped. Earlier
five-minute runs remain untouched. Their source snapshot is preserved in
`runs/v4/source-bbfb7ef-before-adaptive-deadlines.tar.gz`.

## Durable state, isolation and reports

`ExperimentRun` owns an exclusive run lock and SQLite WAL journal. It records a request before
calling the model and the full returned response before parsing it. Restart tests interrupt
exactly between response persistence and interpretation, then recover without another model
call. In-flight requests without a complete returned response cannot recover the model's
internal generation state; they are reported as interrupted, not silently replayed.

`SymbolicTaskState` holds versioned definitions, grounded observations, competing hypotheses,
evidence, counterexamples, unresolved choices and immutable artifact references. Definitions
cannot have missing dependencies or cycles. Grounded cells are checked against the raw grids.
It is an initial interface, **not** a completed Phase 2 revisable-scene/compaction experiment.

The solver receives test inputs but never their output labels. Native templates are rendered
and checked against the official Hugging Face tokenizer. The strip-history ablation explicitly
removes assistant reasoning while preserving actions and tool results; it does not rely only
on `preserve_thinking`. Retained text is not opaque provider state or a saved KV cache.

Each pilot writes:

- `submission-direct.json` and `submission-program.json`: complete, ordered predictions,
  initially deterministic input-copy fallbacks, replaced atomically as tasks finish.
- `run.sqlite3` and `artifacts/`: exact request/response/checkpoint history and authoritative grids.
- `server.log`, `server-process.json`, `memory-baseline.json`, `memory.jsonl`: runtime evidence.
- `report.json`: exact-match at two attempts, strict task accuracy, candidate coverage,
  valid-response rate, task errors, known token counts, solve time and memory evidence.
  Token totals are marked as lower bounds when a failed in-flight call has unknown usage.
- `gate.json`: full 20-task completion in both modes, at least 95% usable structured responses
  in each mode, current-source correctness tests, live native-tool/restart checks, Metal and
  enforced-offline evidence, and no memory guard failure. ARC accuracy is reported separately.

Correctness tests include malformed grids/tools, model lineage rejection, label isolation,
frozen splits, missing/corrupt artifacts, definition integrity, five successive **checkpoint
serialization** roundtrips, context overflow, timeout accounting, memory shutdown, atomic
completeness and crash recovery. Five **model compactions** and cross-GPU formatting remain
Phase 2 / Phase 1 tests respectively; checkpoint roundtrips are not substitutes for them.

## Remaining gated phases

| Phase / target dates | Work still to perform | Promotion gate |
|---|---|---|
| 0 · Sep 8–15 | Finish native compatibility and the complete fixed local pilot; evaluate 8K only after memory checks | Current-source correctness, ≥95% structured responses, no OOM or sustained pressure |
| 1 · Sep 16–18 | Offline single-L4 bundle with the same runtime; GGUF vs BF16; validate authors' vLLM branch; 8K then 16K; licenses/provenance | Complete reproducible outputs and measured GPU memory/runtime; retain GGUF if acceleration fails |
| 2 · Sep 19–25 | Direct programs vs fixed scenes vs revisable working models; strip/retain/prose/structured compaction; 16K cap, 8K/12K triggers, ~1K summary; artifact retrieval and five compactions | +3 percentage points mean development exact match over three fixed seeds, or ≥20% faster without mean accuracy loss; state integrity passes |
| 3 · Sep 26–Oct 7 | Up to 3,000 executable-verified deterministic/official/Nanbeige episodes; task/family isolation; one balanced multitask LoRA; HF Jobs only with approval and durable private checkpoints | Useful baseline first, training compatibility/save/reload/resume checks, then Phase 2 promotion rule |
| 4 · Oct 8–14 | Nanbeige grid/program plus deterministic candidates; ancestry/deduplication; training-only out-of-fold non-LLM ranker; 4-grid / 2+2 / 1+3 worker allocations on L4×4 | Better than the best individual strategy at equal runtime |
| 5 · Oct 15–20 | Freeze; local regression of each final artifact; one baseline/candidate lockbox evaluation; full offline L4×4 rehearsal | Complete valid submission; approved lineage; no network dependency; 10h active solving / 11h completion target under the specified 12h limit |

Promote representation, retention and compaction independently. These effect thresholds are
engineering decisions, not statistical significance claims. Failed experiments retain the
best validated Nanbeige configuration; they never switch to another LLM.

Phase 3 initial settings remain rank 16, learning rate `1e-4`, maximum two epochs, 8K sequences,
microbatch one, effective batch 16. Preserve the tokenizer and native chat format; verify
loss masks, shared-layer adapter attachment, optimizer/scheduler/RNG/data-cursor recovery and
restructure overlength examples instead of deleting necessary evidence. Exclude other models'
traces, cached answers and judgments. Preserve failed attempts as context; supervise only
executable-verified useful turns. Reserve generator families and withheld demonstrations for
evaluation. Full pretraining, online RL and tokenizer replacement are out of scope.

Phase 4 must provide every task initial coverage before refinement, bound queues, enforce a
global deadline and keep complete predictions durable. Phase 5 evaluates the frozen baseline
and candidate on the sealed lockbox once, falls back on regression, and never tunes against
revealed lockbox results. Current competition resource availability and eligibility must be
verified before the Kaggle phase; L4×4 and the runtime limits here are the supplied plan's
requirements, not a new claim about available allocations.

## Billing and source notes

OpenAI Docs was used to preserve the distinction between included ChatGPT/Codex allowance and
separately metered API use: [official pricing documentation](https://learn.chatgpt.com/docs/pricing).
No OpenAI API is part of the V4 solver. HF Jobs remains a separately approved external service;
see [HF Jobs overview](https://huggingface.co/docs/hub/en/jobs-overview). No job was launched by
this local-first implementation. The Hugging Face CLI and LLM Trainer skills informed pinned
artifact retrieval, full-vocabulary conversion and training-compatibility gates.

Model/runtime sources: [official Nanbeige checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
and [authors' pinned runtime](https://github.com/Nanbeige/llama.cpp/tree/c6640a1c0cf7b38df342b67021a3900b04d092e7).
