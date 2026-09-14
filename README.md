# ARC-AGI-2 — Astra symbolic teacher, Nanbeige student

The research pipeline extends [V5: symbolic distillation](docs/V5_SYMBOLIC_DISTILLATION.md):
**`gpt-6-astra` through the ChatGPT subscription** is the symbolic teacher and online reference
solver; the pinned **Nanbeige4.2-3B** is the intended fine-tuning and offline-inference student.
The hypothesis is that learning grounded representations, hypotheses and revisions improves
generalisation—not simply reproducing final puzzle grids. That transfer is not yet demonstrated.
No API-key billing or alternative teacher/student fallback is enabled.

The active [Astra–Nanbeige phased plan](docs/ASTRA_NANBEIGE_PLAN.md) covers symbolic-only
distillation, training-data isolation, student promotion gates and the October 20 offline target.

## Latest experiments

Status checked **15 September 2026**; latest model experiments ran **8–9 September**.
Under the current Astra–Nanbeige roadmap, **Phase 0's teacher reference baseline is complete;
Phase 1 is incomplete**. The symbolic-data pilot passed, but the latest untrained student run
stopped on a memory safeguard. No official-weight Nanbeige fine-tuning, GPU validation,
lockbox evaluation or competition submission has run.

| Experiment | Observed result | Interpretation |
| --- | --- | --- |
| [V2 public evaluation](docs/V2_FAILURE_ANALYSIS.md) | **3/167 outputs (1.80%)**, **3/120 whole tasks (2.50%)**; 115/121 model requests timed out. | Timeout-dominated retrieval/fallback result, not a complete test of model resynthesis. |
| [V4 local Nanbeige](docs/V4_IMPLEMENTATION_STATUS.md) | Metal/native-tool/restart fixtures passed at 4K and 8K. Adaptive-context and extended-deadline trials reached 12K, then stopped on memory pressure without correct completed predictions. | Compatibility passed on fixtures; the full ARC pilot did not. Longer context/time alone did not establish improvement. |
| [V5 symbolic-data pilot](docs/V5_SYMBOLIC_DISTILLATION.md) | One training task produced one grounding-only target; **zero full-symbolic targets** accepted. | Motivated separating symbolic representations from a more expressive executable witness. |
| [V6 Astra reference solver](docs/V6_LOCAL_SCORE.md) | **20/22 outputs (90.91%)**, **19/20 whole tasks (95%)**, in **65m16s**; one timeout counted as two wrong outputs. Independent scoring and saved-response replay passed. | Completed online teacher result on the frozen development pilot—not Nanbeige or private-competition performance. |
| [V7 symbolic witness bridge](docs/V7_SYMBOLIC_BRIDGE.md) | All 20 training tasks processed; **21/21 structured-valid responses**; **39 admitted targets**: 20 grounding, 18 interpretation, one repair. Post-completion replay audit passed. | A small, verified curriculum pilot. One sealed-example failure stayed rejected; no heldout-driven repair. |
| [V8 latest Nanbeige baseline](docs/V8_STUDENT_BASELINE.md) | Direct seed 42 completed 20 tasks: **1/22 outputs (4.55%)**, **0/20 whole tasks**, 100% structured validity. The overall three-seed/two-mode run stopped incomplete after **2h04m49s**. | An untrained direct-mode result within an incomplete, non-qualifying baseline—not a trained-student gain. |

Output scores above are exact match using either of two predictions per test input. The V2
public-evaluation cohort differs from the V6/V8 pilot; these rows are not a controlled comparison.
V6 and V8 use the same frozen **20 development tasks / 22 test inputs** from the official training
corpus. The underlying split is **792 training / 103 development / 105 lockbox tasks**;
V7 uses training tasks only. Development traces never become student targets. Public-task
pretraining contamination cannot be ruled out, even with correct harness-level isolation.

### What changed in V8

- **Reference and witness repair:** valid JSON alone still produced undefined concepts or
  incorrect grounding. Nanbeige-authored reference patches and later code-only repair of
  statically rejected witnesses passed fresh 4K/8K fixtures while preserving verified state.
  The reference-patch development run nevertheless failed its structured-validity gate.
- **Compact cells:** lossless `[row,column,color]` encoding reduced aggregate payload tokens
  by **16.7% across 26 complete saved states**, with exact round trips. This is serialization,
  not reasoning compaction or an end-to-end speedup. Its development run stopped after 40
  stages: the maximum achievable symbolic validity was **94.29%**, below 95%; all nine malformed
  replies had hit the **2K output cap**, rather than the context limit.
- **Larger answer allowance:** a new, separate protocol raised generation to **4K tokens per
  call**. Offline 8K/12K fixtures passed. The baseline started at **12,288 context tokens**,
  with 15 minutes per task/mode/seed and an 11-hour run cap; the full answer allowance is
  reserved before generation. The 2K experiments remain preserved, not retroactively changed.

The latest run, `runs/v8/untrained-output4k-baseline-01`, stopped on **9 September at 09:31
Brisbane** after host-wide additional swap reached **1.099 GiB**, exceeding the unchanged
1 GiB limit. Warning-level pressure was permitted and was **not** itself the stop trigger;
host-wide telemetry does not prove Nanbeige alone caused the swap growth.

It retained **68 completed responses and one interrupted request with unknown usage**.
Direct seed 43 completed four tasks with no correct outputs; seed 44 was unstarted.
Symbolic seed 42 returned 37/40 structured-valid replies, but neither started symbolic group
produced a grounded state or terminal task. Complete two-prediction fallback files exist for
all six mode/seed groups; file completeness is not experiment completion.

The stopped score/evidence audit passed. The supplemental recovery audit correctly rejected
the new runtime failure: an earlier startup-only exception does not authorize continuation.
Raw failed safety flags and checkpoints remain intact; follow-up is paused. **Do not resume
this run under its old approval or repeat the unknown-usage call.** Details and evidence paths
are in the [V8 protocol and safety-stop record](docs/V8_STUDENT_BASELINE.md).

### Training readiness and remaining gates

The implemented training stack is **PyTorch + Transformers + TRL SFTTrainer + PEFT LoRA**, with
[pinned dependencies](configs/nanbeige-trainer-requirements.txt) and a separate
[CUDA compatibility/SFT runner](scripts/train_nanbeige_symbolic.py). The local CPU probe checked
completion-only data-loader masks for all **39 examples**, **154 unique LoRA attachments across
22 shared physical layers**, and adapter updates/save/reload/exact resume on a **tiny random
Nanbeige architecture fixture**. It did not load or train the official 3B weights. All 39
compact native-format examples fit 8K without truncation; they are not 39 independent tasks.

The supervision targets are explicit symbolic state, not final grids, witness code or opaque
provider reasoning. Executable witnesses check consequences; they do not prove every prose
definition or guarantee generalisation. Trained-checkpoint provenance and adapter-payload
checks exist, but trained-model export and the separate candidate evaluation path remain unfinished.

Before advancing:

1. Agree on a resource-safe replacement student baseline; preserve failed runs and require a
   complete comparison with the configured validity and memory gates intact.
2. Resolve the documented [Astra output-use gate](docs/ASTRA_OUTPUT_USE_REVIEW.md) before further
   teacher collection, dataset upload or real student training. Existing training drafts are
   unsealed and tied to failed baselines; they cannot be silently rebound or treated as approval.
3. Validate the official checkpoint on approved GPU hardware, then train and compare the
   actual student under the same frozen solver protocol. Separately billed jobs need approval.
   Kaggle portability, controlled retention/compaction ablations and lockbox evaluation remain
   unproven; teacher accuracy and local fixtures do not establish L4 throughput or eligibility.

Current regression verification: **794 tests passed** on 15 September, with two dependency
deprecation warnings. Solver/test lint passes; the historical V2 report generator retains
formatting warnings. These are harness checks, not model accuracy or full-size training evidence.
Run the suite in the prepared local environment with:

```sh
.runtime/nanbeige/venv/bin/python -m pytest
```

Some tests require the pinned local tokenizer and llama.cpp grammar tools; a bare clone is
not sufficient for the full suite. See the [local setup guide](docs/V4_LOCAL_FIRST.md) and
[V8 verification notes](docs/V8_STUDENT_BASELINE.md). Raw `runs/`, `.runtime/`, weights and local
account screenshots are excluded from Git; linked documentation records their local evidence
paths. No experiment was restarted to prepare this README.

## Preserved V4 baseline

The prior local-first iteration is documented in [V4: Nanbeige-only, local-first](docs/V4_LOCAL_FIRST.md).
It uses only the pinned official Nanbeige4.2-3B checkpoint, an isolated Metal runtime,
label-blind evaluation, durable checkpoints and explicit promotion gates. There are no
monetary budget controls. Kaggle and cloud training remain gated on local validation;
separately billed services require approval.

To reproduce V4, use `python -m arc_agent.v4_cli` in the isolated Nanbeige environment, not the legacy
solver commands below. Existing models and experiments are preserved but excluded from V4.

## Historical V1–V3 implementation

The architecture, model setup and commands below describe preserved historical experiments,
not the active Astra–Nanbeige student pipeline. Bonsai/Qwen artifacts are not active fallbacks.

A clean-holdout, verifier-guided ARC-AGI-2 solver designed for a small local reasoning
model. The harness combines deterministic program search, learned Markdown skills,
three adaptive difficulty levels, LLM-guided program synthesis, exact execution
feedback, and two-attempt Kaggle output generation.

The target of **50%+** remains the research north star. V1 reports the score it actually
achieves; it does not tune against the public evaluation set.

See [the V1 design](docs/V1_DESIGN.md), [measured baseline](docs/BASELINE.md), and the
[V3 typed-DSL pilot](docs/V3_DESIGN.md).

## Architecture

1. The typed grid DSL searches cheap transformations first.
2. Training tasks solved by compact programs become provenance-bearing skill cards.
3. A structural router assigns Level 1 (primitive), Level 2 (compositional), or Level 3
   (contextual/symbolic), with automatic escalation after verifier failure.
4. Bonsai proposes DSL or tightly sandboxed Python programs.
5. Only programs that reproduce every demonstration exactly are marked verified.
6. Two distinct predictions are selected and written in Kaggle's `submission.json` format.

The integer matrices are authoritative. PNG rendering is available for diagnostics, but
the selected 1-bit MLX model receives text-only exact grids because its current MLX
backend does not support image input.

## Quick start

Requires Python 3.11 and `uv`.

```bash
uv sync --extra dev
uv run pytest
scripts/fetch_arc_data.sh
uv run arc-agent learn \
  --data data/ARC-AGI-2/data/training \
  --output skills/generated-dev \
  --validation-percent 20
uv run arc-agent evaluate \
  --data data/ARC-AGI-2/data/evaluation \
  --config configs/deterministic.yaml \
  --skills skills/generated-dev
```

The development `learn` command holds out a deterministic 20% of training IDs. For the
frozen public-evaluation run, rebuild the same finalized learner over all 1,000 training
tasks into a new directory and retain its manifest:

```bash
uv run arc-agent learn \
  --data data/ARC-AGI-2/data/training \
  --output skills/generated-full \
  --validation-percent 0
```

## Local Bonsai 27B MLX

The model runtime is deliberately separate from the harness environment. It is roughly
5.13 GB on disk and fits the detected 16 GB M1 Pro at the configured 8K context.

```bash
scripts/setup_bonsai_mlx.sh
scripts/start_bonsai_mlx.sh
uv run arc-agent benchmark-model --config configs/local-mlx.yaml
uv run arc-agent evaluate \
  --data data/ARC-AGI-2/data/evaluation \
  --config configs/local-mlx.yaml \
  --skills skills/generated-full \
  --public-eval data/ARC-AGI-2/data/evaluation
```

If the installed Xcode/Metal toolchain cannot build PrismML's MLX fork, the same
binary weights are supported through the official prebuilt llama.cpp/Metal runtime:

```bash
scripts/setup_bonsai_llama.sh
scripts/start_bonsai_llama.sh
uv run arc-agent benchmark-model --config configs/local-llama.yaml
```

## ARC-specialized Qwen3-4B TTT

The experimental small-model path uses `mlx-community/Qwen3-4B-ARC-MLX-4bit`,
an optional converted DiARC preference adapter, per-puzzle LoRA training, exact
grid grammar, augmented candidate scoring, and bounded DFS. It requires an
Apple-silicon environment with MLX-LM 0.31.x, NumPy 2.x, and Safetensors 0.5+.
Convert the adapter with:

```bash
uv run python scripts/convert_diarc_adapter.py \
  --source .runtime/models/diarc-adapters/qwen3-4b/arc-agi-2 \
  --output .runtime/models/diarc-qwen3-arc2-mlx-adapter
```

Fuse the converted adapter with `mlx_lm fuse`, then evaluate with
`configs/local-mlx-arc-diarc-scored.yaml`. The bounded-search experiment is
`configs/local-mlx-arc-diarc-dfs.yaml`. These are research configurations, not
claimed 50% solutions; see [the measured experiment analysis](docs/EXPERIMENTS_2026-08-20.md).

## Hybrid object world model + symbolic executor

The experimental hybrid path combines a lossless object/relation scene model,
demonstration transition inference, a typed symbolic executor, exact verification,
and an optional compact recurrent MLX world model. Run the safe symbolic-only
configuration with:

```bash
uv run arc-agent evaluate \
  --data data/ARC-AGI-2/data/evaluation \
  --config configs/hybrid-symbolic-fast.yaml
```

Train and benchmark a leakage-resistant local MLX prior with:

```bash
uv run python scripts/run_hybrid_experiment.py \
  --data data/ARC-AGI-2/data/training \
  --public-eval-data data/ARC-AGI-2/data/evaluation \
  --solver-config configs/hybrid-symbolic-fast.yaml \
  --output runs/hybrid-mlx \
  --backend mlx \
  --max-learn-tasks 100 \
  --max-holdout-tasks 20
```

V1 is implemented but remains at **2.40% public pass@2 / 2.50% strict**, with
no newly verified public programs. Treat it as research infrastructure, not a
50% solution; see [the full result and rejected hypotheses](docs/EXPERIMENTS_2026-08-21.md).

The solver budget is 41,400 seconds (11.5 hours), leaving 30 minutes inside Kaggle's
12-hour limit for startup and artifact writing. V1 caps thinking at 512 tokens, allows
up to 1,024 total completion tokens, and uses one Level 2 call plus at most two Level 3
calls so the measured local throughput fits the wall-clock envelope.

## Kaggle offline bundle

Kaggle Linux cannot run MLX, so V1 uses the same Bonsai 27B binary weights in GGUF form
with llama.cpp/CUDA. Download the public 1-bit GGUF and a compatible `llama-server`, then:

```bash
uv run python scripts/prepare_kaggle_bundle.py \
  --gguf /path/to/Bonsai-27B-Q1_0.gguf \
  --llama-server /path/to/llama-server \
  --skills skills/generated-full
```

Upload `dist/arc-bonsai-v1` as a private Kaggle dataset, attach it to
`kaggle/arc_agi_2_v1.ipynb` together with the competition data, keep internet disabled,
and run the notebook. The runner detects one to four NVIDIA GPUs, starts one local model
server per GPU, covers every task, and writes `/kaggle/working/submission.json` even when
the solver must fall back to deterministic attempts.

## Commands

- `arc-agent learn`: mine programs and generate Markdown/JSON skills.
- `arc-agent solve`: solve unlabelled challenge tasks.
- `arc-agent evaluate`: solve labelled tasks and report official pass@2 plus strict task accuracy.
- `arc-agent validate-submission`: validate task coverage, attempts, dimensions, and colors.
- `arc-agent benchmark-model`: smoke-test the local OpenAI-compatible model server.
- `arc-agent v2-auth-login`: sign a V2 workspace into Codex with ChatGPT subscription access.
- `arc-agent v2-auth-status`: inspect that workspace-isolated subscription login.
- `arc-agent v2-build-bank`: sequentially synthesize and checkpoint the Luna xhigh program bank.
- `arc-agent v2-evaluate`: resume-safe blind evaluation using direct reuse plus Luna resynthesis.
- `arc-agent v2-status`: inspect active task, quota pauses, usage, and the exact resume command.
- `arc-agent v3-auth-login`: sign the isolated V3 pilot workspace into Codex.
- `arc-agent v3-build-pilot`: synthesize or resume the ten typed JSON signatures with Sol xhigh.
- `arc-agent v3-evaluate`: run the frozen signature matcher and bounded DSL search offline.
- `arc-agent v3-status`: inspect pilot/evaluation checkpoints and the exact resume command.

V2 defaults to ChatGPT subscription access through the Codex App Server, so it does not require
`OPENAI_API_KEY`. Authenticate the run workspace once with `v2-auth-login`, then start the bank
build. A subscription quota exhaustion pauses without advancing the active task; rerun the same
command with `--resume` after the account resets. See [the V2 design and
runbook](docs/V2_DESIGN.md).

Run artifacts contain dataset/config hashes, per-task tiers, candidate programs, verifier
outcomes, time, calls, token usage, `submission.json`, and Markdown/HTML reports.

## Evaluation integrity

The 1,000 official training tasks may be used to acquire priors. The 120-task public
evaluation set is held out from skill learning and prompt/router tuning. Skill manifests
record source hashes and task provenance, and `--public-eval` rejects overlapping IDs.
