# ARC-AGI-2 Bonsai Agent Harness V1

A clean-holdout, verifier-guided ARC-AGI-2 solver designed for a small local reasoning
model. The harness combines deterministic program search, learned Markdown skills,
three adaptive difficulty levels, LLM-guided program synthesis, exact execution
feedback, and two-attempt Kaggle output generation.

The target of **50%+** remains the research north star. V1 reports the score it actually
achieves; it does not tune against the public evaluation set.

See [the V1 design](docs/V1_DESIGN.md) and [measured baseline](docs/BASELINE.md).

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

Run artifacts contain dataset/config hashes, per-task tiers, candidate programs, verifier
outcomes, time, calls, token usage, `submission.json`, and Markdown/HTML reports.

## Evaluation integrity

The 1,000 official training tasks may be used to acquire priors. The 120-task public
evaluation set is held out from skill learning and prompt/router tuning. Skill manifests
record source hashes and task provenance, and `--public-eval` rejects overlapping IDs.
