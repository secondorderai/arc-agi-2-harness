# V1 Baseline — 19 August 2026

## Environment

- Machine: Apple M1 Pro, 10 CPU cores, 16 GB unified memory.
- Model: Bonsai 27B Q1 binary weights through the official llama.cpp/Metal runtime.
- Context: 8,192 tokens.
- Generation: temperature 0.7, top-p 0.95, 512-token thinking cap.
- Dataset: official ARC-AGI-2 repository.

The preferred MLX setup was attempted first. Full Xcode 26.6 is installed, but its
Metal Toolchain is missing and `xcodebuild -downloadComponent MetalToolchain` fails due
to an incompatible `IDESimulatorFoundation`/`DVTDownloads` installation. The harness's
MLX scripts remain ready; the measured run therefore uses PrismML's supported prebuilt
llama.cpp/Metal backend with the same binary model weights.

## Measured results

| Measurement | Scope | Result |
|---|---|---:|
| Deterministic development baseline | 206 held-out training tasks, 224 outputs | 2.68% pass@2 |
| Deterministic strict accuracy | Same 206 tasks | 2.91% tasks |
| Frozen deterministic public baseline | 120 evaluation tasks, 167 outputs | 0.00% pass@2 |
| Bonsai health benchmark | Exact JSON request | 13.70 generated tokens/s |
| Model-guided integration smoke | One internal holdout task | 0/1 outputs |

The integration smoke is not a leaderboard estimate. It was repeatedly used to repair
response budgeting and parsing, so it is reported only as an end-to-end systems check.
The saved final proposal re-parses under the finalized narrow quote repair, executes in
the sandbox, and is correctly rejected after matching 0/3 demonstrations.

The public deterministic result was opened only after the learner, router, DSL, scoring,
and full 1,000-task skill artifact were frozen. No public-evaluation result was used to
change those components.

## Reproducible artifacts

- Internal holdout report: `runs/deterministic-20260819T084215Z/report.md`
- Frozen public report: `runs/deterministic-20260819T081039Z/report.md`
- Model throughput: `runs/model-benchmark-llama.json`
- Final integration smoke: `runs/local-llama-smoke-20260819T083821Z/report.md`
- Full skill provenance: `skills/generated-full/manifest.json`

## Reproduction commands

```bash
uv sync --extra dev
uv run pytest
scripts/fetch_arc_data.sh
uv run arc-agent learn \
  --data data/ARC-AGI-2/data/training \
  --output skills/generated-full \
  --validation-percent 0
scripts/setup_bonsai_llama.sh
scripts/start_bonsai_llama.sh
uv run arc-agent benchmark-model --config configs/local-llama.yaml
.venv/bin/python scripts/validate_kaggle_bundle.py dist/arc-bonsai-v1
```

A full model-guided evaluation is intentionally not labeled as complete in V1's measured
baseline: at the observed expensive-task latency, the 120-task run requires several
hours. The configured global budget and fallback behavior are ready for that unattended
run.
