# ARC-AGI-2 Agent Harness V1 Design

## Objective

V1 is a small-model research harness, not a claim of 50% accuracy. Its job is to make
progress toward that target measurable and repeatable: learn only from the official
training set, generate executable rules, reject rules that do not reproduce every
demonstration, preserve two distinct attempts, and finish within Kaggle's wall-clock
envelope.

## Frozen evaluation protocol

- Development: a deterministic 80/20 split of the 1,000 official training tasks.
- Skill acquisition: verified programs and general strategy cards only.
- Final skill build: all 1,000 training tasks, with task IDs and source hashes recorded.
- Public evaluation: all 120 evaluation tasks remain excluded from learning, prompt
  tuning, router tuning, and verifier design.
- Leakage gate: evaluation aborts if any evaluation task ID appears in skill provenance.

## Solve pipeline

1. Load and validate exact integer grids.
2. Search the deterministic DSL for compact programs.
3. Extract structural features and assign one of three difficulty levels.
4. Retrieve only the most relevant training-derived skill cards.
5. Ask Bonsai for one concise DSL or sandboxed Python program.
6. Execute the same program against every training pair.
7. Feed exact shape/cell/error diagnostics into the next bounded call.
8. Rank verified candidates first and preserve behaviorally distinct predictions.
9. Fill any missing attempts with deterministic identity/zero fallbacks.
10. Validate and write Kaggle's two-attempt submission format.

## Difficulty levels

| Level | Intended tasks | Main mechanism | Model budget |
|---|---|---|---|
| 1 | Primitive geometry, recoloring, crop, scale, tile, padding | Deterministic typed DSL | No model call |
| 2 | Low-complexity compositions | Skills plus one generated program | One call, 180-second cap |
| 3 | Contextual rules, symbolic roles, object interaction | Python synthesis plus verifier refinement | Up to two calls, 360-second cap |

Failure at Level 2 promotes the task to Level 3 with verifier feedback attached. Each
model call returns one candidate so the answer is not truncated; diversity comes from
separate feedback-conditioned calls.

## Execution and safety

- DSL operations have a strict argument whitelist and validate every output grid.
- Generated Python accepts one grid and runs in an isolated interpreter with no imports,
  files, network, process access, or external state.
- CPU, memory, and wall-clock limits apply to generated code.
- Model, parse, server, and candidate failures become recorded diagnostics.
- Kaggle server failure falls back to deterministic solving and still writes a valid
  submission covering every task.

## Runtime design

- Preferred local path: Bonsai 27B MLX 1-bit, text-only exact grids.
- Validated local fallback: the same Bonsai 27B binary weights in Q1 GGUF through
  llama.cpp/Metal.
- Kaggle path: Q1 GGUF through llama.cpp/CUDA, one server per detected GPU.
- Completion envelope: 512 thinking tokens, 1,024 total tokens, 11.5-hour global solver
  budget, leaving 30 minutes for startup and artifacts.

The input matrices remain authoritative. PNG rendering is diagnostic only in V1; the
selected binary MLX backend cannot currently serve the model's vision tower.

## V1 acceptance status

- Official ARC-AGI-2 data loader and Kaggle challenge loader: complete.
- Training-only Markdown skills with provenance: complete.
- Three-level router and promotion: complete.
- Typed program search and exact verifier: complete.
- Sandboxed Python synthesis: complete.
- Bonsai OpenAI-compatible adapter and measured local runtime: complete.
- Two-attempt scoring, reports, hashes, and submission validation: complete.
- Offline Kaggle bundle/notebook runner: complete.
- Automated tests: 21 passing.

## Known ceiling and V2 research priorities

The V1 deterministic library solves only 25 of 1,000 training tasks, and its public
evaluation baseline is 0%. Reaching 50% will require a substantially richer object DSL
and stronger inference, not prompt wording alone. The highest-value next work is:

1. Add object translation, alignment, painting, ray/line drawing, counting, repetition,
   conditional selection, and relational placement primitives.
2. Synthesize typed Python from object graphs rather than raw matrices alone.
3. Generate task-specific invariants and verifier properties before proposing a rule.
4. Distill successful verified traces into reusable parameterized programs.
5. Evaluate a stronger teacher/ensemble while retaining Bonsai as the efficient executor.

