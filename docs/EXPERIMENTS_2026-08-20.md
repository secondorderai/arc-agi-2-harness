# ARC-AGI-2 V1 experiment analysis — 20 August 2026

## Measured status

The 50% public-evaluation target has **not** been reached. The strongest frozen
full-public result remains the deterministic ensemble at **2.40% output pass@2**
and **2.50% strict task accuracy** (`runs/deterministic-20260819T152533Z`).
The corresponding 1,000-task training score is **8.09% output pass@2** and
**8.00% strict task accuracy** (`runs/deterministic-20260819T152614Z`).

The ARC-specialized Qwen3-4B + DiARC + per-puzzle TTT path scored **1/4 public
test outputs (25%)** on the only two public tasks whose grids fit its <=10x10
training regime, but solved neither whole task. This is a diagnostic subset,
not a 25% estimate for the full 120-task set.

## What improved

- Exact-shape grammar constrained decoding fixed malformed grid serialization.
- Seeded LoRA initialization made TTT comparisons reproducible.
- Early stopping and augmented teacher-forced NLL reduced wasted inference and
  made candidate ranking evidence-based.
- New general DSL primitives raised deterministic training accuracy from 7.20%
  to 8.00% strict without losing previously solved tasks.
- Bounded probability DFS generated 32 distinct valid candidates on the frozen
  public failure `28a6681f`, proving cache rewind and pruning work correctly.
- Iterative Bonsai repair now preserves reasoning, deduplicates feedback, and
  uses non-degenerate repair sampling.

## Rejected default hypotheses

### Full BF16 DiARC + TTT

The DiARC paper used BF16 Qwen3 weights, whereas the deployable local model is
4-bit. A dequantized 6.8 GB model was built to test lost adapter signal. It took
more than 39 minutes without finishing the first 10x10 puzzle on the 16 GB M1
Pro, versus minutes for the quantized path. It is not viable under the 12-hour
competition budget and was stopped before scoring.

### More random samples

Nine high-temperature samples still omitted the correct candidates on the two
failed public outputs. Candidate ranking cannot recover an answer that was
never generated.

### Probability-bounded DFS alone

DFS explored 843 nodes and filled its 32-candidate cap on `28a6681f`, but all
candidates were pixel-level variants of the same wrong semantic hypothesis.
The bottleneck is representation/program induction, not local token search.

### Longer Bonsai refinement as the default

Increasing reasoning from 512 to 1,024 tokens and diversifying four repair
rounds eliminated duplicate code, but still produced zero exact training-pair
matches on holdout `234bbc79`. Best raw cell accuracy rose only from 44.2% to
50.9% while latency rose from 319 to 432 seconds. Retain this only as a
selectively routed Level-3 option.

## Evidence-backed next priorities

1. Add an object-and-relation intermediate representation: typed objects,
   holes, containment, adjacency, repeated glyphs, and color-as-symbol links.
2. Search programs over that representation and verify every hypothesis on all
   demonstrations. Public failures show that direct pixels cannot express the
   required semantics efficiently.
3. Integrate a reproducible complementary small recursive model only after its
   checkpoint and claimed score can be independently verified. TOPAS-DSPL
   reports 24%, but the visible repository requires a `best_model.pt` checkpoint
   that was not available during this run.
4. Learn reusable abstractions from the 1,000 training tasks with clean
   task-level holdouts. Do not train on or import programs for the 120 public
   evaluation solutions; that would inflate the score without measuring
   generalization or predicting Kaggle private performance.
5. Calibrate routing so deterministic search handles easy tasks, quantized TTT
   handles small-grid distribution matches, and expensive program repair is
   reserved for high-value semantic tasks.

## External references

- DiARC paper: https://arxiv.org/html/2606.26530
- DiARC implementation: https://github.com/szu-tera/DiARC
- ARC Prize 2026 rules and metric: https://arcprize.org/competitions/2026/arc-agi-2
- TOPAS-DSPL repository: https://github.com/Bitterbot-AI/topas_DSLPv1
- Poetiq solver: https://github.com/poetiq-ai/poetiq-arc-agi-solver
