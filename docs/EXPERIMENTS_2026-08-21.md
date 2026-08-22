# Hybrid object-centric world model + symbolic executor — 21 August 2026

## Outcome

V1 is implemented end to end, but it does **not** improve the frozen public
score. The final 120-task public evaluation remains **2.40% pass@2** and
**2.50% strict task accuracy**, identical to the deterministic baseline.
The run took 251.6 seconds and generated no additional verified symbolic
programs.

Canonical public report:
`runs/hybrid-transition-public-final-20260821/hybrid-symbolic-fast-20260820T235121Z/report.md`.

## Implemented architecture

1. A lossless object scene IR extracts connected objects, bounding boxes,
   holes, D4-invariant signatures, colors, containment, adjacency, alignment,
   relative position, and color references.
2. Scene-transition inference globally matches input/output objects and records
   preservation, movement, recoloring, transformation, addition, removal, and
   relation changes.
3. A typed JSON-serializable executor supports recolor, crop/select, translate,
   copy/remove, fill holes, rotate/reflect, repeat/propagate, and color/relation
   chains.
4. Bounded synthesis proposes parameters from scene transitions, composes short
   programs, and admits a candidate only when it exactly reproduces every
   demonstration.
5. A compact recurrent object-slot world model has both a portable NumPy
   reference and an end-to-end trainable MLX backend. Its program head can rank
   symbolic operation families; direct neural grids remain optional and are
   rejected unless leave-one-demonstration-out verification succeeds.
6. A leakage-resistant experiment runner creates deterministic task-level
   learn/holdout splits, records manifests, trains either backend, compares the
   baseline and hybrid solver, and keeps the public evaluation set evaluation-only.

## Measured experiments

| Experiment | Result | Decision |
|---|---:|---|
| NumPy 4-task smoke | 44.1% holdout cell, 0% exact | Reference path works; direct grids rejected |
| Transition-guided 20-task holdout | 4.76% pass@2, 5.00% strict | Flat versus baseline |
| Initial MLX program head | 73.9% holdout program accuracy | Rejected: exactly matched a catch-all class frequency |
| Corrected, balanced MLX program head | 13.5% known-class, 20.0% macro, 0% exact grids | Rejected as a useful learned prior at this scale |
| Corrected-prior 20-task solver | 4.76% pass@2, 5.00% strict | One extra verified program, only on an already solved task |
| Frozen 120-task public run | 2.40% pass@2, 2.50% strict | Zero newly verified symbolic programs |

The corrected MLX training report is
`runs/hybrid-mlx-balanced-loss-100x20-20260821/hybrid_experiment.md`. The final
corrected-prior holdout comparison is
`runs/hybrid-mlx-balanced-prior-heldout20-20260821/hybrid_experiment.md`.

## Engineering findings

- Exact global object matching originally recomputed relational contexts and
  D4 signatures for every pair. Caching them once per scene reduced the hybrid
  solver from 102 seconds to 39 seconds on the same 20-task holdout.
- Symbolic search originally exhausted its node budget on primitives before
  trying any composition. It now reserves composition budget and prioritizes
  transition-evidenced parameters.
- Catch-all auxiliary labels can manufacture impressive classifier accuracy.
  Unresolved transitions now map to `unknown`; unknown examples train the grid
  model but are masked out of the program loss, and known classes are balanced.
- Demonstration verification prevents the weak neural grid decoder from
  lowering the score, but exact demonstration fit alone does not guarantee
  test generalization.

## Rejected V1 hypothesis and next bottleneck

The hypothesis “connected-component object transitions plus a short generic
executor will materially improve ARC-AGI-2 recall” is rejected. The current
executor cannot express role-dependent selection, masks and logical overlays,
panel correspondence, counting, conditional object rewrite, dynamic output
canvas construction, or relational iteration. A learned operation-family prior
cannot recover programs absent from that language.

The next high-reward experiment should therefore expand the declarative object
language and synthesize arguments from relations, rather than scale the current
grid decoder or add more search nodes.
