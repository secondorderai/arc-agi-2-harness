# V3 typed DSL pilot

V3 tests whether a small, auditable bank of typed game signatures transfers better than V2's
one-Python-program-per-task bank. The teacher model writes JSON only. Public evaluation is fully
offline and uses the frozen signatures as search priors rather than replaying them unchanged.

## Data flow

1. The harness fingerprints official training demonstrations and deterministically selects two
   related-but-different tasks in each of five transformation families.
2. `gpt-5.6-sol` at `xhigh` proposes a strict-schema `GameSignature`. The signature is a typed
   pipeline with symbolic parameters such as `background`, `output_only_color`, or an inferred
   translation offset.
3. The local executor rejects unknown fields, arbitrary Python, embedded grids, task IDs,
   excessive literals, bad type transitions, and unbounded pipelines. A signature advances only
   after it reproduces every demonstration exactly and passes its declared invariants.
4. The ten accepted signatures, DSL/config/dataset hashes, selection rationale, prompts, and
   usage are frozen before hidden training-test validation.
5. Source-isolated validation removes each task's own signature and tries to solve it from the
   remaining nine signatures plus bounded DSL search.
6. Public evaluation strips test outputs, ranks three signatures using deterministic fingerprint
   similarity and parameter feasibility, mutates their parameters, then falls back to bounded
   family and global grammar search. Two behaviorally distinct predictions are frozen before the
   scoring-only copy of the labels is consulted.

SQLite uses WAL and a workspace lock. Logical requests, idempotency keys, every raw response,
response IDs, usage, candidates, cursors, and frozen outputs are committed transactionally.
Quota failures pause the pilot without moving the task cursor. A restart retrieves a checkpointed
background/Codex response or reparses its stored completed body before creating another request.

If a task reaches 12 unsuccessful teacher attempts, V3 writes an `expressivity_gaps/<task>.json`
packet and stops. Add only a reusable typed primitive backed by the two family tasks or synthetic
counterfactual tests. Because the DSL implementation and schema are part of the DSL hash, the
complete ten-task pilot must then start in a new workspace.

## Runbook

Install and authenticate the isolated workspace:

```bash
uv sync --extra dev
uv run arc-agent v3-auth-login \
  --config configs/v3-sol-xhigh.yaml \
  --workspace runs/v3-pilot
```

Build the pilot. Running the same command resumes a matching unfinished workspace; `--resume`
requires that the workspace already exist.

```bash
uv run arc-agent v3-build-pilot \
  --data data/ARC-AGI-2/data/training \
  --config configs/v3-sol-xhigh.yaml \
  --workspace runs/v3-pilot
```

Inspect progress:

```bash
uv run arc-agent v3-status --workspace runs/v3-pilot --phase pilot
```

After the pilot freezes, run the 120-task public evaluation. This command creates no model client
and makes no network request. It targets ten active solver hours and stops search at that point,
leaving the remaining two hours of the hard cap for recovery and atomic submission freezing.

```bash
uv run arc-agent v3-evaluate \
  --data data/ARC-AGI-2/data/evaluation \
  --config configs/v3-sol-xhigh.yaml \
  --workspace runs/v3-pilot
```

Outputs are `pilot_manifest.json`, `pilot_prompts.jsonl`, `signatures.jsonl`,
`pilot_validation.json`, `v3_submission.json`, and `v3_evaluation_report.json` in the workspace.
The report separates
retrieval, mutation, family-search, and global-search contributions and records the go/no-go gate.
The same command accepts an unlabelled challenge set in a copied/forked frozen-pilot workspace;
it produces the same offline submission but leaves accuracy and the public go/no-go result null.

## Go/no-go policy

Expansion requires all ten tasks to be expressible, at least 8/10 own hidden tests, at least 4/10
source-isolated tasks, at least 6/120 public tasks, at least 9/167 public outputs, and deterministic
completion within ten active solver hours. A failed first iteration is diagnosed using a fresh
training-only holdout. A second miss retires the approach rather than tuning against public labels.
