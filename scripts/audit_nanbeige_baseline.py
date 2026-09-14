"""Read-only V8 score/integrity audit; never imported by the solver or teacher.

Live snapshots are explicitly provisional. This verifies score arithmetic and stored
evidence consistency, not model skill, full inference replay or student promotion.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import sqlite3
import tarfile
from pathlib import Path

MODEL = "Nanbeige/Nanbeige4.2-3B"
SEEDS = (42, 43, 44)
MODES = ("direct", "symbolic")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def valid_response(call, outputs):
    from arc_agent.v4_experiment import predictions_from_json
    from arc_agent.v4_reasoner import final_json, separate_reasoning
    from arc_agent.v5_symbolic import SymbolicModel
    from arc_agent.v8_reasoner import ProgramResponse

    response, prepared = call["response"], call["prepared"]
    raw = response["raw"]
    stage = prepared["_stage"]
    if stage not in {"direct", "symbolic", "witness"}:
        raise ValueError("unknown recorded inference stage")
    reasoning, final = separate_reasoning(raw["content"])
    if stage == "symbolic":
        reasoning, final = "", raw["content"].replace("<|im_end|>", "").strip()
    if (
        response["reasoning"] != reasoning
        or response["final"] != final
        or response["prompt_tokens"] != prepared["_prompt_tokens"]
        or response["generated_tokens"] != raw.get("tokens_predicted", 0)
        or raw.get("truncated")
    ):
        raise ValueError("parsed response differs from raw runtime evidence")
    if prepared.get("_response_format") == "symbolic_reference_patch" and (
        prepared.get("_reference_base")
        != json.loads(call["messages"][-1]["content"]).get("previous_symbolic_model")
        or digest(prepared.get("_reference_base")) != prepared.get("_reference_base_sha256")
    ):
        raise ValueError("reference patch base differs from its authoritative input")
    try:
        value = final_json(final)
        if stage == "direct":
            predictions_from_json(value, outputs)
        elif stage == "symbolic":
            if prepared.get("_response_format") == "symbolic_reference_patch":
                from arc_agent.v8_reference_repair import apply_reference_patch

                apply_reference_patch(
                    prepared["_reference_base"], value, prepared["_reference_base_sha256"]
                )
            elif prepared.get("_response_format") == "symbolic_cell_triples_v1":
                from arc_agent.v8_codec import unpack_symbolic_cells

                unpack_symbolic_cells(value)
            else:
                SymbolicModel.model_validate(value)
        else:
            ProgramResponse.model_validate(value)
    except ValueError:
        return False
    return True


def valid_grid(grid):
    return (
        isinstance(grid, list)
        and 1 <= len(grid) <= 30
        and isinstance(grid[0], list)
        and 1 <= len(grid[0]) <= 30
        and all(
            isinstance(row, list)
            and len(row) == len(grid[0])
            and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
            for row in grid
        )
    )


def score(tasks, submission):
    if set(submission) != set(tasks):
        raise ValueError("predictions must cover exactly the complete cohort")
    correct = strict = total = 0
    hits_by_task = {}
    for task_id, task in tasks.items():
        attempts = submission[task_id]
        if len(attempts) != len(task["test"]):
            raise ValueError("test input order/count changed")
        hits = []
        for pair, guesses in zip(task["test"], attempts, strict=True):
            if set(guesses) != {"attempt_1", "attempt_2"}:
                raise ValueError("exactly two named predictions are required")
            if not all(valid_grid(grid) for grid in guesses.values()):
                raise ValueError("invalid predicted grid")
            if pair.get("output") is None:
                raise ValueError("scoring labels are missing")
            hits.append(any(grid == pair["output"] for grid in guesses.values()))
        hits_by_task[task_id] = hits
        correct += sum(hits)
        total += len(hits)
        strict += all(hits)
    return {
        "correct_outputs": correct,
        "test_outputs": total,
        "exact_match": correct / total,
        "strict_correct_tasks": strict,
        "tasks": len(tasks),
        "strict_accuracy": strict / len(tasks),
        "hits_by_task": hits_by_task,
    }


def archive_hashes(path):
    hashes = {}
    with tarfile.open(path, "r:gz") as archive:
        seen = set()
        for member in archive.getmembers():
            if member.name in seen or not member.isfile():
                raise ValueError("duplicate or non-regular source archive member")
            seen.add(member.name)
            p = Path(member.name)
            if (p.parent.as_posix() == "src/arc_agent" and p.suffix == ".py") or member.name in {
                "tests/test_v4.py",
                "scripts/v4-loopback.sb",
            }:
                hashes[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if not {"tests/test_v4.py", "scripts/v4-loopback.sb"} <= hashes.keys():
        raise ValueError("source archive is incomplete")
    return hashes


def state_statistics(
    task_id, mode, seed, state, work, artifacts, *, outputs, symbolic_cell_encoding="named_fields"
):
    """Account for every completed stage, including invalid/missing model responses."""
    if state is None:
        return {
            "dispatches": 0,
            "responses": 0,
            "generated_tokens": 0,
            "prompt_tokens": 0,
            "valid_responses": 0,
            "invalid_responses": 0,
            "complete": False,
            "unreconciled_dispatches": 0,
            "remaining_call_upper_bound": 4 if mode == "direct" else 8,
        }
    key = f"student:{seed}:{mode}:{task_id}"
    if state["status"] not in {"running", "complete", "exhausted", "timed_out", "context_limit"}:
        raise ValueError("unknown task status")
    for ref in state["artifact_refs"]:
        if len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            raise ValueError("invalid artifact reference")
        if digest(read_json(artifacts / f"{ref}.json")) != ref:
            raise ValueError("corrupt response or verifier artifact")
    calls = []
    for step in range(state["step"]):
        call = work.get(f"{key}:{step}")
        if call and call["dispatched_at"] is not None:
            calls.append(call)
    responses = [c["response"] for c in calls if c["response"] is not None]
    if len(calls) != state["request_count"]:
        raise ValueError("saved request count differs from the dispatch ledger")
    for call in calls:
        if call["response"] is None or call["prepared"]["_stage"] != "symbolic":
            continue
        response_format = call["prepared"].get("_response_format")
        if response_format == "symbolic_reference_patch":
            continue
        expected_format = (
            "symbolic_cell_triples_v1" if symbolic_cell_encoding == "triples_v1" else None
        )
        if response_format != expected_format:
            raise ValueError("symbolic response encoding differs from the frozen protocol")
        if response_format == "symbolic_cell_triples_v1" and valid_response(call, outputs):
            from arc_agent.v4_reasoner import final_json
            from arc_agent.v8_codec import unpack_symbolic_cells

            packed = final_json(call["response"]["final"])
            decoding = {
                "kind": "symbolic_cell_decode",
                "encoding": "triples_v1",
                "source_response_ref": digest(call["response"]),
                "packed_sha256": digest(packed),
                "canonical": unpack_symbolic_cells(packed),
            }
            if digest(decoding) not in state["artifact_refs"]:
                raise ValueError("symbolic cell decoding evidence is missing or altered")
    for response in responses:
        ref = digest(response)
        if ref not in state["artifact_refs"]:
            raise ValueError("completed response missing from authoritative artifacts")
    for field in ("generated_tokens", "prompt_tokens"):
        if sum(r[field] for r in responses) != state[field]:
            raise ValueError("reported token usage differs from returned responses")
    if len(calls) - len(responses) != state["unknown_usage_calls"]:
        raise ValueError("unknown usage was dropped from the ledger")
    if not 0 <= state["valid_responses"] <= len(responses):
        raise ValueError("validity count exceeds returned responses")
    actual_valid = sum(valid_response(c, outputs) for c in calls if c["response"] is not None)
    if actual_valid != state["valid_responses"]:
        raise ValueError("structured validity differs from independently parsed responses")
    terminal = state["status"] != "running"
    remaining = 0
    if not terminal:
        remaining = max(0, 4 - state["cycle"])
        if mode == "symbolic":
            remaining = remaining * 2 - int(state["stage"] == "witness")
    pending = work.get(f"{key}:{state['step']}")
    unreconciled = int(bool(pending and pending["dispatched_at"] is not None))
    if unreconciled:
        if terminal:
            raise ValueError("terminal task has an unexpected pending dispatch")
        if pending["response"] is not None:
            raise ValueError(
                "returned pending response needs saved-response replay or a fresh snapshot"
            )
        # An interrupted dispatch is visible in the report denominator even before
        # resume marks its stage unavailable. Do not drop it or issue it again.
        remaining = max(0, remaining - 1)
    return {
        "dispatches": len(calls) + unreconciled,
        "responses": len(responses),
        "generated_tokens": state["generated_tokens"],
        "prompt_tokens": state["prompt_tokens"],
        "valid_responses": state["valid_responses"],
        "invalid_responses": len(calls) + unreconciled - state["valid_responses"],
        "complete": terminal,
        "unreconciled_dispatches": unreconciled,
        "remaining_call_upper_bound": max(0, remaining),
    }


def validity_upper_bound(result, *, running, direct_task_count=None):
    """A live unresolved reply may still succeed; a stopped lost reply cannot.

    Keep all dispatches in the observed denominator. Only this optimistic bound
    gives live unreconciled requests the benefit of success. Already-accounted
    malformed replies, timeouts and lost responses remain fixed failures.
    Direct mode terminates on its first valid reply, so it can have at most one
    valid response per task. Unused retries cannot all become additional successes.
    """
    fields = (
        "dispatches",
        "invalid_responses",
        "unreconciled_dispatches",
        "remaining_call_upper_bound",
    )
    if any(type(result[key]) is not int or result[key] < 0 for key in fields):
        raise ValueError("validity bound requires nonnegative integer counts")
    dispatched, invalid, pending, remaining = (result[key] for key in fields)
    if not pending <= invalid <= dispatched:
        raise ValueError("inconsistent validity-bound counts")
    possible_pending_successes = pending if running else 0
    fixed_failures = invalid - possible_pending_successes
    maximum = dispatched + remaining
    request_bound = 1 - fixed_failures / maximum if maximum else None
    direct_bound = None
    if direct_task_count is not None:
        if type(direct_task_count) is not int or direct_task_count <= 0:
            raise ValueError("direct validity bound requires a positive task count")
        if dispatched - invalid > direct_task_count:
            raise ValueError("direct mode cannot have more valid replies than tasks")
        direct_bound = direct_task_count / (direct_task_count + fixed_failures)
    bound = request_bound
    if bound is not None and direct_bound is not None:
        bound = min(bound, direct_bound)
    return {
        "potential_pending_successes_for_bound": possible_pending_successes,
        "fixed_failures_for_bound": fixed_failures,
        "request_count_validity_bound": request_bound,
        "direct_task_validity_bound": direct_bound,
        "optimistic_final_validity_bound": bound,
    }


def verify_format_archive(workspace, identity):
    cell_encoding = identity["config"].get("symbolic_cell_encoding", "named_fields")
    if cell_encoding not in {"named_fields", "triples_v1"} or (
        identity.get("symbolic_cell_encoding", "named_fields") != cell_encoding
    ):
        raise ValueError("unknown or inconsistent symbolic cell encoding identity")
    repair_policy = identity["config"].get("witness_repair_policy", "revise_symbolic")
    if repair_policy not in {"revise_symbolic", "repair_static_errors"} or (
        identity.get("witness_repair_policy", "revise_symbolic") != repair_policy
    ):
        raise ValueError("unknown or inconsistent witness-repair policy identity")
    style = identity["config"].get("structured_style", "legacy")
    if style == "legacy":
        if identity.get("structured_style", "legacy") != "legacy":
            raise ValueError("structured-style identity mismatch")
        return
    if (
        style
        not in {
            "compact_rectangular",
            "compact_identifiers",
            "reference_repair",
            "reference_repair_bounded_ws",
        }
        or identity.get("structured_style") != style
    ):
        raise ValueError("unknown or inconsistent structured-style identity")
    from arc_agent.v8_grammar import CONVERTER_SHA256

    if identity.get("grammar_converter_sha256") != CONVERTER_SHA256:
        raise ValueError("grammar converter identity differs from frozen source")
    with tarfile.open(workspace / "source.tar.gz", "r:gz") as archive:
        try:
            source = archive.extractfile("runtime-tools/json_schema_to_grammar.py").read()
        except KeyError as exc:
            raise ValueError("missing archived grammar converter") from exc
    if hashlib.sha256(source).hexdigest() != CONVERTER_SHA256:
        raise ValueError("archived grammar converter hash mismatch")


def audit(workspace, split_path, data_path, *, allow_running=False):
    identity = read_json(workspace / "identity.json")
    if identity["student"] != "untrained_nanbeige":
        raise ValueError("this auditor accepts the original V8 baseline only")
    if identity["config"]["seeds"] != list(SEEDS) or identity["config"]["modes"] != list(MODES):
        raise ValueError("comparison requires both modes and all three fixed seeds")
    if identity["config"]["max_cycles"] != 4:
        raise ValueError("call upper bounds require the frozen four-cycle protocol")
    split = read_json(split_path)
    cohort = split["groups"]["development"][:20]
    if (
        len(set(cohort)) != 20
        or identity["cohort"] != cohort
        or identity["split_hash"] != file_hash(split_path)
    ):
        raise ValueError("frozen cohort or split changed")
    archived = archive_hashes(workspace / "source.tar.gz")
    if digest(archived) != identity["source_hash"]:
        raise ValueError("source archive no longer reproduces the baseline identity")
    # Do not silently use newer parsers when auditing a frozen run.
    import arc_agent.v8_reasoner as protocol

    source_dir = Path(protocol.__file__).parent
    for name, expected in archived.items():
        if (
            name.startswith("src/arc_agent/")
            and file_hash(source_dir / Path(name).name) != expected
        ):
            raise ValueError("audit parser/source changed; load the frozen source archive")
    verify_format_archive(workspace, identity)
    tasks = {}
    for task_id in cohort:
        raw = read_json(data_path / f"{task_id}.json")
        task = {
            "task_id": task_id,
            **{
                group: [{"input": p["input"], "output": p.get("output")} for p in raw[group]]
                for group in ("train", "test")
            },
        }
        if digest(task) != split["task_hashes"][task_id]:
            raise ValueError("evaluation task changed after split freeze")
        tasks[task_id] = task
    # Lock existence is not process evidence. Test the actual advisory lock, and keep a
    # read lock through a stopped-run audit so a writer cannot resume underneath it.
    with (workspace / ".lock").open("r") as lock:
        running = False
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            running = True
            if not allow_running:
                raise ValueError(
                    "run has a live owner; only a provisional snapshot is allowed"
                ) from None
        with sqlite3.connect(
            (workspace / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            db.execute("BEGIN")
            stored_identity = json.loads(
                db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
            )
            work = {k: json.loads(v) for k, v in db.execute("SELECT key,payload FROM work")}
        if stored_identity != identity:
            raise ValueError("database and file identities differ")
        saved_report = read_json(workspace / "report.json")
        metrics = {}
        for mode in MODES:
            for seed in SEEDS:
                guesses, statistics = {}, []
                for task_id, task in tasks.items():
                    state = work.get(f"student:{seed}:{mode}:{task_id}")
                    guesses[task_id] = (
                        state["attempts"]
                        if state
                        else [
                            {"attempt_1": p["input"], "attempt_2": p["input"]} for p in task["test"]
                        ]
                    )
                    statistics.append(
                        state_statistics(
                            task_id,
                            mode,
                            seed,
                            state,
                            work,
                            workspace / "artifacts",
                            outputs=len(task["test"]),
                            symbolic_cell_encoding=identity["config"].get(
                                "symbolic_cell_encoding", "named_fields"
                            ),
                        )
                    )
                result = score(tasks, guesses)
                if result["test_outputs"] != 22:
                    raise ValueError("the original cohort must contain 22 test inputs")
                for field in (
                    "dispatches",
                    "responses",
                    "generated_tokens",
                    "prompt_tokens",
                    "valid_responses",
                    "invalid_responses",
                    "remaining_call_upper_bound",
                    "unreconciled_dispatches",
                ):
                    result[field] = sum(s[field] for s in statistics)
                result["processed_tasks"] = sum(s["complete"] for s in statistics)
                result["valid_response_rate"] = (
                    result["valid_responses"] / result["dispatches"]
                    if result["dispatches"]
                    else None
                )
                result.update(
                    validity_upper_bound(
                        result,
                        running=running,
                        direct_task_count=len(tasks) if mode == "direct" else None,
                    )
                )
                key = f"{mode}:{seed}"
                recorded = saved_report["metrics"][key]
                fields = (
                    "correct_outputs",
                    "test_outputs",
                    "exact_match",
                    "strict_correct_tasks",
                    "strict_accuracy",
                    "processed_tasks",
                    "dispatches",
                    "generated_tokens",
                    "prompt_tokens",
                    "valid_response_rate",
                )
                result["saved_report_matches_snapshot"] = all(
                    recorded[f] == result[f] for f in fields
                )
                saved_submission = read_json(workspace / f"submission-{mode}-{seed}.json")
                # Validate early fallbacks even when a live export is one stage behind.
                score(tasks, saved_submission)
                result["saved_submission_matches_snapshot"] = saved_submission == guesses
                if not running and not (
                    result["saved_report_matches_snapshot"]
                    and result["saved_submission_matches_snapshot"]
                ):
                    raise ValueError(
                        "stopped-run report/submission disagrees with independent audit"
                    )
                metrics[key] = result
        complete = all(m["processed_tasks"] == 20 for m in metrics.values())
        return {
            "scope": "independent score and stored-evidence consistency, not full inference replay",
            "model": MODEL,
            "trained": False,
            "provisional_live_snapshot": running,
            "complete": complete,
            "cohort_verified": True,
            "source_archive_verified": True,
            "stored_evidence_verified": True,
            "auditor_sha256": file_hash(__file__),
            "validity_bound_policy": "live_unreconciled_may_succeed_v2_direct_task_cap",
            "cohort": cohort,
            "split_hash": identity["split_hash"],
            "comparison_config": {
                "solver": {
                    k: v
                    for k, v in identity["config"].items()
                    if k not in {"data", "split", "runtime_config"}
                },
                "runtime": {
                    k: v
                    for k, v in identity["runtime_config"].items()
                    if k not in {"manifest", "base_url"}
                },
            },
            "identity_hash": digest(identity),
            "source_archive_hash": file_hash(workspace / "source.tar.gz"),
            "metrics": metrics,
            "promotion_proven": False,
        }


def paired_accuracy_decision(baseline, candidate):
    """Arithmetic component only; caller still needs training/lineage/runtime audits.

    Compare the candidate's best declared mode against the strongest untrained mode,
    not merely against the weakest control. Three seeds are correlated measurements,
    not independent tasks or statistical proof. Speed promotion needs separate inclusive
    end-to-end timing; sums of per-task durations cannot establish it.
    """
    if (
        baseline.get("model") != MODEL
        or candidate.get("model") != MODEL
        or baseline.get("trained") is not False
        or candidate.get("trained") is not True
    ):
        raise ValueError("compare untrained and trained Nanbeige, never an Astra teacher score")
    for key in ("cohort", "split_hash", "comparison_config"):
        if key not in baseline or baseline[key] != candidate.get(key):
            raise ValueError("comparison data or declared inference protocol differs")
    if len(baseline["cohort"]) != 20 or len(set(baseline["cohort"])) != 20:
        raise ValueError("comparison must cover 20 distinct frozen tasks")
    means = []
    for report in (baseline, candidate):
        if report.get("complete") is not True or report.get("provisional_live_snapshot"):
            raise ValueError("unfinished/live experiments cannot pass a promotion comparison")
        if set(report["metrics"]) != {f"{mode}:{seed}" for mode in MODES for seed in SEEDS}:
            raise ValueError("comparison requires exactly six mode/seed groups")
        mode_means = {}
        for mode in MODES:
            values = []
            for seed in SEEDS:
                metric = report["metrics"][f"{mode}:{seed}"]
                if metric["test_outputs"] != 22 or metric["processed_tasks"] != 20:
                    raise ValueError("comparison denominators differ")
                value = metric["correct_outputs"] / 22
                if (
                    type(metric["correct_outputs"]) is not int
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("invalid exact-match score")
                values.append(value)
            mode_means[mode] = sum(values) / len(SEEDS)
        means.append(mode_means)
    baseline_mode = max(MODES, key=lambda mode: means[0][mode])
    candidate_mode = max(MODES, key=lambda mode: means[1][mode])
    gain = means[1][candidate_mode] - means[0][baseline_mode]
    return {
        "baseline_mode_means": means[0],
        "candidate_mode_means": means[1],
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "mean_exact_match_gain": gain,
        "accuracy_threshold_met": gain >= 0.03 - 1e-12,
        "runtime_threshold_evaluated": False,
        "promotion_proven": False,
        "remaining_checks": [
            "trained artifact lineage",
            "same inference protocol and hardware",
            "state integrity and recovery",
            "complete independent run audits",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("runs/v8/untrained-baseline-01"))
    parser.add_argument("--split", type=Path, default=Path("runs/v4/split.json"))
    parser.add_argument("--data", type=Path, default=Path("data/ARC-AGI-2/data/training"))
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--output", type=Path, help="Create a new audit snapshot; never overwrite")
    args = parser.parse_args()
    result = audit(args.workspace, args.split, args.data, allow_running=args.allow_running)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(
            json.dumps(
                {
                    "snapshot": str(args.output),
                    "complete": result["complete"],
                    "provisional_live_snapshot": result["provisional_live_snapshot"],
                    "promotion_proven": False,
                }
            )
        )
    else:
        print(json.dumps(result, indent=2))
