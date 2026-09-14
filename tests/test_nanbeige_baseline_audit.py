"""Synthetic audit fixtures; no teacher calls, model inference or training."""

import copy
import fcntl
import importlib.util
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baseline_audit", ROOT / "scripts/audit_nanbeige_baseline.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_two_attempts_and_multiple_inputs_are_scored_in_order():
    tasks = {
        "a": {"test": [{"output": [[1]]}, {"output": [[2]]}]},
        "b": {"test": [{"output": [[3]]}]},
    }
    guesses = {
        "a": [{"attempt_1": [[0]], "attempt_2": [[1]]}, {"attempt_1": [[2]], "attempt_2": [[2]]}],
        "b": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    scored = audit.score(tasks, guesses)
    assert scored["correct_outputs"] == 2 and scored["test_outputs"] == 3
    assert scored["exact_match"] == 2 / 3 and scored["strict_correct_tasks"] == 1
    guesses["a"].reverse()
    assert audit.score(tasks, guesses)["correct_outputs"] == 0
    del guesses["b"]
    with pytest.raises(ValueError, match="complete cohort"):
        audit.score(tasks, guesses)


@pytest.mark.parametrize(
    "config_policy,identity_policy,valid",
    [
        (None, None, True),
        ("revise_symbolic", "revise_symbolic", True),
        ("repair_static_errors", "repair_static_errors", True),
        ("repair_static_errors", None, False),
        (None, "repair_static_errors", False),
        ("ignore_sandbox", "ignore_sandbox", False),
    ],
)
def test_witness_repair_policy_identity_is_explicit_and_consistent(
    tmp_path, config_policy, identity_policy, valid
):
    identity = {"config": {}}
    if config_policy is not None:
        identity["config"]["witness_repair_policy"] = config_policy
    if identity_policy is not None:
        identity["witness_repair_policy"] = identity_policy
    if valid:
        audit.verify_format_archive(tmp_path, identity)
    else:
        with pytest.raises(ValueError, match="witness-repair policy identity"):
            audit.verify_format_archive(tmp_path, identity)


@pytest.mark.parametrize("grid", [[], [[True]], [[10]], [[-1]], [[1], [2, 3]], "1", [[1]] * 31])
def test_malformed_grid_rejected(grid):
    assert not audit.valid_grid(grid)


def reports(direct=(5, 5, 5), symbolic=(3, 3, 3), *, trained=False):
    return {
        "model": audit.MODEL,
        "trained": trained,
        "complete": True,
        "provisional_live_snapshot": False,
        "cohort": [f"t{i}" for i in range(20)],
        "split_hash": "fixture-split",
        "comparison_config": {"max_new_tokens": 2048},
        "metrics": {
            f"{mode}:{seed}": {"correct_outputs": hits, "test_outputs": 22, "processed_tasks": 20}
            for mode, counts in (("direct", direct), ("symbolic", symbolic))
            for seed, hits in zip(audit.SEEDS, counts, strict=True)
        },
    }


def test_three_seed_mean_not_best_seed():
    baseline = reports()
    candidate = reports((5, 6, 6), trained=True)
    result = audit.paired_accuracy_decision(baseline, candidate)
    assert result["accuracy_threshold_met"]
    assert result["mean_exact_match_gain"] == pytest.approx(2 / 66)
    assert result["promotion_proven"] is False
    assert result["runtime_threshold_evaluated"] is False
    candidate = reports((5, 5, 6), trained=True)
    assert not audit.paired_accuracy_decision(baseline, candidate)["accuracy_threshold_met"]


def test_compare_against_strongest_baseline_not_weak_control():
    result = audit.paired_accuracy_decision(
        reports((10, 10, 10), (3, 3, 3)), reports((3, 3, 3), (6, 6, 6), trained=True)
    )
    assert result["baseline_mode"] == "direct" and result["candidate_mode"] == "symbolic"
    assert not result["accuracy_threshold_met"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("model", "gpt-6-astra"),
        ("trained", False),
        ("complete", False),
        ("provisional_live_snapshot", True),
        ("cohort", []),
        ("split_hash", "changed"),
        ("comparison_config", {"max_new_tokens": 9999}),
    ],
)
def test_ineligible_comparisons_rejected(key, value):
    candidate = reports(trained=True)
    candidate[key] = value
    with pytest.raises(ValueError):
        audit.paired_accuracy_decision(reports(), candidate)


def state_and_call(tmp_path, *, malformed=False):
    text = (
        '{"predictions":['
        if malformed
        else ('{"predictions":[{"attempt_1":["1"],"attempt_2":"same_as_1"}]}')
    )
    from arc_agent.v4_reasoner import separate_reasoning

    text = "</think>" + text
    reasoning, final = separate_reasoning(text)
    response = {
        "raw": {"content": text, "tokens_predicted": 10},
        "final": final,
        "reasoning": reasoning,
        "prompt_tokens": 100,
        "generated_tokens": 10,
    }
    ref = audit.digest(response)
    (tmp_path / f"{ref}.json").write_text(json.dumps(response))
    state = {
        "artifact_refs": [ref],
        "step": 1,
        "request_count": 1,
        "prompt_tokens": 100,
        "generated_tokens": 10,
        "unknown_usage_calls": 0,
        "valid_responses": int(not malformed),
        "status": "running" if malformed else "complete",
        "cycle": int(malformed),
        "stage": "direct",
    }
    call = {
        "dispatched_at": 123,
        "response": response,
        "prepared": {"_stage": "direct", "_prompt_tokens": 100},
    }
    return state, {"student:42:direct:t:0": call}


@pytest.mark.parametrize("malformed", [False, True])
def test_validity_reparsed_and_counted(tmp_path, malformed):
    state, work = state_and_call(tmp_path, malformed=malformed)
    result = audit.state_statistics("t", "direct", 42, state, work, tmp_path, outputs=1)
    assert result["invalid_responses"] == int(malformed)
    assert result["remaining_call_upper_bound"] == (3 if malformed else 0)
    state["valid_responses"] = int(malformed)
    with pytest.raises(ValueError, match="independently parsed"):
        audit.state_statistics("t", "direct", 42, state, work, tmp_path, outputs=1)


def test_raw_response_and_token_tampering_rejected(tmp_path):
    state, work = state_and_call(tmp_path)
    response = work["student:42:direct:t:0"]["response"]
    response["final"] = "altered after runtime"
    with pytest.raises(ValueError, match="raw runtime"):
        audit.valid_response(work["student:42:direct:t:0"], 1)
    with pytest.raises(ValueError, match="authoritative artifacts"):
        audit.state_statistics("t", "direct", 42, state, work, tmp_path, outputs=1)


def test_missing_and_unknown_responses_are_not_free(tmp_path):
    state = {
        "artifact_refs": [],
        "step": 1,
        "request_count": 1,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "unknown_usage_calls": 1,
        "valid_responses": 0,
        "status": "timed_out",
        "cycle": 1,
        "stage": "symbolic",
    }
    work = {"student:42:symbolic:t:0": {"dispatched_at": 123, "response": None}}
    result = audit.state_statistics("t", "symbolic", 42, state, work, tmp_path, outputs=1)
    assert result["invalid_responses"] == 1 and result["remaining_call_upper_bound"] == 0
    state["unknown_usage_calls"] = 0
    with pytest.raises(ValueError, match="unknown usage"):
        audit.state_statistics("t", "symbolic", 42, state, work, tmp_path, outputs=1)


def test_interrupted_pending_dispatch_counted_before_resume(tmp_path):
    state = {
        "artifact_refs": [],
        "step": 0,
        "request_count": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "unknown_usage_calls": 0,
        "valid_responses": 0,
        "status": "running",
        "cycle": 0,
        "stage": "symbolic",
    }
    work = {"student:42:symbolic:t:0": {"dispatched_at": 123, "response": None}}
    result = audit.state_statistics("t", "symbolic", 42, state, work, tmp_path, outputs=1)
    assert result["dispatches"] == 1 and result["invalid_responses"] == 1
    assert result["unreconciled_dispatches"] == 1
    assert result["remaining_call_upper_bound"] == 7 and not result["complete"]
    work["student:42:symbolic:t:0"]["response"] = {"returned": "not yet consumed"}
    with pytest.raises(ValueError, match="saved-response replay"):
        audit.state_statistics("t", "symbolic", 42, state, work, tmp_path, outputs=1)


@pytest.mark.parametrize(
    "running,pending,remaining,expected",
    [(True, 1, 0, 0.95), (False, 1, 0, 0.9), (True, 0, 0, 0.9), (True, 1, 20, 0.975)],
)
def test_optimistic_gate_does_not_treat_live_reply_as_fixed_failure(
    running, pending, remaining, expected
):
    counts = {
        "dispatches": 20,
        "invalid_responses": 2,
        "unreconciled_dispatches": pending,
        "remaining_call_upper_bound": remaining,
    }
    before = counts.copy()
    result = audit.validity_upper_bound(counts, running=running)
    assert result["optimistic_final_validity_bound"] == pytest.approx(expected)
    assert result["potential_pending_successes_for_bound"] == (pending if running else 0)
    assert result["fixed_failures_for_bound"] == 2 - (pending if running else 0)
    assert counts == before  # Observed failures/usage are never erased by the bound.


def test_empty_bound_is_unavailable_not_success():
    counts = dict.fromkeys(
        (
            "dispatches",
            "invalid_responses",
            "unreconciled_dispatches",
            "remaining_call_upper_bound",
        ),
        0,
    )
    assert (
        audit.validity_upper_bound(counts, running=True)["optimistic_final_validity_bound"] is None
    )


@pytest.mark.parametrize(
    "field,value", [("dispatches", -1), ("invalid_responses", 3), ("unreconciled_dispatches", 2)]
)
def test_inconsistent_validity_bound_rejected(field, value):
    counts = {
        "dispatches": 2,
        "invalid_responses": 1,
        "unreconciled_dispatches": 1,
        "remaining_call_upper_bound": 0,
    }
    counts[field] = value
    with pytest.raises(ValueError, match="counts"):
        audit.validity_upper_bound(counts, running=True)


@pytest.fixture
def frozen_run(tmp_path):
    workspace, data = tmp_path / "run", tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    (workspace / ".lock").touch()
    (workspace / "artifacts").mkdir()
    tasks = {
        f"task{i}": {
            "task_id": f"task{i}",
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[1]], "output": [[1]]}] * (2 if i < 2 else 1),
        }
        for i in range(20)
    }
    for task_id, task in tasks.items():
        (data / f"{task_id}.json").write_text(json.dumps(task))
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "groups": {"development": list(tasks)},
                "task_hashes": {k: audit.digest(t) for k, t in tasks.items()},
            }
        )
    )
    with tarfile.open(workspace / "source.tar.gz", "w:gz") as archive:
        paths = list((ROOT / "src/arc_agent").glob("*.py"))
        paths += [ROOT / "tests/test_v4.py", ROOT / "scripts/v4-loopback.sb"]
        for path in paths:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    identity = {
        "student": "untrained_nanbeige",
        "config": {"seeds": list(audit.SEEDS), "modes": list(audit.MODES), "max_cycles": 4},
        "runtime_config": {},
        "cohort": list(tasks),
        "split_hash": audit.file_hash(split_path),
        "source_hash": audit.digest(audit.archive_hashes(workspace / "source.tar.gz")),
    }
    (workspace / "identity.json").write_text(json.dumps(identity))
    guesses = {
        k: [{"attempt_1": p["input"], "attempt_2": p["input"]} for p in t["test"]]
        for k, t in tasks.items()
    }
    report = {"metrics": {}}
    with sqlite3.connect(workspace / "run.sqlite3") as db:
        db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        db.execute("CREATE TABLE work (key TEXT, payload TEXT)")
        db.execute("INSERT INTO metadata VALUES ('identity', ?)", (json.dumps(identity),))
        for mode in audit.MODES:
            for seed in audit.SEEDS:
                for task_id in tasks:
                    state = {
                        "artifact_refs": [],
                        "step": 0,
                        "request_count": 0,
                        "prompt_tokens": 0,
                        "generated_tokens": 0,
                        "unknown_usage_calls": 0,
                        "valid_responses": 0,
                        "status": "exhausted",
                        "cycle": 4,
                        "stage": mode,
                        "attempts": guesses[task_id],
                    }
                    db.execute(
                        "INSERT INTO work VALUES (?,?)",
                        (f"student:{seed}:{mode}:{task_id}", json.dumps(state)),
                    )
                (workspace / f"submission-{mode}-{seed}.json").write_text(json.dumps(guesses))
                report["metrics"][f"{mode}:{seed}"] = {
                    **audit.score(tasks, guesses),
                    "processed_tasks": 20,
                    "dispatches": 0,
                    "generated_tokens": 0,
                    "prompt_tokens": 0,
                    "valid_response_rate": None,
                }
    (workspace / "report.json").write_text(json.dumps(report))
    return workspace, split_path, data


def test_complete_synthetic_score_audit_does_not_prove_promotion(frozen_run):
    result = audit.audit(*frozen_run)
    assert result["complete"] and not result["promotion_proven"]
    assert result["metrics"]["direct:42"]["correct_outputs"] == 22
    assert result["metrics"]["symbolic:44"]["valid_response_rate"] is None


def test_formatting_archive_is_bound_to_pinned_converter(tmp_path):
    from arc_agent.v8_grammar import CONVERTER_SHA256

    source = ROOT / ".runtime/nanbeige/llama.cpp/examples/json_schema_to_grammar.py"
    identity = {
        "config": {"structured_style": "compact_rectangular"},
        "structured_style": "compact_rectangular",
        "grammar_converter_sha256": CONVERTER_SHA256,
    }
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        archive.add(source, arcname="runtime-tools/json_schema_to_grammar.py")
    audit.verify_format_archive(tmp_path, identity)
    with pytest.raises(ValueError, match="identity differs"):
        audit.verify_format_archive(tmp_path, {**identity, "grammar_converter_sha256": "wrong"})
    with pytest.raises(ValueError, match="inconsistent"):
        audit.verify_format_archive(tmp_path, {**identity, "structured_style": "legacy"})
    changed = tmp_path / "changed.py"
    changed.write_bytes(source.read_bytes() + b"\n# altered\n")
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        archive.add(changed, arcname="runtime-tools/json_schema_to_grammar.py")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.verify_format_archive(tmp_path, identity)
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        archive.add(source, arcname="elsewhere.py")
    with pytest.raises(ValueError, match="missing archived"):
        audit.verify_format_archive(tmp_path, identity)


def test_live_owner_requires_provisional_audit(frozen_run):
    with (frozen_run[0] / ".lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="live owner"):
            audit.audit(*frozen_run)
        assert audit.audit(*frozen_run, allow_running=True)["provisional_live_snapshot"]


def test_actual_audit_uses_live_bound_but_preserves_stopped_missing_call(frozen_run):
    workspace = frozen_run[0]
    key = "student:42:direct:task0"
    with sqlite3.connect(workspace / "run.sqlite3") as db:
        state = json.loads(db.execute("SELECT payload FROM work WHERE key=?", (key,)).fetchone()[0])
        state.update(status="running", cycle=0)
        db.execute("UPDATE work SET payload=? WHERE key=?", (json.dumps(state), key))
        db.execute(
            "INSERT INTO work VALUES (?,?)",
            (key + ":0", json.dumps({"dispatched_at": 123, "response": None})),
        )
    saved = audit.read_json(workspace / "report.json")
    saved["metrics"]["direct:42"].update(processed_tasks=19, dispatches=1, valid_response_rate=0.0)
    (workspace / "report.json").write_text(json.dumps(saved))
    with (workspace / ".lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live = audit.audit(*frozen_run, allow_running=True)
    stopped = audit.audit(*frozen_run)
    for report in (live, stopped):
        metric = report["metrics"]["direct:42"]
        assert metric["dispatches"] == metric["invalid_responses"] == 1
        assert metric["valid_response_rate"] == 0
        assert report["auditor_sha256"] == audit.file_hash(audit.__file__)
        assert report["validity_bound_policy"] == "live_unreconciled_may_succeed_v2_direct_task_cap"
        assert metric["direct_task_validity_bound"] == (1 if report is live else 20 / 21)
        assert report["metrics"]["symbolic:42"]["direct_task_validity_bound"] is None
    assert live["metrics"]["direct:42"]["optimistic_final_validity_bound"] == 1
    assert stopped["metrics"]["direct:42"]["optimistic_final_validity_bound"] == 0.75


@pytest.mark.parametrize(
    "invalid,pending,running,expected",
    [(2, 0, True, 20 / 22), (2, 1, True, 20 / 21), (2, 1, False, 20 / 22), (0, 0, True, 1)],
)
def test_direct_successes_are_capped_by_tasks(invalid, pending, running, expected):
    counts = {
        "dispatches": 10,
        "invalid_responses": invalid,
        "unreconciled_dispatches": pending,
        "remaining_call_upper_bound": 70,
    }
    result = audit.validity_upper_bound(counts, running=running, direct_task_count=20)
    assert result["optimistic_final_validity_bound"] == expected
    assert result["direct_task_validity_bound"] == expected
    # This old request-only bound allowed too many successes from unused retries.
    assert result["request_count_validity_bound"] >= expected
    assert audit.validity_upper_bound(counts, running=running)["direct_task_validity_bound"] is None


@pytest.mark.parametrize("count", [0, -1, True, "20"])
def test_invalid_direct_task_cap_rejected(count):
    counts = {
        "dispatches": 1,
        "invalid_responses": 0,
        "unreconciled_dispatches": 0,
        "remaining_call_upper_bound": 0,
    }
    with pytest.raises(ValueError, match="positive task count"):
        audit.validity_upper_bound(counts, running=True, direct_task_count=count)


def test_impossible_direct_success_count_rejected():
    counts = {
        "dispatches": 21,
        "invalid_responses": 0,
        "unreconciled_dispatches": 0,
        "remaining_call_upper_bound": 0,
    }
    with pytest.raises(ValueError, match="more valid replies"):
        audit.validity_upper_bound(counts, running=True, direct_task_count=20)


def test_stopped_report_tamper_rejected(frozen_run):
    path = frozen_run[0] / "report.json"
    value = audit.read_json(path)
    value["metrics"]["direct:42"]["correct_outputs"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disagrees"):
        audit.audit(*frozen_run)


def test_changed_evaluation_task_rejected(frozen_run):
    path = frozen_run[2] / "task0.json"
    value = copy.deepcopy(audit.read_json(path))
    value["test"][0]["output"] = [[2]]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="evaluation task"):
        audit.audit(*frozen_run)
