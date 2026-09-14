"""Replay saved V8 state transitions without tokenization or new model calls.

Re-executes sandboxed witnesses. Timing/transport failures are explicitly unsupported,
not turned into successes. A live snapshot or partial replay never proves promotion.
Use the original frozen source on PYTHONPATH for a historical experiment.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import io
import json
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arc_agent import v8_runner as runner
from arc_agent.models import ArcTask
from arc_agent.v4_config import content_hash
from arc_agent.v4_state import blind_task
from arc_agent.v8_config import V8Config

SPEC = importlib.util.spec_from_file_location(
    "baseline_audit", Path(__file__).with_name("audit_nanbeige_baseline.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class NoInference:
    def prepare_stage(self, *args, **kwargs):
        raise AssertionError("replay must not prepare a new request")

    def complete(self, *args, **kwargs):
        raise AssertionError("replay must never call a model")


class CheckedArtifacts:
    def __init__(self, root, allowed):
        self.root, self.allowed = Path(root), set(allowed)

    def get(self, key):
        if key not in self.allowed:
            raise ValueError("replay produced an unrecorded artifact")
        value = audit.read_json(self.root / f"{key}.json")
        if content_hash(value) != key:
            raise ValueError("replay artifact hash differs")
        return value

    def put(self, value):
        key = content_hash(value)
        if self.get(key) != value:
            raise ValueError("replayed artifact differs from its recorded value")
        return key


class ReplayStore:
    """All state writes stay in memory; original artifacts are read-only."""

    def __init__(self, calls, artifacts, allowed):
        self.work = copy.deepcopy(calls)
        self.artifacts = CheckedArtifacts(artifacts, allowed)

    def get(self, key):
        return copy.deepcopy(self.work.get(key))

    def save(self, key, value):
        # Match ExperimentRun's SQLite JSON round trip, including nested key order:
        # later native prompts serialize those dictionaries without sorting again.
        self.work[key] = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def replay_state(task, mode, seed, expected, work, artifacts, config):
    key = runner.state_key(task.task_id, mode, seed)
    pending = work.get(f"{key}:{expected['step']}")
    result = {
        "key": key,
        "recorded_steps": expected["step"],
        "replayed_steps": 0,
        "semantic_state_reproduced": False,
        "terminal": expected["status"] != "running",
        "pending_dispatch": bool(pending and pending.get("dispatched_at") is not None),
    }
    # Historical wall time cannot be reconstructed from returned model responses alone.
    # Preserve those failures in scoring; do not invent timestamps or regenerate them.
    if expected["status"] in {"timed_out", "context_limit"} or expected["unknown_usage_calls"]:
        return {**result, "unsupported_reason": "timing/context/unknown-response transition"}
    calls = {}
    for step in range(expected["step"]):
        call_key = f"{key}:{step}"
        call = work.get(call_key)
        if not call or call.get("response") is None or call.get("dispatched_at") is None:
            return {**result, "unsupported_reason": "missing returned stage evidence"}
        if call["prepared"].get("_stage") not in {"direct", "symbolic", "witness"}:
            raise ValueError("unknown replay stage")
        calls[call_key] = call
    store = ReplayStore(calls, artifacts, expected["artifact_refs"])
    for reference in expected["artifact_refs"]:
        store.artifacts.get(reference)
    # A constant replay clock omits all elapsed-time claims. Saved charged_seconds
    # remain accounted by advance_stage, but cannot prove inclusive runtime savings.
    with (
        patch.object(runner, "time", SimpleNamespace(monotonic=lambda: 0.0)),
        redirect_stdout(io.StringIO()),
    ):
        for step in range(expected["step"]):
            previous = store.get(key) or runner.initial_state(task, mode, seed)
            if calls[f"{key}:{step}"]["prepared"]["_stage"] != previous["stage"]:
                raise ValueError("recorded request stage differs from reconstructed state")
            runner.advance_stage(blind_task(task), mode, seed, config, store, NoInference())
        actual = store.get(key) or runner.initial_state(task, mode, seed)
        fields = set(expected) - {"elapsed_seconds"}
        if set(actual) != set(expected) or any(actual[k] != expected[k] for k in fields):
            differing = sorted(k for k in fields if actual.get(k) != expected[k])
            raise ValueError(f"{key}: replayed state differs: {differing}")
        terminal_noop = False
        if expected["status"] != "running":
            before = copy.deepcopy(store.work)
            runner.advance_stage(blind_task(task), mode, seed, config, store, NoInference())
            if store.work != before:
                raise ValueError("terminal replay modified saved work")
            terminal_noop = True
    return {
        **result,
        "replayed_steps": expected["step"],
        "semantic_state_reproduced": True,
        "terminal_noop_verified": terminal_noop,
    }


def replay(workspace, *, allow_running=False):
    workspace = Path(workspace)
    identity = audit.read_json(workspace / "identity.json")
    config = V8Config.model_validate(identity["config"])
    # Verifies the frozen source/format/split and independent stored-evidence arithmetic.
    # This audit may precede a live snapshot: it is not claimed to be the same snapshot.
    audit.audit(workspace, config.split, config.data, allow_running=allow_running)
    with (workspace / ".lock").open("r") as lock:
        running = False
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            running = True
            if not allow_running:
                raise ValueError("live run requires an explicitly provisional replay") from None
        with sqlite3.connect(
            (workspace / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            db.execute("BEGIN")
            recorded = json.loads(
                db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
            )
            work = {k: json.loads(v) for k, v in db.execute("SELECT key,payload FROM work")}
        if recorded != identity:
            raise ValueError("replay identity changed")
        split = audit.read_json(config.split)
        if audit.file_hash(config.split) != identity["split_hash"]:
            raise ValueError("replay split changed")
        results = []
        for seed in config.seeds:
            for mode in config.modes:
                for task_id in identity["cohort"]:
                    key = runner.state_key(task_id, mode, seed)
                    expected = work.get(key)
                    if expected is None:
                        continue
                    raw = audit.read_json(config.data / f"{task_id}.json")
                    task = ArcTask.model_validate({"task_id": task_id, **raw})
                    if content_hash(task.model_dump(mode="json")) != split["task_hashes"][task_id]:
                        raise ValueError("replay task changed")
                    # Validate artifacts, parsing, counters and pending calls in THIS snapshot.
                    audit.state_statistics(
                        task_id,
                        mode,
                        seed,
                        expected,
                        work,
                        workspace / "artifacts",
                        outputs=len(task.test),
                        symbolic_cell_encoding=config.symbolic_cell_encoding,
                    )
                    results.append(
                        replay_state(
                            task, mode, seed, expected, work, workspace / "artifacts", config
                        )
                    )
        fully_replayed = len(results) == 120 and all(
            r["semantic_state_reproduced"] and r["terminal"] and not r["pending_dispatch"]
            for r in results
        )
        return {
            "scope": "saved response state-transition replay, not inference or timing replay",
            "identity_hash": content_hash(identity),
            "work_snapshot_hash": content_hash(work),
            "replay_script_sha256": audit.file_hash(__file__),
            "provisional_live_snapshot": running,
            "states_inspected": len(results),
            "states_reproduced": sum(r["semantic_state_reproduced"] for r in results),
            "stages_replayed": sum(r["replayed_steps"] for r in results),
            "all_120_terminal_states_reproduced": fully_replayed and not running,
            "new_model_calls": 0,
            "elapsed_runtime_verified": False,
            "promotion_proven": False,
            "results": results,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.workspace, allow_running=args.allow_running)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
