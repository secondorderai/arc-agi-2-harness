"""Approved, narrowly scoped recovery for the 2026-09-09 pre-model startup failure.

Never edits frozen code or old evidence. Raw aggregate flags remain false; the
supplemental audit distinguishes one sealed preflight from actual execution.
Run resume through sandbox-exec: it checks network denial before touching the run.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import re
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "runs/v8/untrained-output4k-baseline-01"
IDENTITY = "4255842094cea985e798faf4d2e384ea37b105950c0207a4f35ef86b64525644"
SOURCE = "40a79e960b8d06d3778882e9763777cfef257f7028474af9fd6b597e6eeaa1cc"
REPLAY_WORK = "bf3d6ed70fb4f821128a5aa4f4d178e3cd0bd6270fbe9801ec59b6180ab0d311"
CONFIG = ROOT / "configs/v8-nanbeige-output4k-baseline.yaml"
SMOKE = WORKSPACE.parent / "output4k-smoke-12k-01/smoke.json"
RECEIPT = WORKSPACE / "approved-preflight-recovery-01.json"
PROOFS = (
    "audit-stopped-40-01.json",
    "audit-preflight-stop-01.json",
    "replay-preflight-stop-01.json",
    "resume-preflight-incident-01.md",
)
SPEC = importlib.util.spec_from_file_location(
    "recovery_baseline_audit", Path(__file__).with_name("audit_nanbeige_baseline.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def stopped_snapshot(workspace):
    with (workspace / ".lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with sqlite3.connect(
            (workspace / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            db.execute("BEGIN")
            work = {k: json.loads(v) for k, v in db.execute("SELECT key,payload FROM work")}
        yield work


def no_execution_segment(workspace, segment):
    expected = {
        "context_tokens": 12288,
        "path": "runs/v8/untrained-output4k-baseline-01/runtime-segments/001-context-12288",
        "offline_enforced": False,
        "metal_verified": False,
        "memory_safe": False,
        "prompt_cache_mib": 0,
        "prompt_cache_disabled_verified": False,
        "warning_pressure_policy": "record",
        "warning_pressure_samples": 0,
        "guard_error": None,
        "peak_server_rss_bytes": 0,
        "swap_growth_bytes": 0,
        "peak_pressure_level": 0,
        "initial_pressure_level": None,
        "initial_swap_used_bytes": None,
        "sample_count": 0,
    }
    if segment != expected:
        raise ValueError("not the exact approved pre-model failure")
    path = workspace / "runtime-segments/001-context-12288"
    if path.exists() or path.is_symlink():
        raise ValueError("pre-model segment unexpectedly has runtime files")


def validate_proofs(before, after, replay, work):
    for record in (before, after, replay):
        if record.get("identity_hash") != IDENTITY or record.get("provisional_live_snapshot"):
            raise ValueError("recovery requires the original stopped identity")
    if before["metrics"] != after["metrics"]:
        raise ValueError("model work changed during failed startup")
    if any(m["unreconciled_dispatches"] for m in after["metrics"].values()):
        raise ValueError("unresolved model dispatch cannot be called a preflight")
    if not (
        replay["stages_replayed"] == replay["states_reproduced"] == 40
        and replay["new_model_calls"] == 0
        and replay["work_snapshot_hash"] == REPLAY_WORK == audit.digest(work)
    ):
        raise ValueError("original 40-stage replay/work snapshot changed")


def inspect_execution(workspace, segments, config):
    from arc_agent.v4_cli import runtime_evidence

    for index, segment in enumerate(segments):
        if index == 1:
            no_execution_segment(workspace, segment)
            continue
        expected_path = (
            workspace / "runtime-segments" / (f"{index:03d}-context-{segment['context_tokens']}")
        )
        if (ROOT / segment["path"]).resolve() != expected_path.resolve():
            raise ValueError("runtime segment path differs from its index/context")
        samples = [
            json.loads(line) for line in (expected_path / "memory.jsonl").read_text().splitlines()
        ]
        baseline = audit.read_json(expected_path / "memory-baseline.json")
        if baseline.get("swap_used_bytes") != segments[0].get("initial_swap_used_bytes"):
            raise ValueError("cumulative swap baseline was reset")
        if not samples or segment.get("guard_error") is not None:
            raise ValueError("executed segment lacks verified telemetry or hit a guard")
        for sample in [baseline, *samples]:
            if sample.get("pressure_level") not in {1, 2}:
                raise ValueError("critical, unknown or missing pressure telemetry")
            for key in ("swap_used_bytes", "server_rss_bytes"):
                value = sample.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid or missing memory telemetry")
            if (
                sample["swap_used_bytes"] - baseline["swap_used_bytes"]
                > config.max_swap_growth_gib * 1024**3
            ):
                raise ValueError("cumulative swap safety limit exceeded")
        server = SimpleNamespace(
            samples=samples,
            baseline=baseline,
            log_start_offset=0,
            guard_error=segment["guard_error"],
            config=config,
            offline_verified=segment.get("offline_enforced") is True,
        )
        evidence = runtime_evidence(server, expected_path)
        if any(segment.get(key) != value for key, value in evidence.items()):
            raise ValueError("saved execution evidence differs from telemetry/logs")
        if not all(
            evidence[key] is True
            for key in (
                "offline_enforced",
                "metal_verified",
                "memory_safe",
                "prompt_cache_disabled_verified",
            )
        ):
            raise ValueError("executed segment did not pass existing runtime safeguards")


def validate_receipt(workspace, receipt, work, segments):
    if (
        receipt.get("schema") != "approved_pre_model_recovery_v1"
        or receipt.get("identity_hash") != IDENTITY
    ):
        raise ValueError("unapproved recovery identity/schema")
    if receipt.get("recovery_script_sha256") != audit.file_hash(__file__):
        raise ValueError("recovery implementation changed after approval was sealed")
    for name, expected in receipt["proof_hashes"].items():
        if name not in PROOFS or audit.file_hash(workspace / name) != expected:
            raise ValueError("sealed startup evidence changed")
    if set(receipt["proof_hashes"]) != set(PROOFS):
        raise ValueError("incomplete startup proof inventory")
    if segments[:2] != receipt["original_segments"]:
        raise ValueError("original runtime evidence was altered")
    no_execution_segment(workspace, segments[1])
    for key, expected in receipt["completed_call_hashes"].items():
        if key not in work or audit.digest(work[key]) != expected:
            raise ValueError("a completed model call was altered or removed")
    if len(receipt["completed_call_hashes"]) != 40:
        raise ValueError("original completed call inventory is incomplete")
    if len(receipt["terminal_state_hashes"]) != 20:
        raise ValueError("original terminal task inventory is incomplete")
    for key, expected in receipt["terminal_state_hashes"].items():
        if key not in work or audit.digest(work[key]) != expected:
            raise ValueError("a completed task was altered or removed")


def load_context():
    # Only the sealed solver is imported. No monkeypatching or changed protocols.
    sys.path.insert(0, str(WORKSPACE / "frozen/src"))
    from arc_agent.v4_config import load_v4_config, verify_manifest
    from arc_agent.v8_cli import experiment_binding, require_pilot_smoke
    from arc_agent.v8_config import load_config

    identity = audit.read_json(WORKSPACE / "identity.json")
    if audit.digest(identity) != IDENTITY or identity["source_hash"] != SOURCE:
        raise ValueError("only the explicitly approved frozen run may recover")
    cfg = load_config(CONFIG)
    runtime_cfg = load_v4_config(cfg.runtime_config)
    manifest = verify_manifest(runtime_cfg.manifest, runtime_cfg)
    binding = experiment_binding(cfg, runtime_cfg, manifest)
    require_pilot_smoke(audit.read_json(SMOKE), binding, cfg, runtime_cfg)
    if any(identity.get(key) != value for key, value in binding.items()):
        raise ValueError("runtime/source/smoke binding changed")
    return cfg, runtime_cfg


def review(*, seal=False):
    cfg, runtime_cfg = load_context()
    with stopped_snapshot(WORKSPACE) as work:
        result = audit.audit(WORKSPACE, cfg.split, cfg.data)
        state = audit.read_json(WORKSPACE / "adaptive-state.json")
        segments = state["segments"]
        if seal:
            if len(segments) != 2:
                raise ValueError("seal only immediately after the known preflight failure")
            validate_proofs(*(audit.read_json(WORKSPACE / p) for p in PROOFS[:3]), work)
            no_execution_segment(WORKSPACE, segments[1])
            receipt = {
                "schema": "approved_pre_model_recovery_v1",
                "identity_hash": IDENTITY,
                "source_hash": SOURCE,
                "approval": (
                    "User replied Yes to audited recovery preserving the failure record, "
                    "correct offline wrapper and unchanged safety limits on 2026-09-09."
                ),
                "recovery_script_sha256": audit.file_hash(__file__),
                "proof_hashes": {p: audit.file_hash(WORKSPACE / p) for p in PROOFS},
                "original_segments": segments,
                "completed_call_hashes": {
                    k: audit.digest(v)
                    for k, v in work.items()
                    if re.fullmatch(r"student:\d+:(direct|symbolic):[0-9a-f]{8}:\d+", k)
                },
                "terminal_state_hashes": {
                    k: audit.digest(v)
                    for k, v in work.items()
                    if re.fullmatch(r"student:\d+:(direct|symbolic):[0-9a-f]{8}", k)
                    and v["status"] != "running"
                },
                "raw_safety_flags_unchanged": True,
                "promotion_proven": False,
            }
        else:
            receipt = audit.read_json(RECEIPT)
        validate_receipt(WORKSPACE, receipt, work, segments)
        inspect_execution(WORKSPACE, segments, runtime_cfg)
        if any(m["unreconciled_dispatches"] for m in result["metrics"].values()):
            raise ValueError("pending/unknown model work requires separate recovery")
        possible = all(
            m["optimistic_final_validity_bound"] >= 0.95 for m in result["metrics"].values()
        )
        remaining = cfg.global_seconds - work["timer"]["elapsed_seconds"]
        if seal:
            write_new(RECEIPT, receipt)
        return {
            "scope": (
                "supplemental stopped execution-safety audit; no promotion or raw flag override"
            ),
            "identity_hash": IDENTITY,
            "receipt_sha256": audit.file_hash(RECEIPT),
            "executed_segments_verified": len(segments) - 1,
            "excluded_pre_model_segment": 1,
            "raw_aggregate": audit.read_json(WORKSPACE / "report.json")["memory_and_runtime"],
            "executed_segments_safe": True,
            "validity_gate_achievable": possible,
            "remaining_seconds": remaining,
            "complete": result["complete"],
            "resume_allowed": possible and remaining > 0 and not result["complete"],
            "promotion_proven": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "audit", "resume"))
    parser.add_argument(
        "--output", type=Path, required=True, help="New audit file; never overwrite"
    )
    parser.add_argument(
        "--receipt-sha256", help="Required externally pinned receipt hash after sealing"
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("run from the original project root")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "resume":
        sys.path.insert(0, str(WORKSPACE / "frozen/src"))
        from arc_agent.v4_runtime import external_network_denied

        if not external_network_denied():
            raise ValueError("use sandbox-exec -f scripts/v4-loopback.sb; run untouched")
    if args.mode != "seal" and (
        not args.receipt_sha256 or audit.file_hash(RECEIPT) != args.receipt_sha256
    ):
        raise ValueError("explicitly pinned recovery receipt hash is required and must match")
    result = review(seal=args.mode == "seal")
    write_new(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if args.mode == "resume":
        if not result["resume_allowed"]:
            raise ValueError("baseline gate, completion or deadline prevents resume")
        from arc_agent.v4_runtime import memory_sample, pressure_allows_start
        from arc_agent.v8_cli import pilot

        _, runtime_cfg = load_context()
        sample = memory_sample()
        baseline = audit.read_json(WORKSPACE / "adaptive-state.json")["swap_baseline_bytes"]
        if (
            not pressure_allows_start(sample["pressure_level"], runtime_cfg)
            or sample["swap_used_bytes"] is None
            or sample["server_rss_bytes"] is None
            or sample["swap_used_bytes"] - baseline > runtime_cfg.max_swap_growth_gib * 1024**3
            or shutil.disk_usage(WORKSPACE).free < runtime_cfg.min_free_disk_gib * 1024**3
        ):
            raise ValueError("host safety preflight failed; no model call issued")
        print(json.dumps({"recovery_host_preflight": sample, "bounded_new_stages": 40}), flush=True)
        pilot(config=CONFIG, workspace=WORKSPACE, smoke_report=SMOKE, resume=True, stop_after=40)


if __name__ == "__main__":
    main()
