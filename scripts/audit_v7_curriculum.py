"""Post-completion replay of the sealed training-only teacher pilot; no model calls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import ExperimentRun, atomic_json
from arc_agent.v5_config import V5Config
from arc_agent.v5_pipeline import select_tasks, source_identity
from arc_agent.v7_dataset import audit_training_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    root = args.workspace
    identity = json.loads((root / "identity.json").read_text())
    original = json.loads((root / "pilot-report.json").read_text())
    cfg = V5Config.model_validate(identity["config"])
    if identity["version"] != 7 or not original["complete"]:
        raise ValueError("audit requires the completed V7 training pilot")
    tasks, split = select_tasks(cfg)
    if identity["split_hash"] != content_hash(split):
        raise ValueError("split identity changed")
    run = ExperimentRun(root, identity, resume=True)
    try:
        audit_training_records(run, cfg, tasks, split)
        dataset = json.loads((root / "symbolic-sft.json").read_text())
        if content_hash(dataset) != original["dataset"]["dataset_hash"]:
            raise ValueError("exported dataset changed")
        if not original["data_quality_gate_passed"]:
            raise ValueError("full curriculum pilot quality gate not passed")
        if any(e["source_task_id"] not in split["groups"]["training"] for e in dataset):
            raise ValueError("non-training target in export")
        report = {
            "passed": True,
            "scope": "replayed visible prompts, raw responses, fixed witnesses and sealed outcomes",
            "teacher_calls": 0,
            "tasks": len(tasks),
            "examples": len(dataset),
            "dataset_hash": content_hash(dataset),
            "pilot_report_hash": file_hash(root / "pilot-report.json"),
            "run_identity_hash": content_hash(identity),
            "collector_source_archive_hash": file_hash(root / "source.tar.gz"),
            "audit_source_hash": source_identity(),
            "audit_script_hash": file_hash(Path(__file__)),
            "new_teacher_repairs_after_holdout": 0,
            "student_training_verified": False,
            "promotion_proven": False,
        }
        atomic_json(root / "final-curriculum-audit.json", report)
        print(json.dumps(report, indent=2))
    finally:
        run.close()


if __name__ == "__main__":
    main()
