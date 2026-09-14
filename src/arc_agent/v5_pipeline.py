"""Label-isolated, durable symbolic distillation collection and local SFT export."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from arc_agent.data import load_tasks
from arc_agent.v2_codex import resolve_codex_cli
from arc_agent.v2_models import ResponseSnapshot
from arc_agent.v2_openai import response_output_text
from arc_agent.v4_config import content_hash, file_hash
from arc_agent.v4_state import ExperimentRun, atomic_json, freeze_split
from arc_agent.v4_tools import program_catalog, task_grids
from arc_agent.v5_config import V5Config
from arc_agent.v5_symbolic import TeacherArtifact, teaching_task, validate_symbols, verify_artifact

STUDENT_INSTRUCTION = """Build or revise a compact symbolic working model of the supplied ARC
demonstrations. Return only SymbolicModel JSON: version, entities, definitions, relations,
hypotheses, unresolved, proposed_transfer_skill. Entity cells and relations must be grounded.
Definitions have id, provisional_meaning, depends_on. Hypothesis ordered_actions reference
definition IDs in execution order. Interpretations and transfer skills are provisional, not
proved facts. Use verifier counterexamples when present. Do not return final answer grids,
executable witnesses or private reasoning. Preserve uncertainty and necessary evidence."""


def source_identity() -> str:
    root = Path(__file__).parent
    return content_hash({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def select_tasks(config: V5Config):
    if not config.split.is_file():
        raise ValueError("existing frozen split required; V5 will not create a new split")
    tasks = load_tasks(config.data)
    split = freeze_split(tasks, config.split)  # equality check, no overwrite
    by_id = {task.task_id: task for task in tasks}
    selected = [by_id[key] for key in split["groups"]["training"]][: config.max_tasks]
    return selected, split


def run_identity(config: V5Config, split: dict, *, runtime: bool = True) -> dict:
    identity = {
        "version": 5,
        "config": config.model_dump(mode="json"),
        "split_hash": content_hash(split),
        "source_hash": source_identity(),
    }
    if runtime:
        binary = resolve_codex_cli(config.teacher.codex_cli)
        identity["codex_binary_sha256"] = file_hash(Path(binary))
        identity["codex_version"] = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return identity


def student_prompt(visible, previous=None, feedback=None) -> list[dict]:
    evidence = {"grids": task_grids(visible)}
    if previous is not None:
        evidence["previous_symbolic_model"] = previous
    if feedback is not None:
        evidence["visible_verifier_feedback"] = feedback
    return [
        {"role": "system", "content": STUDENT_INSTRUCTION},
        {"role": "user", "content": json.dumps(evidence, separators=(",", ":"))},
    ]


class DSLTrainingContract:
    """Default V5 contract; later collectors may replace the witness, not the target."""

    artifact_type = TeacherArtifact
    verify = staticmethod(verify_artifact)

    @staticmethod
    def initial_prompt(messages):
        return json.dumps(
            {
                "evidence": json.loads(messages[-1]["content"]),
                "executable_witness_catalog": program_catalog(),
                "contract": "ordered_actions contain definition IDs; "
                "witness is optional {hypothesis_id, program_json}; "
                "program uses {op,args,steps}; never literal grids",
            }
        )


def collect_task(task, config: V5Config, run: ExperimentRun, teacher, *, contract=None) -> dict:
    contract = contract or DSLTrainingContract()
    key = f"task:{task.task_id}"
    state = run.get(key) or {
        "task_id": task.task_id,
        "status": "running",
        "round": 0,
        "started_at": time.time(),
        "records": [],
        "previous_id": None,
        "previous_symbolic": None,
        "feedback": None,
    }
    if state["status"] != "running":
        return state
    run.save(key, state)
    try:
        visible, heldout = teaching_task(task)
    except ValueError as exc:
        state.update(status="insufficient_demonstrations", error=str(exc))
        run.save(key, state)
        return state
    deadline = state["started_at"] + config.task_seconds
    while state["round"] < config.max_rounds:
        if shutil.disk_usage(run.root).free < 10 * 1024**3:
            raise RuntimeError("preserve at least 10 GiB free disk space")
        index = state["round"]
        call_key = f"call:{task.task_id}:{index}"
        call = run.get(call_key)
        if call is None:
            if time.time() >= deadline:
                state["status"] = "timed_out"
                break
            messages = student_prompt(visible, state["previous_symbolic"], state["feedback"])
            if index == 0:
                prompt = contract.initial_prompt(messages)
            else:
                # Provider-native continuity stays in the same teacher thread.
                prompt = json.dumps(
                    {
                        "visible_verifier_feedback": state["feedback"],
                        "request": "Revise the symbolic model and optional witness.",
                    }
                )
            request_key = content_hash({"run": str(run.root.resolve()), "call": call_key})
            call = {
                "request": {
                    "prompt": prompt,
                    "model": config.teacher.model,
                    "token_limit": config.output_token_hint,
                    "request_key": request_key,
                },
                "student_prompt": messages,
                "snapshot": None,
            }
            run.save(call_key, call)
        atomic_json(
            run.root / "active-call.json",
            {
                "task": task.task_id,
                "round": index,
                "teacher": config.teacher.model,
                "deadline": deadline,
                "call_key": call_key,
            },
        )
        request = call["request"]

        def checkpoint(snapshot, call=call, call_key=call_key):
            call["snapshot"] = snapshot.model_dump(mode="json")
            run.save(call_key, call)

        snapshot = ResponseSnapshot.model_validate(call["snapshot"]) if call["snapshot"] else None
        if snapshot is None:
            # No provider request is sent until its exact inputs are durable.
            snapshot = teacher.create(
                prompt=request["prompt"],
                task_id=task.task_id,
                phase="symbolic_training",
                round_index=index,
                max_output_tokens=request["token_limit"],
                previous_response_id=state["previous_id"],
                request_key=request["request_key"],
                model=config.teacher.model,
                checkpoint=checkpoint,
            )
            checkpoint(snapshot)
        while snapshot.status in {"queued", "in_progress"}:
            if time.time() >= deadline:
                teacher.cancel(snapshot.response_id)
                state["status"] = "timed_out"
                run.save(key, state)
                return state
            time.sleep(min(config.poll_seconds, max(0, deadline - time.time())))
            snapshot = teacher.retrieve(snapshot.response_id, request=request)
            checkpoint(snapshot)
        if snapshot.status != "completed":
            state.update(
                status="provider_stopped",
                provider_status=snapshot.status,
                error=snapshot.body.get("error"),
            )
            break
        raw = response_output_text(snapshot.body)
        artifact = None
        try:
            artifact = contract.artifact_type.model_validate_json(raw)
            verification = contract.verify(artifact, visible, heldout)
        except (ValueError, subprocess.TimeoutExpired) as exc:
            verification = {"accepted": False, "visible_verified": False, "error": str(exc)[:1200]}
        record = {
            "task_id": task.task_id,
            "task_hash": content_hash(task.model_dump(mode="json")),
            "split": "training",
            "category": "interpretation" if index == 0 else "repair",
            "teacher_model": config.teacher.model,
            "auth_mode": config.teacher.auth_mode,
            "teacher_response_id": snapshot.response_id,
            "student": config.student,
            "student_revision": config.student_revision,
            "round": index,
            "prompt": call["student_prompt"],
            "artifact": artifact.model_dump(mode="json") if artifact else None,
            "structured_valid": artifact is not None,
            "raw_response_ref": run.artifacts.put({"output_text": raw}),
            "verification": verification,
            "usage": snapshot.usage.model_dump(mode="json"),
            "elapsed_seconds": time.time() - state["started_at"],
        }
        record_ref = run.artifacts.put(record)
        if record_ref not in state["records"]:
            state["records"].append(record_ref)
        state["round"] = index + 1
        state["previous_id"] = snapshot.response_id
        if artifact:
            state["previous_symbolic"] = artifact.symbolic.model_dump(mode="json")
        # Neither the holdout score nor labels are returned to the teacher. Stop at the
        # first visible-demo solution, even if it fails the sealed demonstration.
        if verification.get("visible_verified"):
            state["status"] = "accepted" if verification["accepted"] else "heldout_rejected"
        elif artifact and verification.get("grounding_valid") and artifact.witness is None:
            state["status"] = "symbolic_only"
        elif state["round"] >= config.max_rounds:
            state["status"] = "unresolved"
        else:
            state["feedback"] = verification.get("visible_feedback") or {
                "error": verification.get("error", "No verified witness")
            }
        run.save(key, state)
        print(
            json.dumps(
                {
                    "task": task.task_id,
                    "round": index,
                    "status": state["status"],
                    "accepted": verification["accepted"],
                }
            ),
            flush=True,
        )
        if state["status"] != "running":
            return state
    run.save(key, state)
    return state


def export_examples(
    run: ExperimentRun, config: V5Config, tasks: list, split: dict, *, contract=None
) -> dict:
    """Reverify source membership and artifacts; export no grid-answer supervision."""
    contract = contract or DSLTrainingContract()
    by_id = {task.task_id: task for task in tasks}
    examples, rejected, seen = [], [], set()
    for task in tasks:
        state = run.get(f"task:{task.task_id}") or {}
        for ref in state.get("records", []):
            record = run.artifacts.get(ref)
            task_id = record["task_id"]
            if (
                task_id not in split["groups"]["training"]
                or task_id not in by_id
                or record["split"] != "training"
                or record["teacher_model"] != "gpt-6-astra"
                or record["auth_mode"] != "chatgpt_subscription"
                or record["student"] != config.student
                or record["student_revision"] != config.student_revision
                or record["task_hash"] != split["task_hashes"][task_id]
            ):
                raise ValueError("training isolation or model lineage violation")
            if not record["verification"].get("grounding_valid"):
                rejected.append(ref)
                continue
            artifact = contract.artifact_type.model_validate(record["artifact"])
            visible, heldout = teaching_task(by_id[task_id])
            validate_symbols(artifact.symbolic, visible)
            grounded = {
                "version": 1,
                "entities": [
                    e.model_dump(mode="json", exclude={"proposed_role"})
                    for e in artifact.symbolic.entities
                ],
                "relations": [r.model_dump(mode="json") for r in artifact.symbolic.relations],
            }
            grounding_prompt = [
                {
                    "role": "system",
                    "content": "Describe selected entities and checked relations as "
                    "GroundedScene JSON with version, entities (id, grid, cells), relations. "
                    "Report exact observations only, not inferred roles, hypotheses, rules, "
                    "answer grids or private reasoning.",
                },
                {
                    "role": "user",
                    "content": json.dumps({"grids": task_grids(visible)}, separators=(",", ":")),
                },
            ]
            ground_completion = [
                {"role": "assistant", "content": json.dumps(grounded, separators=(",", ":"))}
            ]
            ground_digest = content_hash(
                {"prompt": grounding_prompt, "completion": ground_completion}
            )
            if ground_digest not in seen:
                examples.append(
                    {
                        "prompt": grounding_prompt,
                        "completion": ground_completion,
                        "source_task_id": task_id,
                        "category": "grounding",
                        "record_ref": ref,
                    }
                )
                seen.add(ground_digest)
            if not record["verification"].get("accepted"):
                rejected.append(ref)
                continue
            if not contract.verify(artifact, visible, heldout)["accepted"]:
                raise ValueError("accepted training artifact failed independent revalidation")
            completion = [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        artifact.symbolic.model_dump(mode="json"), separators=(",", ":")
                    ),
                }
            ]
            example = {
                "prompt": record["prompt"],
                "completion": completion,
                "source_task_id": task_id,
                "category": record["category"],
                "record_ref": ref,
            }
            digest = content_hash({"prompt": example["prompt"], "completion": completion})
            if digest not in seen:
                examples.append(example)
                seen.add(digest)
    # JSON list is HF Datasets-compatible. Write complete file atomically; no append duplication.
    atomic_json(run.root / "symbolic-sft.json", examples)
    report = {
        "examples": len(examples),
        "grounding_examples": sum(e["category"] == "grounding" for e in examples),
        "full_symbolic_examples": sum(e["category"] != "grounding" for e in examples),
        "rejected_records": len(rejected),
        "rejected_refs": rejected,
        "source_tasks": len({e["source_task_id"] for e in examples}),
        "teacher": config.teacher.model,
        "student": config.student,
        "target": "explicit_symbolic_model_not_final_grids",
        "dataset_hash": content_hash(examples),
        "free_form_semantics_verified": False,
        "training_launched": False,
        "tokenization_and_loss_mask_gate_passed": False,
    }
    atomic_json(run.root / "dataset-report.json", report)
    return report
