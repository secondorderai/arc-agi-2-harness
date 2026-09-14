from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc_agent.data import submission_to_json
from arc_agent.models import ArcTask, Attempt, Submission
from arc_agent.scoring import score_submission
from arc_agent.v2_models import ResponseSnapshot
from arc_agent.v2_openai import (
    ConfigurationError,
    QuotaExhausted,
    ResponseExpired,
    TransientAPIError,
    classify_response_failure,
    response_snapshot,
)
from arc_agent.v3_config import V3ExperimentConfig
from arc_agent.v3_dsl import (
    dsl_sha256,
    failure_signature,
    operator_catalog,
    validate_signature,
    verify_signature,
)
from arc_agent.v3_fingerprint import fingerprint_task, select_pilot_tasks
from arc_agent.v3_matcher import generate_candidates, sanitize_task, select_attempts
from arc_agent.v3_models import (
    GameSignature,
    ProposedSignature,
    SignatureVerification,
    V3RunResult,
    finalize_signature,
)
from arc_agent.v3_prompt import build_signature_prompt
from arc_agent.v3_state import V3State
from arc_agent.v3_teacher import create_teacher_client, parse_signature_response


class PhasePaused(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _proposal(signature: GameSignature) -> ProposedSignature:
    payload = signature.model_dump(mode="python")
    payload.pop("canonical_hash")
    payload.pop("source_task_ids")
    payload.pop("complexity")
    payload.pop("provenance")
    return ProposedSignature.model_validate(payload)


def _serialize_signatures(signatures: list[GameSignature]) -> str:
    serialized = "\n".join(
        signature.model_dump_json()
        for signature in sorted(signatures, key=lambda item: item.canonical_hash)
    )
    return serialized + "\n" if serialized else ""


def _score_predictions(task: ArcTask, predictions: list[list[list[int]]]) -> list[bool]:
    return [
        pair.output is not None and prediction == pair.output
        for pair, prediction in zip(task.test, predictions, strict=True)
    ]


class V3Orchestrator:
    def __init__(
        self,
        config: V3ExperimentConfig,
        state: V3State,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.state = state
        self._client = client
        self.sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_teacher_client(
                self.config.teacher,
                workspace=self.state.workspace,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                close()
            self._client = None

    def _pause(self, phase: str, status: str, message: str) -> None:
        self.state.finish_phase(phase, status, message)
        raise PhasePaused(status, message)

    def _obtain_response(self, request: dict[str, Any]) -> ResponseSnapshot:
        request_key = str(request["request_key"])
        poll_delay = self.config.teacher.poll_initial_seconds
        retry_delay = min(
            self.config.teacher.retry_max_seconds,
            self.config.teacher.retry_initial_seconds
            * (2 ** min(12, int(request.get("retry_count") or 0))),
        )
        while True:
            request = self.state.request(request_key)
            try:
                if request.get("body_json") and request.get("status") == "completed":
                    return response_snapshot(json.loads(request["body_json"]))
                response_id = request.get("response_id")
                if response_id:
                    snapshot = self.client.retrieve(str(response_id), request=request)
                else:
                    snapshot = self.client.create(
                        prompt=str(request["prompt"]),
                        task_id=str(request["task_id"]),
                        phase=str(request["phase"]),
                        round_index=int(request["round_index"]),
                        max_output_tokens=int(request["token_limit"]),
                        previous_response_id=request.get("previous_response_id"),
                        request_key=str(request["idempotency_key"]),
                        model=str(request["model"]),
                        checkpoint=lambda item: self.state.record_response(request_key, item),
                    )
                self.state.record_response(request_key, snapshot)
                if snapshot.status == "completed":
                    return snapshot
                if snapshot.status in {"failed", "cancelled"}:
                    raise classify_response_failure(snapshot)
                if snapshot.status == "incomplete":
                    return snapshot
                self.sleep(poll_delay)
                poll_delay = min(self.config.teacher.poll_max_seconds, poll_delay * 1.5)
            except ResponseExpired as exc:
                self.state.record_request_error(request_key, "retrying", str(exc))
                self.state.reset_expired_response(request_key)
                poll_delay = self.config.teacher.poll_initial_seconds
            except QuotaExhausted as exc:
                self.state.record_request_error(request_key, "paused_quota", str(exc))
                self._pause(str(request["phase"]), "paused_quota", str(exc))
            except ConfigurationError as exc:
                if exc.code == "incomplete_response":
                    self.state.record_request_error(request_key, "incomplete", str(exc))
                    raise
                self.state.record_request_error(request_key, "paused_configuration", str(exc))
                self._pause(str(request["phase"]), "paused_configuration", str(exc))
            except TransientAPIError as exc:
                failed_response = self.state.request(request_key).get("status") in {
                    "failed",
                    "cancelled",
                }
                self.state.record_request_error(request_key, "retrying", str(exc))
                if failed_response:
                    self.state.reset_expired_response(request_key)
                delay = exc.retry_after if exc.retry_after is not None else retry_delay
                jitter = random.Random(request_key + str(request.get("retry_count"))).uniform(0, 1)
                self.sleep(min(self.config.teacher.retry_max_seconds, delay + jitter))
                retry_delay = min(self.config.teacher.retry_max_seconds, retry_delay * 2)

    def _latest_candidate(
        self, task_id: str
    ) -> tuple[ProposedSignature | None, SignatureVerification | None]:
        rows = self.state.candidates("pilot", task_id)
        if not rows:
            return None, None
        signature = GameSignature.model_validate_json(rows[0]["signature_json"])
        verification = SignatureVerification.model_validate_json(rows[0]["verification_json"])
        return _proposal(signature), verification

    def _write_expressivity_gap(self, task: ArcTask, family: str) -> Path:
        rows = self.state.candidates("pilot", task.task_id)
        payload = {
            "schema_version": 1,
            "task_id": task.task_id,
            "family": family,
            "dsl_sha256": dsl_sha256(),
            "attempts": len(rows),
            "best_candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "score": row["score"],
                    "verification": json.loads(row["verification_json"]),
                    "signature": json.loads(row["signature_json"]),
                }
                for row in rows[:3]
            ],
            "test_outputs_included": False,
            "required_action": (
                "add a reusable typed primitive using training demonstrations only, then start "
                "a new workspace because the DSL hash has changed"
            ),
        }
        target = self.state.workspace / "expressivity_gaps" / f"{task.task_id}.json"
        _atomic_write(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    def _synthesize_task(self, task: ArcTask, family: str) -> None:
        sanitized = sanitize_task(task)
        row = self.state.task_row("pilot", task.task_id)
        round_index = int(row["round_index"])
        if round_index >= self.config.pilot.max_attempts_per_task:
            path = self._write_expressivity_gap(sanitized, family)
            self.state.update_task(
                "pilot",
                task.task_id,
                status="expressivity_gap",
                error=f"maximum attempts reached; see {path}",
            )
            return
        self.state.update_task("pilot", task.task_id, status="running")
        recent = self.state.recent_failure_signatures(
            "pilot", task.task_id, self.config.pilot.plateau_rounds
        )
        independent = (
            len(recent) == self.config.pilot.plateau_rounds and len(set(recent)) == 1
        ) or int(row["no_progress"]) >= self.config.pilot.plateau_rounds
        previous, verification = self._latest_candidate(task.task_id)
        if independent:
            previous = None
            verification = None
        prompt = build_signature_prompt(
            sanitized,
            family=family,
            round_index=round_index,
            previous=previous,
            verification=verification,
            independent=independent,
            max_attempts=self.config.pilot.max_attempts_per_task,
        )
        token_limits = self.config.teacher.max_output_tokens
        token_limit = token_limits[min(round_index, len(token_limits) - 1)]
        request = self.state.prepare_request(
            phase="pilot",
            task_id=task.task_id,
            round_index=round_index,
            prompt=prompt,
            previous_response_id=None if independent else row.get("previous_response_id"),
            token_limit=token_limit,
            model=self.config.teacher.model,
        )
        try:
            snapshot = self._obtain_response(request)
            proposal = parse_signature_response(snapshot.body)
            if proposal.family != family:
                raise ValueError(
                    f"teacher returned family {proposal.family}, expected selected family {family}"
                )
            validate_signature(
                proposal,
                max_pipeline_steps=self.config.pilot.max_pipeline_steps,
                max_parameters=self.config.pilot.max_parameters,
                max_literal_scalars=self.config.pilot.max_literal_scalars,
                forbidden_identifiers=[task.task_id],
            )
            signature = finalize_signature(
                proposal,
                source_task_ids=[task.task_id],
                provenance={
                    "kind": "teacher",
                    "model": self.config.teacher.model,
                    "response_id": snapshot.response_id,
                    "round": str(round_index),
                },
            )
            candidate_verification = verify_signature(signature, sanitized)
        except ConfigurationError:
            raise
        except (ValueError, TypeError) as exc:
            self.state.record_request_error(str(request["request_key"]), "parse_failed", str(exc))
            self.state.mark_ingested(str(request["request_key"]))
            self.state.update_task(
                "pilot",
                task.task_id,
                round_index=round_index + 1,
                previous_response_id=(
                    snapshot.response_id
                    if "snapshot" in locals()
                    else row.get("previous_response_id")
                ),
                no_progress=int(row["no_progress"]) + 1,
                error=str(exc),
            )
            return
        prior_best = float(row["best_score"])
        observed_failure = failure_signature(candidate_verification)
        candidate_id = self.state.record_candidate(
            phase="pilot",
            task_id=task.task_id,
            round_index=round_index,
            source_kind="teacher",
            response_id=snapshot.response_id,
            signature=signature,
            verification=candidate_verification,
            failure_signature=observed_failure,
        )
        self.state.mark_ingested(str(request["request_key"]))
        if candidate_verification.accepted:
            self.state.accept_signature(
                task_id=task.task_id,
                signature=signature,
                verification=candidate_verification,
                candidate_id=candidate_id,
            )
            return
        improved = candidate_verification.score > prior_best
        self.state.update_task(
            "pilot",
            task.task_id,
            round_index=round_index + 1,
            previous_response_id=snapshot.response_id,
            no_progress=0 if improved else int(row["no_progress"]) + 1,
            error="signature did not reproduce every demonstration",
        )

    def _freeze_pilot(self, tasks: list[ArcTask], selection: dict[str, Any]) -> None:
        signatures = self.state.signatures()
        serialized = _serialize_signatures(signatures)
        signature_path = self.state.workspace / "signatures.jsonl"
        _atomic_write(signature_path, serialized)
        signature_hash = hashlib.sha256(serialized.encode()).hexdigest()
        prompts = "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in self.state.prompts("pilot")
        )
        if prompts:
            prompts += "\n"
        _atomic_write(self.state.workspace / "pilot_prompts.jsonl", prompts)
        prompts_hash = hashlib.sha256(prompts.encode()).hexdigest()
        manifest = {
            "schema_version": 1,
            "experiment": self.config.name,
            "teacher_model": self.config.teacher.model,
            "reasoning_effort": self.config.teacher.reasoning_effort,
            "evaluation_requires_network": False,
            "dataset_sha256": self.state.get_meta("pilot_dataset_hash"),
            "pilot_tasks": [task.task_id for task in tasks],
            "selection": selection,
            "signatures": len(signatures),
            "signature_bank_sha256": signature_hash,
            "prompt_bank_sha256": prompts_hash,
            "dsl_sha256": dsl_sha256(),
            "config_sha256": self.config.sha256(),
            "operator_catalog": operator_catalog(),
            "usage": self.state.usage().model_dump(mode="json"),
            "requests": self.state.request_manifest("pilot"),
            "frozen_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write(
            self.state.workspace / "pilot_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        self.state.set_meta("pilot_manifest", manifest)
        self.state.set_meta("pilot_frozen", True)

    def _validate_pilot(self, tasks: list[ArcTask]) -> dict[str, Any]:
        signatures = self.state.signatures()
        fingerprints = self.state.fingerprints()
        by_source = {
            task_id: signature
            for signature in signatures
            for task_id in signature.source_task_ids
        }
        own_passes = 0
        isolated_passes = 0
        rows: list[dict[str, Any]] = []
        for task in tasks:
            own = by_source[task.task_id]
            own_verification = verify_signature(own, sanitize_task(task))
            own_correct = all(_score_predictions(task, own_verification.predictions))
            own_passes += int(own_correct)
            candidates, matches = generate_candidates(
                sanitize_task(task),
                signatures,
                fingerprints,
                settings=self.config.search,
                excluded_task_ids={task.task_id},
            )
            attempts, provenance = select_attempts(task, candidates)
            isolated_correct = all(
                pair.output in [attempt.attempt_1, attempt.attempt_2]
                for pair, attempt in zip(task.test, attempts, strict=True)
            )
            isolated_passes += int(isolated_correct)
            rows.append(
                {
                    "task_id": task.task_id,
                    "family": fingerprint_task(task).predicted_family,
                    "own_signature_hidden_test_exact": own_correct,
                    "source_isolated_pass_at_2": isolated_correct,
                    "retrieved": [match.model_dump(mode="json") for match in matches],
                    "selection": provenance,
                }
            )
        result = {
            "schema_version": 1,
            "tasks": len(tasks),
            "own_signature_hidden_test_exact": own_passes,
            "source_isolated_pass_at_2_tasks": isolated_passes,
            "own_signature_gate_passed": own_passes >= 8,
            "source_isolated_gate_passed": isolated_passes >= 4,
            "details": rows,
        }
        _atomic_write(
            self.state.workspace / "pilot_validation.json",
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
        self.state.set_meta("pilot_validation", result)
        return result

    def build_pilot(
        self,
        tasks: list[ArcTask],
        *,
        dataset_hash: str,
        resume_only: bool = False,
    ) -> V3RunResult:
        selection = self.state.get_meta("pilot_selection")
        if selection is None:
            selected, selection = select_pilot_tasks(tasks)
        else:
            by_task_id = {task.task_id: task for task in tasks}
            try:
                selected = [by_task_id[task_id] for task_id in selection["task_ids"]]
            except (KeyError, TypeError) as exc:
                raise RuntimeError("stored V3 pilot selection is invalid for this dataset") from exc
            if len(selected) != self.config.pilot.size:
                raise RuntimeError("stored V3 pilot selection has the wrong task count")
        family_by_task = {
            task_id: row["family"]
            for row in selection["pairs"]
            for task_id in row["task_ids"]
        }
        self.state.initialize_phase(
            phase="pilot",
            tasks=[(task.task_id, family_by_task[task.task_id]) for task in selected],
            dataset_hash=dataset_hash,
            config_hash=self.config.sha256(),
            dsl_hash=dsl_sha256(),
            resume_only=resume_only,
        )
        if self.state.get_meta("pilot_frozen", False):
            validation = self.state.get_meta("pilot_validation")
            if validation is None:
                validation = self._validate_pilot(selected)
            self.state.finish_phase("pilot", "complete", "frozen ten-signature pilot")
            return V3RunResult(
                status="complete",
                completed_tasks=10,
                total_tasks=10,
                message=(
                    "pilot already frozen; own hidden tests "
                    f"{validation.get('own_signature_hidden_test_exact', 0)}/10, "
                    "source-isolated "
                    f"{validation.get('source_isolated_pass_at_2_tasks', 0)}/10"
                ),
            )
        self.state.set_meta("pilot_selection", selection)
        for task in selected:
            self.state.record_fingerprint(fingerprint_task(task))
        by_id = {task.task_id: task for task in selected}
        try:
            while row := self.state.next_task("pilot"):
                self._synthesize_task(by_id[str(row["task_id"])], str(row["family"]))
                self.state.checkpoint_active("pilot")
                refreshed = self.state.task_row("pilot", str(row["task_id"]))
                if refreshed["status"] == "expressivity_gap":
                    message = str(refreshed["error"])
                    self.state.finish_phase("pilot", "needs_dsl_extension", message)
                    return V3RunResult(
                        status="needs_dsl_extension",
                        active_task_id=str(row["task_id"]),
                        completed_tasks=self.state.summary("pilot")["completed_tasks"],
                        total_tasks=10,
                        message=message,
                    )
            self._freeze_pilot(selected, selection)
            validation = self._validate_pilot(selected)
            self.state.finish_phase("pilot", "complete", "frozen ten-signature pilot")
            return V3RunResult(
                status="complete",
                completed_tasks=10,
                total_tasks=10,
                message=(
                    "pilot frozen; own hidden tests "
                    f"{validation['own_signature_hidden_test_exact']}/10, source-isolated "
                    f"{validation['source_isolated_pass_at_2_tasks']}/10"
                ),
            )
        except PhasePaused as paused:
            summary = self.state.summary("pilot")
            return V3RunResult(
                status=paused.status,
                active_task_id=summary["active_task_id"],
                completed_tasks=summary["completed_tasks"],
                total_tasks=summary["total_tasks"],
                message=str(paused),
            )
        except KeyboardInterrupt:
            self.state.finish_phase("pilot", "paused_user", "interrupted by user")
            summary = self.state.summary("pilot")
            return V3RunResult(
                status="paused_user",
                active_task_id=summary["active_task_id"],
                completed_tasks=summary["completed_tasks"],
                total_tasks=summary["total_tasks"],
                message="interrupted by user",
            )

    def _fallback_attempts(self, task: ArcTask) -> list[Attempt]:
        return [
            Attempt(
                attempt_1=[list(row) for row in pair.input],
                attempt_2=[[0 for _ in row] for row in pair.input],
            )
            for pair in task.test
        ]

    def _freeze_evaluation(self, tasks: list[ArcTask]) -> dict[str, Any]:
        rows = {row["task_id"]: row for row in self.state.evaluation_outputs()}
        submission: Submission = {
            task.task_id: [
                Attempt.model_validate(item)
                for item in json.loads(rows[task.task_id]["attempts_json"])
            ]
            for task in tasks
        }
        encoded = json.dumps(submission_to_json(submission), sort_keys=True, separators=(",", ":"))
        submission_hash = hashlib.sha256(encoded.encode()).hexdigest()
        _atomic_write(self.state.workspace / "v3_submission.json", encoded)
        self.state.set_meta("evaluation_submission_sha256", submission_hash)
        self.state.set_meta("evaluation_frozen_at", datetime.now(UTC).isoformat())

        labels_available = all(pair.output is not None for task in tasks for pair in task.test)
        measured = score_submission(tasks, submission) if labels_available else None
        candidate_oracle_outputs = 0
        candidate_oracle_tasks = 0
        exact_demo_tasks = 0
        retrieval_contribution_tasks = 0
        global_search_contribution_tasks = 0
        contributions: Counter[str] = Counter()
        per_family: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {"tasks": 0, "correct": 0, "outputs": 0, "correct_outputs": 0}
        )
        task_details: list[dict[str, Any]] = []
        for task in tasks:
            row = rows[task.task_id]
            candidates = json.loads(row["candidates_json"])
            attempts = submission[task.task_id]
            exact_outputs = (
                [
                    pair.output in [attempt.attempt_1, attempt.attempt_2]
                    for pair, attempt in zip(task.test, attempts, strict=True)
                ]
                if labels_available
                else []
            )
            oracle_outputs = (
                [
                    any(
                        len(candidate["predictions"]) == len(task.test)
                        and candidate["predictions"][index] == pair.output
                        for candidate in candidates
                    )
                    for index, pair in enumerate(task.test)
                ]
                if labels_available
                else []
            )
            candidate_oracle_outputs += sum(oracle_outputs)
            candidate_oracle_tasks += int(all(oracle_outputs))
            exact_demo_tasks += int(any(candidate["accepted"] for candidate in candidates))
            family = fingerprint_task(task).predicted_family
            per_family[family]["tasks"] += 1
            if labels_available:
                per_family[family]["correct"] += int(all(exact_outputs))
                per_family[family]["outputs"] += len(exact_outputs)
                per_family[family]["correct_outputs"] += sum(exact_outputs)
            provenance = json.loads(row["provenance_json"])
            selected_ids = {
                provenance.get("attempt_1_candidate"),
                provenance.get("attempt_2_candidate"),
            }
            for candidate in candidates:
                if candidate["candidate_id"] in selected_ids:
                    contributions[candidate["source_kind"]] += 1
            selected_sources = {
                candidate["source_kind"]
                for candidate in candidates
                if candidate["candidate_id"] in selected_ids
            }
            if (
                labels_available
                and all(exact_outputs)
                and selected_sources.intersection({"retrieved", "mutation"})
            ):
                retrieval_contribution_tasks += 1
            if labels_available and all(exact_outputs) and "global_search" in selected_sources:
                global_search_contribution_tasks += 1
            task_details.append(
                {
                    "task_id": task.task_id,
                    "family": family,
                    "correct_outputs": sum(exact_outputs) if labels_available else None,
                    "outputs": len(task.test),
                    "oracle_outputs": sum(oracle_outputs) if labels_available else None,
                    "provenance": provenance,
                }
            )
        validation = self.state.get_meta("pilot_validation", {})
        active_seconds = self.state.checkpoint_active("evaluation")
        if labels_available:
            for metrics in per_family.values():
                metrics["strict_accuracy"] = metrics["correct"] / max(1, metrics["tasks"])
                metrics["pass_at_2"] = metrics["correct_outputs"] / max(1, metrics["outputs"])
        strict_gate = measured is not None and measured.strict_task_accuracy >= 0.05
        pass_gate = measured is not None and measured.pass_at_2 >= 0.05
        runtime_gate = active_seconds <= self.config.evaluation.target_active_seconds
        report = {
            "schema_version": 1,
            "tasks": len(tasks),
            "test_outputs": sum(len(task.test) for task in tasks),
            "labels_available": labels_available,
            "strict_task_accuracy": measured.strict_task_accuracy if measured else None,
            "pass_at_2": measured.pass_at_2 if measured else None,
            "candidate_oracle_outputs": candidate_oracle_outputs if labels_available else None,
            "candidate_oracle_tasks": candidate_oracle_tasks if labels_available else None,
            "candidate_oracle_output_accuracy": (
                candidate_oracle_outputs / max(1, sum(len(task.test) for task in tasks))
                if labels_available
                else None
            ),
            "candidate_oracle_task_accuracy": (
                candidate_oracle_tasks / max(1, len(tasks)) if labels_available else None
            ),
            "exact_demonstration_coverage": exact_demo_tasks / max(1, len(tasks)),
            "top_three_retrieval_contribution_tasks": retrieval_contribution_tasks,
            "global_search_contribution_tasks": global_search_contribution_tasks,
            "contributions": dict(contributions),
            "per_family": dict(per_family),
            "active_seconds": active_seconds,
            "dataset_sha256": self.state.get_meta("evaluation_dataset_hash"),
            "config_sha256": self.config.sha256(),
            "dsl_sha256": dsl_sha256(),
            "submission_sha256": submission_hash,
            "evaluation_used_network": False,
            "pilot_validation": validation,
            "go_no_go": {
                "all_ten_expressible": (
                    self.state.summary("pilot")["completed_tasks"] == 10
                ),
                "own_hidden_test_gate": validation.get("own_signature_hidden_test_exact", 0) >= 8,
                "source_isolated_gate": validation.get("source_isolated_pass_at_2_tasks", 0) >= 4,
                "public_strict_gate": strict_gate if labels_available else None,
                "public_pass_at_2_gate": pass_gate if labels_available else None,
                "runtime_gate": runtime_gate,
            },
            "details": task_details,
        }
        report["go_no_go"]["passed"] = (
            all(report["go_no_go"].values()) if labels_available else None
        )
        _atomic_write(
            self.state.workspace / "v3_evaluation_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        self.state.set_meta("evaluation_report", report)
        self.state.set_meta("evaluation_frozen", True)
        return report

    def evaluate(
        self,
        tasks: list[ArcTask],
        *,
        dataset_hash: str,
        resume_only: bool = False,
    ) -> V3RunResult:
        tasks = sorted(tasks, key=lambda task: task.task_id)
        if not self.state.get_meta("pilot_frozen", False):
            raise RuntimeError("V3 pilot must be complete and frozen before evaluation")
        manifest = self.state.get_meta("pilot_manifest", {})
        if manifest.get("dsl_sha256") != dsl_sha256():
            raise RuntimeError("the DSL changed after the pilot was frozen")
        if manifest.get("config_sha256") != self.config.sha256():
            raise RuntimeError("the V3 configuration changed after the pilot was frozen")
        signatures = self.state.signatures()
        serialized_signatures = _serialize_signatures(signatures)
        expected_bank_hash = manifest.get("signature_bank_sha256")
        if hashlib.sha256(serialized_signatures.encode()).hexdigest() != expected_bank_hash:
            raise RuntimeError("the frozen signature bank no longer matches its manifest")
        signature_path = self.state.workspace / "signatures.jsonl"
        if (
            not signature_path.is_file()
            or hashlib.sha256(signature_path.read_bytes()).hexdigest() != expected_bank_hash
        ):
            raise RuntimeError("the frozen signature artifact is missing or has been modified")
        fingerprints = self.state.fingerprints()
        sanitized = [sanitize_task(task) for task in tasks]
        evaluation_fingerprints = {
            task.task_id: fingerprint_task(task) for task in sanitized
        }
        families = {
            task_id: fingerprint.predicted_family
            for task_id, fingerprint in evaluation_fingerprints.items()
        }
        self.state.initialize_phase(
            phase="evaluation",
            tasks=[(task.task_id, families[task.task_id]) for task in sanitized],
            dataset_hash=dataset_hash,
            config_hash=self.config.sha256(),
            dsl_hash=dsl_sha256(),
            resume_only=resume_only,
        )
        if self.state.get_meta("evaluation_frozen", False):
            report = self.state.get_meta("evaluation_report", {})
            self.state.finish_phase("evaluation", "complete", "submission already frozen")
            return V3RunResult(
                status="complete",
                completed_tasks=len(tasks),
                total_tasks=len(tasks),
                message=(
                    "submission already frozen; "
                    + (
                        f"strict={float(report['strict_task_accuracy']) * 100:.2f}% "
                        f"pass@2={float(report['pass_at_2']) * 100:.2f}%"
                        if report.get("labels_available")
                        else "labels unavailable; not scored"
                    )
                ),
            )
        for fingerprint in evaluation_fingerprints.values():
            self.state.record_fingerprint(fingerprint)
        by_id = {task.task_id: task for task in sanitized}
        try:
            while row := self.state.next_task("evaluation"):
                active = self.state.checkpoint_active("evaluation")
                task = by_id[str(row["task_id"])]
                if active >= self.config.evaluation.target_active_seconds:
                    for remaining in sanitized:
                        current = self.state.task_row("evaluation", remaining.task_id)
                        if current["status"] == "accepted":
                            continue
                        self.state.save_evaluation_output(
                            task_id=remaining.task_id,
                            attempts=self._fallback_attempts(remaining),
                            provenance={"budget_fallback": True},
                            candidates=[],
                        )
                    break
                candidates, matches = generate_candidates(
                    task,
                    signatures,
                    fingerprints,
                    settings=self.config.search,
                    task_fingerprint=evaluation_fingerprints[task.task_id],
                )
                attempts, provenance = select_attempts(task, candidates)
                provenance["retrieved_signatures"] = [
                    match.signature_hash for match in matches
                ]
                self.state.record_matches(
                    task.task_id,
                    [match.model_dump(mode="json") for match in matches],
                )
                self.state.save_evaluation_output(
                    task_id=task.task_id,
                    attempts=attempts,
                    provenance=provenance,
                    candidates=candidates,
                )
            report = self._freeze_evaluation(tasks)
            self.state.finish_phase("evaluation", "complete", "submission frozen and scored")
            return V3RunResult(
                status="complete",
                completed_tasks=len(tasks),
                total_tasks=len(tasks),
                message=(
                    (
                        f"strict={report['strict_task_accuracy'] * 100:.2f}% "
                        f"pass@2={report['pass_at_2'] * 100:.2f}% "
                        f"go/no-go={'pass' if report['go_no_go']['passed'] else 'fail'}"
                    )
                    if report["labels_available"]
                    else "unlabelled submission frozen; scoring was not attempted"
                ),
            )
        except KeyboardInterrupt:
            self.state.finish_phase("evaluation", "paused_user", "interrupted by user")
            summary = self.state.summary("evaluation")
            return V3RunResult(
                status="paused_user",
                active_task_id=summary["active_task_id"],
                completed_tasks=summary["completed_tasks"],
                total_tasks=summary["total_tasks"],
                message="interrupted by user",
            )
