from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from arc_agent.data import submission_to_json
from arc_agent.dsl import search_programs
from arc_agent.models import ArcTask, Attempt, Grid, Submission
from arc_agent.scoring import score_submission
from arc_agent.v2_config import V2ExperimentConfig
from arc_agent.v2_models import (
    BankProgram,
    InductionVerification,
    ResponseSnapshot,
    ResponseUsage,
    RunResult,
    VerificationFailure,
)
from arc_agent.v2_openai import (
    ConfigurationError,
    QuotaExhausted,
    ResponseExpired,
    SynthesisClient,
    TransientAPIError,
    classify_response_failure,
    create_synthesis_client,
    parse_synthesized_program,
    response_output_text,
    response_snapshot,
)
from arc_agent.v2_prompt import (
    SEARCH_LENSES,
    build_synthesis_prompt,
    episodic_features,
    sanitize_task,
)
from arc_agent.v2_ranker import (
    compatibility_features,
    heuristic_rank,
    rank_programs,
    train_ranker,
)
from arc_agent.v2_sandbox import canonicalize_source
from arc_agent.v2_state import V2State, atomic_write
from arc_agent.v2_verifier import verification_feedback, verify_induction_program


class PhasePaused(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def training_task_order(tasks: Iterable[ArcTask]) -> list[ArcTask]:
    material = list(tasks)

    def key(task: ArcTask) -> tuple[float, float, str]:
        features = episodic_features(task)
        deterministic = bool(search_programs(task, limit=1))
        complexity = (
            features["mean_input_colors"]
            + features["mean_components"]
            + features["dimension_change_ratio"] * 4
            + features["mean_separator_lines"] * 0.25
        )
        return (0.0 if deterministic else 1.0, complexity, task.task_id)

    return sorted(material, key=key)


def _request_key(
    *, phase: str, task_id: str, round_index: int, token_index: int, prompt: str
) -> str:
    material = f"{phase}\0{task_id}\0{round_index}\0{token_index}\0{prompt}"
    return hashlib.sha256(material.encode()).hexdigest()


def _candidate_program(row: dict[str, Any]) -> tuple[str, InductionVerification]:
    return row["python_source"], InductionVerification.model_validate_json(row["verification_json"])


def _candidate_feedback(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    source, verification = _candidate_program(row)
    return (
        f"Best prior program (score={verification.score:.2f}):\n"
        f"```python\n{source}\n```\n\n{verification_feedback(verification)}"
    )


def _bank_program(
    task: ArcTask,
    *,
    candidate: dict[str, Any],
    verification: InductionVerification,
) -> BankProgram:
    canonical, complexity = canonicalize_source(candidate["python_source"])
    program_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return BankProgram(
        program_hash=program_hash,
        source_task_ids=[task.task_id],
        hypothesis=candidate["hypothesis"],
        strategy_tags=json.loads(candidate["strategy_tags_json"]),
        invariants=json.loads(candidate["invariants_json"]),
        python_source=candidate["python_source"],
        canonical_ast=canonical,
        features=episodic_features(task),
        verification=verification,
        complexity=complexity,
    )


class V2Orchestrator:
    def __init__(
        self,
        config: V2ExperimentConfig,
        state: V2State,
        *,
        client: SynthesisClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.state = state
        self._client = client
        self.sleep = sleep

    @property
    def client(self) -> SynthesisClient:
        if self._client is None:
            self._client = create_synthesis_client(
                self.config.openai,
                workspace=self.state.workspace,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                close()
            self._client = None

    def _pause_for_error(self, phase: str, request_key: str, error: Exception) -> None:
        status = "paused_quota" if isinstance(error, QuotaExhausted) else "paused_configuration"
        self.state.record_request_error(request_key, status, str(error))
        self.state.pause(phase, status, str(error))
        raise PhasePaused(status, str(error)) from error

    def _obtain_response(self, request: dict[str, Any]) -> ResponseSnapshot | None:
        phase = str(request["phase"])
        request_key = str(request["request_key"])
        retry_delay = min(
            self.config.openai.retry_max_seconds,
            self.config.openai.retry_initial_seconds
            * (2 ** min(20, int(request.get("retry_count") or 0))),
        )
        poll_delay = self.config.openai.poll_initial_seconds
        while True:
            try:
                body_json = request.get("body_json")
                if body_json and request.get("status") in {"completed", "incomplete"}:
                    return response_snapshot(json.loads(body_json))
                response_id = request.get("response_id")
                if response_id:
                    snapshot = self.client.retrieve(str(response_id), request=request)
                else:
                    snapshot = self.client.create(
                        prompt=str(request["prompt"]),
                        task_id=str(request["task_id"]),
                        phase=phase,
                        round_index=int(request["round_index"]),
                        max_output_tokens=int(request["token_limit"]),
                        previous_response_id=request.get("previous_response_id"),
                        request_key=request_key,
                        checkpoint=lambda pending: self.state.record_response(request_key, pending),
                    )
                self.state.record_response(request_key, snapshot)
                request = {
                    **request,
                    "response_id": snapshot.response_id,
                    "status": snapshot.status,
                    "body_json": json.dumps(snapshot.body),
                }
                if snapshot.status in {"queued", "in_progress"}:
                    self.sleep(poll_delay)
                    poll_delay = min(self.config.openai.poll_max_seconds, poll_delay * 2)
                    continue
                if snapshot.status == "failed":
                    error = classify_response_failure(snapshot)
                    if isinstance(error, (QuotaExhausted, ConfigurationError)):
                        self._pause_for_error(phase, request_key, error)
                    self.state.record_request_error(request_key, "failed", str(error))
                    self.state.mark_request_ingested(request_key)
                    task = self.state.task_row(phase, str(request["task_id"]))
                    self.state.update_task(
                        phase,
                        str(request["task_id"]),
                        round_index=int(task["round_index"]) + 1,
                        previous_response_id=None,
                        no_progress=int(task["no_progress"]) + 1,
                    )
                    return None
                if snapshot.status == "cancelled":
                    self.state.record_request_error(
                        request_key, "cancelled", "background response was cancelled"
                    )
                    self.state.mark_request_ingested(request_key)
                    task = self.state.task_row(phase, str(request["task_id"]))
                    self.state.update_task(
                        phase,
                        str(request["task_id"]),
                        round_index=int(task["round_index"]) + 1,
                        previous_response_id=None,
                    )
                    return None
                return snapshot
            except ResponseExpired as exc:
                self.state.record_request_error(request_key, "expired", str(exc))
                self.state.mark_request_ingested(request_key)
                task = self.state.task_row(phase, str(request["task_id"]))
                self.state.update_task(
                    phase,
                    str(request["task_id"]),
                    round_index=int(task["round_index"]) + 1,
                    lens_index=int(task["lens_index"]) + 1,
                    previous_response_id=None,
                    no_progress=0,
                )
                return None
            except (QuotaExhausted, ConfigurationError) as exc:
                self._pause_for_error(phase, request_key, exc)
            except TransientAPIError as exc:
                delay = exc.retry_after if exc.retry_after is not None else retry_delay
                self.state.record_request_error(
                    request_key, "transient", str(exc), retry_after=delay
                )
                self.sleep(max(0.0, delay))
                retry_delay = min(self.config.openai.retry_max_seconds, retry_delay * 2)

    def _retrieved_programs(self, task: ArcTask, *, phase: str) -> list[BankProgram]:
        programs = self.state.list_programs()
        if not programs:
            return []
        if phase == "evaluation":
            return rank_programs(
                task,
                programs,
                ranker_path=self.state.workspace / "ranker.pkl",
            )
        return heuristic_rank(task, programs)

    def _prepare_active_request(self, task: ArcTask, *, phase: str) -> dict[str, Any]:
        row = self.state.task_row(phase, task.task_id)
        lens_index = int(row["lens_index"])
        token_index = min(int(row["token_index"]), len(self.config.openai.max_output_tokens) - 1)
        best = self.state.best_candidate(phase, task.task_id)
        prompt = build_synthesis_prompt(
            sanitize_task(task),
            phase=phase,
            search_lens=SEARCH_LENSES[lens_index % len(SEARCH_LENSES)],
            feedback=_candidate_feedback(best),
            retrieved_programs=self._retrieved_programs(task, phase=phase),
            retrieved_limit=self.config.retrieved_programs,
        )
        request_key = _request_key(
            phase=phase,
            task_id=task.task_id,
            round_index=int(row["round_index"]),
            token_index=token_index,
            prompt=prompt,
        )
        return self.state.prepare_request(
            request_key=request_key,
            phase=phase,
            task_id=task.task_id,
            round_index=int(row["round_index"]),
            prompt=prompt,
            previous_response_id=row["previous_response_id"],
            token_limit=self.config.openai.max_output_tokens[token_index],
        )

    def _ingest_response(
        self,
        *,
        task: ArcTask,
        phase: str,
        request: dict[str, Any],
        snapshot: ResponseSnapshot,
    ) -> bool:
        task_row = self.state.task_row(phase, task.task_id)
        request_key = str(request["request_key"])
        if snapshot.status == "incomplete":
            token_index = min(
                int(task_row["token_index"]) + 1,
                len(self.config.openai.max_output_tokens) - 1,
            )
            self.state.mark_request_ingested(request_key)
            self.state.update_task(
                phase,
                task.task_id,
                round_index=int(task_row["round_index"]) + 1,
                token_index=token_index,
                previous_response_id=snapshot.response_id,
                no_progress=int(task_row["no_progress"]) + 1,
            )
            return False
        if snapshot.status != "completed":
            raise TransientAPIError(f"unexpected terminal response status: {snapshot.status}")

        try:
            generated = parse_synthesized_program(snapshot.body)
            verification = verify_induction_program(
                generated.python_source,
                task if phase == "training" else sanitize_task(task),
                guards=self.config.guards,
                include_test_labels=phase == "training",
                run_transformations=True,
                seed=self.config.seed,
            )
        except ValueError as exc:
            generated_text = response_output_text(snapshot.body)[
                : self.config.guards.max_source_chars
            ]
            from arc_agent.v2_models import SynthesizedProgram

            generated = SynthesizedProgram(
                hypothesis="Structured response could not be parsed.",
                python_source=generated_text or "def solve(train, grid):\n    return None",
            )
            verification = InductionVerification(
                failures=[VerificationFailure(case="response_parse", error=str(exc))]
            )

        old_best = float(task_row["best_score"])
        candidate_id = self.state.record_candidate(
            phase=phase,
            task_id=task.task_id,
            round_index=int(request["round_index"]),
            source_kind="luna",
            response_id=snapshot.response_id,
            hypothesis=generated.hypothesis,
            strategy_tags=generated.strategy_tags,
            invariants=generated.invariants,
            python_source=generated.python_source,
            verification=verification,
        )
        self.state.mark_request_ingested(request_key)
        if verification.accepted and phase == "training":
            candidate = self.state.best_candidate(phase, task.task_id)
            if candidate is None or candidate["candidate_id"] != candidate_id:
                candidate = next(
                    row
                    for row in self.state.candidates(phase, task.task_id)
                    if row["candidate_id"] == candidate_id
                )
            self.state.accept_training_task(
                task.task_id,
                _bank_program(task, candidate=candidate, verification=verification),
                candidate_id,
            )
            return True

        improved = verification.score > old_best
        no_progress = 0 if improved else int(task_row["no_progress"]) + 1
        reset = no_progress >= self.config.no_progress_reset_rounds
        self.state.update_task(
            phase,
            task.task_id,
            round_index=int(task_row["round_index"]) + 1,
            token_index=0,
            no_progress=0 if reset else no_progress,
            lens_index=int(task_row["lens_index"]) + int(reset),
            previous_response_id=None if reset else snapshot.response_id,
        )
        return verification.accepted

    def synthesize_once(self, task: ArcTask, *, phase: str) -> bool:
        pending = self.state.pending_request(phase, task.task_id)
        request = pending or self._prepare_active_request(task, phase=phase)
        snapshot = self._obtain_response(request)
        if snapshot is None:
            return False
        return self._ingest_response(
            task=task,
            phase=phase,
            request=request,
            snapshot=snapshot,
        )

    def build_bank(
        self,
        tasks: list[ArcTask],
        *,
        dataset_hash: str,
        resume_only: bool = False,
        restart_task: bool = False,
        finalize: bool = True,
    ) -> RunResult:
        ordered = training_task_order(tasks)
        self.state.set_meta("openai_auth_mode", self.config.openai.auth_mode)
        self.state.initialize_phase(
            phase="training",
            task_ids=[task.task_id for task in ordered],
            dataset_hash=dataset_hash,
            config_hash=self.config.sha256(),
            resume_only=resume_only,
        )
        if restart_task:
            self.state.restart_active_task("training")
        by_id = {task.task_id: task for task in ordered}
        try:
            while (active := self.state.active_task("training")) is not None:
                task = by_id[str(active["task_id"])]
                self.synthesize_once(task, phase="training")
            if finalize:
                self.finalize_bank(ordered, dataset_hash=dataset_hash)
            else:
                self.state.complete_phase("training")
        except PhasePaused as paused:
            active = self.state.active_task("training")
            return RunResult(
                status=paused.status,
                active_task_id=str(active["task_id"]) if active else None,
                completed_tasks=self._completed_count("training"),
                total_tasks=len(ordered),
                message=str(paused),
            )
        except KeyboardInterrupt:
            self.state.pause("training", "paused_user", "interrupted by user")
            active = self.state.active_task("training")
            return RunResult(
                status="paused_user",
                active_task_id=str(active["task_id"]) if active else None,
                completed_tasks=self._completed_count("training"),
                total_tasks=len(ordered),
                message="interrupted by user",
            )
        return RunResult(
            status="complete",
            completed_tasks=len(ordered),
            total_tasks=len(ordered),
            message=f"frozen program bank: {self.state.workspace / 'program_bank.jsonl'}",
        )

    def _completed_count(self, phase: str) -> int:
        row = self.state.connection.execute(
            "SELECT COUNT(*) AS count FROM phase_tasks WHERE phase=? AND status='accepted'",
            (phase,),
        ).fetchone()
        return int(row["count"])

    def finalize_bank(self, tasks: list[ArcTask], *, dataset_hash: str) -> None:
        programs = self.state.list_programs()
        if not programs:
            raise RuntimeError("cannot finalize an empty program bank")
        self.state.set_meta("training_stage", "compatibility")
        for task in tasks:
            for program in programs:
                if self.state.has_compatibility(program.program_hash, task.task_id):
                    continue
                verification = verify_induction_program(
                    program.python_source,
                    task,
                    guards=self.config.guards,
                    include_test_labels=True,
                    run_transformations=False,
                    seed=self.config.seed,
                )
                self.state.record_compatibility(
                    program_hash=program.program_hash,
                    task_id=task.task_id,
                    features=compatibility_features(task, program),
                    verification=verification,
                )
        ranker_metrics: dict[str, Any] = {"enabled": False}
        if self.config.ranker.enabled:
            compatibility_rows = self.state.compatibility_rows()
            self.state.set_meta(
                "ranker_state",
                {
                    "status": "fitting",
                    "examples": len(compatibility_rows),
                    "epochs": self.config.ranker.epochs,
                },
            )
            self.state.set_meta("training_stage", "ranker")
            ranker_metrics = {
                "enabled": True,
                **train_ranker(
                    compatibility_rows,
                    settings=self.config.ranker,
                    output_path=self.state.workspace / "ranker.pkl",
                ),
            }
            self.state.set_meta("ranker_state", {"status": "complete", **ranker_metrics})
        bank_content = "\n".join(program.model_dump_json() for program in programs) + "\n"
        bank_path = atomic_write(self.state.workspace / "program_bank.jsonl", bank_content)
        bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
        timing = self.state.summary("training")
        manifest = {
            "schema_version": 1,
            "experiment": self.config.name,
            "model": self.config.openai.model,
            "reasoning_effort": self.config.openai.reasoning_effort,
            "reasoning_mode": "standard",
            "authentication": self.config.openai.auth_mode,
            "dataset_sha256": dataset_hash,
            "config_sha256": self.config.sha256(),
            "accepted_training_tasks": len(tasks),
            "deduplicated_programs": len(programs),
            "program_bank_sha256": bank_hash,
            "ranker": ranker_metrics,
            "usage": self.state.usage_totals(),
            "quota_pause_count": timing["quota_pause_count"],
            "paused_seconds": timing["paused_seconds"],
            "active_seconds": timing["active_seconds"],
            "pricing_snapshot": {
                "input_per_million_usd": 0.20,
                "cached_input_per_million_usd": 0.02,
                "cache_write_input_per_million_usd": 0.25,
                "output_per_million_usd": 1.20,
                "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
                "billing_basis": (
                    "api_equivalent_estimate_not_subscription_charge"
                    if self.config.openai.auth_mode == "chatgpt_subscription"
                    else "api_usage"
                ),
            },
        }
        atomic_write(
            self.state.workspace / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        self.state.set_meta("training_manifest", manifest)
        self.state.set_meta("training_stage", "frozen")
        self.state.complete_phase("training")

    def _direct_evaluation(self, task: ArcTask) -> list[dict[str, Any]]:
        programs = rank_programs(
            task,
            self.state.list_programs(),
            ranker_path=self.state.workspace / "ranker.pkl",
        )
        task_row = self.state.task_row("evaluation", task.task_id)
        cursor = int(task_row["direct_cursor"])
        accepted = [
            row
            for row in self.state.accepted_candidates("evaluation", task.task_id)
            if row["source_kind"] == "direct"
        ]
        while cursor < len(programs) and len(accepted) < self.config.ranker.direct_verified_target:
            program = programs[cursor]
            verification = verify_induction_program(
                program.python_source,
                sanitize_task(task),
                guards=self.config.guards,
                include_test_labels=False,
                run_transformations=True,
                seed=self.config.seed,
            )
            self.state.record_candidate(
                phase="evaluation",
                task_id=task.task_id,
                round_index=-(cursor + 1),
                source_kind="direct",
                response_id=None,
                hypothesis=program.hypothesis,
                strategy_tags=program.strategy_tags,
                invariants=program.invariants,
                python_source=program.python_source,
                verification=verification,
            )
            cursor += 1
            self.state.update_task("evaluation", task.task_id, direct_cursor=cursor)
            if verification.accepted:
                accepted = [
                    row
                    for row in self.state.accepted_candidates("evaluation", task.task_id)
                    if row["source_kind"] == "direct"
                ]
        return accepted

    def _select_attempts(self, task: ArcTask) -> tuple[list[Attempt], dict[str, Any]]:
        rows = self.state.candidates("evaluation", task.task_id)
        viable = []
        for row in rows:
            verification = InductionVerification.model_validate_json(row["verification_json"])
            if len(verification.predictions) == len(task.test):
                viable.append((row, verification))
        groups: dict[str, list[tuple[dict[str, Any], InductionVerification]]] = defaultdict(list)
        for row, verification in viable:
            key = hashlib.sha256(
                json.dumps(verification.predictions, sort_keys=True).encode()
            ).hexdigest()
            groups[key].append((row, verification))
        ranked = sorted(
            groups.values(),
            key=lambda group: (
                not any(bool(row["accepted"]) for row, _ in group),
                -len(group),
                -max(float(row["score"]) for row, _ in group),
                min(str(row["candidate_id"]) for row, _ in group),
            ),
        )
        selected: list[tuple[dict[str, Any] | None, list[Grid]]] = []
        for group in ranked[:2]:
            row, verification = max(group, key=lambda item: float(item[0]["score"]))
            selected.append((row, verification.predictions))
        fallbacks = [
            [[list(row) for row in pair.input] for pair in task.test],
            [[[0 for _ in row] for row in pair.input] for pair in task.test],
        ]
        for predictions in fallbacks:
            if len(selected) == 2:
                break
            if all(predictions != existing for _, existing in selected):
                selected.append((None, predictions))
        if len(selected) == 1:
            selected.append(selected[0])
        attempts = [
            Attempt(attempt_1=selected[0][1][index], attempt_2=selected[1][1][index])
            for index in range(len(task.test))
        ]
        provenance = {
            "attempt_1_candidate": selected[0][0]["candidate_id"] if selected[0][0] else None,
            "attempt_2_candidate": selected[1][0]["candidate_id"] if selected[1][0] else None,
        }
        return attempts, provenance

    def evaluate(
        self,
        labelled_tasks: list[ArcTask],
        *,
        dataset_hash: str,
        resume_only: bool = False,
        restart_task: bool = False,
    ) -> RunResult:
        if self.state.get_meta("training_status") != "complete":
            raise RuntimeError("the training program bank must be complete before evaluation")
        self.state.set_meta("openai_auth_mode", self.config.openai.auth_mode)
        ordered_labelled = sorted(labelled_tasks, key=lambda task: task.task_id)
        sanitized = [sanitize_task(task) for task in ordered_labelled]
        self.state.initialize_phase(
            phase="evaluation",
            task_ids=[task.task_id for task in sanitized],
            dataset_hash=dataset_hash,
            config_hash=self.config.sha256(),
            resume_only=resume_only,
        )
        if restart_task:
            self.state.restart_active_task("evaluation")
        by_id = {task.task_id: task for task in sanitized}
        try:
            while (active := self.state.active_task("evaluation")) is not None:
                task = by_id[str(active["task_id"])]
                self._direct_evaluation(task)
                luna = [
                    row
                    for row in self.state.accepted_candidates("evaluation", task.task_id)
                    if row["source_kind"] == "luna"
                ]
                while not luna:
                    self.synthesize_once(task, phase="evaluation")
                    luna = [
                        row
                        for row in self.state.accepted_candidates("evaluation", task.task_id)
                        if row["source_kind"] == "luna"
                    ]
                attempts, provenance = self._select_attempts(task)
                self.state.save_evaluation_output(
                    task.task_id,
                    [attempt.model_dump(mode="json") for attempt in attempts],
                    provenance,
                )
            self._freeze_and_score(ordered_labelled)
        except PhasePaused as paused:
            active = self.state.active_task("evaluation")
            return RunResult(
                status=paused.status,
                active_task_id=str(active["task_id"]) if active else None,
                completed_tasks=self._completed_count("evaluation"),
                total_tasks=len(sanitized),
                message=str(paused),
            )
        except KeyboardInterrupt:
            self.state.pause("evaluation", "paused_user", "interrupted by user")
            active = self.state.active_task("evaluation")
            return RunResult(
                status="paused_user",
                active_task_id=str(active["task_id"]) if active else None,
                completed_tasks=self._completed_count("evaluation"),
                total_tasks=len(sanitized),
                message="interrupted by user",
            )
        return RunResult(
            status="complete",
            completed_tasks=len(sanitized),
            total_tasks=len(sanitized),
            message=f"frozen submission: {self.state.workspace / 'submission.json'}",
        )

    def _freeze_and_score(self, labelled_tasks: list[ArcTask]) -> None:
        raw = self.state.evaluation_outputs()
        submission: Submission = {
            task_id: [Attempt.model_validate(attempt) for attempt in attempts]
            for task_id, attempts in raw.items()
        }
        payload = json.dumps(submission_to_json(submission), separators=(",", ":"))
        submission_path = atomic_write(self.state.workspace / "submission.json", payload)
        submission_hash = hashlib.sha256(submission_path.read_bytes()).hexdigest()
        self.state.set_meta("evaluation_submission_sha256", submission_hash)
        self.state.set_meta("evaluation_frozen_at", time.time())

        measured = score_submission(labelled_tasks, submission)
        timing = self.state.summary("evaluation")
        per_task: list[dict[str, Any]] = []
        for task in labelled_tasks:
            candidates = self.state.candidates("evaluation", task.task_id)
            sources = {row["candidate_id"]: row["source_kind"] for row in candidates}
            output = self.state.connection.execute(
                "SELECT provenance_json FROM evaluation_outputs WHERE task_id=?",
                (task.task_id,),
            ).fetchone()
            provenance = json.loads(output["provenance_json"]) if output else {}
            request_rows = self.state.connection.execute(
                """
                SELECT response_id, usage_json, created_at, updated_at FROM requests
                WHERE phase='evaluation' AND task_id=? AND response_id IS NOT NULL
                """,
                (task.task_id,),
            ).fetchall()
            usage = ResponseUsage()
            for row in request_rows:
                if not row["usage_json"]:
                    continue
                item = ResponseUsage.model_validate_json(row["usage_json"])
                usage.input_tokens += item.input_tokens
                usage.cached_input_tokens += item.cached_input_tokens
                usage.cache_write_input_tokens += item.cache_write_input_tokens
                usage.output_tokens += item.output_tokens
                usage.reasoning_tokens += item.reasoning_tokens
            selected_ids = [
                provenance.get("attempt_1_candidate"),
                provenance.get("attempt_2_candidate"),
            ]
            per_task.append(
                {
                    "task_id": task.task_id,
                    "direct_candidates": sum(row["source_kind"] == "direct" for row in candidates),
                    "direct_verified": sum(
                        row["source_kind"] == "direct" and bool(row["accepted"])
                        for row in candidates
                    ),
                    "luna_candidates": sum(row["source_kind"] == "luna" for row in candidates),
                    "luna_verified": sum(
                        row["source_kind"] == "luna" and bool(row["accepted"]) for row in candidates
                    ),
                    "selected_candidate_ids": selected_ids,
                    "selected_sources": [
                        sources.get(candidate_id) if candidate_id else "fallback"
                        for candidate_id in selected_ids
                    ],
                    "model_calls": len(request_rows),
                    "usage": usage.model_dump(mode="json"),
                    "estimated_cost_usd": usage.estimated_cost_usd,
                }
            )
        report = {
            "pass_at_2": measured.pass_at_2,
            "strict_task_accuracy": measured.strict_task_accuracy,
            "correct_outputs": measured.correct_outputs,
            "total_outputs": measured.total_outputs,
            "correct_tasks": measured.correct_tasks,
            "total_tasks": measured.total_tasks,
            "v1_pass_at_2": 0.024,
            "v1_strict_task_accuracy": 0.025,
            "pass_at_2_delta": measured.pass_at_2 - 0.024,
            "strict_delta": measured.strict_task_accuracy - 0.025,
            "submission_sha256": submission_hash,
            "usage": self.state.usage_totals(),
            "quota_pause_count": self.state.get_meta("evaluation_quota_pause_count", 0),
            "paused_seconds": self.state.get_meta("evaluation_paused_seconds", 0.0),
            "active_seconds": timing["active_seconds"],
            "per_task": per_task,
        }
        atomic_write(
            self.state.workspace / "evaluation_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        markdown = f"""# ARC-AGI-2 V2 Frozen Public Evaluation

- Pass@2: **{measured.pass_at_2 * 100:.2f}%** ({measured.correct_outputs}/{measured.total_outputs})
- Strict task accuracy: **{measured.strict_task_accuracy * 100:.2f}%** \
({measured.correct_tasks}/{measured.total_tasks})
- Pass@2 delta from V1: **{(measured.pass_at_2 - 0.024) * 100:+.2f} points**
- Strict delta from V1: **{(measured.strict_task_accuracy - 0.025) * 100:+.2f} points**
- Submission SHA-256: `{submission_hash}`
- Quota pauses: {report["quota_pause_count"]}
- Recorded paused seconds: {report["paused_seconds"]:.1f}
- Recorded active seconds: {report["active_seconds"]:.1f}
"""
        atomic_write(self.state.workspace / "evaluation_report.md", markdown)
        self.state.set_meta("evaluation_report", report)
        self.state.complete_phase("evaluation")
