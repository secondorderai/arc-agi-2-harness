from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
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
from arc_agent.v2_verifier import (
    failed_induction_guards,
    verification_feedback,
    verify_induction_program,
)

ESCALATION_MODEL = "gpt-5.6-sol"
MODEL_ESCALATION_AFTER_ROUNDS = 20
SYNTHESIS_SOURCE_KINDS = {"luna", "terra", "sol"}
PLATEAU_REPEAT_THRESHOLD = 3
PLATEAU_WINDOW = 12


def adaptive_synthesis_policy() -> dict[str, Any]:
    return {
        "version": 2,
        "initial_model": "gpt-5.6-luna",
        "escalation_model": ESCALATION_MODEL,
        "escalation_starts_at_attempt": MODEL_ESCALATION_AFTER_ROUNDS + 1,
        "reasoning_effort": "xhigh",
        "plateau_repeat_threshold": PLATEAU_REPEAT_THRESHOLD,
        "plateau_window": PLATEAU_WINDOW,
        "independent_resynthesis_omits_prior_source": True,
        "independent_resynthesis_omits_retrieved_programs": True,
    }


class PhasePaused(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class WorkerStopped(RuntimeError):
    pass


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
    *,
    phase: str,
    task_id: str,
    round_index: int,
    token_index: int,
    model: str,
    strategy_mode: str,
    prompt: str,
) -> str:
    material = (
        f"{phase}\0{task_id}\0{round_index}\0{token_index}\0"
        f"{model}\0{strategy_mode}\0{prompt}"
    )
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


def _independent_plateau_feedback(
    row: dict[str, Any] | None, plateau: dict[str, Any]
) -> str:
    header = (
        "A repeated verifier-failure signature was observed "
        f"{plateau['count']} times after the last independent restart, in rounds "
        f"{plateau['rounds']}. The previous program source and hypothesis are intentionally "
        "withheld. Use the counterexamples below only as constraints for a new derivation."
    )
    if row is None:
        return header
    _, verification = _candidate_program(row)
    return f"{header}\n\n{verification_feedback(verification)}"


def _model_for_round(configured_model: str, round_index: int) -> str:
    if configured_model == "gpt-5.6-luna" and round_index >= MODEL_ESCALATION_AFTER_ROUNDS:
        return ESCALATION_MODEL
    return configured_model


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
        stop_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.state = state
        self._client = client
        self.sleep = sleep
        self.stop_event = stop_event

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

    def _check_stopped(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise WorkerStopped

    def _wait(self, seconds: float) -> None:
        if self.stop_event is not None:
            if self.stop_event.wait(max(0.0, seconds)):
                raise WorkerStopped
            return
        self.sleep(seconds)

    def _pause_for_error(self, phase: str, request_key: str, error: Exception) -> None:
        status = "paused_quota" if isinstance(error, QuotaExhausted) else "paused_configuration"
        self.state.record_request_error(request_key, status, str(error))
        self.state.pause(phase, status, str(error))
        raise PhasePaused(status, str(error)) from error

    def _obtain_response(
        self,
        request: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> ResponseSnapshot | None:
        phase = str(request["phase"])
        request_key = str(request["request_key"])
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        )
        resuming_quota_failure = request.get("status") == "paused_quota"
        retry_delay = min(
            self.config.openai.retry_max_seconds,
            self.config.openai.retry_initial_seconds
            * (2 ** min(20, int(request.get("retry_count") or 0))),
        )
        poll_delay = self.config.openai.poll_initial_seconds
        while True:
            self._check_stopped()
            if deadline is not None and time.monotonic() >= deadline:
                response_id = request.get("response_id")
                if response_id:
                    with suppress(Exception):
                        self.client.cancel(str(response_id))
                self.state.record_request_error(
                    request_key,
                    "timed_out",
                    f"model turn exceeded {timeout_seconds:.1f} seconds",
                )
                self.state.mark_request_ingested(request_key)
                task = self.state.task_row(phase, str(request["task_id"]))
                self.state.update_task(
                    phase,
                    str(request["task_id"]),
                    round_index=max(
                        int(task["round_index"]),
                        int(request["round_index"]) + 1,
                    ),
                    previous_response_id=None,
                    no_progress=int(task["no_progress"]) + 1,
                )
                return None
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
                        model=str(request.get("model") or self.config.openai.model),
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
                    wait_seconds = poll_delay
                    if deadline is not None:
                        wait_seconds = min(wait_seconds, max(0.0, deadline - time.monotonic()))
                    self._wait(wait_seconds)
                    poll_delay = min(self.config.openai.poll_max_seconds, poll_delay * 2)
                    continue
                if snapshot.status == "failed":
                    error = classify_response_failure(snapshot)
                    if isinstance(error, QuotaExhausted) and resuming_quota_failure:
                        # A terminal Codex turn cannot change after the account quota
                        # becomes available again.  Preserve that failed response, retire
                        # its request, and start a fresh chain on the next loop iteration
                        # instead of replaying the same immutable failure forever.
                        self.state.mark_request_ingested(request_key)
                        task = self.state.task_row(phase, str(request["task_id"]))
                        self.state.update_task(
                            phase,
                            str(request["task_id"]),
                            round_index=max(
                                int(task["round_index"]),
                                int(request["round_index"]) + 1,
                            ),
                            previous_response_id=None,
                        )
                        return None
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
                self._wait(delay)
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
        round_index = int(row["round_index"])
        token_index = min(int(row["token_index"]), len(self.config.openai.max_output_tokens) - 1)
        best = self.state.best_candidate(phase, task.task_id)
        evaluation_sol_after = self.config.evaluation.sol_after_model_attempts
        if phase == "evaluation" and (
            evaluation_sol_after is not None and round_index >= evaluation_sol_after
        ):
            model = ESCALATION_MODEL
        else:
            model = _model_for_round(self.config.openai.model, round_index)
        plateau = self.state.repeated_failure_plateau(
            phase,
            task.task_id,
            threshold=PLATEAU_REPEAT_THRESHOLD,
            window=PLATEAU_WINDOW,
        )
        independent = plateau is not None
        strategy_mode = "independent_plateau" if independent else "refinement"
        feedback = (
            _independent_plateau_feedback(
                self.state.candidate(str(plateau["candidate_id"])) or best,
                plateau,
            )
            if plateau is not None
            else _candidate_feedback(best)
        )
        retrieved = [] if independent else self._retrieved_programs(task, phase=phase)
        prompt = build_synthesis_prompt(
            sanitize_task(task),
            phase=phase,
            search_lens=SEARCH_LENSES[lens_index % len(SEARCH_LENSES)],
            feedback=feedback,
            retrieved_programs=retrieved,
            retrieved_limit=0 if independent else self.config.retrieved_programs,
            independent_resynthesis=independent,
        )
        previous_response_id = row["previous_response_id"]
        if independent:
            previous_response_id = None
        elif previous_response_id:
            previous_model = self.state.request_model_for_response(str(previous_response_id))
            if (previous_model or self.config.openai.model) != model:
                previous_response_id = None
        request_key = _request_key(
            phase=phase,
            task_id=task.task_id,
            round_index=round_index,
            token_index=token_index,
            model=model,
            strategy_mode=strategy_mode,
            prompt=prompt,
        )
        return self.state.prepare_request(
            request_key=request_key,
            phase=phase,
            task_id=task.task_id,
            round_index=round_index,
            prompt=prompt,
            previous_response_id=previous_response_id,
            token_limit=self.config.openai.max_output_tokens[token_index],
            model=model,
            strategy_mode=strategy_mode,
            plateau_signature=str(plateau["signature"]) if plateau is not None else None,
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
            source_kind=str(request.get("model") or self.config.openai.model).rsplit("-", 1)[-1],
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

    def synthesize_once(
        self,
        task: ArcTask,
        *,
        phase: str,
        timeout_seconds: float | None = None,
    ) -> bool:
        self._check_stopped()
        pending = self.state.pending_request(phase, task.task_id)
        request = pending or self._prepare_active_request(task, phase=phase)
        snapshot = self._obtain_response(request, timeout_seconds=timeout_seconds)
        if snapshot is None:
            return False
        return self._ingest_response(
            task=task,
            phase=phase,
            request=request,
            snapshot=snapshot,
        )

    def _finalize_best_effort_training_task(
        self,
        task: ArcTask,
        *,
        max_refinement_rounds: int,
    ) -> None:
        candidate: dict[str, Any] | None = None
        verification: InductionVerification | None = None
        for row in self.state.candidates("training", task.task_id):
            measured = InductionVerification.model_validate_json(row["verification_json"])
            if measured.static_safe:
                candidate = row
                verification = measured
                break
        if candidate is None or verification is None:
            fallback_source = "def solve(train, grid):\n    return [row[:] for row in grid]"
            verification = verify_induction_program(
                fallback_source,
                task,
                guards=self.config.guards,
                include_test_labels=True,
                run_transformations=True,
                seed=self.config.seed,
            )
            candidate_id = self.state.record_candidate(
                phase="training",
                task_id=task.task_id,
                round_index=max(0, max_refinement_rounds - 1),
                source_kind="round_limit_fallback",
                response_id=None,
                hypothesis=(
                    "Safe identity fallback used because no statically safe candidate existed."
                ),
                strategy_tags=["best-effort", "identity-fallback"],
                invariants=["Preserve grid dimensions and cell values."],
                python_source=fallback_source,
                verification=verification,
            )
            candidate = self.state.candidate(candidate_id)
            if candidate is None:
                raise RuntimeError("best-effort fallback candidate was not persisted")
        acceptance_kind = "full" if verification.accepted else "best_effort"
        failed_guards = (
            []
            if verification.accepted
            else failed_induction_guards(verification, task, guards=self.config.guards)
        )
        self.state.accept_training_task(
            task.task_id,
            _bank_program(task, candidate=candidate, verification=verification),
            str(candidate["candidate_id"]),
            acceptance_kind=acceptance_kind,
            failed_guards=failed_guards,
        )

    def _run_training_task(
        self,
        task: ArcTask,
        stop_event: threading.Event,
        max_refinement_rounds: int,
    ) -> PhasePaused | None:
        shared_client = self._client
        with V2State(
            self.state.workspace,
            acquire_workspace_lock=False,
        ) as worker_state:
            worker = V2Orchestrator(
                self.config,
                worker_state,
                client=shared_client,
                sleep=self.sleep,
                stop_event=stop_event,
            )
            try:
                while True:
                    task_row = worker_state.task_row("training", task.task_id)
                    if task_row["status"] == "accepted":
                        break
                    if int(task_row["round_index"]) >= max_refinement_rounds:
                        worker._finalize_best_effort_training_task(
                            task,
                            max_refinement_rounds=max_refinement_rounds,
                        )
                        break
                    worker.synthesize_once(task, phase="training")
                return None
            except PhasePaused as paused:
                stop_event.set()
                return paused
            except WorkerStopped:
                return None
            finally:
                if shared_client is None:
                    worker.close()

    def _run_training_pool(
        self,
        tasks_by_id: dict[str, ArcTask],
        *,
        concurrency: int,
        max_refinement_rounds: int,
    ) -> PhasePaused | None:
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="arc-v2-training",
        )
        futures: dict[Future[PhasePaused | None], str] = {}

        def fill_slots() -> None:
            available = concurrency - len(futures)
            for row in self.state.claim_pending_tasks("training", limit=available):
                task_id = str(row["task_id"])
                future = executor.submit(
                    self._run_training_task,
                    tasks_by_id[task_id],
                    stop_event,
                    max_refinement_rounds,
                )
                futures[future] = task_id

        paused: PhasePaused | None = None
        try:
            fill_slots()
            while futures and paused is None:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    result = future.result()
                    if result is not None:
                        paused = result
                        stop_event.set()
                if paused is None:
                    fill_slots()
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            self.state.requeue_running_tasks("training")
            raise
        stop_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        self.state.requeue_running_tasks("training")
        return paused

    def build_bank(
        self,
        tasks: list[ArcTask],
        *,
        dataset_hash: str,
        resume_only: bool = False,
        restart_task: bool = False,
        finalize: bool = True,
        concurrency: int | None = None,
        max_refinement_rounds: int | None = None,
    ) -> RunResult:
        worker_count = concurrency or self.config.training_concurrency
        refinement_limit = max_refinement_rounds or self.config.max_refinement_rounds
        if not 1 <= worker_count <= 16:
            raise ValueError("training concurrency must be between 1 and 16")
        if refinement_limit < 1:
            raise ValueError("max refinement rounds must be positive")
        ordered = training_task_order(tasks)
        self.state.set_meta("openai_auth_mode", self.config.openai.auth_mode)
        self.state.set_meta("training_adaptive_synthesis_policy", adaptive_synthesis_policy())
        self.state.set_meta("training_concurrency", worker_count)
        self.state.set_meta("training_max_refinement_rounds", refinement_limit)
        self.state.initialize_phase(
            phase="training",
            task_ids=[task.task_id for task in ordered],
            dataset_hash=dataset_hash,
            config_hash=self.config.sha256(),
            resume_only=resume_only,
        )
        self.state.requeue_running_tasks("training")
        if restart_task:
            self.state.restart_active_task("training")
        by_id = {task.task_id: task for task in ordered}
        try:
            paused = self._run_training_pool(
                by_id,
                concurrency=worker_count,
                max_refinement_rounds=refinement_limit,
            )
            if paused is not None:
                active = self.state.active_task("training")
                return RunResult(
                    status=paused.status,
                    active_task_id=str(active["task_id"]) if active else None,
                    completed_tasks=self._completed_count("training"),
                    total_tasks=len(ordered),
                    message=str(paused),
                )
            if finalize:
                self.finalize_bank(
                    ordered,
                    dataset_hash=dataset_hash,
                    concurrency=worker_count,
                )
            else:
                self.state.complete_phase("training")
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

    def _compute_compatibility(
        self,
        task: ArcTask,
        program: BankProgram,
    ) -> tuple[str, str, list[float], InductionVerification]:
        verification = verify_induction_program(
            program.python_source,
            task,
            guards=self.config.guards,
            include_test_labels=True,
            run_transformations=False,
            seed=self.config.seed,
        )
        return (
            program.program_hash,
            task.task_id,
            compatibility_features(task, program),
            verification,
        )

    def _build_compatibility_matrix(
        self,
        tasks: list[ArcTask],
        programs: list[BankProgram],
        *,
        concurrency: int,
    ) -> None:
        if not 1 <= concurrency <= 16:
            raise ValueError("compatibility concurrency must be between 1 and 16")

        pending = (
            (task, program)
            for task in tasks
            for program in programs
            if not self.state.has_compatibility(program.program_hash, task.task_id)
        )
        executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="arc-v2-compatibility",
        )
        futures: dict[
            Future[tuple[str, str, list[float], InductionVerification]],
            tuple[str, str],
        ] = {}

        def fill_slots() -> None:
            while len(futures) < concurrency:
                try:
                    task, program = next(pending)
                except StopIteration:
                    return
                future = executor.submit(self._compute_compatibility, task, program)
                futures[future] = (program.program_hash, task.task_id)

        try:
            fill_slots()
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    program_hash, task_id, features, verification = future.result()
                    self.state.record_compatibility(
                        program_hash=program_hash,
                        task_id=task_id,
                        features=features,
                        verification=verification,
                    )
                fill_slots()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def finalize_bank(
        self,
        tasks: list[ArcTask],
        *,
        dataset_hash: str,
        concurrency: int = 1,
    ) -> None:
        programs = self.state.list_programs()
        if not programs:
            raise RuntimeError("cannot finalize an empty program bank")
        self.state.set_meta("training_stage", "compatibility")
        self.state.set_meta("compatibility_concurrency", concurrency)
        self._build_compatibility_matrix(tasks, programs, concurrency=concurrency)
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
        acceptance_counts = self.state.acceptance_counts("training")
        best_effort_tasks = self.state.best_effort_tasks("training")
        manifest = {
            "schema_version": 1,
            "experiment": self.config.name,
            "model": self.config.openai.model,
            "models_used": self.state.request_counts_by_model(self.config.openai.model),
            "adaptive_synthesis_policy": adaptive_synthesis_policy(),
            "reasoning_effort": self.config.openai.reasoning_effort,
            "reasoning_mode": "standard",
            "authentication": self.config.openai.auth_mode,
            "dataset_sha256": dataset_hash,
            "config_sha256": self.config.sha256(),
            "accepted_training_tasks": len(tasks),
            "full_guard_acceptances": acceptance_counts.get("full", 0),
            "best_effort_acceptances": acceptance_counts.get("best_effort", 0),
            "best_effort_tasks": best_effort_tasks,
            "max_refinement_rounds": self.state.get_meta(
                "training_max_refinement_rounds",
                self.config.max_refinement_rounds,
            ),
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
        policy = self.config.evaluation
        target = policy.direct_verified_target or self.config.ranker.direct_verified_target
        initial_limit = min(len(programs), policy.direct_scan_limit or len(programs))
        expand_limit = min(len(programs), policy.direct_expand_limit or initial_limit)
        current_limit = initial_limit

        def verify(program: BankProgram) -> InductionVerification:
            sanitized = sanitize_task(task)
            if policy.staged_verification:
                quick_guards = self.config.guards.model_copy(
                    update={"require_leave_one_out": False}
                )
                quick = verify_induction_program(
                    program.python_source,
                    sanitized,
                    guards=quick_guards,
                    include_test_labels=False,
                    run_transformations=False,
                    seed=self.config.seed,
                )
                if not quick.accepted:
                    return quick
            return verify_induction_program(
                program.python_source,
                sanitized,
                guards=self.config.guards,
                include_test_labels=False,
                run_transformations=True,
                seed=self.config.seed,
            )

        with ThreadPoolExecutor(
            max_workers=policy.direct_concurrency,
            thread_name_prefix="arc-v2-direct",
        ) as executor:
            while cursor < len(programs) and len(accepted) < target:
                self._check_stopped()
                if cursor >= current_limit:
                    if accepted and current_limit < expand_limit:
                        current_limit = expand_limit
                    else:
                        break
                batch_end = min(current_limit, cursor + policy.direct_concurrency)
                batch = programs[cursor:batch_end]
                # executor.map preserves ranking order. Only this main thread
                # writes checkpoints, so cursor advancement stays deterministic.
                results = list(executor.map(verify, batch))
                for program, verification in zip(batch, results, strict=True):
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
                            for row in self.state.accepted_candidates(
                                "evaluation", task.task_id
                            )
                            if row["source_kind"] == "direct"
                        ]
        return accepted

    def _evaluation_budget_remaining(self) -> float | None:
        limit = self.config.evaluation.active_time_limit_seconds
        if limit is None:
            return None
        return max(0.0, limit - float(self.state.summary("evaluation")["active_seconds"]))

    def _evaluation_solve_budget_exhausted(self) -> bool:
        remaining = self._evaluation_budget_remaining()
        return bool(
            remaining is not None
            and remaining <= self.config.evaluation.freeze_reserve_seconds
        )

    def _set_evaluation_stage(self, stage: str) -> None:
        if self.state.get_meta("evaluation_stage") == stage:
            return
        self.state.set_meta("evaluation_stage", stage)
        self.state.set_meta(
            "evaluation_stage_started_active_seconds",
            float(self.state.summary("evaluation")["active_seconds"]),
        )

    def _evaluation_stage_budget_exhausted(self, limit: float | None) -> bool:
        if limit is None:
            return False
        started = float(
            self.state.get_meta("evaluation_stage_started_active_seconds", 0.0)
        )
        current = float(self.state.summary("evaluation")["active_seconds"])
        return current - started >= limit

    def _save_selected_evaluation_output(
        self,
        task: ArcTask,
        *,
        complete: bool,
        selection_stage: str,
    ) -> None:
        attempts, provenance = self._select_attempts(task)
        provenance["selection_stage"] = selection_stage
        self.state.save_evaluation_output(
            task.task_id,
            [attempt.model_dump(mode="json") for attempt in attempts],
            provenance,
            complete=complete,
        )

    def _run_evaluation_direct_task(
        self,
        task: ArcTask,
        stop_event: threading.Event,
    ) -> PhasePaused | None:
        with V2State(self.state.workspace, acquire_workspace_lock=False) as worker_state:
            worker = V2Orchestrator(
                self.config,
                worker_state,
                sleep=self.sleep,
                stop_event=stop_event,
            )
            try:
                worker._direct_evaluation(task)
                worker._save_selected_evaluation_output(
                    task,
                    complete=False,
                    selection_stage="direct_coverage",
                )
                return None
            except PhasePaused as paused:
                stop_event.set()
                return paused
            except WorkerStopped:
                return None
            finally:
                worker.close()

    @staticmethod
    def _acquire_evaluation_llm_slot(
        semaphore: threading.Semaphore,
        stop_event: threading.Event,
    ) -> bool:
        while not stop_event.is_set():
            if semaphore.acquire(timeout=1.0):
                return True
        return False

    def _run_evaluation_model_task(
        self,
        task: ArcTask,
        stop_event: threading.Event,
        llm_semaphore: threading.Semaphore,
        *,
        selection_stage: str,
        skip_if_accepted: bool,
    ) -> PhasePaused | None:
        shared_client = self._client
        with V2State(self.state.workspace, acquire_workspace_lock=False) as worker_state:
            worker = V2Orchestrator(
                self.config,
                worker_state,
                client=shared_client,
                sleep=self.sleep,
                stop_event=stop_event,
            )
            try:
                accepted = worker_state.accepted_candidates("evaluation", task.task_id)
                if not (skip_if_accepted and accepted):
                    if not self._acquire_evaluation_llm_slot(llm_semaphore, stop_event):
                        raise WorkerStopped
                    try:
                        worker.synthesize_once(
                            task,
                            phase="evaluation",
                            timeout_seconds=self.config.evaluation.model_turn_timeout_seconds,
                        )
                    finally:
                        llm_semaphore.release()
                worker._save_selected_evaluation_output(
                    task,
                    complete=True,
                    selection_stage=selection_stage,
                )
                return None
            except PhasePaused as paused:
                stop_event.set()
                return paused
            except WorkerStopped:
                return None
            finally:
                if shared_client is None:
                    worker.close()

    def _run_evaluation_pool(
        self,
        tasks_by_id: dict[str, ArcTask],
        *,
        source_status: str,
        requeue_status: str,
        mode: str,
        llm_semaphore: threading.Semaphore,
        stage_budget_seconds: float | None = None,
    ) -> tuple[PhasePaused | None, bool, bool]:
        concurrency = self.config.evaluation.task_concurrency
        stop_event = threading.Event()
        executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=f"arc-v2-evaluation-{mode}",
        )
        futures: dict[Future[PhasePaused | None], str] = {}

        def fill_slots() -> None:
            if stop_event.is_set():
                return
            available = concurrency - len(futures)
            for row in self.state.claim_tasks(
                "evaluation",
                source_status=source_status,
                limit=available,
            ):
                task_id = str(row["task_id"])
                if mode == "direct":
                    future = executor.submit(
                        self._run_evaluation_direct_task,
                        tasks_by_id[task_id],
                        stop_event,
                    )
                else:
                    future = executor.submit(
                        self._run_evaluation_model_task,
                        tasks_by_id[task_id],
                        stop_event,
                        llm_semaphore,
                        selection_stage=mode,
                        skip_if_accepted=mode == "first_model",
                    )
                futures[future] = task_id

        paused: PhasePaused | None = None
        budget_exhausted = False
        stage_exhausted = False
        try:
            fill_slots()
            while (
                futures
                and paused is None
                and not budget_exhausted
                and not stage_exhausted
            ):
                completed, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    result = future.result()
                    if result is not None:
                        paused = result
                        stop_event.set()
                if self._evaluation_solve_budget_exhausted():
                    budget_exhausted = True
                    stop_event.set()
                elif self._evaluation_stage_budget_exhausted(stage_budget_seconds):
                    stage_exhausted = True
                    stop_event.set()
                elif paused is None:
                    fill_slots()
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            self.state.requeue_running_tasks("evaluation", status=requeue_status)
            raise
        stop_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        self.state.requeue_running_tasks("evaluation", status=requeue_status)
        return paused, budget_exhausted, stage_exhausted

    def _eligible_evaluation_improvements(self) -> list[str]:
        maximum = self.config.evaluation.max_model_attempts_per_task
        if maximum is None:
            return []
        rows = self.state.connection.execute(
            """
            SELECT task_id, round_index, no_progress, best_score, position
            FROM phase_tasks
            WHERE phase='evaluation' AND status='accepted'
            ORDER BY no_progress ASC, best_score DESC, position ASC
            """
        ).fetchall()
        eligible: list[str] = []
        for row in rows:
            task_id = str(row["task_id"])
            if int(row["round_index"]) >= maximum or int(row["no_progress"]) > 0:
                continue
            if self.state.accepted_candidates("evaluation", task_id):
                continue
            eligible.append(task_id)
        return eligible

    def _schedule_evaluation_improvements(self, task_ids: list[str]) -> None:
        if not task_ids:
            return
        with self.state.transaction(immediate=True) as connection:
            placeholders = ",".join("?" for _ in task_ids)
            connection.execute(
                f"""
                UPDATE phase_tasks SET status='improve_pending'
                WHERE phase='evaluation' AND status='accepted'
                    AND task_id IN ({placeholders})
                """,
                task_ids,
            )

    def _finalize_all_evaluation_outputs(
        self,
        tasks: list[ArcTask],
        *,
        selection_stage: str,
    ) -> None:
        for task in tasks:
            row = self.state.task_row("evaluation", task.task_id)
            if row["status"] == "accepted":
                continue
            self._save_selected_evaluation_output(
                task,
                complete=True,
                selection_stage=selection_stage,
            )

    def _cover_all_evaluation_tasks(
        self,
        tasks: list[ArcTask],
        *,
        selection_stage: str,
    ) -> None:
        for task in tasks:
            row = self.state.task_row("evaluation", task.task_id)
            if row["status"] in {"accepted", "covered"}:
                continue
            self._save_selected_evaluation_output(
                task,
                complete=False,
                selection_stage=selection_stage,
            )

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

    def _budgeted_evaluation_result(
        self,
        *,
        status: str,
        total_tasks: int,
        message: str,
    ) -> RunResult:
        active = self.state.active_task("evaluation")
        return RunResult(
            status=status,
            active_task_id=str(active["task_id"]) if active else None,
            completed_tasks=self._completed_count("evaluation"),
            total_tasks=total_tasks,
            message=message,
        )

    def _evaluate_with_active_time_budget(
        self,
        sanitized: list[ArcTask],
        labelled: list[ArcTask],
    ) -> RunResult:
        by_id = {task.task_id: task for task in sanitized}
        policy = self.config.evaluation
        llm_semaphore = threading.Semaphore(policy.llm_concurrency)
        stage = str(self.state.get_meta("evaluation_stage", "direct_coverage"))
        self._set_evaluation_stage(stage)

        try:
            while stage != "frozen":
                if self._evaluation_solve_budget_exhausted():
                    self.state.set_meta("evaluation_budget_exhausted", True)
                    stage = "freezing"
                    self._set_evaluation_stage(stage)

                if stage == "direct_coverage":
                    self.state.requeue_running_tasks("evaluation", status="pending")
                    paused, exhausted, stage_exhausted = self._run_evaluation_pool(
                        by_id,
                        source_status="pending",
                        requeue_status="pending",
                        mode="direct",
                        llm_semaphore=llm_semaphore,
                        stage_budget_seconds=policy.direct_coverage_budget_seconds,
                    )
                    if paused is not None:
                        return self._budgeted_evaluation_result(
                            status=paused.status,
                            total_tasks=len(sanitized),
                            message=str(paused),
                        )
                    if stage_exhausted:
                        self._cover_all_evaluation_tasks(
                            sanitized,
                            selection_stage="direct_coverage_budget",
                        )
                    stage = "freezing" if exhausted else "first_model"
                    self._set_evaluation_stage(stage)
                    continue

                if stage == "first_model":
                    self.state.requeue_running_tasks("evaluation", status="covered")
                    paused, exhausted, stage_exhausted = self._run_evaluation_pool(
                        by_id,
                        source_status="covered",
                        requeue_status="covered",
                        mode="first_model",
                        llm_semaphore=llm_semaphore,
                        stage_budget_seconds=policy.first_model_budget_seconds,
                    )
                    if paused is not None:
                        return self._budgeted_evaluation_result(
                            status=paused.status,
                            total_tasks=len(sanitized),
                            message=str(paused),
                        )
                    if stage_exhausted:
                        self._finalize_all_evaluation_outputs(
                            sanitized,
                            selection_stage="first_model_budget",
                        )
                    stage = "freezing" if exhausted else "improvement"
                    self._set_evaluation_stage(stage)
                    continue

                if stage == "improvement":
                    self.state.requeue_running_tasks(
                        "evaluation", status="improve_pending"
                    )
                    queued = self.state.connection.execute(
                        """
                        SELECT COUNT(*) AS count FROM phase_tasks
                        WHERE phase='evaluation' AND status='improve_pending'
                        """
                    ).fetchone()["count"]
                    if not queued:
                        self._schedule_evaluation_improvements(
                            self._eligible_evaluation_improvements()
                        )
                        queued = self.state.connection.execute(
                            """
                            SELECT COUNT(*) AS count FROM phase_tasks
                            WHERE phase='evaluation' AND status='improve_pending'
                            """
                        ).fetchone()["count"]
                    if not queued:
                        stage = "freezing"
                        self._set_evaluation_stage(stage)
                        continue
                    paused, exhausted, _ = self._run_evaluation_pool(
                        by_id,
                        source_status="improve_pending",
                        requeue_status="improve_pending",
                        mode="improvement",
                        llm_semaphore=llm_semaphore,
                    )
                    if paused is not None:
                        return self._budgeted_evaluation_result(
                            status=paused.status,
                            total_tasks=len(sanitized),
                            message=str(paused),
                        )
                    if exhausted:
                        stage = "freezing"
                        self._set_evaluation_stage(stage)
                    continue

                if stage == "freezing":
                    self._finalize_all_evaluation_outputs(
                        sanitized,
                        selection_stage=(
                            "budget_deadline"
                            if self.state.get_meta("evaluation_budget_exhausted", False)
                            else "final"
                        ),
                    )
                    self._freeze_and_score(labelled)
                    stage = "frozen"
                    self._set_evaluation_stage(stage)
                    continue

                raise RuntimeError(f"unknown evaluation stage {stage!r}")
        except KeyboardInterrupt:
            self.state.pause("evaluation", "paused_user", "interrupted by user")
            return self._budgeted_evaluation_result(
                status="paused_user",
                total_tasks=len(sanitized),
                message="interrupted by user",
            )

        return RunResult(
            status="complete",
            completed_tasks=len(sanitized),
            total_tasks=len(sanitized),
            message=f"frozen submission: {self.state.workspace / 'submission.json'}",
        )

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
        self.state.set_meta("evaluation_adaptive_synthesis_policy", adaptive_synthesis_policy())
        self.state.set_meta(
            "evaluation_direct_policy",
            self.config.evaluation.model_dump(mode="json"),
        )
        ordered_labelled = sorted(labelled_tasks, key=lambda task: task.task_id)
        sanitized = [sanitize_task(task) for task in ordered_labelled]
        self.state.initialize_phase(
            phase="evaluation",
            task_ids=[task.task_id for task in sanitized],
            dataset_hash=dataset_hash,
            config_hash=self.config.evaluation_sha256(),
            resume_only=resume_only,
        )
        self.state.set_meta(
            "evaluation_active_time_limit_seconds",
            self.config.evaluation.active_time_limit_seconds,
        )
        self.state.set_meta(
            "evaluation_freeze_reserve_seconds",
            self.config.evaluation.freeze_reserve_seconds,
        )
        if restart_task:
            self.state.restart_active_task("evaluation")
        if self.config.evaluation.active_time_limit_seconds is not None:
            return self._evaluate_with_active_time_budget(sanitized, ordered_labelled)
        by_id = {task.task_id: task for task in sanitized}
        try:
            while (active := self.state.active_task("evaluation")) is not None:
                task = by_id[str(active["task_id"])]
                self._direct_evaluation(task)
                synthesized = [
                    row
                    for row in self.state.accepted_candidates("evaluation", task.task_id)
                    if row["source_kind"] in SYNTHESIS_SOURCE_KINDS
                ]
                while not synthesized:
                    self.synthesize_once(task, phase="evaluation")
                    synthesized = [
                        row
                        for row in self.state.accepted_candidates("evaluation", task.task_id)
                        if row["source_kind"] in SYNTHESIS_SOURCE_KINDS
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
                SELECT response_id, model, usage_json, created_at, updated_at FROM requests
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
                    "terra_candidates": sum(row["source_kind"] == "terra" for row in candidates),
                    "terra_verified": sum(
                        row["source_kind"] == "terra" and bool(row["accepted"])
                        for row in candidates
                    ),
                    "sol_candidates": sum(row["source_kind"] == "sol" for row in candidates),
                    "sol_verified": sum(
                        row["source_kind"] == "sol" and bool(row["accepted"])
                        for row in candidates
                    ),
                    "resynthesized_candidates": sum(
                        row["source_kind"] in SYNTHESIS_SOURCE_KINDS for row in candidates
                    ),
                    "selected_candidate_ids": selected_ids,
                    "selected_sources": [
                        sources.get(candidate_id) if candidate_id else "fallback"
                        for candidate_id in selected_ids
                    ],
                    "model_calls": len(request_rows),
                    "model_calls_by_model": {
                        model: sum(
                            (row["model"] or self.config.openai.model) == model
                            for row in request_rows
                        )
                        for model in sorted(
                            {
                                str(row["model"] or self.config.openai.model)
                                for row in request_rows
                            }
                        )
                    },
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
