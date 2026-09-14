from __future__ import annotations

import json
import threading
from pathlib import Path

from arc_agent.models import ArcPair, ArcTask
from arc_agent.v2_config import (
    EvaluationConfig,
    GuardConfig,
    RankerConfig,
    ResponsesConfig,
    V2ExperimentConfig,
)
from arc_agent.v2_experiment import V2Orchestrator
from arc_agent.v2_models import (
    BankProgram,
    InductionVerification,
    ResponseSnapshot,
    ResponseUsage,
    VerificationFailure,
)
from arc_agent.v2_openai import QuotaExhausted
from arc_agent.v2_ranker import compatibility_features
from arc_agent.v2_state import StateMismatch, V2State


def _completed(
    response_id: str = "resp_ok",
    source: str = "def solve(train, grid):\n    return [row[:] for row in grid]",
) -> ResponseSnapshot:
    payload = {
        "hypothesis": "identity",
        "strategy_tags": ["identity"],
        "invariants": ["preserve cells"],
        "python_source": source,
    }
    body = {
        "id": response_id,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    return ResponseSnapshot(
        response_id=response_id,
        status="completed",
        body=body,
        usage=ResponseUsage(input_tokens=10, output_tokens=20),
    )


def _failed_quota(response_id: str = "resp_quota") -> ResponseSnapshot:
    body = {
        "id": response_id,
        "status": "failed",
        "error": {
            "message": "You've hit your usage limit.",
            "code": "usageLimitExceeded",
        },
        "usage": {},
    }
    return ResponseSnapshot(
        response_id=response_id,
        status="failed",
        body=body,
        usage=ResponseUsage(),
    )


class ScriptedClient:
    def __init__(self, events: list[object]) -> None:
        self.events = list(events)
        self.settings = ResponsesConfig()
        self.request_keys: list[str] = []

    def create(self, **kwargs):
        self.request_keys.append(kwargs["request_key"])
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    def retrieve(self, response_id: str, *, request=None):
        del response_id, request
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


class ConcurrentIdentityClient:
    def __init__(self) -> None:
        self.settings = ResponsesConfig()
        self.barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def create(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=5)
            return _completed(f"response-{kwargs['task_id']}")
        finally:
            with self.lock:
                self.active -= 1

    def retrieve(self, response_id: str, *, request=None):
        del response_id, request
        raise AssertionError("no response should require polling")


class ConcurrentQuotaClient(ConcurrentIdentityClient):
    def create(self, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=5)
            if kwargs["task_id"] == "a-quota":
                raise QuotaExhausted("quota exhausted")
            return _completed(f"response-{kwargs['task_id']}")
        finally:
            with self.lock:
                self.active -= 1


class NeverCompletesClient:
    def __init__(self) -> None:
        self.settings = ResponsesConfig()
        self.cancelled: list[str] = []

    def create(self, **kwargs):
        del kwargs
        return ResponseSnapshot(
            response_id="response-running",
            status="in_progress",
            body={"id": "response-running", "status": "in_progress"},
        )

    def retrieve(self, response_id: str, *, request=None):
        del request
        return ResponseSnapshot(
            response_id=response_id,
            status="in_progress",
            body={"id": response_id, "status": "in_progress"},
        )

    def cancel(self, response_id: str) -> None:
        self.cancelled.append(response_id)


def _config() -> V2ExperimentConfig:
    return V2ExperimentConfig(
        guards=GuardConfig(d4_transforms=False, color_permutations=0),
        ranker=RankerConfig(enabled=False),
        retrieved_programs=0,
    )


def test_state_deduplicates_response_usage(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    with V2State(workspace) as state:
        state.initialize_phase(
            phase="training",
            task_ids=["a"],
            dataset_hash="data",
            config_hash="config",
        )
        request = state.prepare_request(
            request_key="key",
            phase="training",
            task_id="a",
            round_index=0,
            prompt="prompt",
            previous_response_id=None,
            token_limit=32_768,
        )
        assert request["status"] == "prepared"
        state.record_response("key", _completed())
        state.record_response("key", _completed())
        assert state.usage_totals()["input_tokens"] == 10
        assert state.active_task("training")["task_id"] == "a"


def test_state_rejects_changed_dataset_on_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    with V2State(workspace) as state:
        state.initialize_phase(
            phase="training",
            task_ids=["a"],
            dataset_hash="data-a",
            config_hash="config",
        )
    with V2State(workspace) as state:
        try:
            state.initialize_phase(
                phase="training",
                task_ids=["a"],
                dataset_hash="data-b",
                config_hash="config",
                resume_only=True,
            )
        except StateMismatch:
            pass
        else:
            raise AssertionError("changed datasets must not resume the same workspace")


def test_quota_pause_resumes_same_task_and_request(tmp_path: Path, identity_task: ArcTask) -> None:
    workspace = tmp_path / "run"
    quota_client = ScriptedClient([QuotaExhausted("credits exhausted")])
    with V2State(workspace) as state:
        result = V2Orchestrator(
            _config(), state, client=quota_client, sleep=lambda _: None
        ).build_bank([identity_task], dataset_hash="data", finalize=False)
        assert result.status == "paused_quota"
        assert result.active_task_id == identity_task.task_id
        prepared = state.pending_request("training", identity_task.task_id)
        assert prepared is not None
        original_key = prepared["request_key"]

    success_client = ScriptedClient([_completed()])
    with V2State(workspace) as state:
        result = V2Orchestrator(
            _config(), state, client=success_client, sleep=lambda _: None
        ).build_bank(
            [identity_task],
            dataset_hash="data",
            resume_only=True,
            finalize=False,
        )
        assert result.status == "complete"
        assert success_client.request_keys == [original_key]
        assert state.active_task("training") is None
        assert len(state.list_programs()) == 1
        assert state.get_meta("training_quota_pause_count") == 1


def test_terminal_quota_response_is_retired_before_fresh_resume(
    tmp_path: Path, identity_task: ArcTask
) -> None:
    workspace = tmp_path / "run"
    failed = _failed_quota()
    with V2State(workspace) as state:
        result = V2Orchestrator(
            _config(), state, client=ScriptedClient([failed]), sleep=lambda _: None
        ).build_bank([identity_task], dataset_hash="data", finalize=False)
        assert result.status == "paused_quota"
        paused = state.pending_request("training", identity_task.task_id)
        assert paused is not None
        assert paused["response_id"] == failed.response_id
        old_key = paused["request_key"]

    resumed_client = ScriptedClient([failed, _completed("resp_after_reset")])
    with V2State(workspace) as state:
        result = V2Orchestrator(
            _config(), state, client=resumed_client, sleep=lambda _: None
        ).build_bank(
            [identity_task],
            dataset_hash="data",
            resume_only=True,
            finalize=False,
        )
        assert result.status == "complete"
        retired = state.connection.execute(
            "SELECT ingested, status FROM requests WHERE request_key=?", (old_key,)
        ).fetchone()
        assert retired["ingested"] == 1
        assert retired["status"] == "failed"
        requests = state.connection.execute(
            "SELECT request_key, round_index FROM requests ORDER BY created_at"
        ).fetchall()
        assert len(requests) == 2
        assert requests[1]["request_key"] != old_key
        assert requests[1]["round_index"] == 1
        assert resumed_client.request_keys == [requests[1]["request_key"]]
        assert state.active_task("training") is None


def test_completed_uningested_response_resumes_without_new_api_call(
    tmp_path: Path, identity_task: ArcTask
) -> None:
    workspace = tmp_path / "run"
    config = _config()
    with V2State(workspace) as state:
        state.initialize_phase(
            phase="training",
            task_ids=[identity_task.task_id],
            dataset_hash="data",
            config_hash=config.sha256(),
        )
        state.prepare_request(
            request_key="already-paid",
            phase="training",
            task_id=identity_task.task_id,
            round_index=0,
            prompt="stored prompt",
            previous_response_id=None,
            token_limit=32_768,
        )
        state.record_response("already-paid", _completed("resp_stored"))

    no_call_client = ScriptedClient([])
    with V2State(workspace) as state:
        result = V2Orchestrator(
            config, state, client=no_call_client, sleep=lambda _: None
        ).build_bank(
            [identity_task],
            dataset_hash="data",
            resume_only=True,
            finalize=False,
        )
        assert result.status == "complete"
        assert no_call_client.request_keys == []
        request = state.connection.execute(
            "SELECT ingested FROM requests WHERE request_key='already-paid'"
        ).fetchone()
        assert request["ingested"] == 1


def test_default_training_concurrency_runs_two_tasks_in_parallel(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    tasks = [
        ArcTask(
            task_id=f"identity-{index}",
            train=[ArcPair(input=[[index]], output=[[index]])],
            test=[ArcPair(input=[[index + 2]], output=[[index + 2]])],
        )
        for index in range(2)
    ]
    config = _config()
    client = ConcurrentIdentityClient()

    with V2State(workspace) as state:
        result = V2Orchestrator(
            config, state, client=client, sleep=lambda _: None
        ).build_bank(tasks, dataset_hash="data", finalize=False)

        assert result.status == "complete"
        assert client.max_active == 2
        assert state.get_meta("training_concurrency") == 2
        assert state.active_tasks("training") == []
        assert all(
            row["status"] == "accepted"
            for row in state.connection.execute(
                "SELECT status FROM phase_tasks WHERE phase='training'"
            ).fetchall()
        )


def test_concurrent_quota_pause_requeues_unfinished_claims(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    tasks = [
        ArcTask(
            task_id=task_id,
            train=[ArcPair(input=[[1]], output=[[1]])],
            test=[ArcPair(input=[[2]], output=[[2]])],
        )
        for task_id in ("a-quota", "b-success")
    ]
    client = ConcurrentQuotaClient()

    with V2State(workspace) as state:
        result = V2Orchestrator(
            _config(), state, client=client, sleep=lambda _: None
        ).build_bank(tasks, dataset_hash="data", finalize=False)

        statuses = {
            row["task_id"]: row["status"]
            for row in state.connection.execute(
                "SELECT task_id, status FROM phase_tasks WHERE phase='training'"
            ).fetchall()
        }
        assert result.status == "paused_quota"
        assert client.max_active == 2
        assert "running" not in statuses.values()
        assert statuses["a-quota"] == "pending"
        assert statuses["b-success"] == "accepted"
        assert state.get_meta("training_quota_pause_count") == 1


def test_training_concurrency_is_excluded_from_experiment_hash() -> None:
    one = V2ExperimentConfig(training_concurrency=1)
    two = V2ExperimentConfig(training_concurrency=2)
    different_limit = V2ExperimentConfig(max_refinement_rounds=81)

    assert one.sha256() == two.sha256()
    assert one.sha256() == different_limit.sha256()


def test_round_limit_freezes_best_program_and_records_failed_guards(
    tmp_path: Path, identity_task: ArcTask
) -> None:
    workspace = tmp_path / "run"
    wrong_source = (
        "def solve(train, grid):\n"
        "    return [[0 for cell in row] for row in grid]"
    )
    client = ScriptedClient([_completed("response-wrong", wrong_source)])

    with V2State(workspace) as state:
        result = V2Orchestrator(
            _config(), state, client=client, sleep=lambda _: None
        ).build_bank(
            [identity_task],
            dataset_hash="data",
            finalize=False,
            max_refinement_rounds=1,
        )

        task = state.task_row("training", identity_task.task_id)
        failed = json.loads(task["failed_guards_json"])
        failed_names = {item["guard"] for item in failed}
        assert result.status == "complete"
        assert len(client.request_keys) == 1
        assert task["status"] == "accepted"
        assert task["acceptance_kind"] == "best_effort"
        assert task["round_index"] == 1
        assert {
            "full_demonstrations_exact",
            "labelled_source_tests_exact",
            "leave_one_out_exact",
        } <= failed_names
        assert state.acceptance_counts("training") == {"best_effort": 1}
        assert state.best_effort_tasks()[0]["task_id"] == identity_task.task_id


def test_repeated_failure_signature_forces_source_free_independent_chain(
    tmp_path: Path, identity_task: ArcTask
) -> None:
    workspace = tmp_path / "run"
    config = _config()
    source = (
        "def solve(train, grid):\n"
        "    unique_old_heuristic = 1\n"
        "    return [[0 for cell in row] for row in grid]"
    )
    failure = InductionVerification(
        static_safe=True,
        exact_cases=0,
        total_cases=3,
        source_tests_total=1,
        leave_one_out_total=1,
        failures=[
            VerificationFailure(
                case="source_test_0",
                expected=[[2, 0], [0, 2]],
                actual=[[0, 0], [0, 0]],
                detail="(0,0)=0/2, (1,1)=0/2",
            )
        ],
        score=25.0,
    )
    with V2State(workspace) as state:
        state.initialize_phase(
            phase="training",
            task_ids=[identity_task.task_id],
            dataset_hash="data",
            config_hash=config.sha256(),
        )
        # Reusing the same source exercises the per-attempt ledger instead of
        # relying on the source-deduplicated candidates table.
        for round_index in range(3):
            state.record_candidate(
                phase="training",
                task_id=identity_task.task_id,
                round_index=round_index,
                source_kind="luna",
                response_id=f"response-{round_index}",
                hypothesis="same failed behaviour",
                strategy_tags=[],
                invariants=[],
                python_source=source,
                verification=failure,
            )
        state.update_task(
            "training",
            identity_task.task_id,
            round_index=3,
            previous_response_id="response-2",
        )

        request = V2Orchestrator(config, state)._prepare_active_request(
            identity_task, phase="training"
        )

        assert request["strategy_mode"] == "independent_plateau"
        assert request["plateau_signature"]
        assert request["previous_response_id"] is None
        assert "INDEPENDENT RESYNTHESIS" in request["prompt"]
        assert "unique_old_heuristic" not in request["prompt"]
        assert state.repeated_failure_plateau(
            "training", identity_task.task_id, threshold=3, window=12
        ) is None


def test_refinement_attempt_21_switches_to_sol_and_breaks_luna_chain(
    tmp_path: Path, identity_task: ArcTask
) -> None:
    workspace = tmp_path / "run"
    config = _config()
    with V2State(workspace) as state:
        state.initialize_phase(
            phase="training",
            task_ids=[identity_task.task_id],
            dataset_hash="data",
            config_hash=config.sha256(),
        )
        previous = state.prepare_request(
            request_key="luna-round-20",
            phase="training",
            task_id=identity_task.task_id,
            round_index=19,
            prompt="old prompt",
            previous_response_id=None,
            token_limit=32_768,
            model="gpt-5.6-luna",
        )
        state.record_response("luna-round-20", _completed("luna-response"))
        state.mark_request_ingested(previous["request_key"])
        state.update_task(
            "training",
            identity_task.task_id,
            round_index=20,
            previous_response_id="luna-response",
        )

        request = V2Orchestrator(config, state)._prepare_active_request(
            identity_task, phase="training"
        )

        assert request["model"] == "gpt-5.6-sol"
        assert request["previous_response_id"] is None
        assert request["strategy_mode"] == "refinement"


def test_evaluation_is_blind_then_freezes_and_scores(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    source = "def solve(train, grid):\n    return [row[::-1] for row in grid]"
    training = ArcTask(
        task_id="train-reverse",
        train=[ArcPair(input=[[1, 2]], output=[[2, 1]])],
        test=[ArcPair(input=[[3, 4]], output=[[4, 3]])],
    )
    evaluation = ArcTask(
        task_id="eval-reverse",
        train=[ArcPair(input=[[5, 6]], output=[[6, 5]])],
        test=[ArcPair(input=[[8, 9]], output=[[9, 8]])],
    )
    config = _config()
    with V2State(workspace) as state:
        build = V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_completed("resp_train", source)]),
            sleep=lambda _: None,
        ).build_bank([training], dataset_hash="training", finalize=False)
        assert build.status == "complete"

    with V2State(workspace) as state:
        result = V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_completed("resp_eval", source)]),
            sleep=lambda _: None,
        ).evaluate([evaluation], dataset_hash="evaluation")
        assert result.status == "complete"
        report = json.loads((workspace / "evaluation_report.json").read_text())
        assert report["pass_at_2"] == 1.0
        request = state.connection.execute(
            "SELECT prompt FROM requests WHERE phase='evaluation'"
        ).fetchone()
        assert "[[9,8]]" not in request["prompt"].replace(" ", "")
        assert state.get_meta("evaluation_submission_sha256")


def _ranked_program(index: int) -> BankProgram:
    source = (
        "def solve(train, grid):\n"
        f"    # ranked candidate {index}\n"
        "    return [row[:] for row in grid]"
    )
    return BankProgram(
        program_hash=f"program-{index}",
        source_task_ids=[f"source-{index}"],
        hypothesis=f"candidate {index}",
        strategy_tags=["test"],
        invariants=[],
        python_source=source,
        canonical_ast=f"ast-{index}",
        features={},
        verification=InductionVerification(),
        complexity=index,
    )


def test_optimized_direct_evaluation_caps_then_adaptively_expands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = ArcTask(
        task_id="eval",
        train=[ArcPair(input=[[1]], output=[[1]])],
        test=[ArcPair(input=[[2]], output=None)],
    )
    programs = [_ranked_program(index) for index in range(10)]
    config = V2ExperimentConfig(
        guards=GuardConfig(d4_transforms=False, color_permutations=0),
        evaluation=EvaluationConfig(
            direct_scan_limit=4,
            direct_expand_limit=8,
            direct_verified_target=2,
            direct_concurrency=1,
            staged_verification=True,
        ),
    )
    calls: list[tuple[int, bool]] = []

    def ranked(*args, **kwargs):
        del args, kwargs
        return programs

    def verify(source: str, ignored_task: ArcTask, **kwargs) -> InductionVerification:
        del ignored_task
        index = int(source.split("ranked candidate ")[1].splitlines()[0])
        full = bool(kwargs["run_transformations"])
        calls.append((index, full))
        accepted = index in {1, 5}
        return InductionVerification(
            accepted=accepted,
            static_safe=True,
            exact_cases=int(accepted),
            total_cases=1,
            predictions=[[[2]]],
            score=float(index),
        )

    monkeypatch.setattr("arc_agent.v2_experiment.rank_programs", ranked)
    monkeypatch.setattr("arc_agent.v2_experiment.verify_induction_program", verify)
    with V2State(tmp_path / "run") as state:
        state.initialize_phase(
            phase="evaluation",
            task_ids=[task.task_id],
            dataset_hash="evaluation",
            config_hash=config.evaluation_sha256(),
        )
        accepted = V2Orchestrator(config, state)._direct_evaluation(task)

        assert len(accepted) == 2
        assert int(state.task_row("evaluation", task.task_id)["direct_cursor"]) == 6
        assert [index for index, full in calls if not full] == list(range(6))
        assert [index for index, full in calls if full] == [1, 5]


def test_optimized_direct_evaluation_verifies_one_batch_concurrently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = ArcTask(
        task_id="eval",
        train=[ArcPair(input=[[1]], output=[[1]])],
        test=[ArcPair(input=[[2]], output=None)],
    )
    programs = [_ranked_program(index) for index in range(8)]
    config = V2ExperimentConfig(
        evaluation=EvaluationConfig(
            direct_scan_limit=4,
            direct_expand_limit=8,
            direct_verified_target=2,
            direct_concurrency=4,
            staged_verification=True,
        )
    )
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def ranked(*args, **kwargs):
        del args, kwargs
        return programs

    def verify(source: str, task: ArcTask, **kwargs) -> InductionVerification:
        del source, task, kwargs
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=5)
            return InductionVerification(
                static_safe=True,
                predictions=[[[2]]],
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("arc_agent.v2_experiment.rank_programs", ranked)
    monkeypatch.setattr("arc_agent.v2_experiment.verify_induction_program", verify)
    with V2State(tmp_path / "run") as state:
        state.initialize_phase(
            phase="evaluation",
            task_ids=[task.task_id],
            dataset_hash="evaluation",
            config_hash=config.evaluation_sha256(),
        )
        accepted = V2Orchestrator(config, state)._direct_evaluation(task)

        assert accepted == []
        assert max_active == 4
        assert int(state.task_row("evaluation", task.task_id)["direct_cursor"]) == 4


def test_evaluation_policy_has_separate_hash_from_training() -> None:
    legacy = V2ExperimentConfig()
    optimized = V2ExperimentConfig(
        evaluation=EvaluationConfig(
            direct_scan_limit=64,
            direct_expand_limit=128,
            direct_verified_target=2,
            direct_concurrency=4,
            staged_verification=True,
        )
    )

    assert optimized.sha256() == legacy.sha256()
    assert legacy.evaluation_sha256() == legacy.sha256()
    assert optimized.evaluation_sha256() != optimized.sha256()


def _budgeted_config(**evaluation_updates) -> V2ExperimentConfig:
    settings = {
        "direct_scan_limit": 1,
        "direct_expand_limit": 1,
        "direct_verified_target": 1,
        "direct_concurrency": 1,
        "staged_verification": True,
        "active_time_limit_seconds": 100.0,
        "freeze_reserve_seconds": 1.0,
        "task_concurrency": 1,
        "llm_concurrency": 1,
        "model_turn_timeout_seconds": 5.0,
        "max_model_attempts_per_task": 1,
    }
    settings.update(evaluation_updates)
    return V2ExperimentConfig(
        guards=GuardConfig(d4_transforms=False, color_permutations=0),
        ranker=RankerConfig(enabled=False),
        evaluation=EvaluationConfig(**settings),
        retrieved_programs=0,
        training_concurrency=1,
    )


def test_budgeted_evaluation_skips_model_when_direct_program_passes(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    reverse_source = "def solve(train, grid):\n    return [row[::-1] for row in grid]"
    training = ArcTask(
        task_id="train",
        train=[ArcPair(input=[[1, 2]], output=[[2, 1]])],
        test=[ArcPair(input=[[3, 4]], output=[[4, 3]])],
    )
    evaluation = ArcTask(
        task_id="eval",
        train=[ArcPair(input=[[5, 6]], output=[[6, 5]])],
        test=[ArcPair(input=[[8, 9]], output=[[9, 8]])],
    )
    config = _budgeted_config()
    with V2State(workspace) as state:
        V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_completed("train", reverse_source)]),
            sleep=lambda _: None,
        ).build_bank([training], dataset_hash="training", finalize=False)

    unused_client = ScriptedClient([])
    with V2State(workspace) as state:
        result = V2Orchestrator(
            config,
            state,
            client=unused_client,
            sleep=lambda _: None,
        ).evaluate([evaluation], dataset_hash="evaluation")

        assert result.status == "complete"
        assert unused_client.request_keys == []
        assert state.get_meta("evaluation_stage") == "frozen"
        assert len(state.evaluation_outputs()) == 1


def test_budgeted_evaluation_freezes_fallbacks_at_deadline(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    task = ArcTask(
        task_id="eval",
        train=[ArcPair(input=[[1]], output=[[2]])],
        test=[ArcPair(input=[[3]], output=[[4]])],
    )
    config = _budgeted_config(
        active_time_limit_seconds=0.000001,
        freeze_reserve_seconds=0.0,
    )
    with V2State(workspace) as state:
        state.set_meta("training_status", "complete")
        result = V2Orchestrator(
            config,
            state,
            client=ScriptedClient([]),
            sleep=lambda _: None,
        ).evaluate([task], dataset_hash="evaluation")

        assert result.status == "complete"
        assert state.get_meta("evaluation_budget_exhausted") is True
        assert state.get_meta("evaluation_stage") == "frozen"
        assert state.task_row("evaluation", task.task_id)["direct_cursor"] == 0
        assert len(state.evaluation_outputs()) == 1


def test_budgeted_evaluation_resumes_first_model_after_quota_pause(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    identity_source = "def solve(train, grid):\n    return [row[:] for row in grid]"
    reverse_source = "def solve(train, grid):\n    return [row[::-1] for row in grid]"
    training = ArcTask(
        task_id="train",
        train=[ArcPair(input=[[1]], output=[[1]])],
        test=[ArcPair(input=[[2]], output=[[2]])],
    )
    evaluation = ArcTask(
        task_id="eval",
        train=[ArcPair(input=[[1, 2]], output=[[2, 1]])],
        test=[ArcPair(input=[[3, 4]], output=[[4, 3]])],
    )
    config = _budgeted_config()
    with V2State(workspace) as state:
        V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_completed("train", identity_source)]),
            sleep=lambda _: None,
        ).build_bank([training], dataset_hash="training", finalize=False)

    with V2State(workspace) as state:
        paused = V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_failed_quota()]),
            sleep=lambda _: None,
        ).evaluate([evaluation], dataset_hash="evaluation")
        assert paused.status == "paused_quota"
        assert state.get_meta("evaluation_stage") == "first_model"
        assert state.task_row("evaluation", evaluation.task_id)["status"] == "covered"
        assert state.task_row("evaluation", evaluation.task_id)["direct_cursor"] == 1

    with V2State(workspace) as state:
        resumed = V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_completed("eval", reverse_source)]),
            sleep=lambda _: None,
        ).evaluate(
            [evaluation],
            dataset_hash="evaluation",
            resume_only=True,
        )
        assert resumed.status == "complete"
        assert state.task_row("evaluation", evaluation.task_id)["direct_cursor"] == 1
        assert state.get_meta("evaluation_stage") == "frozen"


def test_budgeted_evaluation_caps_concurrent_model_turns(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    identity_source = "def solve(train, grid):\n    return [row[:] for row in grid]"
    training = ArcTask(
        task_id="train",
        train=[ArcPair(input=[[1]], output=[[1]])],
        test=[ArcPair(input=[[2]], output=[[2]])],
    )
    evaluations = [
        ArcTask(
            task_id=f"eval-{index}",
            train=[ArcPair(input=[[1, 2]], output=[[2, 1]])],
            test=[ArcPair(input=[[3, 4]], output=[[4, 3]])],
        )
        for index in range(4)
    ]
    config = _budgeted_config(
        task_concurrency=4,
        llm_concurrency=2,
        max_model_attempts_per_task=None,
    )
    with V2State(workspace) as state:
        V2Orchestrator(
            config,
            state,
            client=ScriptedClient([_completed("train", identity_source)]),
            sleep=lambda _: None,
        ).build_bank([training], dataset_hash="training", finalize=False)

    client = ConcurrentIdentityClient()
    with V2State(workspace) as state:
        result = V2Orchestrator(
            config,
            state,
            client=client,
            sleep=lambda _: None,
        ).evaluate(evaluations, dataset_hash="evaluation")

        assert result.status == "complete"
        assert client.max_active == 2
        assert state.connection.execute(
            "SELECT COUNT(*) FROM requests WHERE phase='evaluation'"
        ).fetchone()[0] == 4


def test_model_turn_timeout_retires_and_cancels_request(tmp_path: Path) -> None:
    task = ArcTask(
        task_id="eval",
        train=[ArcPair(input=[[1]], output=[[1]])],
        test=[ArcPair(input=[[2]], output=None)],
    )
    config = _budgeted_config(model_turn_timeout_seconds=0.001)
    client = NeverCompletesClient()
    with V2State(tmp_path / "run") as state:
        state.initialize_phase(
            phase="evaluation",
            task_ids=[task.task_id],
            dataset_hash="evaluation",
            config_hash=config.evaluation_sha256(),
        )
        synthesized = V2Orchestrator(
            config,
            state,
            client=client,
            sleep=lambda _: None,
        ).synthesize_once(
            task,
            phase="evaluation",
            timeout_seconds=config.evaluation.model_turn_timeout_seconds,
        )

        request = state.connection.execute(
            "SELECT status, ingested FROM requests WHERE phase='evaluation'"
        ).fetchone()
        assert synthesized is False
        assert dict(request) == {"status": "timed_out", "ingested": 1}
        assert state.task_row("evaluation", task.task_id)["round_index"] == 1
        assert client.cancelled == ["response-running"]


def test_full_bank_finalization_builds_compatibility_ranker_and_manifest(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "run"
    identity_source = "def solve(train, grid):\n    return [row[:] for row in grid]"
    reverse_source = "def solve(train, grid):\n    return [row[::-1] for row in grid]"
    tasks = [
        ArcTask(
            task_id="a-identity",
            train=[ArcPair(input=[[1, 2]], output=[[1, 2]])],
            test=[ArcPair(input=[[3, 4]], output=[[3, 4]])],
        ),
        ArcTask(
            task_id="b-reverse",
            train=[ArcPair(input=[[1, 2]], output=[[2, 1]])],
            test=[ArcPair(input=[[3, 4]], output=[[4, 3]])],
        ),
    ]
    config = V2ExperimentConfig(
        guards=GuardConfig(d4_transforms=False, color_permutations=0),
        ranker=RankerConfig(enabled=True, folds=2, epochs=20),
        retrieved_programs=0,
        training_concurrency=1,
    )
    client = ScriptedClient(
        [
            _completed("resp_identity", identity_source),
            _completed("resp_reverse", reverse_source),
        ]
    )
    with V2State(workspace) as state:
        result = V2Orchestrator(config, state, client=client, sleep=lambda _: None).build_bank(
            tasks, dataset_hash="training"
        )
        assert result.status == "complete"
        assert len(state.compatibility_rows()) == 4
        assert state.get_meta("training_stage") == "frozen"
        assert state.get_meta("ranker_state")["status"] == "complete"
    manifest = json.loads((workspace / "manifest.json").read_text())
    assert manifest["accepted_training_tasks"] == 2
    assert manifest["deduplicated_programs"] == 2
    assert (workspace / "ranker.pkl").is_file()


def test_compatibility_concurrency_preserves_rows_and_serializes_commits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "run"
    identity_source = "def solve(train, grid):\n    return [row[:] for row in grid]"
    reverse_source = "def solve(train, grid):\n    return [row[::-1] for row in grid]"
    tasks = [
        ArcTask(
            task_id="a-identity",
            train=[ArcPair(input=[[1, 2]], output=[[1, 2]])],
            test=[ArcPair(input=[[3, 4]], output=[[3, 4]])],
        ),
        ArcTask(
            task_id="b-reverse",
            train=[ArcPair(input=[[1, 2]], output=[[2, 1]])],
            test=[ArcPair(input=[[3, 4]], output=[[4, 3]])],
        ),
    ]
    config = V2ExperimentConfig(
        guards=GuardConfig(d4_transforms=False, color_permutations=0),
        ranker=RankerConfig(enabled=False),
        retrieved_programs=0,
        training_concurrency=1,
    )
    client = ScriptedClient(
        [
            _completed("resp_identity", identity_source),
            _completed("resp_reverse", reverse_source),
        ]
    )

    with V2State(workspace) as state:
        orchestrator = V2Orchestrator(config, state, client=client, sleep=lambda _: None)
        result = orchestrator.build_bank(
            tasks,
            dataset_hash="training",
            finalize=False,
            concurrency=1,
        )
        assert result.status == "complete"
        programs = state.list_programs()
        persisted = InductionVerification(
            static_safe=True,
            accepted=True,
            exact_cases=1,
            total_cases=1,
        )
        for program in programs:
            state.record_compatibility(
                program_hash=program.program_hash,
                task_id=tasks[0].task_id,
                features=compatibility_features(tasks[0], program),
                verification=persisted,
            )

        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = 0
        max_active = 0
        calls: list[str] = []

        def verify_concurrently(source: str, task: ArcTask, **kwargs) -> InductionVerification:
            del source, kwargs
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(task.task_id)
            try:
                barrier.wait(timeout=5)
                return persisted
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr(
            "arc_agent.v2_experiment.verify_induction_program",
            verify_concurrently,
        )
        orchestrator.finalize_bank(tasks, dataset_hash="training", concurrency=2)

        assert max_active == 2
        assert calls == [tasks[1].task_id, tasks[1].task_id]
        assert len(state.compatibility_rows()) == 4
        assert state.get_meta("compatibility_concurrency") == 2
