from __future__ import annotations

import json
from pathlib import Path

from arc_agent.models import ArcPair, ArcTask
from arc_agent.v2_config import GuardConfig, RankerConfig, ResponsesConfig, V2ExperimentConfig
from arc_agent.v2_experiment import V2Orchestrator
from arc_agent.v2_models import ResponseSnapshot, ResponseUsage
from arc_agent.v2_openai import QuotaExhausted
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
