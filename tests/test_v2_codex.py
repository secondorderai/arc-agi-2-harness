from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from arc_agent import v2_codex
from arc_agent.v2_codex import CodexSubscriptionClient, _response_id, resolve_codex_cli
from arc_agent.v2_config import ResponsesConfig
from arc_agent.v2_models import ResponseUsage
from arc_agent.v2_openai import (
    ConfigurationError,
    QuotaExhausted,
    classify_response_failure,
)

PROGRAM = {
    "hypothesis": "identity",
    "strategy_tags": ["identity"],
    "invariants": ["preserve cells"],
    "python_source": "def solve(train, grid):\n    return grid",
}


class FakeRPC:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.latest_rate_limits = None
        self.turns: list[dict[str, object]] = []
        self.turn_start_params: dict[str, object] | None = None

    def request(self, method: str, params, *, timeout=None):
        del timeout
        self.calls.append((method, params))
        if method == "account/read":
            return {
                "account": {"type": "chatgpt", "planType": "plus"},
                "requiresOpenaiAuth": True,
            }
        if method == "model/list":
            return {
                "data": [
                    {
                        "id": "gpt-5.6-luna",
                        "supportedReasoningEfforts": [{"reasoningEffort": "xhigh"}],
                    },
                    {
                        "id": "gpt-5.6-terra",
                        "supportedReasoningEfforts": [{"reasoningEffort": "xhigh"}],
                    },
                ]
            }
        if method == "account/rateLimits/read":
            return {"rateLimits": self.latest_rate_limits or {"rateLimitReachedType": None}}
        if method == "thread/start":
            return {"thread": {"id": "thread-1", "turns": []}}
        if method == "thread/resume":
            return {"thread": {"id": "thread-1", "turns": self.turns}}
        if method == "turn/start":
            self.turn_start_params = params
            return {"turn": {"id": "turn-1", "status": "inProgress", "items": []}}
        if method == "thread/read":
            return {"thread": {"id": "thread-1", "turns": self.turns}}
        raise AssertionError(method)

    def turn_usage(self, thread_id: str, turn_id: str, *, wait: float = 0.0):
        del thread_id, turn_id, wait
        return ResponseUsage(input_tokens=7, output_tokens=8, reasoning_tokens=3)

    def close(self):
        pass


def _completed_turn(*, item_type: str = "agentMessage") -> dict[str, object]:
    return {
        "id": "turn-1",
        "status": "completed",
        "items": [
            {
                "type": "userMessage",
                "clientId": "stable-key",
                "content": [{"type": "text", "text": "prompt"}],
            },
            {
                "type": item_type,
                "phase": "final_answer",
                "text": json.dumps(PROGRAM),
            },
        ],
    }


def test_subscription_is_default_and_checkpoints_thread_before_turn(tmp_path: Path) -> None:
    settings = ResponsesConfig()
    rpc = FakeRPC()
    client = CodexSubscriptionClient(settings=settings, workspace=tmp_path, rpc=rpc)
    checkpoints = []
    created = client.create(
        prompt="prompt",
        task_id="task",
        phase="training",
        round_index=0,
        max_output_tokens=32_768,
        previous_response_id=None,
        request_key="stable-key",
        checkpoint=lambda snapshot: checkpoints.append((snapshot, len(rpc.calls))),
    )

    assert settings.auth_mode == "chatgpt_subscription"
    assert checkpoints[0][0].status == "queued"
    assert checkpoints[0][1] < len(rpc.calls)
    assert created.status == "in_progress"
    assert rpc.turn_start_params is not None
    assert rpc.turn_start_params["effort"] == "xhigh"
    assert rpc.turn_start_params["model"] == "gpt-5.6-luna"
    assert rpc.turn_start_params["outputSchema"]["additionalProperties"] is False
    assert "serviceTier" not in rpc.turn_start_params


def test_subscription_completed_turn_maps_to_responses_shape(tmp_path: Path) -> None:
    rpc = FakeRPC()
    rpc.turns = [_completed_turn()]
    client = CodexSubscriptionClient(settings=ResponsesConfig(), workspace=tmp_path, rpc=rpc)
    response_id = _response_id("thread-1", "turn-1", "stable-key")
    completed = client.retrieve(response_id)

    assert completed.status == "completed"
    assert completed.usage.input_tokens == 7
    assert completed.usage.reasoning_tokens == 3
    assert completed.body["provider"] == "chatgpt_subscription"
    assert json.loads(completed.body["output"][0]["content"][0]["text"]) == PROGRAM


def test_subscription_turn_can_override_model_with_terra(tmp_path: Path) -> None:
    rpc = FakeRPC()
    client = CodexSubscriptionClient(settings=ResponsesConfig(), workspace=tmp_path, rpc=rpc)

    client.create(
        prompt="prompt",
        task_id="task",
        phase="training",
        round_index=20,
        max_output_tokens=32_768,
        previous_response_id=None,
        request_key="terra-key",
        model="gpt-5.6-terra",
    )

    assert rpc.turn_start_params is not None
    assert rpc.turn_start_params["model"] == "gpt-5.6-terra"
    assert rpc.turn_start_params["effort"] == "xhigh"


def test_subscription_ignores_context_compaction_item(tmp_path: Path) -> None:
    rpc = FakeRPC()
    turn = _completed_turn()
    items = turn["items"]
    assert isinstance(items, list)
    items.insert(1, {"type": "contextCompaction", "id": "compact-1"})
    rpc.turns = [turn]
    client = CodexSubscriptionClient(settings=ResponsesConfig(), workspace=tmp_path, rpc=rpc)

    completed = client.retrieve(_response_id("thread-1", "turn-1", "stable-key"))

    assert completed.status == "completed"
    assert json.loads(completed.body["output"][0]["content"][0]["text"]) == PROGRAM


def test_pending_checkpoint_recovers_existing_turn_without_duplicate(tmp_path: Path) -> None:
    rpc = FakeRPC()
    rpc.turns = [_completed_turn()]
    client = CodexSubscriptionClient(settings=ResponsesConfig(), workspace=tmp_path, rpc=rpc)
    pending = _response_id("thread-1", "pending", "stable-key")
    completed = client.retrieve(
        pending,
        request={"prompt": "prompt", "token_limit": 32_768},
    )

    assert completed.status == "completed"
    assert not any(method == "turn/start" for method, _ in rpc.calls)


def test_subscription_quota_and_tool_use_pause_configuration(tmp_path: Path) -> None:
    quota_rpc = FakeRPC()
    quota_rpc.latest_rate_limits = {
        "rateLimitReachedType": "workspace_member_usage_limit_reached",
        "primary": {"resetsAt": 123},
    }
    quota_client = CodexSubscriptionClient(
        settings=ResponsesConfig(), workspace=tmp_path, rpc=quota_rpc
    )
    with pytest.raises(QuotaExhausted):
        quota_client.create(
            prompt="prompt",
            task_id="task",
            phase="training",
            round_index=0,
            max_output_tokens=32_768,
            previous_response_id=None,
            request_key="stable-key",
        )

    tool_rpc = FakeRPC()
    tool_rpc.turns = [_completed_turn(item_type="commandExecution")]
    tool_client = CodexSubscriptionClient(
        settings=ResponsesConfig(), workspace=tmp_path, rpc=tool_rpc
    )
    with pytest.raises(ConfigurationError, match="forbidden external item"):
        tool_client.retrieve(_response_id("thread-1", "turn-1", "stable-key"))


def test_codex_usage_limit_failure_is_quota() -> None:
    rpc = FakeRPC()
    rpc.turns = [
        {
            "id": "turn-1",
            "status": "failed",
            "items": [],
            "error": {
                "message": "weekly limit reached",
                "codexErrorInfo": "usageLimitExceeded",
            },
        }
    ]
    client = CodexSubscriptionClient(settings=ResponsesConfig(), workspace=Path("run"), rpc=rpc)
    failed = client.retrieve(_response_id("thread-1", "turn-1", "stable-key"))
    assert isinstance(classify_response_failure(failed), QuotaExhausted)


def test_codex_cli_resolution_accepts_explicit_executable(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "codex-custom"
    executable.write_text("binary")
    executable.chmod(0o700)
    monkeypatch.setattr(v2_codex.shutil, "which", lambda _: None)

    assert resolve_codex_cli(str(executable)) == str(executable.resolve())


def test_codex_cli_resolution_finds_chatgpt_bundle(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "ChatGPT.app" / "Contents" / "Resources" / "codex"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary")
    executable.chmod(0o700)
    monkeypatch.setattr(v2_codex.shutil, "which", lambda _: None)
    monkeypatch.setattr(v2_codex, "_MAC_CHATGPT_CODEX", executable)

    assert os.access(executable, os.X_OK)
    assert resolve_codex_cli("codex") == str(executable)
