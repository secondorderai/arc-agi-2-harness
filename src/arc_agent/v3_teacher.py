from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arc_agent.v2_codex import (
    CodexSubscriptionClient,
    _response_id,
    _snapshot,
    subscription_home,
)
from arc_agent.v2_models import ResponseSnapshot
from arc_agent.v2_openai import (
    ConfigurationError,
    ResponsesClient,
    SynthesisClient,
    TransientAPIError,
    response_output_text,
)
from arc_agent.v3_config import V3TeacherConfig
from arc_agent.v3_models import ProposedSignature

V3_TEACHER_INSTRUCTIONS = """You are a non-agentic ARC-AGI-2 typed-DSL configurator.
Return exactly one JSON game signature conforming to the supplied output schema. Use only the
listed operations and semantic parameter sources. Never return Python, prose outside JSON,
task IDs, literal grids, test outputs, tool calls, or external actions. Infer a compact rule from
the demonstrations; the test inputs are unlabelled and must never be treated as an oracle.
"""

V3_SUBSCRIPTION_INSTRUCTIONS = """Operate only as a structured configuration model. Do not call
tools, inspect files, browse, delegate, execute commands, or infer hidden evaluation labels. The
complete problem, DSL, and verifier evidence are in the user message. Return only schema-valid
JSON.
"""


def signature_output_schema() -> dict[str, Any]:
    schema = ProposedSignature.model_json_schema()

    def make_strict(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                make_strict(value)
        elif isinstance(node, list):
            for value in node:
                make_strict(value)

    make_strict(schema)
    return schema


def parse_signature_response(body: dict[str, Any]) -> ProposedSignature:
    raw = response_output_text(body)
    if not raw:
        raise ValueError("completed teacher response contained no output_text")
    try:
        return ProposedSignature.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid structured game signature: {exc}") from exc


class V3ResponsesClient(ResponsesClient):
    settings: V3TeacherConfig  # type: ignore[assignment]

    def request_payload(
        self,
        *,
        prompt: str,
        task_id: str,
        phase: str,
        round_index: int,
        max_output_tokens: int,
        previous_response_id: str | None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload = super().request_payload(
            prompt=prompt,
            task_id=task_id,
            phase=phase,
            round_index=round_index,
            max_output_tokens=max_output_tokens,
            previous_response_id=previous_response_id,
            model=model,
        )
        payload["instructions"] = V3_TEACHER_INSTRUCTIONS
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "arc_v3_game_signature",
                "strict": True,
                "schema": signature_output_schema(),
            },
            "verbosity": "low",
        }
        payload["prompt_cache_key"] = hashlib.sha256(
            f"arc-v3:{phase}:{task_id}".encode()
        ).hexdigest()[:64]
        payload["metadata"] = {
            "experiment": "arc-agi-2-v3",
            "phase": phase,
            "task_id": task_id,
            "round": str(round_index),
        }
        return payload


class V3CodexSubscriptionClient(CodexSubscriptionClient):
    settings: V3TeacherConfig  # type: ignore[assignment]

    def _ensure_ready(self, model: str) -> None:
        try:
            super()._ensure_ready(model)
        except ConfigurationError as exc:
            if exc.code != "missing_chatgpt_subscription":
                raise
            command = f"uv run arc-agent v3-auth-login --workspace {str(self.workspace)!r}"
            raise ConfigurationError(
                "No ChatGPT subscription login is available for this V3 workspace. "
                f"Run {command} and retry.",
                code=exc.code,
            ) from exc

    def _thread_params(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "cwd": str((subscription_home(self.workspace) / "model-sandbox").resolve()),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": False,
            "baseInstructions": V3_TEACHER_INSTRUCTIONS,
            "developerInstructions": V3_SUBSCRIPTION_INSTRUCTIONS,
        }

    def _start_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        request_key: str,
        token_limit: int,
        model: str,
    ) -> ResponseSnapshot:
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "effort": self.settings.reasoning_effort,
                "model": model,
                "clientUserMessageId": request_key,
                "outputSchema": signature_output_schema(),
                "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
            },
        )
        turn = result.get("turn")
        if not isinstance(turn, dict) or not turn.get("id"):
            raise TransientAPIError("Codex turn/start did not return a turn id")
        turn_id = str(turn["id"])
        response_id = _response_id(thread_id, turn_id, request_key)
        return _snapshot(
            response_id=response_id,
            status="in_progress",
            thread_id=thread_id,
            turn_id=turn_id,
            token_limit=token_limit,
        )


def create_teacher_client(
    settings: V3TeacherConfig,
    *,
    workspace: str | Path,
) -> SynthesisClient:
    if settings.auth_mode == "api_key":
        return V3ResponsesClient.from_environment(settings)  # type: ignore[arg-type,return-value]
    return V3CodexSubscriptionClient(
        settings=settings,  # type: ignore[arg-type]
        workspace=Path(workspace),
    )
