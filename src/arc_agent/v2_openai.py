from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from arc_agent.v2_config import ResponsesConfig
from arc_agent.v2_models import ResponseSnapshot, ResponseUsage, SynthesizedProgram


class ResponsesAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after = retry_after


class QuotaExhausted(ResponsesAPIError):
    pass


class ConfigurationError(ResponsesAPIError):
    pass


class TransientAPIError(ResponsesAPIError):
    pass


class ResponseExpired(ResponsesAPIError):
    pass


class IncompleteResponse(ResponsesAPIError):
    pass


class SynthesisClient(Protocol):
    settings: ResponsesConfig

    def create(
        self,
        *,
        prompt: str,
        task_id: str,
        phase: str,
        round_index: int,
        max_output_tokens: int,
        previous_response_id: str | None,
        request_key: str,
        checkpoint: Callable[[ResponseSnapshot], None] | None = None,
    ) -> ResponseSnapshot: ...

    def retrieve(
        self,
        response_id: str,
        *,
        request: dict[str, Any] | None = None,
    ) -> ResponseSnapshot: ...

    def close(self) -> None: ...


def _error_payload(response: httpx.Response) -> tuple[str, str]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return "", response.text[-2_000:]
    raw = payload.get("error", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return "", str(raw)
    return str(raw.get("code") or raw.get("type") or ""), str(raw.get("message") or raw)


def classify_http_error(response: httpx.Response, *, retrieving: bool = False) -> ResponsesAPIError:
    code, message = _error_payload(response)
    lowered_code = code.lower()
    lowered_message = message.lower()
    retry_after: float | None = None
    with suppress(KeyError, TypeError, ValueError):
        retry_after = float(response.headers["retry-after"])
    kwargs = {
        "status_code": response.status_code,
        "code": code,
        "retry_after": retry_after,
    }
    if response.status_code == 404 and retrieving:
        return ResponseExpired(message or "stored response was not found", **kwargs)
    if lowered_code in {"rate_limit_exceeded", "requests", "tokens"}:
        return TransientAPIError(message or "rate limit exceeded", **kwargs)
    quota_markers = (
        "insufficient_quota",
        "billing_hard_limit",
        "billing hard limit",
        "credit balance",
        "credits exhausted",
        "exceeded your current quota",
        "payment required",
        "billing limit",
    )
    if response.status_code == 402 or any(
        marker in lowered_code or marker in lowered_message for marker in quota_markers
    ):
        return QuotaExhausted(message or "OpenAI quota exhausted", **kwargs)
    if response.status_code == 429:
        return TransientAPIError(message or "rate limit exceeded", **kwargs)
    if response.status_code in {408, 409, 425} or response.status_code >= 500:
        return TransientAPIError(message or f"OpenAI HTTP {response.status_code}", **kwargs)
    if response.status_code in {400, 401, 403, 404, 422}:
        return ConfigurationError(message or f"OpenAI HTTP {response.status_code}", **kwargs)
    return TransientAPIError(message or f"OpenAI HTTP {response.status_code}", **kwargs)


def response_usage(body: dict[str, Any]) -> ResponseUsage:
    raw = body.get("usage") or {}
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    return ResponseUsage(
        input_tokens=int(raw.get("input_tokens", 0)),
        cached_input_tokens=int(input_details.get("cached_tokens", 0)),
        cache_write_input_tokens=int(input_details.get("cache_write_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0)),
    )


def response_snapshot(body: dict[str, Any]) -> ResponseSnapshot:
    response_id = str(body.get("id") or "")
    status = str(body.get("status") or "")
    if not response_id:
        raise TransientAPIError("OpenAI response did not include an id")
    valid = {"queued", "in_progress", "completed", "failed", "cancelled", "incomplete"}
    if status not in valid:
        raise TransientAPIError(f"OpenAI returned unknown response status: {status!r}")
    return ResponseSnapshot(
        response_id=response_id,
        status=status,  # type: ignore[arg-type]
        body=body,
        usage=response_usage(body),
    )


def classify_response_failure(snapshot: ResponseSnapshot) -> ResponsesAPIError:
    raw = snapshot.body.get("error") or {}
    code = str(raw.get("code") if isinstance(raw, dict) else "")
    message = str(raw.get("message") if isinstance(raw, dict) else raw)
    lowered = f"{code} {message}".lower()
    if any(
        marker in lowered
        for marker in (
            "insufficient_quota",
            "usage_limit_exceeded",
            "usagelimitexceeded",
            "usage_limit_reached",
            "credits_depleted",
            "billing_hard_limit",
            "billing hard limit",
            "credit balance",
            "credits exhausted",
            "exceeded your current quota",
            "payment required",
            "billing limit",
        )
    ):
        return QuotaExhausted(message or "OpenAI quota exhausted", code=code)
    if any(marker in lowered for marker in ("invalid_request", "authentication", "permission")):
        return ConfigurationError(message or "OpenAI response failed", code=code)
    return TransientAPIError(message or "OpenAI response failed", code=code)


def response_output_text(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                chunks.append(str(content.get("text") or ""))
    return "".join(chunks)


def parse_synthesized_program(body: dict[str, Any]) -> SynthesizedProgram:
    raw = response_output_text(body)
    if not raw:
        raise ValueError("completed response contained no output_text")
    try:
        payload = json.loads(raw)
        return SynthesizedProgram.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid structured program response: {exc}") from exc


PROGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "maxLength": 2_000},
        "strategy_tags": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 12,
        },
        "invariants": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "maxItems": 12,
        },
        "python_source": {"type": "string", "maxLength": 20_000},
    },
    "required": ["hypothesis", "strategy_tags", "invariants", "python_source"],
    "additionalProperties": False,
}


INSTRUCTIONS = """You are a precise ARC-AGI-2 induction-program synthesizer.
Return one safe, general Python program that infers task parameters from demonstrations.
The program must define exactly def solve(train, grid): and return a rectangular grid of
integers 0-9. It may define pure helper functions. Use no imports, I/O, files, network,
classes, external state, hard-coded task IDs, input grids, or output grids. Do not compare
the input or demonstrations against literal matrices. Finish executable code within 20,000
characters. Put all prose in the structured hypothesis/invariants fields, never in code.
"""


@dataclass(frozen=True)
class ResponsesClient:
    settings: ResponsesConfig
    api_key: str
    client: httpx.Client

    @classmethod
    def from_environment(
        cls,
        settings: ResponsesConfig,
        *,
        client: httpx.Client | None = None,
    ) -> ResponsesClient:
        api_key = os.getenv(settings.api_key_env, "")
        if not api_key:
            raise ConfigurationError(
                f"environment variable {settings.api_key_env} is not set",
                code="missing_api_key",
            )
        return cls(
            settings=settings,
            api_key=api_key,
            client=client or httpx.Client(timeout=settings.request_timeout_seconds),
        )

    def request_payload(
        self,
        *,
        prompt: str,
        task_id: str,
        phase: str,
        round_index: int,
        max_output_tokens: int,
        previous_response_id: str | None,
    ) -> dict[str, Any]:
        cache_key = hashlib.sha256(f"arc-v2:{phase}:{task_id}".encode()).hexdigest()[:64]
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": INSTRUCTIONS,
            "input": prompt,
            "reasoning": {
                "effort": self.settings.reasoning_effort,
                "context": self.settings.reasoning_context,
            },
            "background": self.settings.background,
            "store": self.settings.store,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "arc_induction_program",
                    "strict": True,
                    "schema": PROGRAM_SCHEMA,
                },
                "verbosity": "low",
            },
            "prompt_cache_key": cache_key,
            "metadata": {
                "experiment": "arc-agi-2-v2",
                "phase": phase,
                "task_id": task_id,
                "round": str(round_index),
            },
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        return payload

    @property
    def responses_url(self) -> str:
        return f"{self.settings.base_url.rstrip('/')}/responses"

    def _headers(self, *, request_key: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if request_key:
            headers["Idempotency-Key"] = request_key
        return headers

    def create(
        self,
        *,
        prompt: str,
        task_id: str,
        phase: str,
        round_index: int,
        max_output_tokens: int,
        previous_response_id: str | None,
        request_key: str,
        checkpoint: Callable[[ResponseSnapshot], None] | None = None,
    ) -> ResponseSnapshot:
        payload = self.request_payload(
            prompt=prompt,
            task_id=task_id,
            phase=phase,
            round_index=round_index,
            max_output_tokens=max_output_tokens,
            previous_response_id=previous_response_id,
        )
        try:
            response = self.client.post(
                self.responses_url,
                headers=self._headers(request_key=request_key),
                json=payload,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientAPIError(str(exc)) from exc
        if response.is_error:
            raise classify_http_error(response)
        snapshot = response_snapshot(response.json())
        if checkpoint is not None:
            checkpoint(snapshot)
        return snapshot

    def retrieve(
        self,
        response_id: str,
        *,
        request: dict[str, Any] | None = None,
    ) -> ResponseSnapshot:
        del request
        try:
            response = self.client.get(
                f"{self.responses_url}/{response_id}", headers=self._headers()
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise TransientAPIError(str(exc)) from exc
        if response.is_error:
            raise classify_http_error(response, retrieving=True)
        return response_snapshot(response.json())

    def close(self) -> None:
        self.client.close()


def create_synthesis_client(
    settings: ResponsesConfig,
    *,
    workspace: str | Path,
) -> SynthesisClient:
    if settings.auth_mode == "api_key":
        return ResponsesClient.from_environment(settings)
    from arc_agent.v2_codex import CodexSubscriptionClient

    return CodexSubscriptionClient(settings=settings, workspace=Path(workspace))
