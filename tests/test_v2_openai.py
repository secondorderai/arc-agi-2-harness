from __future__ import annotations

import json

import httpx

from arc_agent.v2_config import ResponsesConfig
from arc_agent.v2_openai import (
    QuotaExhausted,
    ResponsesClient,
    TransientAPIError,
    classify_http_error,
    parse_synthesized_program,
    response_snapshot,
)


def _body(response_id: str = "resp_1", status: str = "completed") -> dict[str, object]:
    program = {
        "hypothesis": "identity",
        "strategy_tags": ["identity"],
        "invariants": ["preserve every cell"],
        "python_source": "def solve(train, grid):\n    return grid",
    }
    return {
        "id": response_id,
        "status": status,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(program)}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 30},
        },
    }


def test_responses_payload_is_luna_xhigh_standard() -> None:
    client = ResponsesClient(settings=ResponsesConfig(), api_key="test", client=httpx.Client())
    payload = client.request_payload(
        prompt="prompt",
        task_id="task",
        phase="training",
        round_index=2,
        max_output_tokens=32_768,
        previous_response_id="resp_previous",
    )
    assert payload["model"] == "gpt-5.6-luna"
    assert payload["reasoning"] == {"effort": "xhigh", "context": "all_turns"}
    assert "mode" not in payload["reasoning"]
    assert payload["background"] is True
    assert payload["previous_response_id"] == "resp_previous"
    assert payload["text"]["format"]["type"] == "json_schema"


def test_quota_and_rate_limit_are_classified_separately() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    rate = httpx.Response(
        429,
        json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
        request=request,
    )
    quota = httpx.Response(
        429,
        json={"error": {"code": "insufficient_quota", "message": "credits exhausted"}},
        request=request,
    )
    assert isinstance(classify_http_error(rate), TransientAPIError)
    assert isinstance(classify_http_error(quota), QuotaExhausted)


def test_structured_program_and_usage_are_parsed() -> None:
    snapshot = response_snapshot(_body())
    program = parse_synthesized_program(snapshot.body)
    assert program.hypothesis == "identity"
    assert snapshot.usage.input_tokens == 100
    assert snapshot.usage.cached_input_tokens == 20
    assert snapshot.usage.cache_write_input_tokens == 10
    assert snapshot.usage.reasoning_tokens == 30


def test_background_create_and_retrieve() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=_body(status="queued"))
        return httpx.Response(200, json=_body(status="completed"))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = ResponsesClient(settings=ResponsesConfig(), api_key="test", client=http_client)
    created = client.create(
        prompt="prompt",
        task_id="task",
        phase="training",
        round_index=0,
        max_output_tokens=32_768,
        previous_response_id=None,
        request_key="stable-key",
    )
    completed = client.retrieve(created.response_id)
    assert created.status == "queued"
    assert completed.status == "completed"
    assert seen[0].headers["idempotency-key"] == "stable-key"
