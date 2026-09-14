"""Subscription-native continuity, explicit symbolic artifacts, no hidden-state export."""

from arc_agent.v2_codex import CodexSubscriptionClient, _response_id, _snapshot
from arc_agent.v2_openai import ConfigurationError, TransientAPIError
from arc_agent.v5_symbolic import output_schema

TEACHER_INSTRUCTIONS = """Author a compact symbolic working model for an ARC transformation.
Return only the supplied JSON schema. This is an explicit scientific artifact, not a request
for private chain of thought. Ground entities in exact grid cells; define task-local concepts,
checked spatial relations, competing hypotheses, an ordered action description, and uncertainty.
Keep observations distinct from interpretation. Free-form meanings, roles, hypotheses and skills
are provisional, even when an executable witness fits examples. Revise them on counterexamples.
Do not output final answer grids, task identifiers, hard-coded grid lookup tables or Python.
Optionally lower one hypothesis to witness_program_json using only the supplied bounded DSL.
The witness is separate from the symbolic state and is used only by the deterministic verifier.
Keep the symbolic state concise, preferably under 1500 tokens; select representative entities.
All problem evidence is supplied. Do not browse, inspect files, use tools/MCP, execute commands,
delegate, or recall benchmark answers. Unseen demonstration and test outputs are never an oracle.
"""


class AstraTeacher(CodexSubscriptionClient):
    def _request(self, method, params):
        result = super()._request(method, params)
        if method in {"thread/start", "thread/resume"} and (
            result.get("model") != "gpt-6-astra"
            or result.get("modelProvider") != "openai"
            or (result.get("activePermissionProfile") or {}).get("id") != "arc-symbolic-teacher"
            or result.get("sandbox") != {"type": "readOnly", "networkAccess": False}
            or result.get("approvalPolicy") != "never"
            or result.get("instructionSources")
        ):
            raise ConfigurationError(
                "unexpected teacher model, permissions or instructions",
                code="teacher_isolation_violation",
            )
        return result

    def _ensure_ready(self, model: str) -> None:
        if model != "gpt-6-astra" or self.settings.auth_mode != "chatgpt_subscription":
            raise ConfigurationError("V5 permits only subscription gpt-6-astra")
        super()._ensure_ready(model)

    def preflight(self) -> dict:
        self._ensure_ready("gpt-6-astra")
        return {
            "model": "gpt-6-astra",
            "auth_mode": "chatgpt_subscription",
            "reasoning_effort": self.settings.reasoning_effort,
            "available": True,
            "api_fallback": False,
            "output_token_cap_enforced": False,
        }

    def _thread_params(self, model: str) -> dict:
        params = super()._thread_params(model)
        params["baseInstructions"] = TEACHER_INSTRUCTIONS
        params["developerInstructions"] = TEACHER_INSTRUCTIONS
        params["modelProvider"] = "openai"
        # The pinned runtime replaced readOnly.access with named permission profiles.
        # Configure it for this isolated thread only; never rewrite user/global settings.
        params.pop("sandbox")
        params["config"] = {
            "default_permissions": "arc-symbolic-teacher",
            "permissions.arc-symbolic-teacher.filesystem": {
                ":minimal": "read",
                params["cwd"]: "read",
            },
            "permissions.arc-symbolic-teacher.network.enabled": False,
        }
        return params

    def _start_turn(self, *, thread_id, prompt, request_key, token_limit, model):
        self._ensure_ready(model)
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "effort": self.settings.reasoning_effort,
                "model": model,
                "clientUserMessageId": request_key,
                "outputSchema": output_schema(),
            },
        )
        turn = result.get("turn")
        if not isinstance(turn, dict) or not turn.get("id"):
            raise TransientAPIError("Astra turn/start returned no turn id")
        return _snapshot(
            response_id=_response_id(thread_id, str(turn["id"]), request_key),
            status="in_progress",
            thread_id=thread_id,
            turn_id=str(turn["id"]),
            token_limit=token_limit,
        )
