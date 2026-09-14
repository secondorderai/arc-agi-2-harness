"""Exact Astra subscription route with the symbolic/Python witness output contract."""

from arc_agent.v2_codex import _response_id, _snapshot
from arc_agent.v2_openai import TransientAPIError
from arc_agent.v5_teacher import AstraTeacher
from arc_agent.v7_symbolic import WITNESS_CONTRACT, output_schema

TEACHER_INSTRUCTIONS = (
    """Author a compact symbolic working model for an ARC transformation.
Work autonomously from the supplied evidence; return only the supplied JSON schema.
This is an explicit scientific artifact, not a request for private chain of thought.
Ground representative entities in exact cells. Define task-local concepts, checked relations,
competing hypotheses, ordered actions and unresolved choices. Keep roles, meanings and transfer
skills provisional. Choose concise representative entities, not an exhaustive grid transcription.
Aim for under 1500 symbolic tokens. Check all visible demonstrations and revise on counterexamples.
The separate witness tests consequences; passing code does not prove the explanation's semantics.
Do not place final predictions, code or opaque reasoning into the symbolic model.
All evidence is supplied. No tools, commands, files, browsing, MCP, delegation or remembered
benchmark answers. Unseen outputs are never an oracle. Follow the contract below.
"""
    + WITNESS_CONTRACT
)


class BridgedAstraTeacher(AstraTeacher):
    def _thread_params(self, model):
        params = super()._thread_params(model)
        params["baseInstructions"] = TEACHER_INSTRUCTIONS
        params["developerInstructions"] = TEACHER_INSTRUCTIONS
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
            raise TransientAPIError("bridged teacher turn/start returned no turn id")
        return _snapshot(
            response_id=_response_id(thread_id, str(turn["id"]), request_key),
            status="in_progress",
            thread_id=thread_id,
            turn_id=str(turn["id"]),
            token_limit=token_limit,
        )
