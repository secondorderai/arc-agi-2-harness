"""Nanbeige-authored reference patches over an immutable, explicit symbolic state."""

from __future__ import annotations

import copy
import json

from pydantic import Field

from arc_agent.v4_config import content_hash
from arc_agent.v5_symbolic import StrictModel, SymbolicModel


class HypothesisReferences(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    concepts: list[str] | None = Field(default=None, max_length=16)
    ordered_actions: list[str] | None = Field(default=None, min_length=1, max_length=16)


class ReferencePatch(StrictModel):
    hypotheses: list[HypothesisReferences] = Field(min_length=1, max_length=8)


def broken_references(base: dict) -> dict[str, set[str]]:
    state = SymbolicModel.model_validate(base)
    definitions = {d.id for d in state.definitions}
    if len(definitions) != len(state.definitions) or any(
        set(d.depends_on) - definitions for d in state.definitions
    ):
        return {}
    if len({h.id for h in state.hypotheses}) != len(state.hypotheses):
        return {}
    return {
        h.id: fields
        for h in state.hypotheses
        if (
            fields := {
                field
                for field in ("concepts", "ordered_actions")
                if set(getattr(h, field)) - definitions
            }
        )
    }


def reference_patch_schema(base: dict) -> dict:
    broken = broken_references(base)
    if not broken:
        raise ValueError("no eligible broken hypothesis references")
    definitions = [d["id"] for d in base["definitions"]]
    variants = []
    for hypothesis in base["hypotheses"]:
        if hypothesis["id"] not in broken:
            continue
        fields = {"id": {"type": "string", "const": hypothesis["id"]}}
        for field in ("concepts", "ordered_actions"):
            if field in broken[hypothesis["id"]]:
                fields[field] = {
                    "type": "array",
                    "items": {"type": "string", "enum": definitions},
                    "minItems": int(field == "ordered_actions"),
                    "maxItems": 16,
                }
        variants.append(
            {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "minItems": len(broken),
                "maxItems": len(broken),
                "items": {"oneOf": variants},
            }
        },
    }


def prepare_reference_repair(messages: list[dict]):
    evidence = json.loads(messages[-1]["content"])
    base = evidence.get("previous_symbolic_model")
    if (
        not base
        or evidence.get("visible_verifier_feedback", {}).get("error")
        != "undefined symbolic concept"
        or not broken_references(base)
    ):
        return None
    base = copy.deepcopy(base)
    system = (
        "Repair ONLY broken references in the supplied symbolic model. Return JSON with"
        " hypotheses: [{id, ...broken reference fields}], one entry for every hypothesis with"
        " undefined references. Use the exact existing definition IDs, selected by their"
        " meanings. Lists contain IDs, never sentences or invented action names. Include only"
        " reference fields with undefined IDs; omit already-valid fields. Other state fields,"
        " definitions"
        " and observations are immutable for this call. The patch will be applied explicitly"
        " and checked against the raw grids; this does not certify the transformation rule."
        " Do not predict test outputs. Return only the requested reference patch."
    )
    return {
        "messages": [{"role": "system", "content": system}, copy.deepcopy(messages[-1])],
        "schema": reference_patch_schema(base),
        "metadata": {
            "_response_format": "symbolic_reference_patch",
            "_reference_base": base,
            "_reference_base_sha256": content_hash(base),
        },
    }


def apply_reference_patch(base: dict, payload: dict, expected_hash: str) -> dict:
    if content_hash(base) != expected_hash:
        raise RuntimeError("symbolic patch base changed; refusing stale-state repair")
    patch = ReferencePatch.model_validate(payload)
    broken = broken_references(base)
    ids = [h.id for h in patch.hypotheses]
    if len(ids) != len(set(ids)) or set(ids) != set(broken):
        raise ValueError("patch must cover exactly the broken hypotheses, without duplicates")
    result = copy.deepcopy(base)
    definitions = {d["id"] for d in base["definitions"]}
    hypotheses = {h["id"]: h for h in result["hypotheses"]}
    for update in patch.hypotheses:
        hypothesis = hypotheses[update.id]
        for field in ("concepts", "ordered_actions"):
            values = getattr(update, field)
            if field not in broken[update.id]:
                if field in update.model_fields_set:
                    raise ValueError("patch includes an already-valid reference field")
                continue
            if values is None:
                raise ValueError("patch omits a broken reference field")
            if set(values) - definitions:
                raise ValueError("patch references an undefined definition")
            hypothesis[field] = values
    # Full grounding, cycle and relationship verification remains the caller's responsibility.
    SymbolicModel.model_validate(result)
    return result
