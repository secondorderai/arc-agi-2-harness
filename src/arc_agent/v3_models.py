from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arc_agent.models import Grid

SignatureFamily = Literal[
    "color-attribute",
    "rigid-geometry",
    "object-relational",
    "counting-compression",
    "repetition-pattern",
]

ValueType = Literal["Grid", "Scene", "ObjectSet", "Object", "Mask", "Color", "Scalar", "Vector"]

ParameterSource = Literal[
    "constant",
    "background",
    "rarest_input_color",
    "most_frequent_input_color",
    "output_only_color",
    "most_frequent_output_color",
    "ranked_object_color",
    "input_height",
    "input_width",
    "output_height",
    "output_width",
    "row_scale",
    "column_scale",
    "inferred_translation_row",
    "inferred_translation_column",
    "inferred_translation",
    "demonstrated_object_rank",
    "train_pair_count",
]

InvariantName = Literal[
    "preserve_shape",
    "change_shape",
    "preserve_background",
    "d4_equivariant",
    "color_equivariant",
    "object_count_preserved",
]

PreconditionKind = Literal[
    "preserve_shape",
    "change_shape",
    "has_output_only_color",
    "single_nonbackground_object",
    "multiple_objects",
    "has_separator",
    "square_input",
    "consistent_integer_scale",
]

ArgumentValue = int | float | str | bool | list[int]


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    value_type: ValueType
    source: ParameterSource
    value: ArgumentValue | None = None

    @model_validator(mode="after")
    def valid_constant(self) -> ParameterSpec:
        if self.source == "constant" and self.value is None:
            raise ValueError("constant parameters require value")
        if self.source != "constant" and self.value is not None:
            raise ValueError("semantic parameters must not include a literal value")
        return self


class StepArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    value: ArgumentValue


class SignatureStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str = Field(min_length=1, max_length=80)
    input_type: ValueType = "Grid"
    output_type: ValueType = "Grid"
    args: list[StepArgument] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def unique_arguments(self) -> SignatureStep:
        names = [argument.name for argument in self.args]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate arguments for {self.op}")
        return self


class SignaturePrecondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PreconditionKind
    required: bool = True


class ProposedSignature(BaseModel):
    """Strict model-authored portion of a V3 signature."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    family: SignatureFamily
    hypothesis: str = Field(default="", max_length=2_000)
    preconditions: list[SignaturePrecondition] = Field(default_factory=list, max_length=12)
    parameters: list[ParameterSpec] = Field(default_factory=list, max_length=20)
    pipeline: list[SignatureStep] = Field(min_length=1, max_length=16)
    invariants: list[InvariantName] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def unique_parameters(self) -> ProposedSignature:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("signature parameter names must be unique")
        return self


class GameSignature(ProposedSignature):
    """Executable signature with identity and provenance owned by the harness."""

    canonical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_task_ids: list[str] = Field(default_factory=list)
    complexity: int = Field(ge=0)
    provenance: dict[str, str] = Field(default_factory=dict)


class TaskFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task_id: str
    predicted_family: SignatureFamily
    family_scores: dict[str, float]
    features: dict[str, float]


class SignatureFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: str
    detail: str
    expected: Grid | None = None
    actual: Grid | None = None


class SignatureVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool = False
    exact_pairs: int = 0
    total_pairs: int = 0
    cell_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    balanced_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    leave_one_out_exact: int = 0
    leave_one_out_total: int = 0
    invariant_exact: int = 0
    invariant_total: int = 0
    predictions: list[Grid] = Field(default_factory=list)
    failures: list[SignatureFailure] = Field(default_factory=list)
    score: float = 0.0


class SignatureMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signature_hash: str
    source_task_ids: list[str]
    family: SignatureFamily
    hard_preconditions_passed: bool
    fingerprint_similarity: float
    parameter_feasibility: float
    demonstration_score: float
    score: float


class SignatureCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source_kind: Literal[
        "retrieved",
        "mutation",
        "family_search",
        "global_search",
        "fallback",
    ]
    signature: GameSignature | None = None
    verification: SignatureVerification
    retrieval_rank: int | None = None


class V3RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    active_task_id: str | None = None
    completed_tasks: int = 0
    total_tasks: int = 0
    message: str = ""


def signature_material(signature: ProposedSignature | GameSignature) -> dict[str, Any]:
    payload = signature.model_dump(mode="json")
    payload.pop("canonical_hash", None)
    payload.pop("source_task_ids", None)
    payload.pop("complexity", None)
    payload.pop("provenance", None)
    payload.pop("hypothesis", None)
    return payload


def signature_hash(signature: ProposedSignature | GameSignature) -> str:
    encoded = json.dumps(
        signature_material(signature), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def finalize_signature(
    proposal: ProposedSignature,
    *,
    source_task_ids: list[str],
    provenance: dict[str, str] | None = None,
) -> GameSignature:
    complexity = (
        len(proposal.pipeline) * 2
        + len(proposal.parameters)
        + len(proposal.preconditions)
        + sum(len(step.args) for step in proposal.pipeline)
    )
    return GameSignature(
        **proposal.model_dump(mode="python"),
        canonical_hash=signature_hash(proposal),
        source_task_ids=sorted(set(source_task_ids)),
        complexity=complexity,
        provenance=provenance or {},
    )
