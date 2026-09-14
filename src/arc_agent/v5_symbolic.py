"""Symbolic working models are not programs; witnesses are independently executed."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from arc_agent.models import ArcPair, ArcTask
from arc_agent.v4_tools import run_program, task_grids


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cell(StrictModel):
    row: int = Field(ge=0, le=29, strict=True)
    column: int = Field(ge=0, le=29, strict=True)
    color: int = Field(ge=0, le=9, strict=True)


class Entity(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    grid: str = Field(min_length=1, max_length=64)
    cells: list[Cell] = Field(min_length=1, max_length=128)
    proposed_role: str = Field(max_length=240)


class Definition(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    provisional_meaning: str = Field(min_length=1, max_length=600)
    depends_on: list[str] = Field(max_length=16)


class Relation(StrictModel):
    subject: str
    predicate: Literal["left_of", "above", "same_shape", "same_colors", "same_size", "overlaps"]
    object: str


class Hypothesis(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    statement: str = Field(min_length=1, max_length=800)
    concepts: list[str] = Field(max_length=16)
    ordered_actions: list[str] = Field(min_length=1, max_length=16)
    status: Literal["proposed", "refuted"]
    counterevidence: list[str] = Field(max_length=16)


class SymbolicModel(StrictModel):
    version: Literal[1] = 1
    entities: list[Entity] = Field(min_length=1, max_length=32)
    definitions: list[Definition] = Field(min_length=1, max_length=16)
    relations: list[Relation] = Field(max_length=64)
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=8)
    unresolved: list[str] = Field(max_length=16)
    proposed_transfer_skill: str = Field(min_length=1, max_length=1000)


class Witness(StrictModel):
    hypothesis_id: str
    program_json: str = Field(min_length=1, max_length=24000)


class TeacherArtifact(StrictModel):
    symbolic: SymbolicModel
    witness: Witness | None


class GroundedEntity(StrictModel):
    id: str
    grid: str
    cells: list[Cell]


class GroundedScene(StrictModel):
    """Only exact, machine-checked observations; no inferred roles or explanations."""

    version: Literal[1] = 1
    entities: list[GroundedEntity]
    relations: list[Relation]


def output_schema() -> dict:
    schema = TeacherArtifact.model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return schema


def teaching_task(task: ArcTask) -> tuple[ArcTask, ArcTask]:
    """Withhold last demonstration entirely; never expose original test labels."""
    if len(task.train) < 2:
        raise ValueError("at least two demonstrations required for held-out verification")
    visible = ArcTask(
        task_id=task.task_id,
        train=task.train[:-1],
        test=[ArcPair(input=p.input) for p in task.test],
    )
    heldout = ArcTask(
        task_id=task.task_id, train=[task.train[-1]], test=[ArcPair(input=task.train[-1].input)]
    )
    return visible, heldout


def validate_symbols(state: SymbolicModel, task: ArcTask) -> None:
    grids = task_grids(task)
    entities = {entity.id: entity for entity in state.entities}
    definitions = {definition.id: definition for definition in state.definitions}
    hypotheses = {hypothesis.id: hypothesis for hypothesis in state.hypotheses}
    if (
        len(entities) != len(state.entities)
        or len(definitions) != len(state.definitions)
        or len(hypotheses) != len(state.hypotheses)
    ):
        raise ValueError("duplicate symbolic identifier")
    for entity in entities.values():
        if entity.grid not in grids:
            raise ValueError("entity references an unavailable grid")
        positions = set()
        for cell in entity.cells:
            position = (cell.row, cell.column)
            if position in positions:
                raise ValueError("duplicate cell in entity")
            positions.add(position)
            try:
                actual = grids[entity.grid][cell.row][cell.column]
            except IndexError as exc:
                raise ValueError("entity outside grid") from exc
            if actual != cell.color:
                raise ValueError("entity contradicts authoritative cell")
    visited, active = set(), set()

    def visit(name):
        if name not in definitions:
            raise ValueError("undefined symbolic concept")
        if name in active:
            raise ValueError("cyclic definition")
        if name in visited:
            return
        active.add(name)
        for dependency in definitions[name].depends_on:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in definitions:
        visit(name)
    for hypothesis in hypotheses.values():
        for name in hypothesis.concepts + hypothesis.ordered_actions:
            visit(name)
    for relation in state.relations:
        if relation.subject not in entities or relation.object not in entities:
            raise ValueError("relation references undefined entity")
        a, b = entities[relation.subject], entities[relation.object]
        ac = {(c.row, c.column) for c in a.cells}
        bc = {(c.row, c.column) for c in b.cells}

        def shape(cells):
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            return {(r - r0, c - c0) for r, c in cells}

        if relation.predicate in {"left_of", "above", "overlaps"} and a.grid != b.grid:
            raise ValueError("spatial relation spans different grids")
        checks = {
            "left_of": max(c for _, c in ac) < min(c for _, c in bc),
            "above": max(r for r, _ in ac) < min(r for r, _ in bc),
            "same_shape": shape(ac) == shape(bc),
            "same_colors": {c.color for c in a.cells} == {c.color for c in b.cells},
            "same_size": len(ac) == len(bc),
            "overlaps": bool(ac & bc),
        }
        if not checks[relation.predicate]:
            raise ValueError(f"false grounded relation: {relation.predicate}")


def verify_artifact(artifact: TeacherArtifact, visible: ArcTask, heldout: ArcTask) -> dict:
    """No LLM judge. A passing witness does NOT prove the prose interpretation."""
    result = {
        "grounding_valid": False,
        "visible_verified": False,
        "heldout_verified": False,
        "accepted": False,
        "free_form_semantics_verified": False,
    }
    try:
        validate_symbols(artifact.symbolic, visible)
        result["grounding_valid"] = True
        if artifact.witness is None:
            raise ValueError("no executable witness; retain as unverified research artifact")
        hypothesis = next(
            (h for h in artifact.symbolic.hypotheses if h.id == artifact.witness.hypothesis_id),
            None,
        )
        if hypothesis is None or hypothesis.status != "proposed":
            raise ValueError("witness must reference a proposed hypothesis")
        program = json.loads(artifact.witness.program_json)
        observed = run_program(visible, program)
        result["visible_verified"] = observed["verified"]
        # Only visible-demo predictions may return to the teacher as counterexamples.
        result["visible_feedback"] = {
            k: observed[k]
            for k in ("exact_train_pairs", "train_pairs", "verified", "train_predictions")
        }
        if observed["verified"]:
            hidden = run_program(heldout, program)
            result["heldout_verified"] = hidden["verified"]
            result["accepted"] = hidden["verified"]
    except (ValueError, TypeError, KeyError, TimeoutError) as exc:
        result["error"] = str(exc)[:1200]
    return result
