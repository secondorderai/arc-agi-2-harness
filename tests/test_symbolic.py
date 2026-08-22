from __future__ import annotations

from dataclasses import dataclass

import pytest

from arc_agent.symbolic import (
    OpKind,
    SymbolicError,
    SymbolicOp,
    SymbolicProgram,
    apply_program,
    execute,
    extract_objects,
    object_relations,
    synthesize,
    synthesize_programs,
    verify_program,
)


def op(kind: str | OpKind, **args: object) -> SymbolicOp:
    return SymbolicOp.make(kind, **args)


def test_typed_program_round_trip_and_defensive_execution() -> None:
    program = SymbolicProgram.from_ops(
        op(OpKind.RECOLOR, mapping={1: 2}),
        op(OpKind.REFLECT, axis="vertical"),
    )
    encoded = program.to_json()
    restored = SymbolicProgram.from_json(encoded)
    source = [[1, 0], [0, 0]]
    assert restored.to_dict() == program.to_dict()
    assert execute(restored, source) == [[0, 0], [2, 0]]
    assert source == [[1, 0], [0, 0]]
    assert apply_program(op("transform", transform="rotate_90"), source) == [[0, 1], [0, 0]]


def test_object_ir_is_stable_and_serializable() -> None:
    grid = [[0, 1, 0, 2], [0, 1, 0, 2], [0, 0, 0, 0]]
    objects = extract_objects(grid, background=0, multicolor=False)
    assert [obj.area for obj in objects] == [2, 2]
    assert objects[0].bbox == (0, 1, 1, 1)
    assert objects[0].shape == ((0, 0, 1), (1, 0, 1))
    assert objects[0].from_dict(objects[0].to_dict()) == objects[0]
    relations = object_relations(grid, background=0)
    assert any(relation.kind == "same_row" for relation in relations)


def test_geometry_color_and_object_operations() -> None:
    source = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert apply_program(op("recolor", mapping={1: 3}), source)[0][1] == 3
    assert apply_program(op("crop", background=0), source) == [[1, 0, 0, 0], [1, 0, 2, 2]]
    assert apply_program(
        op("select_object", background=0, criterion="smallest"), source
    ) == [[1], [1]]
    removed = apply_program(op("remove_object", background=0, criterion="largest"), source)
    assert removed[0][1] == 0 and removed[1][3:] == [2, 2]
    translated = apply_program(op("translate", rows=1, columns=-1, background=0), source)
    assert translated[1][0] == 1 and translated[2][3] == 2
    copied = apply_program(
        op("copy", background=0, criterion="smallest", rows=1, columns=2), source
    )
    assert copied[2][3] == 1
    assert apply_program(op("rotate", turns=1), [[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
    assert apply_program(op("reflect", axis="horizontal"), [[1, 2], [3, 4]]) == [[2, 1], [4, 3]]


def test_fill_holes_repeat_propagate_and_color_chain() -> None:
    ring = [[0, 3, 3, 3, 0], [0, 3, 0, 3, 0], [0, 3, 3, 3, 0]]
    assert apply_program(op("fill_holes", background=0, fill_color=8), ring)[1][2] == 8
    line = [[0, 2, 0, 0, 0, 0]]
    assert apply_program(
        op("repeat", background=0, criterion="smallest", direction="right", step=2, count=2), line
    ) == [[0, 2, 0, 2, 0, 2]]
    assert apply_program(
        op("propagate", background=0, criterion="smallest", direction="right"), line
    ) == [[0, 2, 2, 2, 2, 2]]
    assert apply_program(
        op("color_reference_chain", mapping={1: 2, 2: 3}, steps=2), [[1, 2]]
    ) == [[3, 3]]
    assert apply_program(op("relation_chain", mapping={1: 2}), [[1, 0]]) == [[2, 0]]


def test_verifier_rejects_false_hypothesis_across_all_demos() -> None:
    demos = [
        ([[1, 0], [0, 0]], [[2, 0], [0, 0]]),
        ([[0, 1], [0, 0]], [[0, 2], [0, 0]]),
    ]
    false = SymbolicProgram.from_ops(op("rotate", turns=1))
    result = verify_program(false, demos)
    assert not result.perfect
    assert result.exact_pairs == 0
    assert result.errors


def test_synthesizer_finds_verified_composition_and_deduplicates_behavior() -> None:
    # The output requires a spatial operation and a color operation.  No
    # single recolor or single transform can explain both dimensions.
    demos = [
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 2]]),
        ([[0, 0, 1], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [2, 0, 0]]),
    ]
    candidates = synthesize_programs(
        demos,
        max_depth=2,
        max_nodes=4_000,
        max_candidates=8,
        max_seconds=3,
    )
    assert candidates
    assert all(candidate.verified for candidate in candidates)
    assert all(candidate.exact_pairs == 2 for candidate in candidates)
    assert any(candidate.program.depth == 2 for candidate in candidates)
    assert len({candidate.behavior_signature for candidate in candidates}) == len(candidates)


def test_task_wrapper_returns_test_predictions() -> None:
    @dataclass
    class Pair:
        input: list[list[int]]
        output: list[list[int]] | None = None

    @dataclass
    class Task:
        train: list[Pair]
        test: list[Pair]

    task = Task(
        train=[Pair([[1, 0]], [[2, 0]])],
        test=[Pair([[0, 1]])],
    )
    candidates = synthesize(task, max_depth=1, max_nodes=1_000, max_candidates=4, timeout_seconds=2)
    assert candidates
    assert candidates[0].predictions == ([[0, 2]],)


def test_object_transition_guides_a_tight_compositional_search() -> None:
    demos = [
        (
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0]],
        ),
        (
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]],
        ),
    ]
    candidates = synthesize_programs(
        demos,
        max_depth=2,
        max_nodes=80,
        max_candidates=4,
        max_seconds=1,
    )
    assert any(
        [operation.kind for operation in candidate.program.operations]
        == [OpKind.TRANSLATE, OpKind.RECOLOR]
        for candidate in candidates
    )


def test_object_transition_proposes_copy_parameters() -> None:
    demos = [
        (
            [[0, 3, 0, 0, 0]],
            [[0, 3, 0, 3, 0]],
        )
    ]
    candidates = synthesize_programs(
        demos,
        max_depth=1,
        max_nodes=16,
        max_candidates=4,
        max_seconds=1,
    )
    assert any(candidate.program.operations[0].kind == OpKind.COPY for candidate in candidates)


def test_invalid_grid_and_operation_are_safe() -> None:
    with pytest.raises(SymbolicError):
        execute(op("translate", rows=1, columns=0), [[1, 2], [3]])
    with pytest.raises(SymbolicError):
        SymbolicOp.make("not-an-operation")
    with pytest.raises(SymbolicError):
        apply_program(op("repeat", direction="sideways", count=1), [[1]])
