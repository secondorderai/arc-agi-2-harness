from dataclasses import FrozenInstanceError

import pytest

from arc_agent.object_ir import (
    BoundingBox,
    color_reference_features,
    connected_components,
    extract_objects,
    extract_scene,
    infer_background,
    infer_scene_transition,
    relative_position,
)


def test_background_inference_is_frequency_then_zero_then_first_seen() -> None:
    assert infer_background([[0, 1], [1, 0]]) == 0
    assert infer_background([[2, 3], [3, 2]]) == 2
    assert infer_background([[4, 5], [5, 4]]) == 4
    assert infer_background([[7, 7, 1], [1, 2, 2]]) == 7


def test_grid_validation_rejects_empty_ragged_and_non_arc_values() -> None:
    with pytest.raises(ValueError):
        infer_background([])
    with pytest.raises(ValueError):
        infer_background([[0], [0, 1]])
    with pytest.raises(ValueError):
        infer_background([[0, 10]])


def test_four_and_eight_connected_components_are_deterministic() -> None:
    grid = [[1, 0], [0, 2]]
    assert connected_components(grid, background=0, connectivity=4) == (((0, 0),), ((1, 1),))
    assert connected_components(grid, background=0, connectivity=8, multicolor=True) == (
        ((0, 0), (1, 1)),
    )
    assert connected_components(
        grid, background=0, connectivity=8, multicolor=False
    ) == (((0, 0),), ((1, 1),))


def test_object_bbox_masks_area_and_holes() -> None:
    ring = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    obj = extract_objects(ring)[0]
    assert obj.object_id == 0
    assert obj.color == 1
    assert obj.colors == (1,)
    assert obj.area == 9
    assert obj.bbox == BoundingBox(1, 1, 3, 3)
    assert obj.mask == ((True, True, True), (True, True, True), (True, True, True))
    assert obj.hole_count == 0

    ring[2][2] = 0
    obj = extract_objects(ring)[0]
    assert obj.area == 8
    assert obj.mask == ((True, True, True), (True, False, True), (True, True, True))
    assert len(obj.holes) == 1
    assert obj.holes[0].cells == ((2, 2),)
    assert obj.holes[0].bbox == BoundingBox(2, 2, 2, 2)
    assert obj.holes[0].mask == ((True,),)
    assert obj.holes[0].shape_signature == ((0, 0),)


def test_canonical_shape_signature_ignores_d4_orientation() -> None:
    first = extract_objects(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )[0]
    rotated = extract_objects(
        [
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
        ],
        background=0,
    )[0]
    assert first.canonical_shape_signature() == rotated.canonical_shape_signature()
    assert first.canonical_color_signature() == rotated.canonical_color_signature()
    assert first.shape_signature != first.canonical_shape_signature()


def test_multicolor_objects_keep_per_cell_colors() -> None:
    grid = [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
    objects = extract_objects(grid, connectivity=8, multicolor=True)
    assert len(objects) == 1
    obj = objects[0]
    assert obj.color is None
    assert obj.colors == (1, 2)
    assert obj.cells == ((0, 1), (1, 2))
    assert obj.cell_colors == (1, 2)
    assert obj.color_mask == ((1, None), (None, 2))


def test_scene_round_trip_and_color_reference_inside_ring() -> None:
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    scene = extract_scene(grid)
    assert scene.background == 0
    assert scene.shape == (5, 5)
    assert scene.render() == grid
    assert tuple(tuple(row) for row in scene.grid) == tuple(tuple(row) for row in grid)
    assert len(scene.objects) == 2
    outer, inner = scene.objects
    assert outer.hole_count == 1
    relation = scene.relations_between(outer.object_id, inner.object_id)[0]
    assert relation.has("contains")
    assert relation.direction == "overlap"
    reverse = scene.relations_between(inner.object_id, outer.object_id)[0]
    assert reverse.has("inside")
    references = scene.references_from(outer.object_id)
    assert references == (
        references[0],
    )
    assert references[0].target_id == inner.object_id
    assert references[0].color == 2
    assert references[0].kind == "interior_color"
    assert references[0].coordinates == ((2, 2),)


def test_relations_cover_touching_alignment_and_relative_position() -> None:
    grid = [
        [0, 1, 1, 2, 0],
        [0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
    ]
    scene = extract_scene(grid)
    one, two, three = scene.objects
    one_two = scene.relations_between(one.object_id, two.object_id)[0]
    assert one_two.has("adjacent")
    assert one_two.has("touching")
    assert one_two.has("horizontal_alignment")
    assert one_two.direction == "east"
    assert relative_position(one, two) == "east"
    one_three = scene.relations_between(one.object_id, three.object_id)[0]
    assert one_three.direction == "southwest"


def test_diagonal_touching_is_distinguished_from_orthogonal_adjacency() -> None:
    scene = extract_scene([[1, 0], [0, 2]])
    relation = scene.relations_between(0, 1)[0]
    assert relation.has("touching")
    assert relation.has("diagonal_touching")
    assert not relation.has("adjacent")


def test_direct_color_reference_extraction_accepts_explicit_objects() -> None:
    grid = [[0, 1, 0], [0, 2, 0], [0, 0, 0]]
    objects = extract_objects(grid)
    assert color_reference_features(grid, objects) == ()


def test_dataclasses_are_immutable_and_object_ids_are_stable() -> None:
    obj = extract_objects([[0, 3]])[0]
    with pytest.raises(FrozenInstanceError):
        obj.object_id = 4  # type: ignore[misc]
    assert [item.object_id for item in extract_objects([[0, 3, 0], [2, 0, 0]])] == [0, 1]


def test_transition_matches_a_translated_object() -> None:
    input_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    output_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    transition = infer_scene_transition(input_grid, output_grid)
    assert len(transition.matches) == 1
    match = transition.matches[0]
    assert match.changes == ("moved",)
    assert match.translation == (1.0, 2.0)
    assert match.shape_transform == "identity"
    assert transition.moved == (match,)
    assert transition.added_objects == ()
    assert transition.removed_objects == ()


def test_transition_detects_recolor_without_movement() -> None:
    input_grid = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    output_grid = [[0, 0, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0]]
    transition = infer_scene_transition(input_grid, output_grid)
    match = transition.matches[0]
    assert match.changes == ("recolored",)
    assert match.translation == (0.0, 0.0)
    assert match.same_shape
    assert not match.same_colored_shape
    assert transition.recolored == (match,)


def test_transition_detects_d4_shape_transform() -> None:
    input_grid = [
        [0, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 4, 4, 0],
        [0, 0, 0, 0],
    ]
    output_grid = [
        [0, 0, 0, 0],
        [0, 4, 4, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0],
    ]
    transition = infer_scene_transition(input_grid, output_grid)
    match = transition.matches[0]
    assert match.changes == ("transformed",)
    assert match.same_shape
    assert match.shape_transform != "identity"
    assert transition.transformed == (match,)


def test_transition_records_added_and_removed_objects() -> None:
    input_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    output_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 3, 0],
        [0, 0, 0, 3, 3, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    transition = infer_scene_transition(input_grid, output_grid)
    assert len(transition.matches) == 1
    assert transition.matches[0].changes == ("preserved",)
    assert [obj.color for obj in transition.removed_objects] == [2]
    assert [obj.color for obj in transition.added_objects] == [3]
    assert transition.event_tokens() == ("preserve:1", "add:1", "remove:1")


def test_global_assignment_handles_duplicate_objects_without_id_matching() -> None:
    input_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    output_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    transition = infer_scene_transition(input_grid, output_grid)
    assignments = {
        match.input_object_id: match.output_object_id
        for match in transition.matches
    }
    assert assignments == {0: 1, 1: 0}
    assert transition.matches[0].translation == (2.0, 0.0)
    assert transition.matches[1].translation == (-2.0, 0.0)


def test_relation_changes_and_compact_transition_features() -> None:
    input_grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    output_grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    transition = infer_scene_transition(input_grid, output_grid)
    assert len(transition.relation_changes) == 2
    forward = next(
        change
        for change in transition.relation_changes
        if change.input_source_id == 0 and change.input_target_id == 1
    )
    assert "south" in forward.added_relations
    assert "east" in forward.removed_relations
    assert forward.geometry_changed
    features = transition.compact_features()
    assert features["matches"] == 2
    assert features["moved"] == 1
    assert features["relation_changes"] == 2
    assert "move:1" in features["event_tokens"]
    assert "2 objects" in transition.summary()
    assert "confidence" in transition.matches[0].summary()


def test_transition_inference_is_deterministic_for_grids_and_scenes() -> None:
    input_grid = [[0, 1, 0], [0, 0, 0], [0, 0, 2]]
    output_grid = [[0, 0, 1], [0, 0, 0], [2, 0, 0]]
    first = infer_scene_transition(input_grid, output_grid)
    second = infer_scene_transition(input_grid, output_grid)
    from_scenes = infer_scene_transition(extract_scene(input_grid), extract_scene(output_grid))
    assert first == second == from_scenes
    with pytest.raises(FrozenInstanceError):
        first.matches = ()  # type: ignore[misc]
