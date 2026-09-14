from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_agent.dsl import ProgramError, apply_program, program_family, search_programs
from arc_agent.models import ArcTask, Grid, Program, grid_accuracies, validate_grid
from arc_agent.object_ir import (
    GridObject,
    GridScene,
    extract_objects,
    extract_scene,
    infer_background,
)
from arc_agent.v3_models import (
    GameSignature,
    ParameterSpec,
    ProposedSignature,
    SignatureCandidate,
    SignatureFailure,
    SignatureFamily,
    SignatureStep,
    SignatureVerification,
    StepArgument,
    finalize_signature,
    signature_hash,
)

ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "identity": set(),
    "rotate": {"turns"},
    "flip": {"axis"},
    "transpose": set(),
    "recolor": {"mapping"},
    "recolor_color": {"source", "target"},
    "crop_background": {"background"},
    "scale": {"rows", "columns"},
    "tile": {"rows", "columns"},
    "pad": {"top", "bottom", "left", "right", "color"},
    "select_component": {"background", "criterion", "connectivity", "multicolor"},
    "translate": {"rows", "columns", "background"},
    "translate_by_vector": {"offset", "background"},
    "concat": {"axis", "transform", "order"},
    "complete_symmetry": {"axis", "background"},
    "split_overlay": {"axis", "mode", "background", "separator_color", "output_color"},
    "move_color_to_contact": {"moving_color", "target_color", "background"},
    "pattern_mosaic": {
        "background",
        "empty_color",
        "base_color",
        "marker_color",
        "row_order",
        "column_order",
        "output_rows",
        "output_columns",
    },
    "repair_periodic_panels": set(),
    "select_unique_panel_per_band": set(),
    "deduplicate_repeated_half": set(),
    "fold_quadrants_overlap": set(),
    "mark_most_frequent_above_separator": set(),
    "keep_most_frequent_colors": {"keep_count", "output_color"},
    "pack_points_serpentine": set(),
    "fill_between_row_markers": {"fill_color"},
    "expand_seed_to_width": set(),
    "summarize_nonbackground_count": set(),
    "fill_with_most_frequent_color": set(),
    "frequency_histogram_columns": set(),
    "recolor_singleton_components": {"background"},
    "move_center_block_to_corners": set(),
    "complete_periodic_pattern": {"mode"},
    "complete_bbox_symmetry": set(),
    "mark_closed_square_corners": {"output_color"},
    "extract_scene": {"background", "connectivity", "multicolor"},
    "scene_objects": set(),
    "select_object": {"criterion"},
    "select_object_by_rank": {"rank", "order"},
    "object_mask": set(),
    "translate_mask": {"rows", "columns"},
    "translate_mask_by_vector": {"offset"},
    "render_mask": {"background", "output_color", "crop"},
    "count_objects": set(),
    "render_count": {"background", "output_color", "orientation"},
}

REQUIRED_ARGUMENTS: dict[str, set[str]] = {
    "flip": {"axis"},
    "recolor": {"mapping"},
    "recolor_color": {"source", "target"},
    "scale": {"rows", "columns"},
    "tile": {"rows", "columns"},
    "concat": {"axis"},
    "complete_symmetry": {"axis"},
    "split_overlay": {"axis", "mode", "separator_color", "output_color"},
    "move_color_to_contact": {"moving_color", "target_color"},
    "translate_by_vector": {"offset"},
    "pattern_mosaic": {
        "empty_color",
        "base_color",
        "marker_color",
        "row_order",
        "column_order",
        "output_rows",
        "output_columns",
    },
    "keep_most_frequent_colors": {"keep_count", "output_color"},
    "fill_between_row_markers": {"fill_color"},
    "mark_closed_square_corners": {"output_color"},
    "render_mask": {"output_color"},
    "render_count": {"output_color"},
    "select_object_by_rank": {"rank"},
    "translate_mask_by_vector": {"offset"},
}

TYPED_OPERATORS: dict[str, tuple[str, str]] = {
    **{
        op: ("Grid", "Grid")
        for op in ALLOWED_ARGUMENTS
        if op
        not in {
            "extract_scene",
            "scene_objects",
            "select_object",
            "select_object_by_rank",
            "object_mask",
            "translate_mask",
            "translate_mask_by_vector",
            "render_mask",
            "count_objects",
            "render_count",
        }
    },
    "extract_scene": ("Grid", "Scene"),
    "scene_objects": ("Scene", "ObjectSet"),
    "select_object": ("ObjectSet", "Object"),
    "select_object_by_rank": ("ObjectSet", "Object"),
    "object_mask": ("Object", "Mask"),
    "translate_mask": ("Mask", "Mask"),
    "translate_mask_by_vector": ("Mask", "Mask"),
    "render_mask": ("Mask", "Grid"),
    "count_objects": ("ObjectSet", "Scalar"),
    "render_count": ("Scalar", "Grid"),
}

ENUM_ARGUMENTS: dict[tuple[str, str], tuple[Any, ...]] = {
    ("rotate", "turns"): (1, 2, 3),
    ("flip", "axis"): ("horizontal", "vertical"),
    ("select_component", "criterion"): ("largest", "smallest", "widest", "tallest"),
    ("select_component", "connectivity"): (4, 8),
    ("select_component", "multicolor"): (False, True),
    ("concat", "axis"): ("horizontal", "vertical"),
    ("concat", "transform"): (
        "identity",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "flip_horizontal",
        "flip_vertical",
        "transpose",
    ),
    ("concat", "order"): ("input_first", "transformed_first"),
    ("complete_symmetry", "axis"): ("horizontal", "vertical", "rotational"),
    ("split_overlay", "axis"): ("horizontal", "vertical"),
    ("split_overlay", "mode"): ("union", "intersection", "xor", "neither"),
    ("complete_periodic_pattern", "mode"): ("tile", "diagonal_sum", "diagonal_difference"),
    ("select_object", "criterion"): (
        "largest",
        "smallest",
        "widest",
        "tallest",
        "most_holes",
    ),
    ("select_object_by_rank", "order"): (
        "area_desc",
        "area_asc",
        "left_to_right",
        "top_to_bottom",
    ),
    ("render_mask", "crop"): (False, True),
    ("render_count", "orientation"): ("horizontal", "vertical"),
}

FAMILY_MAP: dict[str, SignatureFamily] = {
    "color-mapping": "color-attribute",
    "rigid-transform": "rigid-geometry",
    "object-selection": "object-relational",
    "object-arrangement": "object-relational",
    "relational-drawing": "object-relational",
    "counting-and-selection": "counting-compression",
    "repetition-compression": "counting-compression",
    "size-and-repetition": "repetition-pattern",
    "pattern-repair": "repetition-pattern",
    "panel-selection": "repetition-pattern",
    "logical-composition": "repetition-pattern",
    "symbolic-composition": "repetition-pattern",
}

SEMANTIC_SOURCES_BY_TYPE: dict[str, tuple[str, ...]] = {
    "Color": (
        "background",
        "rarest_input_color",
        "most_frequent_input_color",
        "output_only_color",
        "most_frequent_output_color",
        "ranked_object_color",
    ),
    "Scalar": (
        "input_height",
        "input_width",
        "output_height",
        "output_width",
        "row_scale",
        "column_scale",
        "inferred_translation_row",
        "inferred_translation_column",
        "demonstrated_object_rank",
        "train_pair_count",
    ),
    "Vector": ("inferred_translation",),
}

ARGUMENT_TYPES: dict[tuple[str, str], str] = {
    ("crop_background", "background"): "Color",
    ("recolor_color", "source"): "Color",
    ("recolor_color", "target"): "Color",
    ("select_component", "background"): "Color",
    ("translate", "background"): "Color",
    ("translate_by_vector", "background"): "Color",
    ("complete_symmetry", "background"): "Color",
    ("split_overlay", "background"): "Color",
    ("split_overlay", "separator_color"): "Color",
    ("split_overlay", "output_color"): "Color",
    ("move_color_to_contact", "moving_color"): "Color",
    ("move_color_to_contact", "target_color"): "Color",
    ("move_color_to_contact", "background"): "Color",
    ("pad", "color"): "Color",
    ("fill_between_row_markers", "fill_color"): "Color",
    ("keep_most_frequent_colors", "output_color"): "Color",
    ("mark_closed_square_corners", "output_color"): "Color",
    ("extract_scene", "background"): "Color",
    ("render_mask", "background"): "Color",
    ("render_mask", "output_color"): "Color",
    ("render_count", "background"): "Color",
    ("render_count", "output_color"): "Color",
    ("rotate", "turns"): "Scalar",
    ("scale", "rows"): "Scalar",
    ("scale", "columns"): "Scalar",
    ("tile", "rows"): "Scalar",
    ("tile", "columns"): "Scalar",
    ("pad", "top"): "Scalar",
    ("pad", "bottom"): "Scalar",
    ("pad", "left"): "Scalar",
    ("pad", "right"): "Scalar",
    ("translate", "rows"): "Scalar",
    ("translate", "columns"): "Scalar",
    ("translate_mask", "rows"): "Scalar",
    ("translate_mask", "columns"): "Scalar",
    ("translate_by_vector", "offset"): "Vector",
    ("translate_mask_by_vector", "offset"): "Vector",
    ("select_object_by_rank", "rank"): "Scalar",
}


class SignatureError(ValueError):
    pass


def operator_catalog() -> dict[str, dict[str, Any]]:
    return {
        op: {
            "arguments": sorted(arguments),
            "required_arguments": sorted(REQUIRED_ARGUMENTS.get(op, set())),
            "argument_types": {
                argument: ARGUMENT_TYPES[(op, argument)]
                for argument in sorted(arguments)
                if (op, argument) in ARGUMENT_TYPES
            },
            "enum_values": {
                argument: list(ENUM_ARGUMENTS[(op, argument)])
                for argument in sorted(arguments)
                if (op, argument) in ENUM_ARGUMENTS
            },
            "input_type": TYPED_OPERATORS[op][0],
            "output_type": TYPED_OPERATORS[op][1],
        }
        for op, arguments in sorted(ALLOWED_ARGUMENTS.items())
    }


def dsl_sha256() -> str:
    digest = hashlib.sha256()
    for module in (
        Path(__file__),
        Path(__file__).with_name("v3_models.py"),
        Path(__file__).with_name("dsl.py"),
        Path(__file__).with_name("object_ir.py"),
    ):
        digest.update(module.name.encode())
        digest.update(module.read_bytes())
    return digest.hexdigest()


def _literal_scalars(value: Any) -> int:
    if isinstance(value, dict):
        return sum(1 + _literal_scalars(item) for key, item in value.items() if str(key))
    if isinstance(value, list):
        return sum(_literal_scalars(item) for item in value)
    return 1


def validate_signature(
    signature: ProposedSignature | GameSignature,
    *,
    max_pipeline_steps: int = 16,
    max_parameters: int = 20,
    max_literal_scalars: int = 64,
    forbidden_identifiers: Iterable[str] = (),
) -> None:
    if isinstance(signature, GameSignature):
        if signature.canonical_hash != signature_hash(signature):
            raise SignatureError("signature canonical hash does not match its typed pipeline")
        if signature.complexity != signature_complexity(signature):
            raise SignatureError("signature complexity does not match its typed pipeline")
    if len(signature.pipeline) > max_pipeline_steps:
        raise SignatureError("signature pipeline is too long")
    if len(signature.parameters) > max_parameters:
        raise SignatureError("signature has too many parameters")
    parameter_names = {parameter.name for parameter in signature.parameters}
    parameter_types = {parameter.name: parameter.value_type for parameter in signature.parameters}
    for parameter in signature.parameters:
        if parameter.source == "constant":
            continue
        allowed_sources = SEMANTIC_SOURCES_BY_TYPE.get(parameter.value_type, ())
        if parameter.source not in allowed_sources:
            raise SignatureError(
                f"parameter {parameter.name!r} source {parameter.source!r} is incompatible "
                f"with {parameter.value_type}"
            )
    literal_count = sum(
        _literal_scalars(parameter.value)
        for parameter in signature.parameters
        if parameter.value is not None
    )
    forbidden = {value for value in forbidden_identifiers if value}
    authored_strings = [signature.hypothesis]
    authored_strings.extend(
        parameter.value
        for parameter in signature.parameters
        if isinstance(parameter.value, str)
    )
    authored_strings.extend(
        argument.value
        for step in signature.pipeline
        for argument in step.args
        if isinstance(argument.value, str)
    )
    if any(
        identifier in value
        for identifier in forbidden
        for value in authored_strings
    ):
        raise SignatureError("signature embeds a forbidden task identifier")
    current_type = "Grid"
    for step in signature.pipeline:
        if step.op not in ALLOWED_ARGUMENTS:
            raise SignatureError(f"unsupported V3 DSL operation: {step.op}")
        supplied = {argument.name for argument in step.args}
        unexpected = supplied - ALLOWED_ARGUMENTS[step.op]
        if unexpected:
            raise SignatureError(f"unexpected arguments for {step.op}: {sorted(unexpected)}")
        missing = REQUIRED_ARGUMENTS.get(step.op, set()) - supplied
        if missing:
            raise SignatureError(f"missing arguments for {step.op}: {sorted(missing)}")
        expected_input, expected_output = TYPED_OPERATORS[step.op]
        if step.input_type != expected_input or step.output_type != expected_output:
            raise SignatureError(
                f"{step.op} must be typed {expected_input}->{expected_output}, got "
                f"{step.input_type}->{step.output_type}"
            )
        if step.input_type != current_type:
            raise SignatureError(
                f"pipeline type mismatch: {step.op} expects {step.input_type}, got {current_type}"
            )
        current_type = step.output_type
        for argument in step.args:
            if isinstance(argument.value, str) and argument.value.startswith("$"):
                name = argument.value[1:]
                if name not in parameter_names:
                    raise SignatureError(f"unknown signature parameter {name!r}")
                expected_type = ARGUMENT_TYPES.get((step.op, argument.name))
                if expected_type is not None and parameter_types[name] != expected_type:
                    raise SignatureError(
                        f"argument {step.op}.{argument.name} needs {expected_type}, "
                        f"but ${name} is {parameter_types[name]}"
                    )
            else:
                literal_count += _literal_scalars(argument.value)
            if step.op == "recolor" and argument.name == "mapping":
                if not isinstance(argument.value, list) or len(argument.value) != 10:
                    raise SignatureError("recolor.mapping must contain exactly ten colors")
                if any(
                    not isinstance(color, int)
                    or isinstance(color, bool)
                    or not 0 <= color <= 9
                    for color in argument.value
                ):
                    raise SignatureError("recolor.mapping colors must be integers from 0 to 9")
    if literal_count > max_literal_scalars:
        raise SignatureError(
            f"signature contains {literal_count} literal scalars; limit is {max_literal_scalars}"
        )
    if current_type != "Grid":
        raise SignatureError(f"signature pipeline must finish with Grid, got {current_type}")


def reject_embedded_task_data(
    signature: ProposedSignature | GameSignature,
    task: ArcTask,
) -> None:
    authored_material = signature.model_dump(mode="json")
    for field in ("canonical_hash", "source_task_ids", "complexity", "provenance"):
        authored_material.pop(field, None)
    serialized = json.dumps(authored_material, sort_keys=True, separators=(",", ":"))
    literal_lists = [
        parameter.value
        for parameter in signature.parameters
        if isinstance(parameter.value, list)
    ] + [
        argument.value
        for step in signature.pipeline
        for argument in step.args
        if isinstance(argument.value, list)
    ]
    grids = [pair.input for pair in task.train]
    grids.extend(pair.output for pair in task.train if pair.output is not None)
    grids.extend(pair.input for pair in task.test)
    for grid in grids:
        encoded = json.dumps(grid, separators=(",", ":"))
        flattened = [cell for row in grid for cell in row]
        if encoded in serialized or flattened in literal_lists:
            raise SignatureError("signature embeds a task grid or serialized task grid")


def _mode(values: Iterable[int]) -> int:
    counter = Counter(values)
    if not counter:
        raise SignatureError("cannot infer a color from an empty collection")
    return min(counter, key=lambda value: (-counter[value], value))


def _rarest(values: Iterable[int]) -> int:
    counter = Counter(values)
    if not counter:
        raise SignatureError("cannot infer a color from an empty collection")
    return min(counter, key=lambda value: (counter[value], value))


def _consistent(values: Sequence[int], name: str) -> int:
    if not values or len(set(values)) != 1:
        raise SignatureError(f"{name} is not consistent across demonstrations")
    return values[0]


def _translation(train: Sequence[tuple[Grid, Grid]]) -> tuple[int, int]:
    offsets: list[tuple[int, int]] = []
    for input_grid, output_grid in train:
        input_objects = extract_objects(input_grid, background=infer_background(input_grid))
        output_objects = extract_objects(output_grid, background=infer_background(output_grid))
        if not input_objects or not output_objects:
            raise SignatureError("translation inference needs foreground objects")
        input_box = max(input_objects, key=lambda item: item.area).bbox
        output_box = max(output_objects, key=lambda item: item.area).bbox
        offsets.append((output_box.top - input_box.top, output_box.left - input_box.left))
    if len(set(offsets)) != 1:
        raise SignatureError("translation offset is not consistent across demonstrations")
    return offsets[0]


def _cropped_object_grid(obj: GridObject) -> Grid:
    colors = dict(zip(obj.cells, obj.cell_colors, strict=True))
    return [
        [
            colors.get((row, column), obj.background)
            for column in range(obj.bbox.left, obj.bbox.right + 1)
        ]
        for row in range(obj.bbox.top, obj.bbox.bottom + 1)
    ]


def _demonstrated_object_rank(train: Sequence[tuple[Grid, Grid]]) -> int:
    ranks: list[int] = []
    for input_grid, output_grid in train:
        objects = sorted(
            extract_objects(input_grid, background=infer_background(input_grid)),
            key=lambda obj: (-obj.area, obj.object_id),
        )
        matching = [
            rank for rank, obj in enumerate(objects) if _cropped_object_grid(obj) == output_grid
        ]
        if len(matching) != 1:
            raise SignatureError(
                "demonstrated object rank needs one area-ranked object matching each output"
            )
        ranks.append(matching[0])
    return _consistent(ranks, "demonstrated object rank")


def resolve_parameter(
    parameter: ParameterSpec,
    train: Sequence[tuple[Grid, Grid]],
    grid: Grid,
) -> Any:
    if parameter.source == "constant":
        return deepcopy(parameter.value)
    input_values = [cell for row in grid for cell in row]
    train_input_values = [cell for left, _ in train for row in left for cell in row]
    train_output_values = [cell for _, right in train for row in right for cell in row]
    if parameter.source == "background":
        return infer_background(grid)
    if parameter.source == "rarest_input_color":
        return _rarest(input_values)
    if parameter.source == "most_frequent_input_color":
        return _mode(input_values)
    if parameter.source == "output_only_color":
        colors = sorted(set(train_output_values) - set(train_input_values))
        if len(colors) != 1:
            raise SignatureError("output_only_color is not unique")
        return colors[0]
    if parameter.source == "most_frequent_output_color":
        return _mode(train_output_values)
    if parameter.source == "ranked_object_color":
        rank = _demonstrated_object_rank(train)
        objects = sorted(
            extract_objects(grid, background=infer_background(grid)),
            key=lambda obj: (-obj.area, obj.object_id),
        )
        if not 0 <= rank < len(objects) or objects[rank].color is None:
            raise SignatureError("ranked object color needs a monochrome object at the rank")
        return objects[rank].color
    if parameter.source == "input_height":
        return len(grid)
    if parameter.source == "input_width":
        return len(grid[0])
    if parameter.source == "output_height":
        return _consistent([len(right) for _, right in train], "output height")
    if parameter.source == "output_width":
        return _consistent([len(right[0]) for _, right in train], "output width")
    if parameter.source == "row_scale":
        if any(len(right) % len(left) for left, right in train):
            raise SignatureError("row scale is not integral across all demonstrations")
        return _consistent(
            [len(right) // len(left) for left, right in train],
            "integer row scale",
        )
    if parameter.source == "column_scale":
        if any(len(right[0]) % len(left[0]) for left, right in train):
            raise SignatureError("column scale is not integral across all demonstrations")
        return _consistent(
            [len(right[0]) // len(left[0]) for left, right in train],
            "integer column scale",
        )
    if parameter.source == "inferred_translation_row":
        return _translation(train)[0]
    if parameter.source == "inferred_translation_column":
        return _translation(train)[1]
    if parameter.source == "inferred_translation":
        return list(_translation(train))
    if parameter.source == "demonstrated_object_rank":
        return _demonstrated_object_rank(train)
    if parameter.source == "train_pair_count":
        return len(train)
    raise SignatureError(f"unsupported parameter source {parameter.source}")


def compile_signature(
    signature: ProposedSignature | GameSignature,
    train: Sequence[tuple[Grid, Grid]],
    grid: Grid,
) -> Program:
    validate_signature(signature)
    if any(step.input_type != "Grid" or step.output_type != "Grid" for step in signature.pipeline):
        raise SignatureError("typed object pipelines cannot be compiled to the legacy Program AST")
    resolved = {
        parameter.name: resolve_parameter(parameter, train, grid)
        for parameter in signature.parameters
    }
    steps: list[Program] = []
    for step in signature.pipeline:
        args = {
            argument.name: (
                deepcopy(resolved[argument.value[1:]])
                if isinstance(argument.value, str) and argument.value.startswith("$")
                else deepcopy(argument.value)
            )
            for argument in step.args
        }
        if step.op == "recolor" and isinstance(args.get("mapping"), list):
            args["mapping"] = dict(enumerate(args["mapping"]))
        if step.op == "recolor_color":
            steps.append(
                Program(
                    op="recolor",
                    args={"mapping": {int(args["source"]): int(args["target"])}},
                )
            )
        elif step.op == "translate_by_vector":
            offset = list(args["offset"])
            if len(offset) != 2:
                raise SignatureError("translation vector must contain row and column offsets")
            translated_args = {"rows": int(offset[0]), "columns": int(offset[1])}
            if "background" in args:
                translated_args["background"] = int(args["background"])
            steps.append(Program(op="translate", args=translated_args))
        else:
            steps.append(Program(op=step.op, args=args))
    return steps[0] if len(steps) == 1 else Program(op="compose", steps=steps)


@dataclass(frozen=True)
class _ObjectSetValue:
    scene: GridScene
    objects: tuple[GridObject, ...]


@dataclass(frozen=True)
class _ObjectValue:
    scene: GridScene
    obj: GridObject


@dataclass(frozen=True)
class _MaskValue:
    height: int
    width: int
    cells: tuple[tuple[int, int], ...]


def _resolved_arguments(
    step: SignatureStep,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        argument.name: (
            deepcopy(parameters[argument.value[1:]])
            if isinstance(argument.value, str) and argument.value.startswith("$")
            else deepcopy(argument.value)
        )
        for argument in step.args
    }


def _select_runtime_object(value: _ObjectSetValue, criterion: str) -> _ObjectValue:
    if not value.objects:
        raise SignatureError("select_object needs at least one object")
    if criterion == "largest":
        selected = max(value.objects, key=lambda obj: (obj.area, -obj.object_id))
    elif criterion == "smallest":
        selected = min(value.objects, key=lambda obj: (obj.area, obj.object_id))
    elif criterion == "widest":
        selected = max(value.objects, key=lambda obj: (obj.bbox.width, obj.area, -obj.object_id))
    elif criterion == "tallest":
        selected = max(value.objects, key=lambda obj: (obj.bbox.height, obj.area, -obj.object_id))
    elif criterion == "most_holes":
        selected = max(value.objects, key=lambda obj: (obj.hole_count, obj.area, -obj.object_id))
    else:
        raise SignatureError(f"unknown object criterion {criterion!r}")
    return _ObjectValue(scene=value.scene, obj=selected)


def _select_runtime_object_by_rank(
    value: _ObjectSetValue,
    rank: int,
    order: str,
) -> _ObjectValue:
    if order == "area_desc":
        ordered = sorted(value.objects, key=lambda obj: (-obj.area, obj.object_id))
    elif order == "area_asc":
        ordered = sorted(value.objects, key=lambda obj: (obj.area, obj.object_id))
    elif order == "left_to_right":
        ordered = sorted(
            value.objects,
            key=lambda obj: (obj.bbox.left, obj.bbox.top, obj.object_id),
        )
    elif order == "top_to_bottom":
        ordered = sorted(
            value.objects,
            key=lambda obj: (obj.bbox.top, obj.bbox.left, obj.object_id),
        )
    else:
        raise SignatureError(f"unknown object ordering {order!r}")
    if not 0 <= rank < len(ordered):
        raise SignatureError(f"object rank {rank} is outside 0..{len(ordered) - 1}")
    return _ObjectValue(scene=value.scene, obj=ordered[rank])


def _execute_typed_step(step: SignatureStep, current: Any, args: dict[str, Any]) -> Any:
    if step.op == "extract_scene":
        return extract_scene(
            current,
            background=(int(args["background"]) if "background" in args else None),
            connectivity=int(args.get("connectivity", 4)),
            multicolor=bool(args.get("multicolor", False)),
        )
    if step.op == "scene_objects":
        if not isinstance(current, GridScene):
            raise SignatureError("scene_objects requires a Scene")
        return _ObjectSetValue(scene=current, objects=current.objects)
    if step.op == "select_object":
        if not isinstance(current, _ObjectSetValue):
            raise SignatureError("select_object requires an ObjectSet")
        return _select_runtime_object(current, str(args.get("criterion", "largest")))
    if step.op == "select_object_by_rank":
        if not isinstance(current, _ObjectSetValue):
            raise SignatureError("select_object_by_rank requires an ObjectSet")
        return _select_runtime_object_by_rank(
            current,
            int(args["rank"]),
            str(args.get("order", "area_desc")),
        )
    if step.op == "object_mask":
        if not isinstance(current, _ObjectValue):
            raise SignatureError("object_mask requires an Object")
        return _MaskValue(
            height=current.scene.height,
            width=current.scene.width,
            cells=current.obj.cells,
        )
    if step.op in {"translate_mask", "translate_mask_by_vector"}:
        if not isinstance(current, _MaskValue):
            raise SignatureError("translate_mask requires a Mask")
        if step.op == "translate_mask_by_vector":
            offset = list(args["offset"])
            if len(offset) != 2:
                raise SignatureError("translation vector must contain row and column offsets")
            row_delta, column_delta = int(offset[0]), int(offset[1])
        else:
            row_delta, column_delta = int(args.get("rows", 0)), int(args.get("columns", 0))
        cells = tuple((row + row_delta, column + column_delta) for row, column in current.cells)
        if any(
            not (0 <= row < current.height and 0 <= column < current.width)
            for row, column in cells
        ):
            raise SignatureError("translated mask leaves the grid")
        return _MaskValue(height=current.height, width=current.width, cells=cells)
    if step.op == "render_mask":
        if not isinstance(current, _MaskValue):
            raise SignatureError("render_mask requires a Mask")
        background = int(args.get("background", 0))
        output_color = int(args["output_color"])
        if bool(args.get("crop", False)):
            top = min(row for row, _ in current.cells)
            bottom = max(row for row, _ in current.cells)
            left = min(column for _, column in current.cells)
            right = max(column for _, column in current.cells)
            occupied = {(row - top, column - left) for row, column in current.cells}
            return [
                [
                    output_color if (row, column) in occupied else background
                    for column in range(right - left + 1)
                ]
                for row in range(bottom - top + 1)
            ]
        occupied = set(current.cells)
        return [
            [
                output_color if (row, column) in occupied else background
                for column in range(current.width)
            ]
            for row in range(current.height)
        ]
    if step.op == "count_objects":
        if not isinstance(current, _ObjectSetValue):
            raise SignatureError("count_objects requires an ObjectSet")
        return len(current.objects)
    if step.op == "render_count":
        if not isinstance(current, int) or isinstance(current, bool):
            raise SignatureError("render_count requires a Scalar")
        if not 1 <= current <= 30:
            raise SignatureError("rendered object count must be between 1 and 30")
        background = int(args.get("background", 0))
        output_color = int(args["output_color"])
        if str(args.get("orientation", "horizontal")) == "horizontal":
            return [[output_color for _ in range(current)]]
        if str(args.get("orientation")) == "vertical":
            return [[output_color] for _ in range(current)]
        raise SignatureError("render_count orientation must be horizontal or vertical")
    raise SignatureError(f"unsupported typed operation {step.op}")


def solve_signature(
    signature: ProposedSignature | GameSignature,
    train: Sequence[tuple[Grid, Grid]],
    grid: Grid,
) -> Grid:
    validate_signature(signature)
    parameters = {
        parameter.name: resolve_parameter(parameter, train, grid)
        for parameter in signature.parameters
    }
    current: Any = [list(row) for row in grid]
    for step in signature.pipeline:
        args = _resolved_arguments(step, parameters)
        if step.op == "recolor" and isinstance(args.get("mapping"), list):
            args["mapping"] = dict(enumerate(args["mapping"]))
        if step.op == "recolor_color":
            current = apply_program(
                Program(
                    op="recolor",
                    args={"mapping": {int(args["source"]): int(args["target"])}},
                ),
                current,
            )
        elif step.op == "translate_by_vector":
            offset = list(args["offset"])
            if len(offset) != 2:
                raise SignatureError("translation vector must contain row and column offsets")
            translated_args = {"rows": int(offset[0]), "columns": int(offset[1])}
            if "background" in args:
                translated_args["background"] = int(args["background"])
            current = apply_program(Program(op="translate", args=translated_args), current)
        elif step.op in TYPED_OPERATORS and TYPED_OPERATORS[step.op] == ("Grid", "Grid"):
            current = apply_program(Program(op=step.op, args=args), current)
        else:
            current = _execute_typed_step(step, current, args)
    if not isinstance(current, list):
        raise SignatureError("signature did not return a Grid")
    return validate_grid(current, label="signature output")


def _has_separator(grid: Grid) -> bool:
    return any(len(set(row)) == 1 for row in grid) or any(
        len({grid[row][column] for row in range(len(grid))}) == 1
        for column in range(len(grid[0]))
    )


def preconditions_pass(signature: ProposedSignature | GameSignature, task: ArcTask) -> bool:
    for precondition in signature.preconditions:
        values: list[bool] = []
        for pair in task.train:
            output = pair.output or [[]]
            same_shape = (len(pair.input), len(pair.input[0])) == (len(output), len(output[0]))
            objects = extract_objects(pair.input, background=infer_background(pair.input))
            if precondition.kind == "preserve_shape":
                value = same_shape
            elif precondition.kind == "change_shape":
                value = not same_shape
            elif precondition.kind == "has_output_only_color":
                value = bool(
                    {cell for row in output for cell in row}
                    - {cell for row in pair.input for cell in row}
                )
            elif precondition.kind == "single_nonbackground_object":
                value = len(objects) == 1
            elif precondition.kind == "multiple_objects":
                value = len(objects) > 1
            elif precondition.kind == "has_separator":
                value = _has_separator(pair.input)
            elif precondition.kind == "square_input":
                value = len(pair.input) == len(pair.input[0])
            elif precondition.kind == "consistent_integer_scale":
                value = (
                    len(output) % len(pair.input) == 0
                    and len(output[0]) % len(pair.input[0]) == 0
                )
            else:
                value = False
            values.append(value)
        observed = all(values)
        if observed != precondition.required:
            return False
    return True


def _grid_diff(actual: Grid | None, expected: Grid) -> tuple[float, float, str]:
    if actual is None:
        return 0.0, 0.0, "no grid returned"
    if (len(actual), len(actual[0])) != (len(expected), len(expected[0])):
        expected_shape = (len(expected), len(expected[0]))
        actual_shape = (len(actual), len(actual[0]))
        return 0.0, 0.0, f"expected shape {expected_shape}, got {actual_shape}"
    raw, balanced = grid_accuracies(actual, expected)
    differences = [
        (row, column, actual[row][column], expected[row][column])
        for row in range(len(expected))
        for column in range(len(expected[0]))
        if actual[row][column] != expected[row][column]
    ]
    detail = ", ".join(f"({r},{c})={got}/{want}" for r, c, got, want in differences[:40])
    return raw, balanced, detail


def _transform_grid(grid: Grid, reflected: bool, turns: int) -> Grid:
    result = [list(row) for row in grid]
    if reflected:
        result = [list(reversed(row)) for row in result]
    for _ in range(turns):
        result = [list(row) for row in zip(*reversed(result), strict=True)]
    return result


def _invariant_cases(
    signature: ProposedSignature | GameSignature,
    train: Sequence[tuple[Grid, Grid]],
) -> list[tuple[str, Sequence[tuple[Grid, Grid]], Grid, Grid]]:
    cases: list[tuple[str, Sequence[tuple[Grid, Grid]], Grid, Grid]] = []
    if "d4_equivariant" in signature.invariants:
        for reflected in (False, True):
            for turns in range(4):
                transformed = [
                    (
                        _transform_grid(left, reflected, turns),
                        _transform_grid(right, reflected, turns),
                    )
                    for left, right in train
                ]
                for index, (left, right) in enumerate(transformed):
                    cases.append((f"d4_{int(reflected)}_{turns}_{index}", transformed, left, right))
    if "color_equivariant" in signature.invariants:
        colors = sorted({cell for pair in train for grid in pair for row in grid for cell in row})
        if len(colors) > 1:
            rotated = colors[1:] + colors[:1]
            mapping = dict(zip(colors, rotated, strict=True))
            mapped = [
                (
                    [[mapping[cell] for cell in row] for row in left],
                    [[mapping[cell] for cell in row] for row in right],
                )
                for left, right in train
            ]
            for index, (left, right) in enumerate(mapped):
                cases.append((f"color_{index}", mapped, left, right))
    return cases


def verify_signature(
    signature: ProposedSignature | GameSignature,
    task: ArcTask,
    *,
    predict_tests: bool = True,
    run_leave_one_out: bool = True,
    run_declared_invariants: bool = True,
    leave_one_out_weight: float = 25.0,
    invariant_weight: float = 10.0,
    complexity_weight: float = 1.0,
) -> SignatureVerification:
    result = SignatureVerification(total_pairs=len(task.train))
    try:
        validate_signature(signature, forbidden_identifiers=[task.task_id])
        reject_embedded_task_data(signature, task)
    except (SignatureError, ValueError) as exc:
        result.failures.append(SignatureFailure(case="static", detail=str(exc)))
        return result
    if not preconditions_pass(signature, task):
        result.failures.append(
            SignatureFailure(case="preconditions", detail="hard precondition failed")
        )
        return result
    train = [(pair.input, pair.output or [[]]) for pair in task.train]
    raw_scores: list[float] = []
    balanced_scores: list[float] = []
    demo_predictions: list[Grid | None] = []
    for index, (input_grid, expected) in enumerate(train):
        actual: Grid | None = None
        try:
            actual = solve_signature(signature, train, input_grid)
        except (ProgramError, SignatureError, ValueError, KeyError) as exc:
            result.failures.append(SignatureFailure(case=f"demo_{index}", detail=str(exc)))
        demo_predictions.append(actual)
        raw, balanced, detail = _grid_diff(actual, expected)
        raw_scores.append(raw)
        balanced_scores.append(balanced)
        if actual == expected:
            result.exact_pairs += 1
        elif not any(failure.case == f"demo_{index}" for failure in result.failures):
            result.failures.append(
                SignatureFailure(
                    case=f"demo_{index}", detail=detail, expected=expected, actual=actual
                )
            )
    if run_leave_one_out and len(train) > 1:
        result.leave_one_out_total = len(train)
        for index, (input_grid, expected) in enumerate(train):
            reduced = [pair for pair_index, pair in enumerate(train) if pair_index != index]
            try:
                actual = solve_signature(signature, reduced, input_grid)
            except (ProgramError, SignatureError, ValueError, KeyError):
                actual = None
            result.leave_one_out_exact += int(actual == expected)
    if run_declared_invariants:
        local_invariants = set(signature.invariants) - {"d4_equivariant", "color_equivariant"}
        for (input_grid, _), actual in zip(train, demo_predictions, strict=True):
            for invariant in sorted(local_invariants):
                result.invariant_total += 1
                if actual is None:
                    continue
                if invariant == "preserve_shape":
                    passed = (len(input_grid), len(input_grid[0])) == (
                        len(actual),
                        len(actual[0]),
                    )
                elif invariant == "change_shape":
                    passed = (len(input_grid), len(input_grid[0])) != (
                        len(actual),
                        len(actual[0]),
                    )
                elif invariant == "preserve_background":
                    passed = infer_background(input_grid) == infer_background(actual)
                elif invariant == "object_count_preserved":
                    passed = len(
                        extract_objects(input_grid, background=infer_background(input_grid))
                    ) == len(extract_objects(actual, background=infer_background(actual)))
                else:
                    passed = False
                result.invariant_exact += int(passed)
        cases = _invariant_cases(signature, train)
        result.invariant_total += len(cases)
        for _, transformed_train, input_grid, expected in cases:
            try:
                actual = solve_signature(signature, transformed_train, input_grid)
            except (ProgramError, SignatureError, ValueError, KeyError):
                actual = None
            result.invariant_exact += int(actual == expected)
    if predict_tests:
        predictions: list[Grid] = []
        for pair in task.test:
            try:
                predictions.append(solve_signature(signature, train, pair.input))
            except (ProgramError, SignatureError, ValueError, KeyError):
                predictions = []
                break
        result.predictions = predictions
    result.cell_accuracy = sum(raw_scores) / max(1, len(raw_scores))
    result.balanced_accuracy = sum(balanced_scores) / max(1, len(balanced_scores))
    invariant_ok = result.invariant_exact == result.invariant_total
    result.accepted = result.exact_pairs == result.total_pairs and invariant_ok
    loo_ratio = result.leave_one_out_exact / max(1, result.leave_one_out_total)
    invariant_ratio = result.invariant_exact / max(1, result.invariant_total)
    result.score = (
        result.exact_pairs * 1_000.0
        + result.balanced_accuracy * 100.0
        + result.cell_accuracy * 10.0
        + loo_ratio * leave_one_out_weight
        + invariant_ratio * invariant_weight
        - signature_complexity(signature) * complexity_weight
    )
    return result


def signature_complexity(signature: ProposedSignature | GameSignature) -> int:
    return (
        len(signature.pipeline) * 2
        + len(signature.parameters)
        + len(signature.preconditions)
        + sum(len(step.args) for step in signature.pipeline)
    )


def failure_signature(verification: SignatureVerification) -> str:
    material = {
        "exact": verification.exact_pairs,
        "total": verification.total_pairs,
        "loo": [verification.leave_one_out_exact, verification.leave_one_out_total],
        "invariants": [verification.invariant_exact, verification.invariant_total],
        "failures": [failure.model_dump(mode="json") for failure in verification.failures],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _argument_value(value: Any) -> Any:
    if isinstance(value, dict):
        mapping = {int(key): int(item) for key, item in value.items()}
        return [mapping.get(color, color) for color in range(10)]
    if isinstance(value, list):
        return [int(item) for item in value]
    return value


def program_to_signature(
    program: Program,
    *,
    task: ArcTask,
    source_task_ids: list[str],
) -> GameSignature:
    steps = program.steps if program.op == "compose" else [program]
    pipeline = [
        SignatureStep(
            op=step.op,
            args=[
                StepArgument(name=name, value=_argument_value(value))
                for name, value in sorted(step.args.items())
            ],
        )
        for step in steps
    ]
    base_family = program_family(program)
    family = FAMILY_MAP.get(base_family, "rigid-geometry")
    proposal = ProposedSignature(
        family=family,
        hypothesis=f"deterministic {base_family} program",
        pipeline=pipeline,
        invariants=[],
    )
    return finalize_signature(
        proposal,
        source_task_ids=source_task_ids,
        provenance={"kind": "deterministic_search"},
    )


def mutate_signature(
    signature: GameSignature,
    *,
    source_task_ids: list[str],
    limit: int,
) -> list[GameSignature]:
    proposals: list[ProposedSignature] = []
    for parameter_index, parameter in enumerate(signature.parameters):
        for source in SEMANTIC_SOURCES_BY_TYPE.get(parameter.value_type, ()):
            if source == parameter.source:
                continue
            payload = signature.model_dump(mode="python")
            payload.pop("canonical_hash")
            payload.pop("source_task_ids")
            payload.pop("complexity")
            payload.pop("provenance")
            payload["parameters"][parameter_index] = ParameterSpec(
                name=parameter.name,
                value_type=parameter.value_type,
                source=source,
            )
            proposals.append(ProposedSignature.model_validate(payload))
    for step_index, step in enumerate(signature.pipeline):
        for argument_index, argument in enumerate(step.args):
            for alternative in ENUM_ARGUMENTS.get((step.op, argument.name), ()):
                if alternative == argument.value:
                    continue
                payload = signature.model_dump(mode="python")
                payload.pop("canonical_hash")
                payload.pop("source_task_ids")
                payload.pop("complexity")
                payload.pop("provenance")
                payload["pipeline"][step_index]["args"][argument_index]["value"] = alternative
                proposals.append(ProposedSignature.model_validate(payload))
    deduplicated: dict[str, GameSignature] = {}
    for proposal in proposals:
        candidate = finalize_signature(
            proposal,
            source_task_ids=source_task_ids,
            provenance={"kind": "mutation", "parent": signature.canonical_hash},
        )
        deduplicated.setdefault(candidate.canonical_hash, candidate)
        if len(deduplicated) >= limit:
            break
    return list(deduplicated.values())


def deterministic_search_signatures(
    task: ArcTask,
    *,
    family: SignatureFamily | None,
    limit: int,
) -> list[GameSignature]:
    found = search_programs(task, limit=max(limit * 2, limit))
    signatures: list[GameSignature] = []
    seen: set[str] = set()
    for candidate in found:
        if candidate.program is None:
            continue
        signature = program_to_signature(candidate.program, task=task, source_task_ids=[])
        if family is not None and signature.family != family:
            continue
        if signature.canonical_hash in seen:
            continue
        seen.add(signature.canonical_hash)
        signatures.append(signature)
        if len(signatures) >= limit:
            break
    return signatures


def candidate_id(
    task_id: str,
    source_kind: str,
    signature: GameSignature | None,
    verification: SignatureVerification,
) -> str:
    material = {
        "task": task_id,
        "source": source_kind,
        "signature": signature.canonical_hash if signature else None,
        "predictions": verification.predictions,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_candidate(
    task: ArcTask,
    source_kind: str,
    signature: GameSignature | None,
    verification: SignatureVerification,
    *,
    retrieval_rank: int | None = None,
) -> SignatureCandidate:
    return SignatureCandidate(
        candidate_id=candidate_id(task.task_id, source_kind, signature, verification),
        source_kind=source_kind,  # type: ignore[arg-type]
        signature=signature,
        verification=verification,
        retrieval_rank=retrieval_rank,
    )
