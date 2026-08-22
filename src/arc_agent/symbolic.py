"""A small object-centric symbolic layer for ARC grids.

The module deliberately sits beside :mod:`arc_agent.dsl` rather than replacing it.
It has two goals:

* provide a typed, JSON-serialisable description of object operations; and
* search a conservative library of short programs, keeping only programs that
  exactly reproduce every demonstration pair.

The executor is deterministic and does not call a model.  Consequently it is
safe to use as a verifier for neural proposals.  Operations operate on ordinary
``list[list[int]]`` grids, so the layer can be used before a learned object
extractor is available.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from itertools import product
from typing import Any

Grid = list[list[int]]
Demo = tuple[Grid, Grid]


class SymbolicError(ValueError):
    """Raised when an operation is invalid for a grid or has invalid arguments."""


class OpKind(StrEnum):
    """The stable wire names of the symbolic executor operations."""

    IDENTITY = "identity"
    RECOLOR = "recolor"
    CROP = "crop"
    SELECT_OBJECT = "select_object"
    CROP_OBJECT = "crop_object"
    TRANSLATE = "translate"
    COPY = "copy"
    COPY_OBJECT = "copy_object"
    FILL_HOLES = "fill_holes"
    REMOVE_OBJECT = "remove_object"
    ROTATE = "rotate"
    REFLECT = "reflect"
    TRANSFORM = "transform"
    REPEAT = "repeat"
    PROPAGATE = "propagate"
    COLOR_REFERENCE = "color_reference"
    COLOR_REFERENCE_CHAIN = "color_reference_chain"
    RELATION_CHAIN = "relation_chain"


@dataclass(frozen=True)
class GridObject:
    """A compact, serialisable object extracted from a grid.

    ``cells`` stores ``(row, column, color)`` triples in absolute coordinates.
    The normalized ``shape`` stores ``(row, column, color)`` relative to the
    object's top-left corner, which makes object matching independent of
    translation.
    """

    index: int
    cells: tuple[tuple[int, int, int], ...]
    bbox: tuple[int, int, int, int]
    colors: tuple[int, ...]
    shape: tuple[tuple[int, int, int], ...]

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def top(self) -> int:
        return self.bbox[0]

    @property
    def left(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[1] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[2] + 1

    @property
    def primary_color(self) -> int:
        return Counter(color for _, _, color in self.cells).most_common(1)[0][0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cells": [list(cell) for cell in self.cells],
            "bbox": list(self.bbox),
            "colors": list(self.colors),
            "shape": [list(cell) for cell in self.shape],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GridObject:
        return cls(
            index=int(value["index"]),
            cells=tuple(tuple(int(x) for x in cell) for cell in value["cells"]),
            bbox=tuple(int(x) for x in value["bbox"]),  # type: ignore[arg-type]
            colors=tuple(int(x) for x in value["colors"]),
            shape=tuple(tuple(int(x) for x in cell) for cell in value["shape"]),
        )


@dataclass(frozen=True)
class Relation:
    """A geometric relation between two extracted objects."""

    source: int
    target: int
    kind: str
    value: int | tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        value: Any = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"source": self.source, "target": self.target, "kind": self.kind, "value": value}


def _validate_grid(grid: Sequence[Sequence[int]], *, label: str = "grid") -> Grid:
    if not grid or not grid[0]:
        raise SymbolicError(f"{label} must not be empty")
    width = len(grid[0])
    if len(grid) > 30 or width > 30:
        raise SymbolicError(f"{label} dimensions must be at most 30x30")
    if any(len(row) != width for row in grid):
        raise SymbolicError(f"{label} must be rectangular")
    if any(
        not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9
        for row in grid
        for cell in row
    ):
        raise SymbolicError(f"{label} cells must be integers from 0 to 9")
    return [list(row) for row in grid]


def _copy(grid: Sequence[Sequence[int]]) -> Grid:
    return [list(row) for row in grid]


def _background(grid: Sequence[Sequence[int]], requested: int | None = None) -> int:
    if requested is not None:
        if not isinstance(requested, int) or not 0 <= requested <= 9:
            raise SymbolicError("background must be an integer from 0 to 9")
        return requested
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _bounds(cells: Iterable[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    points = list(cells)
    if not points:
        raise SymbolicError("object has no cells")
    rows = [point[0] for point in points]
    columns = [point[1] for point in points]
    return min(rows), max(rows), min(columns), max(columns)


def extract_objects(
    grid: Sequence[Sequence[int]],
    *,
    background: int | None = None,
    connectivity: int = 4,
    multicolor: bool = True,
) -> tuple[GridObject, ...]:
    """Extract connected non-background objects in stable scan order.

    With ``multicolor=True`` (the default), touching cells of different colors
    belong to one object.  Set it to ``False`` when colors represent separate
    objects.  The resulting object list is deterministic and JSON serialisable.
    """

    source = _validate_grid(grid)
    if connectivity not in {4, 8}:
        raise SymbolicError("connectivity must be 4 or 8")
    bg = _background(source, background)
    height, width = len(source), len(source[0])
    visited: set[tuple[int, int]] = set()
    objects: list[GridObject] = []
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for row in range(height):
        for column in range(width):
            if (row, column) in visited or source[row][column] == bg:
                continue
            seed_color = source[row][column]
            queue = deque([(row, column)])
            visited.add((row, column))
            cells: list[tuple[int, int, int]] = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column, source[current_row][current_column]))
                for delta_row, delta_column in directions:
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    if not (0 <= next_row < height and 0 <= next_column < width):
                        continue
                    if (next_row, next_column) in visited or source[next_row][next_column] == bg:
                        continue
                    if not multicolor and source[next_row][next_column] != seed_color:
                        continue
                    visited.add((next_row, next_column))
                    queue.append((next_row, next_column))
            bbox = _bounds(cells)
            top, _, left, _ = bbox
            shape = tuple(sorted((r - top, c - left, color) for r, c, color in cells))
            objects.append(
                GridObject(
                    index=len(objects),
                    cells=tuple(sorted(cells)),
                    bbox=bbox,
                    colors=tuple(sorted({color for _, _, color in cells})),
                    shape=shape,
                )
            )
    return tuple(objects)


def object_relations(
    grid: Sequence[Sequence[int]],
    *,
    background: int | None = None,
    connectivity: int = 4,
) -> tuple[Relation, ...]:
    """Return simple typed relations useful to a learned proposal head."""

    objects = extract_objects(grid, background=background, connectivity=connectivity)
    relations: list[Relation] = []
    for source in objects:
        for target in objects:
            if source.index == target.index:
                continue
            dr = target.top - source.top
            dc = target.left - source.left
            if dr == 0:
                relations.append(Relation(source.index, target.index, "same_row", dc))
            if dc == 0:
                relations.append(Relation(source.index, target.index, "same_column", dr))
            if dr == dc:
                relations.append(Relation(source.index, target.index, "diagonal", (dr, dc)))
            if source.primary_color in target.colors:
                relations.append(
                    Relation(source.index, target.index, "color_reference", source.primary_color)
                )
    return tuple(relations)


def _canonical_args(args: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    def normalize(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return tuple(sorted((str(k), normalize(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(normalize(item) for item in value)
        return value

    return tuple(sorted((str(key), normalize(value)) for key, value in args.items()))


def _args_dict(args: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    def denormalize(value: Any) -> Any:
        if isinstance(value, tuple):
            if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
                return {str(k): denormalize(v) for k, v in value}
            return [denormalize(item) for item in value]
        return value

    return {key: denormalize(value) for key, value in args}


@dataclass(frozen=True)
class SymbolicOp:
    """One typed operation and its JSON-compatible arguments."""

    kind: str | OpKind
    args: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        kind = self.kind.value if isinstance(self.kind, OpKind) else str(self.kind)
        try:
            kind = OpKind(kind).value
        except ValueError as exc:
            raise SymbolicError(f"unknown symbolic operation: {kind}") from exc
        object.__setattr__(self, "kind", kind)
        if isinstance(self.args, Mapping):
            object.__setattr__(self, "args", _canonical_args(self.args))
        else:
            object.__setattr__(self, "args", _canonical_args(dict(self.args)))

    @classmethod
    def make(cls, kind: str | OpKind, **args: Any) -> SymbolicOp:
        return cls(kind=kind, args=args)

    @property
    def arguments(self) -> dict[str, Any]:
        return _args_dict(self.args)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "args": _jsonable(self.arguments)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolicOp:
        return cls(kind=str(value["kind"]), args=dict(value.get("args", {})))

    def description(self) -> str:
        if not self.arguments:
            return str(self.kind)
        rendered = ",".join(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.kind}({rendered})"


@dataclass(frozen=True)
class SymbolicProgram:
    """A sequential composition of :class:`SymbolicOp` values."""

    operations: tuple[SymbolicOp, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations:
            raise SymbolicError("a symbolic program needs at least one operation")
        if any(not isinstance(operation, SymbolicOp) for operation in self.operations):
            raise SymbolicError("program operations must be SymbolicOp values")

    @classmethod
    def from_ops(cls, *operations: SymbolicOp) -> SymbolicProgram:
        return cls(tuple(operations))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolicProgram:
        return cls(tuple(SymbolicOp.from_dict(item) for item in value["operations"]))

    def to_dict(self) -> dict[str, Any]:
        return {"operations": [operation.to_dict() for operation in self.operations]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> SymbolicProgram:
        return cls.from_dict(json.loads(value))

    @property
    def depth(self) -> int:
        return len(self.operations)

    @property
    def description_length(self) -> int:
        return sum(1 + len(operation.arguments) for operation in self.operations)

    def description(self) -> str:
        return " |> ".join(operation.description() for operation in self.operations)

    def apply(self, grid: Sequence[Sequence[int]]) -> Grid:
        return execute(self, grid)


@dataclass(frozen=True)
class Verification:
    """Structured exact-verification result with convenient boolean semantics."""

    exact_pairs: int
    total_pairs: int
    predictions: tuple[Grid, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def perfect(self) -> bool:
        return self.exact_pairs == self.total_pairs and not self.errors

    def __bool__(self) -> bool:
        return self.perfect


@dataclass(frozen=True)
class SynthesisCandidate:
    """A verified program returned by :func:`synthesize_programs`."""

    program: SymbolicProgram
    predictions: tuple[Grid, ...]
    description_length: int
    behavior_signature: str
    verified: bool = True
    exact_pairs: int = 0
    total_pairs: int = 0
    train_cell_accuracy: float = 1.0
    train_balanced_accuracy: float = 1.0

    @property
    def score(self) -> float:
        return 100.0 - float(self.description_length)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "predictions": [_jsonable(grid) for grid in self.predictions],
            "description_length": self.description_length,
            "behavior_signature": self.behavior_signature,
            "verified": self.verified,
            "exact_pairs": self.exact_pairs,
            "total_pairs": self.total_pairs,
            "train_cell_accuracy": self.train_cell_accuracy,
            "train_balanced_accuracy": self.train_balanced_accuracy,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _rotate(grid: Grid, turns: int) -> Grid:
    result = _copy(grid)
    for _ in range(turns % 4):
        result = [list(row) for row in zip(*result[::-1], strict=True)]
    return result


def _reflect(grid: Grid, axis: str) -> Grid:
    if axis in {"horizontal", "left_right", "x"}:
        return [list(reversed(row)) for row in grid]
    if axis in {"vertical", "top_bottom", "y"}:
        return _copy(reversed(grid))
    if axis in {"rotational", "both", "180"}:
        return _rotate(grid, 2)
    if axis == "transpose":
        return [list(row) for row in zip(*grid, strict=True)]
    raise SymbolicError(f"unknown reflection axis: {axis}")


def _crop_bounds(grid: Grid, bg: int) -> Grid:
    cells = [
        (row, column, value)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value != bg
    ]
    if not cells:
        return _copy(grid)
    top, bottom, left, right = _bounds(cells)
    return [row[left : right + 1] for row in grid[top : bottom + 1]]


def _choose_object(grid: Grid, args: Mapping[str, Any]) -> GridObject:
    bg = _background(grid, args.get("background"))
    objects = list(
        extract_objects(
            grid,
            background=bg,
            connectivity=int(args.get("connectivity", 4)),
            multicolor=bool(args.get("multicolor", True)),
        )
    )
    if "color" in args:
        color = int(args["color"])
        objects = [obj for obj in objects if color in obj.colors]
    if not objects:
        raise SymbolicError("object selector found no objects")
    if "index" in args:
        index = int(args["index"])
        if not 0 <= index < len(objects):
            raise SymbolicError("object index is out of range")
        return objects[index]
    criterion = str(args.get("criterion", args.get("select", "largest")))
    key_functions = {
        "largest": lambda obj: (obj.area, -obj.top, -obj.left),
        "smallest": lambda obj: (-obj.area, -obj.top, -obj.left),
        "topmost": lambda obj: (-obj.top, -obj.area, -obj.left),
        "bottommost": lambda obj: (obj.top, obj.area, obj.left),
        "leftmost": lambda obj: (-obj.left, -obj.area, -obj.top),
        "rightmost": lambda obj: (obj.left, obj.area, obj.top),
        "widest": lambda obj: (obj.width, obj.area, -obj.top),
        "tallest": lambda obj: (obj.height, obj.area, -obj.left),
        "most_colors": lambda obj: (len(obj.colors), obj.area, -obj.top),
    }
    if criterion not in key_functions:
        raise SymbolicError(f"unknown object criterion: {criterion}")
    return max(objects, key=key_functions[criterion])


def _object_crop(grid: Grid, obj: GridObject, bg: int) -> Grid:
    top, bottom, left, right = obj.bbox
    cells = {(row, column): color for row, column, color in obj.cells}
    return [
        [cells.get((row, column), bg) for column in range(left, right + 1)]
        for row in range(top, bottom + 1)
    ]


def _translate(grid: Grid, rows: int, columns: int, bg: int) -> Grid:
    height, width = len(grid), len(grid[0])
    result = [[bg for _ in range(width)] for _ in range(height)]
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            if value == bg:
                continue
            target_row, target_column = row + rows, column + columns
            if 0 <= target_row < height and 0 <= target_column < width:
                result[target_row][target_column] = value
    return result


def _place_object(
    grid: Grid,
    obj: GridObject,
    row_offset: int,
    column_offset: int,
    *,
    erase_source: bool = False,
    background: int,
) -> Grid:
    result = _copy(grid)
    if erase_source:
        for row, column, _ in obj.cells:
            result[row][column] = background
    for row, column, color in obj.cells:
        target_row, target_column = row + row_offset, column + column_offset
        if 0 <= target_row < len(result) and 0 <= target_column < len(result[0]):
            result[target_row][target_column] = color
    return result


def _holes(grid: Grid, bg: int) -> list[set[tuple[int, int]]]:
    height, width = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    found: list[set[tuple[int, int]]] = []
    for row in range(height):
        for column in range(width):
            if grid[row][column] != bg or (row, column) in visited:
                continue
            queue = deque([(row, column)])
            visited.add((row, column))
            component: set[tuple[int, int]] = set()
            touches_edge = False
            while queue:
                current = queue.popleft()
                component.add(current)
                if current[0] in {0, height - 1} or current[1] in {0, width - 1}:
                    touches_edge = True
                for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    neighbor = (current[0] + dr, current[1] + dc)
                    if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                        continue
                    if neighbor in visited or grid[neighbor[0]][neighbor[1]] != bg:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            if not touches_edge:
                found.append(component)
    return found


def _direction_delta(direction: str) -> tuple[int, int]:
    values = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    if direction not in values:
        raise SymbolicError(f"unknown direction: {direction}")
    return values[direction]


def _execute_op(op: SymbolicOp, grid: Grid) -> Grid:
    args = op.arguments
    kind = str(op.kind)
    source = _validate_grid(grid)
    if kind == OpKind.IDENTITY.value:
        return source
    if kind == OpKind.RECOLOR.value:
        raw_mapping = args.get("mapping", {})
        if not isinstance(raw_mapping, Mapping):
            raise SymbolicError("recolor mapping must be a mapping")
        mapping = {int(key): int(value) for key, value in raw_mapping.items()}
        if any(not 0 <= key <= 9 or not 0 <= value <= 9 for key, value in (*mapping.items(),)):
            raise SymbolicError("recolor values must be from 0 to 9")
        return [[mapping.get(value, value) for value in row] for row in source]
    if kind == OpKind.CROP.value:
        return _crop_bounds(source, _background(source, args.get("background")))
    if kind in {OpKind.SELECT_OBJECT.value, OpKind.CROP_OBJECT.value}:
        bg = _background(source, args.get("background"))
        return _object_crop(source, _choose_object(source, args), bg)
    if kind == OpKind.TRANSLATE.value:
        return _translate(
            source,
            int(args.get("rows", 0)),
            int(args.get("columns", 0)),
            _background(source, args.get("background")),
        )
    if kind in {OpKind.COPY.value, OpKind.COPY_OBJECT.value}:
        bg = _background(source, args.get("background"))
        obj = _choose_object(source, args)
        if "offset" in args:
            offset = args["offset"]
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise SymbolicError("offset must contain row and column")
            dr, dc = int(offset[0]), int(offset[1])
        else:
            dr, dc = int(args.get("rows", 0)), int(args.get("columns", 0))
        if dr == 0 and dc == 0:
            raise SymbolicError("copy offset must be non-zero")
        return _place_object(source, obj, dr, dc, background=bg)
    if kind == OpKind.FILL_HOLES.value:
        bg = _background(source, args.get("background"))
        if "fill_color" in args:
            fill = int(args["fill_color"])
        else:
            non_bg = [value for row in source for value in row if value != bg]
            if not non_bg:
                raise SymbolicError("cannot infer a hole fill color")
            fill = Counter(non_bg).most_common(1)[0][0]
        if not 0 <= fill <= 9:
            raise SymbolicError("fill_color must be from 0 to 9")
        result = _copy(source)
        for component in _holes(source, bg):
            boundary_colors = []
            for row, column in component:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r, c = row + dr, column + dc
                    if 0 <= r < len(source) and 0 <= c < len(source[0]) and source[r][c] != bg:
                        boundary_colors.append(source[r][c])
            selected_fill = fill
            if "fill_color" not in args and boundary_colors:
                selected_fill = Counter(boundary_colors).most_common(1)[0][0]
            for row, column in component:
                result[row][column] = selected_fill
        return result
    if kind == OpKind.REMOVE_OBJECT.value:
        bg = _background(source, args.get("background"))
        obj = _choose_object(source, args)
        result = _copy(source)
        for row, column, _ in obj.cells:
            result[row][column] = bg
        return result
    if kind == OpKind.ROTATE.value:
        turns = int(args.get("turns", 1))
        if turns < 0 or turns > 3:
            turns %= 4
        return _rotate(source, turns)
    if kind == OpKind.REFLECT.value:
        return _reflect(source, str(args.get("axis", "horizontal")))
    if kind == OpKind.TRANSFORM.value:
        transform = str(args.get("transform", args.get("name", "identity")))
        if transform in {"identity", "none"}:
            return source
        if transform in {"rotate_90", "rot90", "cw90"}:
            return _rotate(source, 1)
        if transform in {"rotate_180", "rot180"}:
            return _rotate(source, 2)
        if transform in {"rotate_270", "rot270", "ccw90"}:
            return _rotate(source, 3)
        if transform in {"flip_horizontal", "flip_vertical", "transpose"}:
            axis = {
                "flip_horizontal": "horizontal",
                "flip_vertical": "vertical",
                "transpose": "transpose",
            }[transform]
            return _reflect(source, axis)
        raise SymbolicError(f"unknown transform: {transform}")
    if kind in {OpKind.REPEAT.value, OpKind.PROPAGATE.value}:
        bg = _background(source, args.get("background"))
        obj = _choose_object(source, args)
        direction = str(args.get("direction", "right"))
        dr, dc = _direction_delta(direction)
        if "step" in args:
            step = int(args["step"])
        elif dr:
            step = obj.height
        else:
            step = obj.width
        if step <= 0:
            raise SymbolicError("repeat step must be positive")
        if kind == OpKind.REPEAT.value:
            count = int(args.get("count", args.get("repetitions", 1)))
            if count < 1 or count > 30:
                raise SymbolicError("repeat count must be between 1 and 30")
        else:
            count = 29
        result = _copy(source)
        offset = 0
        completed = 0
        while completed < count:
            offset += step
            placed = False
            for row, column, color in obj.cells:
                target_row, target_column = row + dr * offset, column + dc * offset
                if 0 <= target_row < len(result) and 0 <= target_column < len(result[0]):
                    if result[target_row][target_column] == bg or (target_row, target_column) in {
                        (r, c) for r, c, _ in obj.cells
                    }:
                        result[target_row][target_column] = color
                        placed = True
                    elif kind == OpKind.PROPAGATE.value:
                        return result
            if not placed:
                break
            completed += 1
            if kind == OpKind.PROPAGATE.value and offset > 30:
                break
        if kind == OpKind.REPEAT.value and completed < count:
            raise SymbolicError("repeat would leave the grid")
        return result
    if kind in {
        OpKind.COLOR_REFERENCE.value,
        OpKind.COLOR_REFERENCE_CHAIN.value,
        OpKind.RELATION_CHAIN.value,
    }:
        raw_mapping = args.get("mapping", {})
        if not isinstance(raw_mapping, Mapping):
            raise SymbolicError("color_reference mapping must be a mapping")
        mapping = {int(key): int(value) for key, value in raw_mapping.items()}
        steps = int(args.get("steps", 1))
        if steps < 1 or steps > 9:
            raise SymbolicError("reference steps must be between 1 and 9")
        result = source
        for _ in range(steps):
            result = [[mapping.get(value, value) for value in row] for row in result]
        return result
    raise SymbolicError(f"unsupported symbolic operation: {kind}")


def execute(program: SymbolicProgram | SymbolicOp, grid: Sequence[Sequence[int]]) -> Grid:
    """Execute a program with defensive copies and strict output validation."""

    source = _validate_grid(grid, label="input")
    operations = (program,) if isinstance(program, SymbolicOp) else program.operations
    result = source
    for operation in operations:
        result = _validate_grid(_execute_op(operation, result), label="program output")
    return result


apply_program = execute


def _coerce_demos(demos: Iterable[Any]) -> tuple[Demo, ...]:
    normalized: list[Demo] = []
    for index, demo in enumerate(demos):
        if hasattr(demo, "input") and hasattr(demo, "output"):
            input_grid = demo.input
            output_grid = demo.output
        else:
            try:
                input_grid, output_grid = demo
            except (TypeError, ValueError) as exc:
                raise SymbolicError(f"demo {index} must be (input, output)") from exc
        if output_grid is None:
            raise SymbolicError(f"demo {index} has no output")
        normalized.append((_validate_grid(input_grid), _validate_grid(output_grid)))
    if not normalized:
        raise SymbolicError("at least one demonstration is required")
    return tuple(normalized)


def verify_program(program: SymbolicProgram, demos: Iterable[Any]) -> Verification:
    """Run a program against every demonstration and report exact differences."""

    pairs = _coerce_demos(demos)
    predictions: list[Grid] = []
    errors: list[str] = []
    exact = 0
    for index, (input_grid, expected) in enumerate(pairs):
        try:
            prediction = execute(program, input_grid)
            predictions.append(prediction)
            if prediction == expected:
                exact += 1
            else:
                errors.append(
                    f"pair {index}: expected shape {(len(expected), len(expected[0]))}, "
                    f"got {(len(prediction), len(prediction[0]))}"
                )
        except Exception as exc:  # verifier should turn invalid proposals into feedback
            errors.append(f"pair {index}: {type(exc).__name__}: {exc}")
    return Verification(exact, len(pairs), tuple(predictions), tuple(errors))


def _grid_signature(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def _program_signature(program: SymbolicProgram, demos: tuple[Demo, ...]) -> str | None:
    outputs: list[str] = []
    for input_grid, _ in demos:
        try:
            outputs.append(_grid_signature(execute(program, input_grid)))
        except SymbolicError:
            return None
    # Exact demonstrations necessarily share the same output signature.  Keep
    # independent operation families (for example rotate+recolor versus
    # translate+recolor) so the outer solver can retain useful pass@2
    # diversity, while equal-family variants remain behaviorally deduplicated.
    family = ",".join(operation.kind for operation in program.operations)
    return f"{family}::{'|'.join(outputs)}"


def _object_criteria(object_ids: Sequence[int], scenes: Sequence[Any]) -> tuple[str, ...]:
    """Return selectors that identify the same inferred object in every scene."""

    criteria = ("largest", "smallest", "topmost", "leftmost", "widest", "tallest")
    selected: list[str] = []
    for criterion in criteria:
        matches = True
        for object_id, scene in zip(object_ids, scenes, strict=True):
            objects = list(scene.objects)
            keys = {
                "largest": lambda obj: (obj.area, -obj.bbox.top, -obj.bbox.left),
                "smallest": lambda obj: (-obj.area, -obj.bbox.top, -obj.bbox.left),
                "topmost": lambda obj: (-obj.bbox.top, -obj.area, -obj.bbox.left),
                "leftmost": lambda obj: (-obj.bbox.left, -obj.area, -obj.bbox.top),
                "widest": lambda obj: (obj.bbox.width, obj.area, -obj.bbox.top),
                "tallest": lambda obj: (obj.bbox.height, obj.area, -obj.bbox.left),
            }
            if not objects or max(objects, key=keys[criterion]).object_id != object_id:
                matches = False
                break
        if matches:
            selected.append(criterion)
    return tuple(selected)


def _transition_guided_operations(demos: tuple[Demo, ...]) -> tuple[SymbolicOp, ...]:
    """Turn object-level demonstration transitions into a compact proposal set.

    These proposals do not bypass synthesis or verification.  They only put
    parameters evidenced by object correspondences ahead of the much larger
    generic operation vocabulary.
    """

    from arc_agent.object_ir import infer_scene_transition

    try:
        transitions = tuple(
            infer_scene_transition(
                input_grid,
                output_grid,
                maximum_exact_objects=8,
            )
            for input_grid, output_grid in demos
        )
    except ValueError:
        return ()
    proposals: list[SymbolicOp] = []

    # A whole-scene translation is strongly supported when all matched objects
    # move by one integral vector and no object appears or disappears.
    demo_offsets: list[tuple[int, int]] = []
    for transition in transitions:
        offsets = {
            (round(match.translation[0]), round(match.translation[1]))
            for match in transition.matches
            if match.has("moved")
            and match.translation[0].is_integer()
            and match.translation[1].is_integer()
        }
        if (
            not transition.matches
            or transition.added_objects
            or transition.removed_objects
            or len(offsets) != 1
            or len(transition.moved) != len(transition.matches)
        ):
            demo_offsets = []
            break
        demo_offsets.append(next(iter(offsets)))
    if demo_offsets and len(set(demo_offsets)) == 1:
        rows, columns = demo_offsets[0]
        proposals.append(SymbolicOp.make(OpKind.TRANSLATE, rows=rows, columns=columns))

    # Monochrome object correspondences provide a sparse, task-level color map.
    mapping: dict[int, int] = {}
    mapping_consistent = True
    for transition in transitions:
        inputs = {obj.object_id: obj for obj in transition.input_scene.objects}
        outputs = {obj.object_id: obj for obj in transition.output_scene.objects}
        for match in transition.matches:
            source = inputs[match.input_object_id]
            target = outputs[match.output_object_id]
            if source.color is None or target.color is None or source.color == target.color:
                continue
            if source.color in mapping and mapping[source.color] != target.color:
                mapping_consistent = False
                break
            mapping[source.color] = target.color
    if mapping and mapping_consistent:
        proposals.append(SymbolicOp.make(OpKind.RECOLOR, mapping=mapping))

    # Global D4 transformations remain useful when object transitions agree.
    transform_names = {
        match.shape_transform
        for transition in transitions
        for match in transition.matches
        if match.has("transformed")
    }
    any_moved = any(transition.moved for transition in transitions)
    if len(transform_names) == 1 and not any_moved:
        transform = next(iter(transform_names))
        turns = {"rotate_90": 1, "rotate_180": 2, "rotate_270": 3}.get(transform)
        if turns is not None:
            proposals.append(SymbolicOp.make(OpKind.ROTATE, turns=turns))
        elif transform.startswith("reflect"):
            proposals.extend(
                SymbolicOp.make(OpKind.REFLECT, axis=axis)
                for axis in ("horizontal", "vertical", "transpose")
            )

    # A consistent removed object suggests a selector-specific erase.
    if all(len(transition.removed_objects) == 1 for transition in transitions) and all(
        not transition.added_objects for transition in transitions
    ):
        removed_ids = [transition.removed_objects[0].object_id for transition in transitions]
        scenes = [transition.input_scene for transition in transitions]
        proposals.extend(
            SymbolicOp.make(OpKind.REMOVE_OBJECT, criterion=criterion)
            for criterion in _object_criteria(removed_ids, scenes)
        )

    # If each output adds a translated copy of an existing object, infer both
    # selector and displacement directly from the scene correspondence.
    copy_sources: list[int] = []
    copy_offsets: list[tuple[int, int]] = []
    if all(len(transition.added_objects) == 1 for transition in transitions):
        for transition in transitions:
            added = transition.added_objects[0]
            sources = [
                obj
                for obj in transition.input_scene.objects
                if obj.canonical_color_signature() == added.canonical_color_signature()
            ]
            if not sources:
                copy_sources = []
                break
            source = min(
                sources,
                key=lambda obj: (
                    abs(obj.bbox.top - added.bbox.top)
                    + abs(obj.bbox.left - added.bbox.left),
                    obj.object_id,
                ),
            )
            copy_sources.append(source.object_id)
            copy_offsets.append(
                (added.bbox.top - source.bbox.top, added.bbox.left - source.bbox.left)
            )
    if copy_sources and len(set(copy_offsets)) == 1:
        rows, columns = copy_offsets[0]
        scenes = [transition.input_scene for transition in transitions]
        proposals.extend(
            SymbolicOp.make(
                OpKind.COPY,
                criterion=criterion,
                rows=rows,
                columns=columns,
            )
            for criterion in _object_criteria(copy_sources, scenes)
        )

    if any(transition.shape_changed for transition in transitions):
        proposals.append(SymbolicOp.make(OpKind.CROP))
        proposals.extend(
            SymbolicOp.make(OpKind.SELECT_OBJECT, criterion=criterion)
            for criterion in ("largest", "smallest", "topmost", "leftmost")
        )
    if any(
        obj.hole_count
        for transition in transitions
        for obj in transition.input_scene.objects
    ):
        colors = sorted(
            {
                value
                for _, output_grid in demos
                for row in output_grid
                for value in row
            }
        )
        proposals.extend(SymbolicOp.make(OpKind.FILL_HOLES, fill_color=color) for color in colors)

    unique: dict[str, SymbolicOp] = {}
    for proposal in proposals:
        unique.setdefault(proposal.description(), proposal)
    return tuple(unique.values())


def _candidate_operations(
    demos: tuple[Demo, ...],
    *,
    operation_priors: Mapping[str, float] | None = None,
) -> tuple[SymbolicOp, ...]:
    """Generate a conservative, data-aware operation vocabulary."""

    inputs = [input_grid for input_grid, _ in demos]
    outputs = [output_grid for _, output_grid in demos]
    colors = sorted({value for grid in inputs + outputs for row in grid for value in row})
    backgrounds = sorted({_background(grid) for grid in inputs + outputs} | {0})
    guided = list(_transition_guided_operations(demos))
    operations: list[SymbolicOp] = [SymbolicOp.make(OpKind.IDENTITY)]
    operations.extend(SymbolicOp.make(OpKind.ROTATE, turns=turns) for turns in (1, 2, 3))
    operations.extend(
        SymbolicOp.make(OpKind.REFLECT, axis=axis)
        for axis in ("horizontal", "vertical", "transpose")
    )
    # Keep recolor close to the front so it is available in the bounded
    # second-operation beam after a geometric first step.
    operations.extend(
        SymbolicOp.make(OpKind.RECOLOR, mapping={source: target})
        for source, target in product(colors, colors)
        if source != target
    )
    operations.extend(
        SymbolicOp.make(OpKind.CROP, background=background) for background in backgrounds
    )
    operations.extend(
        SymbolicOp.make(OpKind.SELECT_OBJECT, background=background, criterion=criterion)
        for background, criterion in product(
            backgrounds, ("largest", "smallest", "topmost", "leftmost", "widest", "tallest")
        )
    )
    operations.extend(
        SymbolicOp.make(OpKind.REMOVE_OBJECT, background=background, criterion=criterion)
        for background, criterion in product(
            backgrounds, ("largest", "smallest", "topmost", "leftmost")
        )
    )
    operations.extend(
        SymbolicOp.make(OpKind.FILL_HOLES, background=background, fill_color=color)
        for background, color in product(backgrounds, colors)
    )
    # Translation and copy are bounded to the useful ARC displacement range.
    operations.extend(
        SymbolicOp.make(OpKind.TRANSLATE, background=background, rows=rows, columns=columns)
        for background, rows, columns in product(backgrounds, range(-5, 6), range(-5, 6))
        if rows or columns
    )
    operations.extend(
        SymbolicOp.make(
            OpKind.COPY,
            background=background,
            criterion=criterion,
            rows=rows,
            columns=columns,
        )
        for background, criterion, rows, columns in product(
            backgrounds, ("largest", "smallest", "topmost"), range(-5, 6), range(-5, 6)
        )
        if rows or columns
    )
    operations.extend(
        SymbolicOp.make(
            OpKind.PROPAGATE,
            background=background,
            criterion="largest",
            direction=direction,
        )
        for background, direction in product(backgrounds, ("up", "down", "left", "right"))
    )
    # Infer a total color mapping when the demonstrations preserve shape.
    mapping: dict[int, int] = {}
    consistent = True
    for input_grid, output_grid in demos:
        if (len(input_grid), len(input_grid[0])) != (len(output_grid), len(output_grid[0])):
            consistent = False
            break
        for input_row, output_row in zip(input_grid, output_grid, strict=True):
            for source, target in zip(input_row, output_row, strict=True):
                if source in mapping and mapping[source] != target:
                    consistent = False
                mapping[source] = target
    if consistent and mapping:
        operations.append(SymbolicOp.make(OpKind.RECOLOR, mapping=mapping))
        # Mapping composition captures simple color-reference chains.
        chained = dict(mapping)
        for source, target in tuple(mapping.items()):
            chained[source] = mapping.get(target, target)
        if chained != mapping:
            operations.append(SymbolicOp.make(OpKind.COLOR_REFERENCE_CHAIN, mapping=chained))
    # Learned program-head scores only influence ordering.  Object-transition
    # proposals retain the first positions, and exact demo verification remains
    # the admission criterion.
    priors = {str(key): float(value) for key, value in (operation_priors or {}).items()}
    indexed = list(enumerate(operations))
    indexed.sort(key=lambda item: (-priors.get(str(item[1].kind), 0.0), item[0]))
    unique: dict[str, SymbolicOp] = {}
    for operation in [*guided, *(item[1] for item in indexed)]:
        unique.setdefault(operation.description(), operation)
    return tuple(unique.values())


def _partial_score(outputs: Sequence[Grid], expected: Sequence[Grid]) -> float:
    """Score a not-yet-verified state for bounded beam ordering."""

    scores: list[float] = []
    for actual, wanted in zip(outputs, expected, strict=True):
        if (len(actual), len(actual[0])) != (len(wanted), len(wanted[0])):
            scores.append(0.0)
            continue
        total = len(wanted) * len(wanted[0])
        scores.append(
            sum(
                actual[row][column] == wanted[row][column]
                for row in range(len(wanted))
                for column in range(len(wanted[0]))
            )
            / total
        )
    return sum(scores) / len(scores) if scores else 0.0


def synthesize_programs(
    demos: Iterable[Any],
    *,
    max_depth: int = 2,
    max_nodes: int = 8_000,
    max_candidates: int = 16,
    max_seconds: float = 5.0,
    operation_priors: Mapping[str, float] | None = None,
) -> list[SynthesisCandidate]:
    """Enumerate short programs and retain exact, behaviorally distinct ones.

    The first two levels use a bounded meet-in-the-middle beam: all legal
    one-step states are scored, then promising first operations are composed
    with the front of the operation vocabulary.  This is still enumerative,
    but avoids spending the entire node budget on parameter variants before a
    useful two-step composition can be reached.  Every admitted program is
    verified against *all* demonstrations.  Time, node, depth, and candidate
    limits are hard bounds.
    """

    if max_depth < 1 or max_nodes < 1 or max_candidates < 1 or max_seconds <= 0:
        raise SymbolicError("search limits must be positive")
    pairs = _coerce_demos(demos)
    vocabulary = _candidate_operations(pairs, operation_priors=operation_priors)
    start = time.monotonic()
    behavior_to_candidate: dict[str, SynthesisCandidate] = {}
    nodes = 0

    expected = tuple(output_grid for _, output_grid in pairs)
    one_step: list[tuple[SymbolicOp, tuple[Grid, ...], float]] = []

    def add_if_exact(program: SymbolicProgram, outputs: tuple[Grid, ...]) -> None:
        if outputs != expected:
            return
        signature = _program_signature(program, pairs)
        if signature is None:
            return
        candidate = SynthesisCandidate(
            program=program,
            predictions=outputs,
            description_length=program.description_length,
            behavior_signature=signature,
            exact_pairs=len(pairs),
            total_pairs=len(pairs),
        )
        previous = behavior_to_candidate.get(signature)
        if previous is None or candidate.description_length < previous.description_length:
            behavior_to_candidate[signature] = candidate

    # Evaluate each primitive once.  Invalid object selectors are simply
    # rejected from the beam; they remain safe to propose through the API.
    # Preserve composition budget.  Previously the primitive scan could consume
    # every node when the parameter vocabulary was larger than ``max_nodes``, so
    # even an object-guided two-operation solution was never attempted.
    primitive_limit = len(vocabulary)
    if max_depth >= 2:
        primitive_limit = min(len(vocabulary), max(1, max_nodes // 3))
    for operation in vocabulary[:primitive_limit]:
        if nodes >= max_nodes or time.monotonic() - start > max_seconds:
            break
        nodes += 1
        try:
            outputs = tuple(execute(operation, input_grid) for input_grid, _ in pairs)
        except (SymbolicError, ValueError):
            continue
        one_step.append((operation, outputs, _partial_score(outputs, expected)))
        add_if_exact(SymbolicProgram.from_ops(operation), outputs)

    # A pair beam is useful even when a one-step exact solution was found:
    # distinct operation families produce useful pass@2 alternatives.
    if max_depth >= 2 and one_step and len(behavior_to_candidate) < max_candidates:
        vocabulary_order = {
            operation.description(): index for index, operation in enumerate(vocabulary)
        }
        ranked_first = sorted(
            one_step,
            key=lambda item: (
                -item[2],
                vocabulary_order[item[0].description()],
                item[0].description(),
            ),
        )
        first_width = min(
            len(ranked_first),
            max(1, min(128, max(32, int(max_nodes**0.5) * 2))),
        )
        second_width = min(
            len(vocabulary),
            max(1, max(32, max_nodes // max(1, first_width))),
        )
        for first_operation, first_outputs, _ in ranked_first[:first_width]:
            if nodes >= max_nodes or time.monotonic() - start > max_seconds:
                break
            for second_operation in vocabulary[:second_width]:
                if nodes >= max_nodes or time.monotonic() - start > max_seconds:
                    break
                nodes += 1
                try:
                    outputs = tuple(
                        execute(second_operation, intermediate)
                        for intermediate in first_outputs
                    )
                except (SymbolicError, ValueError):
                    continue
                add_if_exact(
                    SymbolicProgram.from_ops(first_operation, second_operation), outputs
                )
                if len(behavior_to_candidate) >= max_candidates:
                    break

    # For depth > 2, spend any remaining budget on a conventional breadth-first
    # tail.  The pair beam is intentionally the high-value path for the default
    # depth-two solver; this tail preserves the documented depth cap without
    # allowing unbounded expansion.
    if max_depth > 2 and nodes < max_nodes and time.monotonic() - start <= max_seconds:
        queue: deque[SymbolicProgram] = deque(
            SymbolicProgram.from_ops(operation) for operation, _, _ in one_step
        )
        seen_programs = {program.to_json() for program in queue}
        while queue and nodes < max_nodes and time.monotonic() - start <= max_seconds:
            program = queue.popleft()
            if program.depth >= max_depth:
                continue
            for operation in vocabulary:
                if nodes >= max_nodes or time.monotonic() - start > max_seconds:
                    break
                child = SymbolicProgram.from_ops(*program.operations, operation)
                key = child.to_json()
                if key in seen_programs:
                    continue
                seen_programs.add(key)
                nodes += 1
                try:
                    outputs = tuple(
                        execute(child, input_grid) for input_grid, _ in pairs
                    )
                except (SymbolicError, ValueError):
                    continue
                add_if_exact(child, outputs)
                queue.append(child)
                if len(behavior_to_candidate) >= max_candidates:
                    break
    return sorted(
        behavior_to_candidate.values(),
        key=lambda candidate: (candidate.description_length, candidate.program.description()),
    )[:max_candidates]


def search_programs(*args: Any, **kwargs: Any) -> list[SynthesisCandidate]:
    """Compatibility alias for :func:`synthesize_programs`."""

    return synthesize_programs(*args, **kwargs)


def synthesize(
    task_or_demos: Any,
    *,
    max_depth: int = 2,
    max_nodes: int = 8_000,
    max_candidates: int = 16,
    timeout_seconds: float = 5.0,
    operation_priors: Mapping[str, float] | None = None,
) -> list[SynthesisCandidate]:
    """Integration-friendly synthesizer entry point.

    ``task_or_demos`` may be an ``ArcTask``-like object with a ``train``
    attribute or any iterable of ``(input, output)`` pairs.  Every returned
    record is fully verified on the training pairs and includes predictions,
    exact-pair counts, and accuracy fields expected by the outer harness.
    """

    train = getattr(task_or_demos, "train", task_or_demos)
    candidates = synthesize_programs(
        train,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_seconds=timeout_seconds,
        operation_priors=operation_priors,
    )
    test_pairs = getattr(task_or_demos, "test", None)
    if test_pairs is None:
        return candidates
    predictions: list[SynthesisCandidate] = []
    for candidate in candidates:
        test_outputs = tuple(execute(candidate.program, pair.input) for pair in test_pairs)
        predictions.append(replace(candidate, predictions=test_outputs))
    return predictions


__all__ = [
    "Grid",
    "GridObject",
    "OpKind",
    "Relation",
    "SymbolicError",
    "SymbolicOp",
    "SymbolicProgram",
    "SynthesisCandidate",
    "Verification",
    "apply_program",
    "execute",
    "extract_objects",
    "object_relations",
    "search_programs",
    "synthesize",
    "synthesize_programs",
    "verify_program",
]
