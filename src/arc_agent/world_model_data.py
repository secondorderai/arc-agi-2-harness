"""Data structures and deterministic preprocessing for the hybrid world model.

The rest of the harness works with Python lists and Pydantic models.  This module
keeps the experimental object-centric model independent from that inference path:
it converts ARC tasks into padded NumPy arrays and records masks for every padded
dimension.  No training-set answers are read outside the task passed to the
builder.

The representation is intentionally small and inspectable.  Objects are 4-connected
non-background components and each object is represented by colour, area, bounding
box, centroid, fill ratio and border contact.  It is a useful baseline for the
neural model, not a claim that connected components are sufficient for ARC-AGI-2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from arc_agent.models import ArcTask, Grid

try:
    # Reuse the repository's richer scene extractor when available.  The compact
    # adapter below keeps the neural batch contract independent of its dataclasses.
    from arc_agent.object_ir import extract_objects as _extract_ir_objects
    from arc_agent.object_ir import infer_scene_transition as _infer_scene_transition
except ImportError:  # pragma: no cover - relevant only to a deliberately partial wheel
    _extract_ir_objects = None
    _infer_scene_transition = None

OBJECT_FEATURE_DIM = 13
MAX_COLORS = 10

Geometry = str


@dataclass(frozen=True)
class GridAugmentation:
    """A geometry transform and a colour permutation that keeps colour zero fixed."""

    geometry: Geometry = "identity"
    color_map: tuple[int, ...] = tuple(range(MAX_COLORS))

    def __post_init__(self) -> None:
        if self.geometry not in {
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "transpose",
            "flip_horizontal",
            "flip_vertical",
        }:
            raise ValueError(f"unsupported geometry: {self.geometry}")
        if len(self.color_map) != MAX_COLORS or set(self.color_map) != set(range(MAX_COLORS)):
            raise ValueError("color_map must be a permutation of 0..9")
        if self.color_map[0] != 0:
            raise ValueError("background colour zero must remain fixed")


@dataclass(frozen=True)
class GridObject:
    """A connected object and geometry features in grid coordinates."""

    color: int
    pixels: tuple[tuple[int, int], ...]
    row_min: int
    row_max: int
    column_min: int
    column_max: int
    height: int
    width: int

    @property
    def area(self) -> int:
        return len(self.pixels)

    def feature_vector(self, grid_shape: tuple[int, int]) -> np.ndarray:
        """Return a fixed, normalized feature vector for this object."""
        rows, columns = grid_shape
        pixel_set = set(self.pixels)
        perimeter = sum(
            int((row - 1, column) not in pixel_set)
            + int((row + 1, column) not in pixel_set)
            + int((row, column - 1) not in pixel_set)
            + int((row, column + 1) not in pixel_set)
            for row, column in self.pixels
        )
        border_pixels = sum(
            row in (0, rows - 1) or column in (0, columns - 1)
            for row, column in self.pixels
        )
        centroid_row = sum(row for row, _ in self.pixels) / max(self.area, 1)
        centroid_column = sum(column for _, column in self.pixels) / max(self.area, 1)
        box_area = self.height * self.width
        # The final feature is deliberately redundant: it lets a small model
        # distinguish a sparse line from a filled rectangle without a CNN.
        return np.asarray(
            [
                self.color / 9.0,
                self.area / max(rows * columns, 1),
                self.height / max(rows, 1),
                self.width / max(columns, 1),
                self.row_min / max(rows - 1, 1),
                self.column_min / max(columns - 1, 1),
                self.row_max / max(rows - 1, 1),
                self.column_max / max(columns - 1, 1),
                centroid_row / max(rows - 1, 1),
                centroid_column / max(columns - 1, 1),
                self.area / max(box_area, 1),
                perimeter / max(4 * self.area, 1),
                border_pixels / max(self.area, 1),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class WorldModelExample:
    """One supervised transition with the remaining demonstrations as context."""

    task_id: str
    query_input: Grid
    target_output: Grid
    demonstrations: tuple[tuple[Grid, Grid], ...]
    program_label: int
    query_objects: tuple[GridObject, ...] = ()
    program_op: str = "unknown"


@dataclass
class WorldModelBatch:
    """Padded numerical batch consumed by :class:`HybridWorldModel`."""

    query_inputs: np.ndarray
    targets: np.ndarray
    query_mask: np.ndarray
    target_mask: np.ndarray
    demo_inputs: np.ndarray
    demo_outputs: np.ndarray
    demo_masks: np.ndarray
    demo_pair_mask: np.ndarray
    query_objects: np.ndarray
    query_object_mask: np.ndarray
    demo_input_objects: np.ndarray
    demo_output_objects: np.ndarray
    demo_object_mask: np.ndarray
    program_targets: np.ndarray
    query_shapes: tuple[tuple[int, int], ...]
    target_shapes: tuple[tuple[int, int], ...]
    # Optional separate masks preserve correctness when a demonstration changes
    # shape. ``demo_masks`` remains the backwards-compatible union mask.
    demo_input_masks: np.ndarray | None = None
    demo_output_masks: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return int(self.query_inputs.shape[0])

    @property
    def height(self) -> int:
        return int(self.query_inputs.shape[1])

    @property
    def width(self) -> int:
        return int(self.query_inputs.shape[2])


def _as_grid_array(grid: Sequence[Sequence[int]]) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("grid must be a non-empty rectangular matrix")
    if np.any(array < 0) or np.any(array > 9):
        raise ValueError("grid colours must be integers from 0 to 9")
    if any(len(row) != array.shape[1] for row in grid):
        raise ValueError("grid must be rectangular")
    return array


def transform_grid(grid: Grid, augmentation: GridAugmentation) -> Grid:
    """Apply an ARC-safe geometry and colour permutation."""
    array = _as_grid_array(grid)
    if augmentation.geometry == "rotate_90":
        array = np.rot90(array, 1)
    elif augmentation.geometry == "rotate_180":
        array = np.rot90(array, 2)
    elif augmentation.geometry == "rotate_270":
        array = np.rot90(array, 3)
    elif augmentation.geometry == "transpose":
        array = array.T
    elif augmentation.geometry == "flip_horizontal":
        array = np.fliplr(array)
    elif augmentation.geometry == "flip_vertical":
        array = np.flipud(array)
    mapping = np.asarray(augmentation.color_map, dtype=np.int64)
    return mapping[array].astype(int).tolist()


def make_augmentations(seed: int = 0, count: int = 1) -> tuple[GridAugmentation, ...]:
    """Create reproducible geometry/colour augmentations for training only."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    geometries = (
        "identity",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "transpose",
        "flip_horizontal",
        "flip_vertical",
    )
    values: list[GridAugmentation] = []
    for index in range(count):
        geometry = geometries[index % len(geometries)] if index < len(geometries) else str(
            rng.choice(geometries)
        )
        permutation = list(range(MAX_COLORS))
        if index:
            tail = np.arange(1, MAX_COLORS)
            rng.shuffle(tail)
            permutation[1:] = tail.tolist()
        values.append(GridAugmentation(geometry=geometry, color_map=tuple(permutation)))
    return tuple(values)


def augment_pair(
    pair: tuple[Grid, Grid], augmentation: GridAugmentation
) -> tuple[Grid, Grid]:
    return transform_grid(pair[0], augmentation), transform_grid(pair[1], augmentation)


def extract_objects(grid: Grid, *, include_background: bool = False) -> tuple[GridObject, ...]:
    """Extract deterministic 4-connected monochromatic objects.

    Background components are omitted by default.  If a grid has no foreground
    components, callers still receive an empty tuple and should rely on masks.
    """
    array = _as_grid_array(grid)
    if _extract_ir_objects is not None and not include_background:
        # Consume the shared lossless IR when available. It provides inferred
        # background, holes, multicolour objects and stable identity; the compact
        # adapter retains the geometry features required by the neural encoder.
        converted: list[GridObject] = []
        for rich in _extract_ir_objects(grid):
            colors = tuple(int(color) for color in rich.cell_colors)
            dominant = max(sorted(set(colors)), key=lambda color: (colors.count(color), -color))
            converted.append(
                GridObject(
                    color=dominant,
                    pixels=tuple(rich.cells),
                    row_min=rich.bbox.top,
                    row_max=rich.bbox.bottom,
                    column_min=rich.bbox.left,
                    column_max=rich.bbox.right,
                    height=rich.bbox.height,
                    width=rich.bbox.width,
                )
            )
        return tuple(converted)
    rows, columns = array.shape
    visited = np.zeros_like(array, dtype=bool)
    objects: list[GridObject] = []
    for row in range(rows):
        for column in range(columns):
            color = int(array[row, column])
            if visited[row, column] or (color == 0 and not include_background):
                continue
            stack = [(row, column)]
            visited[row, column] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                pixels.append((current_row, current_column))
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if not (0 <= next_row < rows and 0 <= next_column < columns):
                        continue
                    if visited[next_row, next_column] or int(array[next_row, next_column]) != color:
                        continue
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
            objects.append(
                GridObject(
                    color=color,
                    pixels=tuple(sorted(pixels)),
                    row_min=min(item[0] for item in pixels),
                    row_max=max(item[0] for item in pixels),
                    column_min=min(item[1] for item in pixels),
                    column_max=max(item[1] for item in pixels),
                    height=max(item[0] for item in pixels) - min(item[0] for item in pixels) + 1,
                    width=max(item[1] for item in pixels) - min(item[1] for item in pixels) + 1,
                )
            )
    # Stable ordering is important: object slots must not depend on DFS details.
    return tuple(sorted(objects, key=lambda item: (item.row_min, item.column_min, item.color)))


def object_feature_matrix(
    grid: Grid, *, max_objects: int | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[GridObject, ...]]:
    objects = extract_objects(grid)
    selected = objects if max_objects is None else objects[:max_objects]
    features = np.zeros((len(selected), OBJECT_FEATURE_DIM), dtype=np.float32)
    for index, item in enumerate(selected):
        features[index] = item.feature_vector((len(grid), len(grid[0])))
    return features, np.ones(len(selected), dtype=bool), objects


def program_label_from_verified_program(program: object | None) -> int:
    """Map a verified ``Program`` (or dict-like program) to an operation class.

    Unknown/custom DSL operators intentionally map to ``unknown`` rather than
    pretending that the auxiliary target has been semantically understood.
    """
    from arc_agent.world_model import PROGRAM_OPS

    if program is None:
        return PROGRAM_OPS.index("unknown")
    operator = program.get("op") if isinstance(program, dict) else getattr(program, "op", None)
    if not isinstance(operator, str):
        return PROGRAM_OPS.index("unknown")
    aliases = {
        "identity": "identity",
        "recolor": "recolor",
        "map_color": "recolor",
        "rotate": "rotate",
        "reflect": "reflect",
        "flip": "reflect",
        "crop": "crop",
        "tile": "tile",
        "translate": "translate",
        "move": "translate",
        "object_transform": "object_transform",
        "relation_follow": "relation_follow",
    }
    return PROGRAM_OPS.index(aliases.get(operator, "unknown"))


def infer_program_label(
    input_grid: Grid,
    output_grid: Grid,
    *,
    verified_program: object | None = None,
) -> int:
    """Infer a weak symbolic target used as auxiliary supervision.

    If a verified DSL program is supplied, it is authoritative; otherwise the
    label is conservatively inferred from the input/output pair.
    """
    from arc_agent.world_model import PROGRAM_OPS

    if verified_program is not None:
        return program_label_from_verified_program(verified_program)
    source = _as_grid_array(input_grid)
    target = _as_grid_array(output_grid)
    if np.array_equal(source, target):
        return PROGRAM_OPS.index("identity")
    if source.shape == target.shape:
        # A globally consistent colour mapping is a useful, cheap symbolic cue.
        mapping: dict[int, int] = {}
        consistent = True
        for old, new in zip(source.flat, target.flat, strict=True):
            old_int, new_int = int(old), int(new)
            if old_int in mapping and mapping[old_int] != new_int:
                consistent = False
                break
            mapping[old_int] = new_int
        if consistent:
            return PROGRAM_OPS.index("recolor")
        if any(
            np.array_equal(target, transform_grid(input_grid, GridAugmentation(geometry=name)))
            for name in ("rotate_90", "rotate_180", "rotate_270")
        ):
            return PROGRAM_OPS.index("rotate")
        if any(
            np.array_equal(target, transform_grid(input_grid, GridAugmentation(geometry=name)))
            for name in ("flip_horizontal", "flip_vertical")
        ):
            return PROGRAM_OPS.index("reflect")
    if target.shape[0] <= source.shape[0] and target.shape[1] <= source.shape[1]:
        for row in range(source.shape[0] - target.shape[0] + 1):
            for column in range(source.shape[1] - target.shape[1] + 1):
                if np.array_equal(
                    source[row : row + target.shape[0], column : column + target.shape[1]],
                    target,
                ):
                    return PROGRAM_OPS.index("crop")
    if (
        target.shape[0] % source.shape[0] == 0
        and target.shape[1] % source.shape[1] == 0
        and target.shape != source.shape
    ):
        return PROGRAM_OPS.index("tile")
    if _infer_scene_transition is not None:
        transition = _infer_scene_transition(
            input_grid,
            output_grid,
            maximum_exact_objects=8,
        )
        transforms = {
            match.shape_transform
            for match in transition.transformed
            if match.shape_transform != "identity"
        }
        if transforms and all(name.startswith("rotate_") for name in transforms):
            return PROGRAM_OPS.index("rotate")
        if transforms and all(name.startswith("reflect") for name in transforms):
            return PROGRAM_OPS.index("reflect")
        if transition.transformed:
            return PROGRAM_OPS.index("object_transform")
        if (
            transition.moved
            and not transition.added_objects
            and not transition.removed_objects
        ):
            offsets = {match.translation for match in transition.moved}
            if len(offsets) == 1 and len(transition.moved) == len(transition.matches):
                return PROGRAM_OPS.index("translate")
            if transition.relation_changes:
                return PROGRAM_OPS.index("relation_follow")
            return PROGRAM_OPS.index("translate")
        if transition.recolored and not transition.moved:
            return PROGRAM_OPS.index("recolor")
        if transition.relation_changes:
            return PROGRAM_OPS.index("relation_follow")
        if transition.added_objects and not transition.removed_objects:
            input_signatures = {
                obj.canonical_color_signature() for obj in transition.input_scene.objects
            }
            if any(
                obj.canonical_color_signature() in input_signatures
                for obj in transition.added_objects
            ):
                return PROGRAM_OPS.index("tile")
    # Unresolved pairs are not object-transform supervision.  Treating every
    # residual task as that class created a misleading 74% majority baseline.
    return PROGRAM_OPS.index("unknown")


def build_world_model_examples(
    tasks: Iterable[ArcTask],
    *,
    augmentation_count: int = 1,
    seed: int = 0,
    leave_one_out: bool = True,
    max_examples: int | None = None,
) -> list[WorldModelExample]:
    """Build deterministic, augmentation-aware examples from labelled tasks."""
    augmentations = make_augmentations(seed=seed, count=augmentation_count)
    examples: list[WorldModelExample] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        for augmentation in augmentations:
            pairs = [
                augment_pair((pair.input, pair.output or []), augmentation)
                for pair in task.train
            ]
            for query_index, (query_input, target_output) in enumerate(pairs):
                if leave_one_out and len(pairs) > 1:
                    demos = tuple(pair for index, pair in enumerate(pairs) if index != query_index)
                else:
                    demos = tuple(pairs)
                if not demos:
                    demos = (pairs[query_index],)
                program_label = infer_program_label(query_input, target_output)
                from arc_agent.world_model import PROGRAM_OPS

                examples.append(
                    WorldModelExample(
                        task_id=task.task_id,
                        query_input=query_input,
                        target_output=target_output,
                        demonstrations=demos,
                        program_label=program_label,
                        query_objects=extract_objects(query_input),
                        program_op=PROGRAM_OPS[program_label],
                    )
                )
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
    return examples


def collate_world_model_examples(
    examples: Sequence[WorldModelExample], *, max_objects: int = 16
) -> WorldModelBatch:
    """Pad variable-size grids, demonstrations and object sets into NumPy arrays."""
    if not examples:
        raise ValueError("at least one example is required")
    max_height = max(
        max(
            len(item.query_input),
            len(item.target_output),
            *(len(pair[0]) for pair in item.demonstrations),
            *(len(pair[1]) for pair in item.demonstrations),
        )
        for item in examples
    )
    max_width = max(
        max(
            len(item.query_input[0]),
            len(item.target_output[0]),
            *(len(pair[0][0]) for pair in item.demonstrations),
            *(len(pair[1][0]) for pair in item.demonstrations),
        )
        for item in examples
    )
    max_demos = max(len(item.demonstrations) for item in examples)
    query_inputs = np.zeros((len(examples), max_height, max_width), dtype=np.int64)
    targets = np.zeros_like(query_inputs)
    query_mask = np.zeros_like(query_inputs, dtype=np.float32)
    target_mask = np.zeros_like(query_inputs, dtype=np.float32)
    demo_inputs = np.zeros((len(examples), max_demos, max_height, max_width), dtype=np.int64)
    demo_outputs = np.zeros_like(demo_inputs)
    demo_masks = np.zeros((len(examples), max_demos, max_height, max_width), dtype=np.float32)
    demo_input_masks = np.zeros_like(demo_masks)
    demo_output_masks = np.zeros_like(demo_masks)
    demo_pair_mask = np.zeros((len(examples), max_demos), dtype=np.float32)
    query_objects = np.zeros((len(examples), max_objects, OBJECT_FEATURE_DIM), dtype=np.float32)
    query_object_mask = np.zeros((len(examples), max_objects), dtype=np.float32)
    demo_input_objects = np.zeros(
        (len(examples), max_demos, max_objects, OBJECT_FEATURE_DIM), dtype=np.float32
    )
    demo_output_objects = np.zeros_like(demo_input_objects)
    demo_object_mask = np.zeros((len(examples), max_demos, max_objects), dtype=np.float32)
    program_targets = np.asarray([item.program_label for item in examples], dtype=np.int64)
    query_shapes: list[tuple[int, int]] = []
    target_shapes: list[tuple[int, int]] = []

    def copy_grid(
        destination: np.ndarray, mask: np.ndarray, index: tuple[int, ...], grid: Grid
    ) -> None:
        array = _as_grid_array(grid)
        height, width = array.shape
        destination[index + (slice(0, height), slice(0, width))] = array
        mask[index + (slice(0, height), slice(0, width))] = 1.0

    for batch_index, item in enumerate(examples):
        query_shapes.append((len(item.query_input), len(item.query_input[0])))
        target_shapes.append((len(item.target_output), len(item.target_output[0])))
        copy_grid(query_inputs, query_mask, (batch_index,), item.query_input)
        copy_grid(targets, target_mask, (batch_index,), item.target_output)
        query_features, query_valid, _ = object_feature_matrix(
            item.query_input, max_objects=max_objects
        )
        query_objects[batch_index, : len(query_features)] = query_features
        query_object_mask[batch_index, : len(query_valid)] = query_valid
        for demo_index, (demo_input, demo_output) in enumerate(item.demonstrations):
            demo_pair_mask[batch_index, demo_index] = 1.0
            copy_grid(demo_inputs, demo_input_masks, (batch_index, demo_index), demo_input)
            copy_grid(demo_outputs, demo_output_masks, (batch_index, demo_index), demo_output)
            demo_masks[batch_index, demo_index] = np.maximum(
                demo_input_masks[batch_index, demo_index],
                demo_output_masks[batch_index, demo_index],
            )
            input_features, input_valid, _ = object_feature_matrix(
                demo_input, max_objects=max_objects
            )
            output_features, output_valid, _ = object_feature_matrix(
                demo_output, max_objects=max_objects
            )
            demo_input_objects[batch_index, demo_index, : len(input_features)] = input_features
            demo_output_objects[batch_index, demo_index, : len(output_features)] = output_features
            demo_object_mask[
                batch_index, demo_index, : max(len(input_valid), len(output_valid))
            ] = 1.0
    return WorldModelBatch(
        query_inputs=query_inputs,
        targets=targets,
        query_mask=query_mask,
        target_mask=target_mask,
        demo_inputs=demo_inputs,
        demo_outputs=demo_outputs,
        demo_masks=demo_masks,
        demo_pair_mask=demo_pair_mask,
        query_objects=query_objects,
        query_object_mask=query_object_mask,
        demo_input_objects=demo_input_objects,
        demo_output_objects=demo_output_objects,
        demo_object_mask=demo_object_mask,
        program_targets=program_targets,
        query_shapes=tuple(query_shapes),
        target_shapes=tuple(target_shapes),
        demo_input_masks=demo_input_masks,
        demo_output_masks=demo_output_masks,
    )


def iter_world_model_batches(
    examples: Sequence[WorldModelExample],
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    seed: int = 0,
    max_objects: int = 16,
) -> Iterable[WorldModelBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield collate_world_model_examples(
            [examples[int(index)] for index in indices[start : start + batch_size]],
            max_objects=max_objects,
        )


__all__ = [
    "GridAugmentation",
    "GridObject",
    "OBJECT_FEATURE_DIM",
    "WorldModelBatch",
    "WorldModelExample",
    "augment_pair",
    "build_world_model_examples",
    "collate_world_model_examples",
    "extract_objects",
    "infer_program_label",
    "iter_world_model_batches",
    "make_augmentations",
    "object_feature_matrix",
    "transform_grid",
]
