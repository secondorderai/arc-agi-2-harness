"""A small, deterministic object-centric intermediate representation for ARC grids.

The module deliberately keeps the representation independent of the solver and of
any ML framework.  A grid is converted into immutable objects, geometric
signatures, and typed-enough relations that a symbolic executor can consume.  It
is intended to be useful both as a feature extractor and as a lossless scene
container: ``extract_scene(grid).render()`` reproduces the input grid.

Coordinates are ``(row, column)`` pairs and bounding boxes are inclusive:
``BoundingBox(top, left, bottom, right)``.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import permutations

Coord = tuple[int, int]
GridInput = Sequence[Sequence[int]]
GridTuple = tuple[tuple[int, ...], ...]


def _validate_grid(grid: GridInput) -> GridTuple:
    """Validate and freeze an ARC grid."""

    if not grid:
        raise ValueError("grid must not be empty")
    if any(not row for row in grid):
        raise ValueError("grid rows must not be empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular")
    frozen: GridTuple = tuple(tuple(row) for row in grid)
    if any(
        not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9
        for row in frozen
        for cell in row
    ):
        raise ValueError("grid cells must be integers from 0 to 9")
    return frozen


def _validate_connectivity(connectivity: int) -> int:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    return connectivity


def _neighbors(row: int, column: int, connectivity: int) -> tuple[Coord, ...]:
    if connectivity == 4:
        deltas = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        deltas = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    return tuple((row + dr, column + dc) for dr, dc in deltas)


def infer_background(grid: GridInput) -> int:
    """Infer the background color using a deterministic ARC-friendly policy.

    The most frequent color wins.  Ties prefer color ``0`` (when present), then
    the color whose first occurrence is earliest in row-major order.  This makes
    inference stable even for deliberately ambiguous tiny grids.
    """

    frozen = _validate_grid(grid)
    counts = Counter(cell for row in frozen for cell in row)
    maximum = max(counts.values())
    tied = {color for color, count in counts.items() if count == maximum}
    if 0 in tied:
        return 0
    for row in frozen:
        for cell in row:
            if cell in tied:
                return cell
    raise AssertionError("a non-empty grid must contain a background candidate")


@dataclass(frozen=True, slots=True, order=True)
class BoundingBox:
    """An inclusive, immutable axis-aligned bounding box."""

    top: int
    left: int
    bottom: int
    right: int

    def __post_init__(self) -> None:
        if self.top < 0 or self.left < 0:
            raise ValueError("bounding-box coordinates must be non-negative")
        if self.bottom < self.top or self.right < self.left:
            raise ValueError("bounding-box bottom/right must not precede top/left")

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> tuple[float, float]:
        return ((self.top + self.bottom) / 2, (self.left + self.right) / 2)

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.top, self.left, self.bottom, self.right

    def contains(self, coordinate: Coord) -> bool:
        row, column = coordinate
        return self.top <= row <= self.bottom and self.left <= column <= self.right

    def strict_contains(self, other: BoundingBox) -> bool:
        return (
            self.top < other.top
            and self.left < other.left
            and self.bottom > other.bottom
            and self.right > other.right
        )

    def intersects(self, other: BoundingBox) -> bool:
        return not (
            self.bottom < other.top
            or other.bottom < self.top
            or self.right < other.left
            or other.right < self.left
        )

    def coordinates(self) -> tuple[Coord, ...]:
        return tuple(
            (row, column)
            for row in range(self.top, self.bottom + 1)
            for column in range(self.left, self.right + 1)
        )

    @classmethod
    def from_cells(cls, cells: Iterable[Coord]) -> BoundingBox:
        points = tuple(cells)
        if not points:
            raise ValueError("a bounding box needs at least one coordinate")
        return cls(
            min(row for row, _ in points),
            min(column for _, column in points),
            max(row for row, _ in points),
            max(column for _, column in points),
        )


def _mask(cells: Iterable[Coord], bbox: BoundingBox) -> tuple[tuple[bool, ...], ...]:
    occupied = set(cells)
    return tuple(
        tuple((row, column) in occupied for column in range(bbox.left, bbox.right + 1))
        for row in range(bbox.top, bbox.bottom + 1)
    )


def _normalize_coordinates(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    points = tuple(cells)
    if not points:
        return ()
    top = min(row for row, _ in points)
    left = min(column for _, column in points)
    return tuple(sorted((row - top, column - left) for row, column in points))


def _transformed_coordinates(cells: Iterable[Coord]) -> tuple[tuple[Coord, ...], ...]:
    """Return the eight D4 transforms of a shape before normalization."""

    points = tuple(cells)
    transforms: list[tuple[Coord, ...]] = []
    for reflected in (False, True):
        for turns in range(4):
            transformed: list[Coord] = []
            for row, column in points:
                if reflected:
                    column = -column
                for _ in range(turns):
                    row, column = column, -row
                transformed.append((row, column))
            transforms.append(tuple(transformed))
    return tuple(transforms)


def _canonical_shape(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    return min(_normalize_coordinates(candidate) for candidate in _transformed_coordinates(cells))


def _canonical_colored_shape(
    cells: Iterable[Coord], colors: Iterable[int]
) -> tuple[tuple[int, int, int], ...]:
    pairs = tuple(zip(cells, colors, strict=True))
    if not pairs:
        return ()
    transforms: list[tuple[tuple[int, int, int], ...]] = []
    for reflected in (False, True):
        for turns in range(4):
            transformed: list[tuple[int, int, int]] = []
            for (row, column), color in pairs:
                if reflected:
                    column = -column
                for _ in range(turns):
                    row, column = column, -row
                transformed.append((row, column, color))
            top = min(row for row, _, _ in transformed)
            left = min(column for _, column, _ in transformed)
            transforms.append(
                tuple(
                    sorted(
                        (row - top, column - left, color)
                        for row, column, color in transformed
                    )
                )
            )
    return min(transforms)


@dataclass(frozen=True, slots=True)
class Hole:
    """A background-region-shaped cavity inside an object's bounding box."""

    cells: tuple[Coord, ...]
    bbox: BoundingBox
    mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        cells = tuple(sorted(set(self.cells)))
        if not cells:
            raise ValueError("a hole must contain at least one cell")
        if any(not self.bbox.contains(cell) for cell in cells):
            raise ValueError("hole cells must lie within its bounding box")
        object.__setattr__(self, "cells", cells)
        expected = _mask(cells, self.bbox)
        if self.mask != expected:
            object.__setattr__(self, "mask", expected)

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def shape_signature(self) -> tuple[Coord, ...]:
        return _normalize_coordinates(self.cells)


def _find_holes(cells: tuple[Coord, ...]) -> tuple[Hole, ...]:
    """Find 4-connected empty regions enclosed by the object's bounding box."""

    bbox = BoundingBox.from_cells(cells)
    occupied = set(cells)
    empty = set(bbox.coordinates()) - occupied
    outside: set[Coord] = set()
    queue: deque[Coord] = deque()
    for coordinate in sorted(empty):
        row, column = coordinate
        if row in (bbox.top, bbox.bottom) or column in (bbox.left, bbox.right):
            outside.add(coordinate)
            queue.append(coordinate)
    while queue:
        coordinate = queue.popleft()
        for neighbor in _neighbors(*coordinate, 4):
            if neighbor in empty and neighbor not in outside:
                outside.add(neighbor)
                queue.append(neighbor)
    remaining = empty - outside
    holes: list[Hole] = []
    while remaining:
        seed = min(remaining)
        component: set[Coord] = {seed}
        queue = deque([seed])
        remaining.remove(seed)
        while queue:
            coordinate = queue.popleft()
            for neighbor in _neighbors(*coordinate, 4):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        hole_cells = tuple(sorted(component))
        hole_bbox = BoundingBox.from_cells(hole_cells)
        holes.append(Hole(hole_cells, hole_bbox, _mask(hole_cells, hole_bbox)))
    return tuple(holes)


@dataclass(frozen=True, slots=True)
class GridObject:
    """An immutable connected component and its derived geometric features."""

    object_id: int
    cells: tuple[Coord, ...]
    cell_colors: tuple[int, ...]
    bbox: BoundingBox
    connectivity: int
    background: int
    holes: tuple[Hole, ...] = ()

    def __post_init__(self) -> None:
        cells = tuple(sorted(set(self.cells)))
        colors = tuple(self.cell_colors)
        if not cells:
            raise ValueError("an object must contain at least one cell")
        if len(cells) != len(self.cells) or len(colors) != len(cells):
            raise ValueError("object cells must be unique and match cell_colors")
        if any(row < 0 or column < 0 for row, column in cells):
            raise ValueError("object coordinates must be non-negative")
        if any(not isinstance(color, int) or isinstance(color, bool) for color in colors):
            raise ValueError("object colors must be integers")
        if any(not self.bbox.contains(cell) for cell in cells):
            raise ValueError("object cells must lie within its bounding box")
        _validate_connectivity(self.connectivity)
        if not 0 <= self.background <= 9:
            raise ValueError("background must be an ARC color")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "cell_colors", colors)
        object.__setattr__(self, "holes", tuple(self.holes))

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def color(self) -> int | None:
        """The color for a monochrome object, otherwise ``None``."""

        return self.cell_colors[0] if len(set(self.cell_colors)) == 1 else None

    @property
    def colors(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.cell_colors)))

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center

    @property
    def normalized_cells(self) -> tuple[Coord, ...]:
        return tuple((row - self.bbox.top, column - self.bbox.left) for row, column in self.cells)

    @property
    def shape_signature(self) -> tuple[Coord, ...]:
        return self.normalized_cells

    @property
    def mask(self) -> tuple[tuple[bool, ...], ...]:
        return _mask(self.cells, self.bbox)

    @property
    def color_mask(self) -> tuple[tuple[int | None, ...], ...]:
        colors = dict(zip(self.cells, self.cell_colors, strict=True))
        return tuple(
            tuple(
                colors.get((row, column))
                for column in range(self.bbox.left, self.bbox.right + 1)
            )
            for row in range(self.bbox.top, self.bbox.bottom + 1)
        )

    @property
    def hole_count(self) -> int:
        return len(self.holes)

    @property
    def hole_area(self) -> int:
        return sum(hole.area for hole in self.holes)

    def canonical_shape_signature(self, *, include_reflections: bool = True) -> tuple[Coord, ...]:
        if include_reflections:
            return _canonical_shape(self.cells)
        rotations = _transformed_coordinates(self.cells)[:4]
        return min(_normalize_coordinates(candidate) for candidate in rotations)

    def canonical_color_signature(
        self, *, include_reflections: bool = True
    ) -> tuple[tuple[int, int, int], ...]:
        if include_reflections:
            return _canonical_colored_shape(self.cells, self.cell_colors)
        transforms: list[tuple[tuple[int, int, int], ...]] = []
        for candidate in _transformed_coordinates(self.cells)[:4]:
            values = tuple(zip(candidate, self.cell_colors, strict=True))
            top = min(row for (row, _), _ in values)
            left = min(column for (_, column), _ in values)
            transforms.append(
                tuple(sorted((row - top, column - left, color) for (row, column), color in values))
            )
        return min(transforms)

    def contains_coordinate(self, coordinate: Coord) -> bool:
        return coordinate in set(self.cells)


def connected_components(
    grid: GridInput,
    *,
    background: int | None = None,
    connectivity: int = 4,
    multicolor: bool = False,
) -> tuple[tuple[Coord, ...], ...]:
    """Extract non-background components in deterministic row-major order.

    With ``multicolor=False`` (the default), cells of different colors never
    share a component.  With ``multicolor=True``, all non-background cells are
    connected according to the requested neighborhood.
    """

    frozen = _validate_grid(grid)
    _validate_connectivity(connectivity)
    inferred = infer_background(frozen) if background is None else background
    if not isinstance(inferred, int) or isinstance(inferred, bool) or not 0 <= inferred <= 9:
        raise ValueError("background must be an ARC color")
    height, width = len(frozen), len(frozen[0])
    visited: set[Coord] = set()
    components: list[tuple[Coord, ...]] = []
    for row in range(height):
        for column in range(width):
            seed = (row, column)
            if seed in visited or frozen[row][column] == inferred:
                continue
            seed_color = frozen[row][column]
            component: list[Coord] = []
            queue: deque[Coord] = deque([seed])
            visited.add(seed)
            while queue:
                coordinate = queue.popleft()
                component.append(coordinate)
                for neighbor in _neighbors(*coordinate, connectivity):
                    nrow, ncolumn = neighbor
                    if not (0 <= nrow < height and 0 <= ncolumn < width):
                        continue
                    if neighbor in visited or frozen[nrow][ncolumn] == inferred:
                        continue
                    if not multicolor and frozen[nrow][ncolumn] != seed_color:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            components.append(tuple(sorted(component)))
    return tuple(components)


# A descriptive alias is convenient to callers that use "extract" terminology.
extract_components = connected_components


def extract_objects(
    grid: GridInput,
    *,
    background: int | None = None,
    connectivity: int = 4,
    multicolor: bool = False,
) -> tuple[GridObject, ...]:
    """Extract immutable objects with masks, boxes, and enclosed holes."""

    frozen = _validate_grid(grid)
    inferred = infer_background(frozen) if background is None else background
    components = connected_components(
        frozen,
        background=inferred,
        connectivity=connectivity,
        multicolor=multicolor,
    )
    objects: list[GridObject] = []
    for object_id, cells in enumerate(components):
        colors = tuple(frozen[row][column] for row, column in cells)
        bbox = BoundingBox.from_cells(cells)
        objects.append(
            GridObject(
                object_id=object_id,
                cells=cells,
                cell_colors=colors,
                bbox=bbox,
                connectivity=connectivity,
                background=inferred,
                holes=_find_holes(cells),
            )
        )
    return tuple(objects)


def _bbox_contains_object(container: GridObject, target: GridObject) -> bool:
    if not container.bbox.strict_contains(target.bbox):
        return False
    target_cells = set(target.cells)
    return any(target_cells.issubset(set(hole.cells)) for hole in container.holes)


def relative_position(source: GridObject, target: GridObject) -> str:
    """Return the target's coarse position relative to the source."""

    source_row = source.bbox.top + source.bbox.bottom
    source_column = source.bbox.left + source.bbox.right
    target_row = target.bbox.top + target.bbox.bottom
    target_column = target.bbox.left + target.bbox.right
    row_sign = (target_row > source_row) - (target_row < source_row)
    column_sign = (target_column > source_column) - (target_column < source_column)
    names = {
        (-1, -1): "northwest",
        (-1, 0): "north",
        (-1, 1): "northeast",
        (0, -1): "west",
        (0, 0): "overlap",
        (0, 1): "east",
        (1, -1): "southwest",
        (1, 0): "south",
        (1, 1): "southeast",
    }
    return names[(row_sign, column_sign)]


def _minimum_cell_distance(source: GridObject, target: GridObject) -> int:
    return min(
        abs(row - other_row) + abs(column - other_column)
        for row, column in source.cells
        for other_row, other_column in target.cells
    )


@dataclass(frozen=True, slots=True)
class ObjectRelation:
    """All deterministic geometric facts observed for an ordered object pair."""

    source_id: int
    target_id: int
    relations: tuple[str, ...]
    direction: str
    offset: tuple[int, int]
    center_offset: tuple[float, float]
    manhattan_distance: int
    bbox_overlap: tuple[int, int]

    @property
    def relation(self) -> str:
        """Return the first relation for callers that expect a singular label."""

        return self.relations[0] if self.relations else ""

    def has(self, relation: str) -> bool:
        return relation in self.relations


def _make_relation(source: GridObject, target: GridObject) -> ObjectRelation:
    target_cells = set(target.cells)
    orthogonal_touch = any(
        (row + dr, column + dc) in target_cells
        for row, column in source.cells
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0))
    )
    touching = any(
        abs(row - other_row) <= 1 and abs(column - other_column) <= 1
        for row, column in source.cells
        for other_row, other_column in target.cells
    )
    relations: list[str] = []
    if _bbox_contains_object(source, target):
        relations.append("contains")
    if _bbox_contains_object(target, source):
        relations.append("inside")
    if orthogonal_touch:
        relations.append("adjacent")
    if touching:
        relations.append("touching")
        if not orthogonal_touch:
            relations.append("diagonal_touching")
    if source.bbox.top <= target.bbox.bottom and target.bbox.top <= source.bbox.bottom:
        relations.append("horizontal_alignment")
    if source.bbox.left <= target.bbox.right and target.bbox.left <= source.bbox.right:
        relations.append("vertical_alignment")
    if source.bbox.center[0] == target.bbox.center[0]:
        relations.append("same_row")
    if source.bbox.center[1] == target.bbox.center[1]:
        relations.append("same_column")
    direction = relative_position(source, target)
    relations.append(direction)
    top_left_offset = (target.bbox.top - source.bbox.top, target.bbox.left - source.bbox.left)
    center_offset = (target.center[0] - source.center[0], target.center[1] - source.center[1])
    row_overlap = max(
        0,
        min(source.bbox.bottom, target.bbox.bottom)
        - max(source.bbox.top, target.bbox.top)
        + 1,
    )
    column_overlap = max(
        0,
        min(source.bbox.right, target.bbox.right)
        - max(source.bbox.left, target.bbox.left)
        + 1,
    )
    return ObjectRelation(
        source_id=source.object_id,
        target_id=target.object_id,
        relations=tuple(relations),
        direction=direction,
        offset=top_left_offset,
        center_offset=center_offset,
        manhattan_distance=_minimum_cell_distance(source, target),
        bbox_overlap=(row_overlap, column_overlap),
    )


def object_relations(objects: Sequence[GridObject]) -> tuple[ObjectRelation, ...]:
    """Compute ordered pairwise containment, contact, alignment, and direction."""

    return tuple(_make_relation(source, target) for source, target in permutations(objects, 2))


@dataclass(frozen=True, slots=True)
class ColorReference:
    """A color-bearing cell or cavity that points to an object's color."""

    source_id: int
    target_id: int
    color: int
    coordinates: tuple[Coord, ...]
    kind: str


def color_reference_features(
    grid: GridInput,
    objects: Sequence[GridObject],
    *,
    background: int | None = None,
) -> tuple[ColorReference, ...]:
    """Find conservative color-as-symbol references.

    The primary signal is a non-background color inside a source object's
    enclosed cavity that matches a target object's monochrome color.  For a
    multicolor source, less frequent marker colors are also considered.  This
    avoids treating every pair of same-colored objects as a reference while
    covering the common ARC ring/glyph-pointer construction.
    """

    frozen = _validate_grid(grid)
    inferred = infer_background(frozen) if background is None else background
    targets_by_color: dict[int, list[GridObject]] = {}
    for target in objects:
        for color in target.colors:
            targets_by_color.setdefault(color, []).append(target)
    references: dict[tuple[int, int, int, str], set[Coord]] = {}
    for source in objects:
        interior: set[Coord] = set()
        for hole in source.holes:
            interior.update(
                coordinate
                for coordinate in hole.cells
                if frozen[coordinate[0]][coordinate[1]] != inferred
            )
        for coordinate in sorted(interior):
            color = frozen[coordinate[0]][coordinate[1]]
            for target in targets_by_color.get(color, ()):
                if target.object_id != source.object_id:
                    key = (source.object_id, target.object_id, color, "interior_color")
                    references.setdefault(key, set()).add(coordinate)

        counts = Counter(source.cell_colors)
        if len(counts) > 1:
            dominant = max(counts.values())
            for color, count in counts.items():
                if count >= dominant:
                    continue
                for target in targets_by_color.get(color, ()):
                    if target.object_id != source.object_id:
                        coordinates = {
                            coordinate
                            for coordinate, cell_color in zip(
                                source.cells, source.cell_colors, strict=True
                            )
                            if cell_color == color
                        }
                        key = (source.object_id, target.object_id, color, "marker_color")
                        references.setdefault(key, set()).update(coordinates)
    return tuple(
        ColorReference(source_id, target_id, color, tuple(sorted(coordinates)), kind)
        for (source_id, target_id, color, kind), coordinates in sorted(references.items())
    )


@dataclass(frozen=True, slots=True)
class GridScene:
    """An immutable, lossless scene plus extracted objects and relations."""

    grid: GridTuple
    background: int
    objects: tuple[GridObject, ...]
    connectivity: int = 4
    multicolor: bool = False
    relations: tuple[ObjectRelation, ...] = ()
    color_references: tuple[ColorReference, ...] = ()

    def __post_init__(self) -> None:
        frozen = _validate_grid(self.grid)
        if frozen != self.grid:
            object.__setattr__(self, "grid", frozen)
        if not 0 <= self.background <= 9:
            raise ValueError("background must be an ARC color")
        _validate_connectivity(self.connectivity)
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "color_references", tuple(self.color_references))

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def render(self) -> list[list[int]]:
        """Render extracted objects on the inferred background."""

        rendered = [[self.background for _ in range(self.width)] for _ in range(self.height)]
        for obj in self.objects:
            for coordinate, color in zip(obj.cells, obj.cell_colors, strict=True):
                row, column = coordinate
                if not (0 <= row < self.height and 0 <= column < self.width):
                    raise ValueError("object cell lies outside scene bounds")
                previous = rendered[row][column]
                if previous != self.background and previous != color:
                    raise ValueError("objects overlap with conflicting colors")
                rendered[row][column] = color
        return rendered

    def object(self, object_id: int) -> GridObject:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(object_id)

    def relations_between(self, source_id: int, target_id: int) -> tuple[ObjectRelation, ...]:
        return tuple(
            relation
            for relation in self.relations
            if relation.source_id == source_id and relation.target_id == target_id
        )

    def references_from(self, source_id: int) -> tuple[ColorReference, ...]:
        return tuple(
            reference
            for reference in self.color_references
            if reference.source_id == source_id
        )

    @classmethod
    def from_grid(
        cls,
        grid: GridInput,
        *,
        background: int | None = None,
        connectivity: int = 4,
        multicolor: bool = False,
    ) -> GridScene:
        frozen = _validate_grid(grid)
        inferred = infer_background(frozen) if background is None else background
        objects = extract_objects(
            frozen,
            background=inferred,
            connectivity=connectivity,
            multicolor=multicolor,
        )
        return cls(
            grid=frozen,
            background=inferred,
            objects=objects,
            connectivity=connectivity,
            multicolor=multicolor,
            relations=object_relations(objects),
            color_references=color_reference_features(frozen, objects, background=inferred),
        )


def extract_scene(
    grid: GridInput,
    *,
    background: int | None = None,
    connectivity: int = 4,
    multicolor: bool = False,
) -> GridScene:
    """Build a complete immutable scene from an ARC grid."""

    return GridScene.from_grid(
        grid,
        background=background,
        connectivity=connectivity,
        multicolor=multicolor,
    )


_TRANSFORM_NAMES = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "reflect",
    "reflect_rotate_90",
    "reflect_rotate_180",
    "reflect_rotate_270",
)


def _shape_transform(source: GridObject, target: GridObject) -> str:
    wanted = target.normalized_cells
    for name, candidate in zip(
        _TRANSFORM_NAMES,
        _transformed_coordinates(source.cells),
        strict=True,
    ):
        if _normalize_coordinates(candidate) == wanted:
            return name
    return "reshape"


def _object_context(scene: GridScene, object_id: int) -> Counter[str]:
    """Return an order-independent relational fingerprint for one object."""

    context: Counter[str] = Counter()
    for relation in scene.relations:
        if relation.source_id != object_id:
            continue
        context[f"direction:{relation.direction}"] += 1
        for label in relation.relations:
            if label != relation.direction:
                context[f"relation:{label}"] += 1
    return context


def _counter_similarity(first: Counter[str], second: Counter[str]) -> float:
    keys = set(first) | set(second)
    if not keys:
        return 1.0
    intersection = sum(min(first[key], second[key]) for key in keys)
    union = sum(max(first[key], second[key]) for key in keys)
    return intersection / union if union else 1.0


def _position_similarity(
    input_scene: GridScene,
    output_scene: GridScene,
    source: GridObject,
    target: GridObject,
) -> float:
    input_row_scale = max(1, input_scene.height - 1)
    input_column_scale = max(1, input_scene.width - 1)
    output_row_scale = max(1, output_scene.height - 1)
    output_column_scale = max(1, output_scene.width - 1)
    source_row = source.center[0] / input_row_scale
    source_column = source.center[1] / input_column_scale
    target_row = target.center[0] / output_row_scale
    target_column = target.center[1] / output_column_scale
    normalized_distance = (abs(source_row - target_row) + abs(source_column - target_column)) / 2
    return max(0.0, 1.0 - normalized_distance)


@dataclass(frozen=True, slots=True)
class _PairScore:
    score: int
    same_shape: bool
    same_colored_shape: bool
    same_oriented_shape: bool
    same_area: bool
    same_color: bool
    position_similarity: float
    relation_similarity: float


def _score_object_pair(
    input_scene: GridScene,
    output_scene: GridScene,
    source: GridObject,
    target: GridObject,
    *,
    source_context: Counter[str] | None = None,
    target_context: Counter[str] | None = None,
    source_shape: tuple[Coord, ...] | None = None,
    target_shape: tuple[Coord, ...] | None = None,
    source_colored_shape: tuple[tuple[int, int, int], ...] | None = None,
    target_colored_shape: tuple[tuple[int, int, int], ...] | None = None,
) -> _PairScore:
    same_shape = (
        source_shape if source_shape is not None else source.canonical_shape_signature()
    ) == (target_shape if target_shape is not None else target.canonical_shape_signature())
    same_colored_shape = (
        source_colored_shape
        if source_colored_shape is not None
        else source.canonical_color_signature()
    ) == (
        target_colored_shape
        if target_colored_shape is not None
        else target.canonical_color_signature()
    )
    same_oriented_shape = source.shape_signature == target.shape_signature
    same_area = source.area == target.area
    same_color = source.colors == target.colors
    same_dimensions = (
        source.bbox.height == target.bbox.height and source.bbox.width == target.bbox.width
    )
    position_similarity = _position_similarity(input_scene, output_scene, source, target)
    relation_similarity = _counter_similarity(
        source_context
        if source_context is not None
        else _object_context(input_scene, source.object_id),
        target_context
        if target_context is not None
        else _object_context(output_scene, target.object_id),
    )
    score = (
        400 * int(same_colored_shape)
        + 300 * int(same_shape)
        + 80 * int(same_oriented_shape)
        + 60 * int(same_area)
        + 60 * int(same_color)
        + 40 * int(same_dimensions)
        + round(100 * position_similarity)
        + round(80 * relation_similarity)
    )
    return _PairScore(
        score=score,
        same_shape=same_shape,
        same_colored_shape=same_colored_shape,
        same_oriented_shape=same_oriented_shape,
        same_area=same_area,
        same_color=same_color,
        position_similarity=position_similarity,
        relation_similarity=relation_similarity,
    )


@dataclass(frozen=True, slots=True)
class ObjectMatch:
    """A globally assigned input/output object correspondence."""

    input_object_id: int
    output_object_id: int
    score: int
    changes: tuple[str, ...]
    translation: tuple[float, float]
    shape_transform: str
    same_shape: bool
    same_colored_shape: bool
    same_area: bool
    same_color: bool
    position_similarity: float
    relation_similarity: float

    @property
    def confidence(self) -> float:
        return min(1.0, self.score / 1120)

    def has(self, change: str) -> bool:
        return change in self.changes

    def compact_features(self) -> dict[str, int | float | bool | str]:
        """Return stable scalar features suitable for a task/world-model encoder."""

        return {
            "input_id": self.input_object_id,
            "output_id": self.output_object_id,
            "score": self.score,
            "confidence": round(self.confidence, 4),
            "same_shape": self.same_shape,
            "same_colored_shape": self.same_colored_shape,
            "same_area": self.same_area,
            "same_color": self.same_color,
            "delta_row": self.translation[0],
            "delta_column": self.translation[1],
            "shape_transform": self.shape_transform,
        }

    def summary(self) -> str:
        changes = "+".join(self.changes)
        delta = f"({self.translation[0]:g},{self.translation[1]:g})"
        return (
            f"object {self.input_object_id}->{self.output_object_id}: {changes}; "
            f"delta={delta}; shape={self.shape_transform}; confidence={self.confidence:.2f}"
        )


@dataclass(frozen=True, slots=True)
class RelationChange:
    """A spatial relation change between two matched object pairs."""

    input_source_id: int
    input_target_id: int
    output_source_id: int
    output_target_id: int
    before: ObjectRelation
    after: ObjectRelation
    added_relations: tuple[str, ...]
    removed_relations: tuple[str, ...]

    @property
    def geometry_changed(self) -> bool:
        return (
            self.before.offset != self.after.offset
            or self.before.center_offset != self.after.center_offset
            or self.before.manhattan_distance != self.after.manhattan_distance
            or self.before.bbox_overlap != self.after.bbox_overlap
        )

    def summary(self) -> str:
        additions = ",".join(self.added_relations) or "-"
        removals = ",".join(self.removed_relations) or "-"
        return (
            f"relation {self.input_source_id}->{self.input_target_id} became "
            f"{self.output_source_id}->{self.output_target_id}; +[{additions}] -[{removals}]"
        )


def _make_object_match(
    source: GridObject,
    target: GridObject,
    pair_score: _PairScore,
) -> ObjectMatch:
    translation = (
        target.center[0] - source.center[0],
        target.center[1] - source.center[1],
    )
    shape_transform = _shape_transform(source, target)
    changes: list[str] = []
    if translation != (0.0, 0.0):
        changes.append("moved")
    if not pair_score.same_colored_shape:
        changes.append("recolored")
    if not pair_score.same_oriented_shape:
        changes.append("transformed")
    if not changes:
        changes.append("preserved")
    return ObjectMatch(
        input_object_id=source.object_id,
        output_object_id=target.object_id,
        score=pair_score.score,
        changes=tuple(changes),
        translation=translation,
        shape_transform=shape_transform,
        same_shape=pair_score.same_shape,
        same_colored_shape=pair_score.same_colored_shape,
        same_area=pair_score.same_area,
        same_color=pair_score.same_color,
        position_similarity=pair_score.position_similarity,
        relation_similarity=pair_score.relation_similarity,
    )


def _exact_assignment(
    input_objects: tuple[GridObject, ...],
    output_objects: tuple[GridObject, ...],
    scores: dict[tuple[int, int], _PairScore],
    *,
    minimum_score: int,
) -> tuple[tuple[int, int], ...]:
    """Maximize accepted matches, then score, with deterministic tie-breaking."""

    @cache
    def solve(input_index: int, used_outputs: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if input_index == len(input_objects):
            return 0, ()
        best_score, best_pairs = solve(input_index + 1, used_outputs)
        source = input_objects[input_index]
        for output_index, target in enumerate(output_objects):
            if used_outputs & (1 << output_index):
                continue
            pair_score = scores[(source.object_id, target.object_id)].score
            if pair_score < minimum_score:
                continue
            remaining_score, remaining_pairs = solve(
                input_index + 1,
                used_outputs | (1 << output_index),
            )
            candidate_score = pair_score + remaining_score
            candidate_pairs = ((source.object_id, target.object_id),) + remaining_pairs
            if (
                len(candidate_pairs) > len(best_pairs)
                or (
                    len(candidate_pairs) == len(best_pairs)
                    and candidate_score > best_score
                )
                or (
                    len(candidate_pairs) == len(best_pairs)
                    and candidate_score == best_score
                    and candidate_pairs < best_pairs
                )
            ):
                best_score, best_pairs = candidate_score, candidate_pairs
        return best_score, best_pairs

    return solve(0, 0)[1]


def _bounded_assignment(
    input_objects: tuple[GridObject, ...],
    output_objects: tuple[GridObject, ...],
    scores: dict[tuple[int, int], _PairScore],
    *,
    minimum_score: int,
) -> tuple[tuple[int, int], ...]:
    """Deterministic fallback for scenes too large for exact bit-mask assignment."""

    edges = sorted(
        (-score.score, source.object_id, target.object_id)
        for source in input_objects
        for target in output_objects
        if (score := scores[(source.object_id, target.object_id)]).score >= minimum_score
    )
    used_inputs: set[int] = set()
    used_outputs: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, input_id, output_id in edges:
        if input_id in used_inputs or output_id in used_outputs:
            continue
        used_inputs.add(input_id)
        used_outputs.add(output_id)
        matches.append((input_id, output_id))
    return tuple(sorted(matches))


def _relation_changes(
    input_scene: GridScene,
    output_scene: GridScene,
    matches: tuple[ObjectMatch, ...],
) -> tuple[RelationChange, ...]:
    output_by_input = {
        match.input_object_id: match.output_object_id
        for match in matches
    }
    input_relations = {
        (relation.source_id, relation.target_id): relation
        for relation in input_scene.relations
    }
    output_relations = {
        (relation.source_id, relation.target_id): relation
        for relation in output_scene.relations
    }
    changes: list[RelationChange] = []
    for input_source_id, input_target_id in permutations(sorted(output_by_input), 2):
        output_source_id = output_by_input[input_source_id]
        output_target_id = output_by_input[input_target_id]
        before = input_relations[(input_source_id, input_target_id)]
        after = output_relations[(output_source_id, output_target_id)]
        before_labels = set(before.relations)
        after_labels = set(after.relations)
        added = tuple(sorted(after_labels - before_labels))
        removed = tuple(sorted(before_labels - after_labels))
        change = RelationChange(
            input_source_id=input_source_id,
            input_target_id=input_target_id,
            output_source_id=output_source_id,
            output_target_id=output_target_id,
            before=before,
            after=after,
            added_relations=added,
            removed_relations=removed,
        )
        if added or removed or change.geometry_changed:
            changes.append(change)
    return tuple(changes)


@dataclass(frozen=True, slots=True)
class SceneTransition:
    """A verified object-level description of one ARC demonstration transition."""

    input_scene: GridScene
    output_scene: GridScene
    matches: tuple[ObjectMatch, ...]
    added_objects: tuple[GridObject, ...]
    removed_objects: tuple[GridObject, ...]
    relation_changes: tuple[RelationChange, ...]

    @property
    def preserved(self) -> tuple[ObjectMatch, ...]:
        return tuple(match for match in self.matches if match.has("preserved"))

    @property
    def moved(self) -> tuple[ObjectMatch, ...]:
        return tuple(match for match in self.matches if match.has("moved"))

    @property
    def recolored(self) -> tuple[ObjectMatch, ...]:
        return tuple(match for match in self.matches if match.has("recolored"))

    @property
    def transformed(self) -> tuple[ObjectMatch, ...]:
        return tuple(match for match in self.matches if match.has("transformed"))

    @property
    def background_changed(self) -> bool:
        return self.input_scene.background != self.output_scene.background

    @property
    def shape_changed(self) -> bool:
        return self.input_scene.shape != self.output_scene.shape

    def compact_features(self) -> dict[str, int | bool | tuple[str, ...]]:
        """Return a compact transition vector and event vocabulary."""

        return {
            "input_objects": len(self.input_scene.objects),
            "output_objects": len(self.output_scene.objects),
            "matches": len(self.matches),
            "preserved": len(self.preserved),
            "moved": len(self.moved),
            "recolored": len(self.recolored),
            "transformed": len(self.transformed),
            "added": len(self.added_objects),
            "removed": len(self.removed_objects),
            "relation_changes": len(self.relation_changes),
            "background_changed": self.background_changed,
            "shape_changed": self.shape_changed,
            "event_tokens": self.event_tokens(),
        }

    def event_tokens(self) -> tuple[str, ...]:
        tokens: list[str] = []
        for label, matches in (
            ("preserve", self.preserved),
            ("move", self.moved),
            ("recolor", self.recolored),
            ("transform", self.transformed),
        ):
            if matches:
                tokens.append(f"{label}:{len(matches)}")
        if self.added_objects:
            tokens.append(f"add:{len(self.added_objects)}")
        if self.removed_objects:
            tokens.append(f"remove:{len(self.removed_objects)}")
        if self.relation_changes:
            tokens.append(f"relation_change:{len(self.relation_changes)}")
        if self.background_changed:
            tokens.append("background_change")
        if self.shape_changed:
            tokens.append("grid_resize")
        return tuple(tokens)

    def summary(self) -> str:
        events = ", ".join(self.event_tokens()) or "no object-level change"
        return (
            f"scene {self.input_scene.height}x{self.input_scene.width}/"
            f"{len(self.input_scene.objects)} objects -> "
            f"{self.output_scene.height}x{self.output_scene.width}/"
            f"{len(self.output_scene.objects)} objects: {events}"
        )


def infer_scene_transition(
    input_grid_or_scene: GridInput | GridScene,
    output_grid_or_scene: GridInput | GridScene,
    *,
    connectivity: int = 4,
    multicolor: bool = False,
    maximum_exact_objects: int = 12,
    minimum_match_score: int = 430,
) -> SceneTransition:
    """Infer a deterministic object-level transition between two ARC scenes.

    Assignment is global and exact for small scenes.  Larger scenes use a
    score-ordered deterministic matching bounded by the pair matrix size.  The
    score combines D4 geometry, colored geometry, area, color, normalized
    position, and pairwise relational context; object IDs never contribute to
    the score.
    """

    if maximum_exact_objects < 1:
        raise ValueError("maximum_exact_objects must be positive")
    if minimum_match_score < 0:
        raise ValueError("minimum_match_score must be non-negative")
    if isinstance(input_grid_or_scene, GridScene):
        input_scene = input_grid_or_scene
    else:
        scene_connectivity = (
            output_grid_or_scene.connectivity
            if isinstance(output_grid_or_scene, GridScene)
            else connectivity
        )
        scene_multicolor = (
            output_grid_or_scene.multicolor
            if isinstance(output_grid_or_scene, GridScene)
            else multicolor
        )
        input_scene = extract_scene(
            input_grid_or_scene,
            connectivity=scene_connectivity,
            multicolor=scene_multicolor,
        )
    if isinstance(output_grid_or_scene, GridScene):
        output_scene = output_grid_or_scene
    else:
        output_scene = extract_scene(
            output_grid_or_scene,
            connectivity=input_scene.connectivity,
            multicolor=input_scene.multicolor,
        )

    input_objects = input_scene.objects
    output_objects = output_scene.objects
    input_contexts = {
        obj.object_id: _object_context(input_scene, obj.object_id) for obj in input_objects
    }
    output_contexts = {
        obj.object_id: _object_context(output_scene, obj.object_id) for obj in output_objects
    }
    input_shapes = {
        obj.object_id: obj.canonical_shape_signature() for obj in input_objects
    }
    output_shapes = {
        obj.object_id: obj.canonical_shape_signature() for obj in output_objects
    }
    input_colored_shapes = {
        obj.object_id: obj.canonical_color_signature() for obj in input_objects
    }
    output_colored_shapes = {
        obj.object_id: obj.canonical_color_signature() for obj in output_objects
    }
    scores = {
        (source.object_id, target.object_id): _score_object_pair(
            input_scene,
            output_scene,
            source,
            target,
            source_context=input_contexts[source.object_id],
            target_context=output_contexts[target.object_id],
            source_shape=input_shapes[source.object_id],
            target_shape=output_shapes[target.object_id],
            source_colored_shape=input_colored_shapes[source.object_id],
            target_colored_shape=output_colored_shapes[target.object_id],
        )
        for source in input_objects
        for target in output_objects
    }
    if max(len(input_objects), len(output_objects)) <= maximum_exact_objects:
        assigned = _exact_assignment(
            input_objects,
            output_objects,
            scores,
            minimum_score=minimum_match_score,
        )
    else:
        assigned = _bounded_assignment(
            input_objects,
            output_objects,
            scores,
            minimum_score=minimum_match_score,
        )
    input_by_id = {obj.object_id: obj for obj in input_objects}
    output_by_id = {obj.object_id: obj for obj in output_objects}
    matches = tuple(
        _make_object_match(
            input_by_id[input_id],
            output_by_id[output_id],
            scores[(input_id, output_id)],
        )
        for input_id, output_id in assigned
    )
    matched_inputs = {match.input_object_id for match in matches}
    matched_outputs = {match.output_object_id for match in matches}
    removed = tuple(obj for obj in input_objects if obj.object_id not in matched_inputs)
    added = tuple(obj for obj in output_objects if obj.object_id not in matched_outputs)
    return SceneTransition(
        input_scene=input_scene,
        output_scene=output_scene,
        matches=matches,
        added_objects=added,
        removed_objects=removed,
        relation_changes=_relation_changes(input_scene, output_scene, matches),
    )


__all__ = [
    "BoundingBox",
    "ColorReference",
    "Coord",
    "GridObject",
    "GridScene",
    "Hole",
    "ObjectMatch",
    "ObjectRelation",
    "RelationChange",
    "SceneTransition",
    "color_reference_features",
    "connected_components",
    "extract_components",
    "extract_objects",
    "extract_scene",
    "infer_background",
    "infer_scene_transition",
    "object_relations",
    "relative_position",
]
