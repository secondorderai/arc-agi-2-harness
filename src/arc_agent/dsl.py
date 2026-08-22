from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Iterable
from itertools import permutations

from arc_agent.models import (
    ArcTask,
    Candidate,
    Grid,
    Program,
    VerificationResult,
    grid_accuracies,
    validate_grid,
)


class ProgramError(ValueError):
    pass


_ALLOWED_ARGS: dict[str, set[str]] = {
    "identity": set(),
    "rotate": {"turns"},
    "flip": {"axis"},
    "transpose": set(),
    "recolor": {"mapping"},
    "crop_background": {"background"},
    "scale": {"rows", "columns"},
    "tile": {"rows", "columns"},
    "pad": {"top", "bottom", "left", "right", "color"},
    "select_component": {
        "background",
        "criterion",
        "connectivity",
        "multicolor",
    },
    "translate": {"rows", "columns", "background"},
    "concat": {"axis", "transform", "order"},
    "complete_symmetry": {"axis", "background"},
    "split_overlay": {
        "axis",
        "mode",
        "background",
        "separator_color",
        "output_color",
    },
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
    "compose": set(),
}


def _validate_program_args(program: Program) -> None:
    unexpected = set(program.args) - _ALLOWED_ARGS[program.op]
    if unexpected:
        raise ProgramError(f"unexpected args for {program.op}: {sorted(unexpected)}")


def _checked(grid: Grid) -> Grid:
    validate_grid(grid, label="program output")
    return grid


def _copy(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _rotate(grid: Grid, turns: int) -> Grid:
    result = _copy(grid)
    for _ in range(turns % 4):
        result = [list(row) for row in zip(*result[::-1], strict=True)]
    return result


def _flip(grid: Grid, axis: str) -> Grid:
    if axis == "horizontal":
        return [list(reversed(row)) for row in grid]
    if axis == "vertical":
        return [list(row) for row in reversed(grid)]
    raise ProgramError(f"unknown flip axis: {axis}")


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _bounding_box(grid: Grid, *, background: int) -> tuple[int, int, int, int] | None:
    points = [
        (row_index, column_index)
        for row_index, row in enumerate(grid)
        for column_index, value in enumerate(row)
        if value != background
    ]
    if not points:
        return None
    rows, columns = zip(*points, strict=True)
    return min(rows), max(rows), min(columns), max(columns)


def _crop(grid: Grid, *, background: int) -> Grid:
    bounds = _bounding_box(grid, background=background)
    if bounds is None:
        return _copy(grid)
    top, bottom, left, right = bounds
    return [row[left : right + 1] for row in grid[top : bottom + 1]]


def connected_components(
    grid: Grid,
    *,
    background: int | None = None,
    connectivity: int = 4,
    multicolor: bool = False,
) -> list[list[tuple[int, int]]]:
    if connectivity not in {4, 8}:
        raise ProgramError("connectivity must be 4 or 8")
    bg = _background(grid) if background is None else background
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for column in range(width):
            if (row, column) in visited or grid[row][column] == bg:
                continue
            color = grid[row][column]
            queue = deque([(row, column)])
            visited.add((row, column))
            component: list[tuple[int, int]] = []
            while queue:
                current_row, current_column = queue.popleft()
                component.append((current_row, current_column))
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if connectivity == 8:
                    directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
                for delta_row, delta_column in directions:
                    neighbor = (current_row + delta_row, current_column + delta_column)
                    if (
                        0 <= neighbor[0] < height
                        and 0 <= neighbor[1] < width
                        and neighbor not in visited
                        and grid[neighbor[0]][neighbor[1]] != bg
                        and (multicolor or grid[neighbor[0]][neighbor[1]] == color)
                    ):
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components


def _select_component(
    grid: Grid,
    *,
    background: int,
    criterion: str,
    connectivity: int,
    multicolor: bool,
) -> Grid:
    components = connected_components(
        grid,
        background=background,
        connectivity=connectivity,
        multicolor=multicolor,
    )
    if not components:
        return _copy(grid)
    if criterion == "largest":
        component = max(components, key=lambda item: (len(item), -min(item)[0], -min(item)[1]))
    elif criterion == "smallest":
        component = min(components, key=lambda item: (len(item), min(item)[0], min(item)[1]))
    elif criterion == "widest":
        component = max(
            components,
            key=lambda item: (
                max(point[1] for point in item) - min(point[1] for point in item) + 1,
                len(item),
                -min(item)[0],
                -min(item)[1],
            ),
        )
    elif criterion == "tallest":
        component = max(
            components,
            key=lambda item: (
                max(point[0] for point in item) - min(point[0] for point in item) + 1,
                len(item),
                -min(item)[0],
                -min(item)[1],
            ),
        )
    else:
        raise ProgramError(f"unknown component criterion: {criterion}")
    rows, columns = zip(*component, strict=True)
    top, bottom, left, right = min(rows), max(rows), min(columns), max(columns)
    selected = set(component)
    return [
        [
            grid[row][column] if (row, column) in selected else background
            for column in range(left, right + 1)
        ]
        for row in range(top, bottom + 1)
    ]


def _translate(grid: Grid, *, rows: int, columns: int, background: int) -> Grid:
    height, width = len(grid), len(grid[0])
    translated = [[background for _ in range(width)] for _ in range(height)]
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            target_row, target_column = row + rows, column + columns
            if value != background and 0 <= target_row < height and 0 <= target_column < width:
                translated[target_row][target_column] = value
    return translated


def _transform(grid: Grid, transform: str) -> Grid:
    if transform == "identity":
        return _copy(grid)
    if transform == "rotate_90":
        return _rotate(grid, 1)
    if transform == "rotate_180":
        return _rotate(grid, 2)
    if transform == "rotate_270":
        return _rotate(grid, 3)
    if transform == "flip_horizontal":
        return _flip(grid, "horizontal")
    if transform == "flip_vertical":
        return _flip(grid, "vertical")
    if transform == "transpose":
        return [list(row) for row in zip(*grid, strict=True)]
    raise ProgramError(f"unknown transform: {transform}")


def _concat(grid: Grid, *, axis: str, transform: str, order: str) -> Grid:
    transformed = _transform(grid, transform)
    parts = (grid, transformed) if order == "input_first" else (transformed, grid)
    if order not in {"input_first", "transformed_first"}:
        raise ProgramError(f"unknown concatenation order: {order}")
    if axis == "horizontal":
        if len(parts[0]) != len(parts[1]):
            raise ProgramError("horizontal concatenation needs equal heights")
        return [left + right for left, right in zip(parts[0], parts[1], strict=True)]
    if axis == "vertical":
        if len(parts[0][0]) != len(parts[1][0]):
            raise ProgramError("vertical concatenation needs equal widths")
        return _copy(parts[0]) + _copy(parts[1])
    raise ProgramError(f"unknown concatenation axis: {axis}")


def _complete_symmetry(grid: Grid, *, axis: str, background: int) -> Grid:
    height, width = len(grid), len(grid[0])
    completed = _copy(grid)
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            if value == background:
                continue
            if axis == "horizontal":
                target = (row, width - column - 1)
            elif axis == "vertical":
                target = (height - row - 1, column)
            elif axis == "rotational":
                target = (height - row - 1, width - column - 1)
            else:
                raise ProgramError(f"unknown symmetry axis: {axis}")
            if completed[target[0]][target[1]] == background:
                completed[target[0]][target[1]] = value
    return completed


def _split_overlay(
    grid: Grid,
    *,
    axis: str,
    mode: str,
    background: int,
    separator_color: int,
    output_color: int,
) -> Grid:
    height, width = len(grid), len(grid[0])
    if axis == "horizontal":
        separator = height // 2
        if height % 2 == 0:
            first, second = grid[:separator], grid[separator:]
        else:
            if any(value != separator_color for value in grid[separator]):
                raise ProgramError("no centered horizontal separator")
            first, second = grid[:separator], grid[separator + 1 :]
    elif axis == "vertical":
        separator = width // 2
        if width % 2 == 0:
            first = [row[:separator] for row in grid]
            second = [row[separator:] for row in grid]
        else:
            if any(grid[row][separator] != separator_color for row in range(height)):
                raise ProgramError("no centered vertical separator")
            first = [row[:separator] for row in grid]
            second = [row[separator + 1 :] for row in grid]
    else:
        raise ProgramError(f"unknown overlay axis: {axis}")
    if not first or not first[0] or len(first) != len(second) or len(first[0]) != len(second[0]):
        raise ProgramError("separator must divide the grid into equal non-empty halves")
    output: Grid = []
    for first_row, second_row in zip(first, second, strict=True):
        output_row: list[int] = []
        for first_value, second_value in zip(first_row, second_row, strict=True):
            left = first_value != background
            right = second_value != background
            if mode == "union":
                occupied = left or right
            elif mode == "intersection":
                occupied = left and right
            elif mode == "xor":
                occupied = left != right
            elif mode == "neither":
                occupied = not left and not right
            else:
                raise ProgramError(f"unknown overlay mode: {mode}")
            output_row.append(output_color if occupied else background)
        output.append(output_row)
    return output


def _move_color_to_contact(
    grid: Grid, *, moving_color: int, target_color: int, background: int
) -> Grid:
    moving = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value == moving_color
    ]
    target = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value == target_color
    ]
    if not moving or not target or moving_color == target_color:
        raise ProgramError("moving and target colors must identify distinct objects")
    moving_rows, moving_columns = zip(*moving, strict=True)
    target_rows, target_columns = zip(*target, strict=True)
    rows_overlap = max(min(moving_rows), min(target_rows)) <= min(
        max(moving_rows), max(target_rows)
    )
    columns_overlap = max(min(moving_columns), min(target_columns)) <= min(
        max(moving_columns), max(target_columns)
    )
    row_delta = 0
    column_delta = 0
    if rows_overlap and max(moving_columns) < min(target_columns):
        column_delta = min(target_columns) - max(moving_columns) - 1
    elif rows_overlap and max(target_columns) < min(moving_columns):
        column_delta = max(target_columns) - min(moving_columns) + 1
    elif columns_overlap and max(moving_rows) < min(target_rows):
        row_delta = min(target_rows) - max(moving_rows) - 1
    elif columns_overlap and max(target_rows) < min(moving_rows):
        row_delta = max(target_rows) - min(moving_rows) + 1
    else:
        raise ProgramError("objects are not separated along one overlapping axis")
    moved = _copy(grid)
    for row, column in moving:
        moved[row][column] = background
    for row, column in moving:
        target_row, target_column = row + row_delta, column + column_delta
        if moved[target_row][target_column] not in {background, moving_color}:
            raise ProgramError("translated object would collide with another color")
        moved[target_row][target_column] = moving_color
    return moved


def _pattern_mosaic(
    grid: Grid,
    *,
    background: int,
    empty_color: int,
    base_color: int,
    marker_color: int,
    row_order: list[int],
    column_order: list[int],
    output_rows: int,
    output_columns: int,
) -> Grid:
    symbol = _crop(grid, background=background)
    symbol_rows, symbol_columns = len(symbol), len(symbol[0])
    if sorted(row_order) != list(range(symbol_rows)):
        raise ProgramError("row_order must permute the cropped symbol rows")
    if sorted(column_order) != list(range(symbol_columns)):
        raise ProgramError("column_order must permute the cropped symbol columns")
    if not 1 <= output_rows <= 30 or not 1 <= output_columns <= 30:
        raise ProgramError("mosaic output dimensions must be between 1 and 30")
    if output_rows < len(grid) or output_columns < len(grid[0]):
        raise ProgramError("mosaic output must contain the centered symbol overlay")

    tile = [
        [
            base_color
            if symbol[row_order[row]][column_order[column]] == background
            else empty_color
            for column in range(symbol_columns)
        ]
        for row in range(symbol_rows)
    ]
    output = [
        [tile[row % symbol_rows][column % symbol_columns] for column in range(output_columns)]
        for row in range(output_rows)
    ]
    overlay_top = (output_rows - len(grid)) // 2
    overlay_left = (output_columns - len(grid[0])) // 2
    for row in range(len(grid)):
        for column in range(len(grid[0])):
            if symbol[row % symbol_rows][column % symbol_columns] != background:
                output[overlay_top + row][overlay_left + column] = marker_color
    return output


def _spans_between_separators(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in separators:
        if start < separator:
            spans.append((start, separator))
        start = separator + 1
    if start < size:
        spans.append((start, size))
    return spans


def _repair_periodic_panels(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    separator_color = grid[0][0]
    separator_rows = [
        row for row in range(height) if all(value == separator_color for value in grid[row])
    ]
    separator_columns = [
        column
        for column in range(width)
        if all(grid[row][column] == separator_color for row in range(height))
    ]
    repaired = _copy(grid)
    for row_start, row_end in _spans_between_separators(height, separator_rows):
        for column_start, column_end in _spans_between_separators(width, separator_columns):
            perimeter = (
                grid[row_start][column_start:column_end]
                + grid[row_end - 1][column_start:column_end]
                + [grid[row][column_start] for row in range(row_start, row_end)]
                + [grid[row][column_end - 1] for row in range(row_start, row_end)]
            )
            border_color = Counter(perimeter).most_common(1)[0][0]
            top, bottom = row_start, row_end
            left, right = column_start, column_end
            while top < bottom and all(value == border_color for value in grid[top][left:right]):
                top += 1
            while top < bottom and all(
                value == border_color for value in grid[bottom - 1][left:right]
            ):
                bottom -= 1
            while left < right and all(
                grid[row][left] == border_color for row in range(top, bottom)
            ):
                left += 1
            while left < right and all(
                grid[row][right - 1] == border_color for row in range(top, bottom)
            ):
                right -= 1
            panel_height, panel_width = bottom - top, right - left
            if panel_height <= 0 or panel_width <= 1:
                continue

            best: tuple[tuple[int, int, int], Grid, int, int, int] | None = None
            for row_period in range(1, min(panel_height, 6) + 1):
                for column_period in range(1, min(panel_width, 8) + 1):
                    if row_period == panel_height and column_period == panel_width:
                        continue
                    buckets = [[Counter() for _ in range(column_period)] for _ in range(row_period)]
                    for row in range(top, bottom):
                        for column in range(left, right):
                            buckets[(row - top) % row_period][(column - left) % column_period][
                                grid[row][column]
                            ] += 1
                    tile = [
                        [
                            buckets[row][column].most_common(1)[0][0]
                            for column in range(column_period)
                        ]
                        for row in range(row_period)
                    ]
                    mismatches = sum(
                        grid[row][column]
                        != tile[(row - top) % row_period][(column - left) % column_period]
                        for row in range(top, bottom)
                        for column in range(left, right)
                    )
                    area = row_period * column_period
                    # A changed cell is substantially more expensive than one template cell:
                    # otherwise a tiny but inaccurate tile wins over the true larger period.
                    key = (4 * mismatches + area, area, mismatches)
                    if mismatches and (best is None or key < best[0]):
                        best = (key, tile, row_period, column_period, mismatches)
            if best is None or best[4] > max(1, panel_height * panel_width // 4):
                continue
            _, tile, row_period, column_period, _ = best
            for row in range(top, bottom):
                for column in range(left, right):
                    repaired[row][column] = tile[(row - top) % row_period][
                        (column - left) % column_period
                    ]
    return repaired


def _select_unique_panel_per_band(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    separator_color = grid[0][0]
    separator_rows = [
        row for row in range(height) if all(value == separator_color for value in grid[row])
    ]
    separator_columns = [
        column
        for column in range(width)
        if all(grid[row][column] == separator_color for row in range(height))
    ]
    if len(separator_rows) < 2 or len(separator_columns) < 3:
        raise ProgramError("unique-panel selection needs a separator-delimited panel grid")
    panel_widths = {
        separator_columns[index + 1] - separator_columns[index]
        for index in range(len(separator_columns) - 1)
    }
    if len(panel_widths) != 1:
        raise ProgramError("panel columns must have equal widths")
    output_width = next(iter(panel_widths)) + 1
    output = [[separator_color for _ in range(output_width)] for _ in range(height)]
    for band in range(len(separator_rows) - 1):
        top, bottom = separator_rows[band], separator_rows[band + 1]
        panels: list[tuple[tuple[int, ...], ...]] = []
        for panel_index in range(len(separator_columns) - 1):
            left = separator_columns[panel_index]
            right = separator_columns[panel_index + 1]
            panels.append(
                tuple(
                    tuple(grid[row][column] for column in range(left, right + 1))
                    for row in range(top, bottom + 1)
                )
            )
        counts = Counter(panels)
        unique = [index for index, panel in enumerate(panels) if counts[panel] == 1]
        if len(unique) != 1:
            raise ProgramError("each row band must contain exactly one unique panel")
        left = separator_columns[unique[0]]
        right = separator_columns[unique[0] + 1]
        for row in range(top, bottom + 1):
            output[row] = grid[row][left : right + 1]
    return output


def _deduplicate_repeated_half(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    if width % 2 == 0:
        middle = width // 2
        if all(row[:middle] == row[middle:] for row in grid):
            return [row[:middle] for row in grid]
    if height % 2 == 0:
        middle = height // 2
        if grid[:middle] == grid[middle:]:
            return _copy(grid[:middle])
    raise ProgramError("grid is not two identical horizontal or vertical halves")


def _fold_quadrants_overlap(grid: Grid) -> Grid:
    background = _background(grid)
    height, width = len(grid), len(grid[0])
    panel_sizes = [
        size
        for size in range(1, min(height, width) // 2 + 1)
        if height - 2 * size >= 1
        and width - 2 * size >= 1
        and all(
            grid[row][column] == background
            for row in range(size, height - size)
            for column in range(width)
        )
        and all(
            grid[row][column] == background
            for row in range(height)
            for column in range(size, width - size)
        )
    ]
    if not panel_sizes:
        raise ProgramError("quadrant fold needs a centered blank cross")
    top_height = left_width = max(panel_sizes)
    bottom_start = height - top_height
    right_start = width - left_width
    output = [
        [background for _ in range(2 * left_width - 1)]
        for _ in range(2 * top_height - 1)
    ]
    for row in range(height):
        if row < top_height:
            output_row = row
        elif row >= bottom_start:
            output_row = top_height - 1 + row - bottom_start
        else:
            continue
        for column in range(width):
            if column < left_width:
                output_column = column
            elif column >= right_start:
                output_column = left_width - 1 + column - right_start
            else:
                continue
            value = grid[row][column]
            previous = output[output_row][output_column]
            if value != background and previous not in {background, value}:
                raise ProgramError("quadrant fold has conflicting overlap colors")
            if value != background:
                output[output_row][output_column] = value
    return output


def _mark_most_frequent_above_separator(grid: Grid) -> Grid:
    if len(set(grid[-1])) != 1:
        raise ProgramError("frequency marker needs a uniform blank bottom row")
    background = grid[-1][0]
    separators = [
        row
        for row, values in enumerate(grid)
        if len(set(values)) == 1 and values[0] != background
    ]
    if len(separators) != 1:
        raise ProgramError("frequency marker needs one solid non-background separator row")
    separator = separators[0]
    if separator == 0 or any(
        value != background for row in grid[separator + 1 :] for value in row
    ):
        raise ProgramError("frequency marker needs data above and blank rows below")
    counts = Counter(value for row in grid[:separator] for value in row)
    if not counts:
        raise ProgramError("frequency marker has no data above separator")
    output = _copy(grid)
    output[-1][len(grid[0]) // 2] = counts.most_common(1)[0][0]
    return output


def _keep_most_frequent_colors(grid: Grid, *, keep_count: int, output_color: int) -> Grid:
    if keep_count < 1:
        raise ProgramError("keep_count must be positive")
    counts = Counter(value for row in grid for value in row)
    if keep_count >= len(counts):
        raise ProgramError("keep_count must be smaller than the number of colors")
    kept = {value for value, _ in counts.most_common(keep_count)}
    return [[value if value in kept else output_color for value in row] for row in grid]


def _pack_points_serpentine(grid: Grid) -> Grid:
    background = _background(grid)
    points = sorted(
        (
            (column, row, value)
            for row, values in enumerate(grid)
            for column, value in enumerate(values)
            if value != background
        )
    )
    if not points:
        raise ProgramError("serpentine packing needs colored points")
    side = 1
    while side * side < len(points):
        side += 1
    values = [value for _, _, value in points]
    output = [[background for _ in range(side)] for _ in range(side)]
    for row in range(side):
        chunk = values[row * side : (row + 1) * side]
        if row % 2:
            chunk = list(reversed(chunk))
        for column, value in enumerate(chunk):
            output[row][column] = value
    return output


def _fill_between_row_markers(grid: Grid, *, fill_color: int) -> Grid:
    background = _background(grid)
    output = _copy(grid)
    changed = False
    for row, values in enumerate(grid):
        columns = [column for column, value in enumerate(values) if value != background]
        if len(columns) < 2:
            continue
        for column in range(min(columns) + 1, max(columns)):
            if output[row][column] == background:
                output[row][column] = fill_color
                changed = True
    if not changed:
        raise ProgramError("no horizontal marker gaps to fill")
    return output


def _expand_seed_to_width(grid: Grid) -> Grid:
    background = _background(grid)
    active_columns = [
        column
        for column in range(len(grid[0]))
        if any(row[column] != background for row in grid)
    ]
    if not active_columns or min(active_columns) != 0:
        raise ProgramError("width expansion needs a left-aligned seed")
    seed_width = max(active_columns) + 1
    width = len(grid[0])
    fill_width = width - (2 * seed_width - 1)
    if seed_width < 2 or fill_width < 1:
        raise ProgramError("grid is too narrow to expand its seed")
    if any(value != background for row in grid for value in row[seed_width:]):
        raise ProgramError("seed must be followed only by background")
    return [
        row[:seed_width] + [row[0]] * fill_width + row[1:seed_width]
        for row in grid
    ]


def _summarize_nonbackground_count(grid: Grid) -> Grid:
    background = 0 if any(0 in row for row in grid) else _background(grid)
    values = [value for row in grid for value in row if value != background]
    if not values or len(set(values)) != 1:
        raise ProgramError("count summary needs one non-background color")
    return [[values[0]] * len(values)]


def _fill_with_most_frequent_color(grid: Grid) -> Grid:
    color = Counter(value for row in grid for value in row).most_common(1)[0][0]
    return [[color for _ in row] for row in grid]


def _frequency_histogram_columns(grid: Grid) -> Grid:
    # This representation treats zero as the empty histogram cell even when the
    # compact source grid contains no empty cells.
    background = 0
    counts = Counter(value for row in grid for value in row if value != background)
    if len(counts) < 2:
        raise ProgramError("frequency histogram needs at least two non-background colors")
    ranked = sorted(counts, key=lambda value: (-counts[value], value))
    if len({counts[value] for value in ranked}) != len(ranked):
        raise ProgramError("frequency histogram needs unique color frequencies")
    height = max(counts.values())
    return [
        [value if row < counts[value] else background for value in ranked]
        for row in range(height)
    ]


def _recolor_singleton_components(grid: Grid, *, background: int) -> Grid:
    color_counts = Counter(value for row in grid for value in row if value != background)
    if len(color_counts) != 1:
        raise ProgramError("singleton recoloring needs one foreground color")
    foreground = next(iter(color_counts))
    available = sorted(set(range(10)) - {background, foreground})
    if not available:
        raise ProgramError("no output color available for singleton recoloring")
    output_color = available[0]
    output = _copy(grid)
    changed = False
    for component in connected_components(grid, background=background):
        if len(component) == 1:
            row, column = component[0]
            output[row][column] = output_color
            changed = True
    if not changed:
        raise ProgramError("no singleton components found")
    return output


def _move_center_block_to_corners(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    bounds = _bounding_box(grid, background=background)
    if bounds is None:
        raise ProgramError("corner expansion needs a foreground block")
    top, bottom, left, right = bounds
    block_height, block_width = bottom - top + 1, right - left + 1
    if block_height != 2 or block_width != 2 or height < 4 or width < 4:
        raise ProgramError("corner expansion needs a centered 2x2 block")
    if top != (height - block_height) // 2 or left != (width - block_width) // 2:
        raise ProgramError("foreground block is not centered")
    block = [row[left : right + 1] for row in grid[top : bottom + 1]]
    if any(value == background for row in block for value in row):
        raise ProgramError("center block must be solid")
    output = [[background for _ in range(width)] for _ in range(height)]
    output[0][0], output[0][-1] = block[0]
    output[-1][0], output[-1][-1] = block[1]
    return output


def _complete_periodic_pattern(grid: Grid, *, mode: str) -> Grid:
    """Infer a diagonal cycle or fully observed tile and fill missing cells."""
    height, width = len(grid), len(grid[0])
    background = _background(grid)
    anchors = [
        (row, column, value)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value != background
    ]
    if len(anchors) < 3 or len({value for _, _, value in anchors}) < 2:
        raise ProgramError("period completion needs multicolor anchors")

    if mode in {"diagonal_sum", "diagonal_difference"}:
        candidates: list[tuple[int, dict[int, int]]] = []
        for period in range(2, max(height, width) + 1):
            cycle: dict[int, int] = {}
            conflict = False
            for row, column, value in anchors:
                position = (
                    (row + column) % period
                    if mode == "diagonal_sum"
                    else (row - column) % period
                )
                previous = cycle.get(position)
                if previous is not None and previous != value:
                    conflict = True
                    break
                cycle[position] = value
            if not conflict and len(cycle) == period:
                candidates.append((period, cycle))
        if not candidates:
            raise ProgramError("anchors do not define a complete diagonal cycle")
        period, cycle = min(candidates, key=lambda item: item[0])
        output = [
            [
                cycle[
                    (row + column) % period
                    if mode == "diagonal_sum"
                    else (row - column) % period
                ]
                for column in range(width)
            ]
            for row in range(height)
        ]
        if output == grid:
            raise ProgramError("periodic completion made no change")
        return output
    if mode != "tile":
        raise ProgramError(f"unsupported period completion mode: {mode}")

    tile_candidates: list[tuple[int, int, int, Grid]] = []
    for row_period in range(1, height + 1):
        for column_period in range(1, width + 1):
            tile_values: dict[tuple[int, int], int] = {}
            conflict = False
            for row, column, value in anchors:
                position = row % row_period, column % column_period
                previous = tile_values.get(position)
                if previous is not None and previous != value:
                    conflict = True
                    break
                tile_values[position] = value
            area = row_period * column_period
            if conflict or len(tile_values) != area:
                continue
            tile = [
                [tile_values[(row, column)] for column in range(column_period)]
                for row in range(row_period)
            ]
            tile_candidates.append((area, row_period + column_period, row_period, tile))
    if not tile_candidates:
        raise ProgramError("anchors do not define a complete periodic tile")
    _, _, row_period, tile = min(tile_candidates)
    column_period = len(tile[0])
    output = [
        [tile[row % row_period][column % column_period] for column in range(width)]
        for row in range(height)
    ]
    if output == grid:
        raise ProgramError("periodic completion made no change")
    return output


def _complete_bbox_symmetry(grid: Grid) -> Grid:
    """Complete reflections, and square-box rotations, around the foreground box."""
    background = _background(grid)
    bounds = _bounding_box(grid, background=background)
    if bounds is None:
        raise ProgramError("bounding-box symmetry needs foreground")
    top, bottom, left, right = bounds
    output = _copy(grid)
    changed = False
    box_height, box_width = bottom - top + 1, right - left + 1
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            if value == background:
                continue
            relative_row, relative_column = row - top, column - left
            targets = {
                (relative_row, relative_column),
                (relative_row, box_width - relative_column - 1),
                (box_height - relative_row - 1, relative_column),
                (box_height - relative_row - 1, box_width - relative_column - 1),
            }
            if box_height == box_width:
                size = box_height
                targets.update(
                    {
                        (relative_column, relative_row),
                        (relative_column, size - relative_row - 1),
                        (size - relative_column - 1, relative_row),
                        (size - relative_column - 1, size - relative_row - 1),
                    }
                )
            for target_row, target_column in targets:
                target_row += top
                target_column += left
                existing = output[target_row][target_column]
                if existing not in {background, value}:
                    raise ProgramError("foreground conflicts with bounding-box symmetry")
                if existing == background:
                    output[target_row][target_column] = value
                    changed = True
    if not changed:
        raise ProgramError("bounding-box symmetry made no change")
    return output


def _mark_closed_square_corners(grid: Grid, *, output_color: int) -> Grid:
    """Mark the two outward neighbors of every closed square corner."""
    background = _background(grid)
    if output_color == background or not 0 <= output_color <= 9:
        raise ProgramError("rectangle marker must be a new non-background color")
    height, width = len(grid), len(grid[0])
    output = _copy(grid)
    changed = False
    foreground_colors = sorted({value for row in grid for value in row} - {background})
    component_sets = [
        set(component)
        for component in connected_components(grid, background=background, connectivity=4)
    ]
    for color in foreground_colors:
        points = {
            (row, column)
            for row, values in enumerate(grid)
            for column, value in enumerate(values)
            if value == color
        }
        rows = sorted({row for row, _ in points})
        columns = sorted({column for _, column in points})
        for top_index, top in enumerate(rows):
            for bottom in rows[top_index + 1 :]:
                for left_index, left in enumerate(columns):
                    for right in columns[left_index + 1 :]:
                        if bottom - top != right - left:
                            continue
                        perimeter = (
                            {(top, column) for column in range(left, right + 1)}
                            | {(bottom, column) for column in range(left, right + 1)}
                            | {(row, left) for row in range(top, bottom + 1)}
                            | {(row, right) for row in range(top, bottom + 1)}
                        )
                        if perimeter not in component_sets or not perimeter <= points:
                            continue
                        markers = {
                            (top - 1, left),
                            (top, left - 1),
                            (top - 1, right),
                            (top, right + 1),
                            (bottom + 1, left),
                            (bottom, left - 1),
                            (bottom + 1, right),
                            (bottom, right + 1),
                        }
                        for row, column in markers:
                            if (
                                0 <= row < height
                                and 0 <= column < width
                                and output[row][column] == background
                            ):
                                output[row][column] = output_color
                                changed = True
    if not changed:
        raise ProgramError("no closed rectangle corners could be marked")
    return output


def apply_program(program: Program, grid: Grid) -> Grid:
    validate_grid(grid)
    _validate_program_args(program)
    op, args = program.op, program.args
    if op == "compose":
        result = _copy(grid)
        if not program.steps:
            raise ProgramError("compose needs at least one step")
        for step in program.steps:
            result = apply_program(step, result)
        return _checked(result)
    if op == "identity":
        return _checked(_copy(grid))
    if op == "rotate":
        turns = int(args.get("turns", 1))
        if turns not in {1, 2, 3}:
            raise ProgramError("rotate turns must be 1, 2, or 3")
        return _checked(_rotate(grid, turns))
    if op == "flip":
        return _checked(_flip(grid, str(args["axis"])))
    if op == "transpose":
        return _checked([list(row) for row in zip(*grid, strict=True)])
    if op == "recolor":
        mapping = {int(key): int(value) for key, value in dict(args["mapping"]).items()}
        if any(not 0 <= cell <= 9 for item in mapping.items() for cell in item):
            raise ProgramError("recolor values must be between 0 and 9")
        return _checked([[mapping.get(cell, cell) for cell in row] for row in grid])
    if op == "crop_background":
        return _checked(_crop(grid, background=int(args.get("background", _background(grid)))))
    if op == "scale":
        rows, columns = int(args["rows"]), int(args["columns"])
        if not 1 <= rows <= 30 or not 1 <= columns <= 30:
            raise ProgramError("scale factors must be between 1 and 30")
        return _checked(
            [[cell for cell in row for _ in range(columns)] for row in grid for _ in range(rows)]
        )
    if op == "tile":
        rows, columns = int(args["rows"]), int(args["columns"])
        if not 1 <= rows <= 30 or not 1 <= columns <= 30:
            raise ProgramError("tile factors must be between 1 and 30")
        wide = [row * columns for row in grid]
        return _checked(wide * rows)
    if op == "pad":
        top = int(args.get("top", 0))
        bottom = int(args.get("bottom", 0))
        left = int(args.get("left", 0))
        right = int(args.get("right", 0))
        color = int(args.get("color", 0))
        if min(top, bottom, left, right) < 0:
            raise ProgramError("pad widths cannot be negative")
        width = left + len(grid[0]) + right
        return _checked(
            [[color] * width for _ in range(top)]
            + [[color] * left + list(row) + [color] * right for row in grid]
            + [[color] * width for _ in range(bottom)]
        )
    if op == "select_component":
        return _checked(
            _select_component(
                grid,
                background=int(args.get("background", _background(grid))),
                criterion=str(args.get("criterion", "largest")),
                connectivity=int(args.get("connectivity", 4)),
                multicolor=bool(args.get("multicolor", False)),
            )
        )
    if op == "translate":
        return _checked(
            _translate(
                grid,
                rows=int(args.get("rows", 0)),
                columns=int(args.get("columns", 0)),
                background=int(args.get("background", _background(grid))),
            )
        )
    if op == "concat":
        return _checked(
            _concat(
                grid,
                axis=str(args["axis"]),
                transform=str(args.get("transform", "identity")),
                order=str(args.get("order", "input_first")),
            )
        )
    if op == "complete_symmetry":
        return _checked(
            _complete_symmetry(
                grid,
                axis=str(args["axis"]),
                background=int(args.get("background", _background(grid))),
            )
        )
    if op == "split_overlay":
        return _checked(
            _split_overlay(
                grid,
                axis=str(args["axis"]),
                mode=str(args["mode"]),
                background=int(args.get("background", _background(grid))),
                separator_color=int(args["separator_color"]),
                output_color=int(args["output_color"]),
            )
        )
    if op == "move_color_to_contact":
        return _checked(
            _move_color_to_contact(
                grid,
                moving_color=int(args["moving_color"]),
                target_color=int(args["target_color"]),
                background=int(args.get("background", _background(grid))),
            )
        )
    if op == "pattern_mosaic":
        return _checked(
            _pattern_mosaic(
                grid,
                background=int(args.get("background", _background(grid))),
                empty_color=int(args["empty_color"]),
                base_color=int(args["base_color"]),
                marker_color=int(args["marker_color"]),
                row_order=[int(value) for value in args["row_order"]],
                column_order=[int(value) for value in args["column_order"]],
                output_rows=int(args["output_rows"]),
                output_columns=int(args["output_columns"]),
            )
        )
    if op == "repair_periodic_panels":
        return _checked(_repair_periodic_panels(grid))
    if op == "select_unique_panel_per_band":
        return _checked(_select_unique_panel_per_band(grid))
    if op == "deduplicate_repeated_half":
        return _checked(_deduplicate_repeated_half(grid))
    if op == "fold_quadrants_overlap":
        return _checked(_fold_quadrants_overlap(grid))
    if op == "mark_most_frequent_above_separator":
        return _checked(_mark_most_frequent_above_separator(grid))
    if op == "keep_most_frequent_colors":
        return _checked(
            _keep_most_frequent_colors(
                grid,
                keep_count=int(args["keep_count"]),
                output_color=int(args["output_color"]),
            )
        )
    if op == "pack_points_serpentine":
        return _checked(_pack_points_serpentine(grid))
    if op == "fill_between_row_markers":
        return _checked(_fill_between_row_markers(grid, fill_color=int(args["fill_color"])))
    if op == "expand_seed_to_width":
        return _checked(_expand_seed_to_width(grid))
    if op == "summarize_nonbackground_count":
        return _checked(_summarize_nonbackground_count(grid))
    if op == "fill_with_most_frequent_color":
        return _checked(_fill_with_most_frequent_color(grid))
    if op == "frequency_histogram_columns":
        return _checked(_frequency_histogram_columns(grid))
    if op == "recolor_singleton_components":
        return _checked(
            _recolor_singleton_components(
                grid,
                background=int(
                    args.get(
                        "background",
                        0 if any(0 in row for row in grid) else _background(grid),
                    )
                ),
            )
        )
    if op == "move_center_block_to_corners":
        return _checked(_move_center_block_to_corners(grid))
    if op == "complete_periodic_pattern":
        return _checked(
            _complete_periodic_pattern(
                grid,
                mode=str(args.get("mode", "tile")),
            )
        )
    if op == "complete_bbox_symmetry":
        return _checked(_complete_bbox_symmetry(grid))
    if op == "mark_closed_square_corners":
        return _checked(
            _mark_closed_square_corners(
                grid,
                output_color=int(args["output_color"]),
            )
        )
    raise ProgramError(f"unsupported operation: {op}")


def verify_program(program: Program, task: ArcTask) -> VerificationResult:
    exact = 0
    pair_scores: list[float] = []
    balanced_scores: list[float] = []
    errors: list[str] = []
    for index, pair in enumerate(task.train):
        try:
            predicted = apply_program(program, pair.input)
            if predicted == pair.output:
                exact += 1
                pair_scores.append(1.0)
                balanced_scores.append(1.0)
            else:
                expected_shape = (len(pair.output or []), len((pair.output or [[]])[0]))
                actual_shape = (len(predicted), len(predicted[0]))
                differing = (
                    sum(
                        left != right
                        for expected_row, actual_row in zip(
                            pair.output or [], predicted, strict=True
                        )
                        for left, right in zip(expected_row, actual_row, strict=True)
                    )
                    if expected_shape == actual_shape
                    else None
                )
                raw_accuracy, balanced_accuracy = grid_accuracies(
                    predicted, pair.output or [[]]
                )
                pair_scores.append(raw_accuracy)
                balanced_scores.append(balanced_accuracy)
                errors.append(
                    f"pair {index}: expected shape {expected_shape}, got {actual_shape}; "
                    f"differing_cells={differing}"
                )
        except Exception as exc:  # verifier must turn candidate failures into feedback
            pair_scores.append(0.0)
            balanced_scores.append(0.0)
            errors.append(f"pair {index}: {type(exc).__name__}: {exc}")
    return VerificationResult(
        exact_pairs=exact,
        total_pairs=len(task.train),
        cell_accuracy=sum(pair_scores) / len(pair_scores),
        balanced_accuracy=sum(balanced_scores) / len(balanced_scores),
        errors=errors,
    )


def _infer_recolor(inputs: Iterable[Grid], outputs: Iterable[Grid]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for input_grid, output_grid in zip(inputs, outputs, strict=True):
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        for input_row, output_row in zip(input_grid, output_grid, strict=True):
            for source, target in zip(input_row, output_row, strict=True):
                previous = mapping.setdefault(source, target)
                if previous != target:
                    return None
    return mapping


def _program_key(program: Program) -> str:
    return json.dumps(program.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _translation_programs(task: ArcTask) -> list[Program]:
    first = task.train[0]
    if first.output is None or (
        len(first.input) != len(first.output) or len(first.input[0]) != len(first.output[0])
    ):
        return []
    programs: list[Program] = []
    backgrounds = sorted({_background(pair.input) for pair in task.train})
    output_background = _background(first.output)
    for background in backgrounds:
        input_bounds = _bounding_box(first.input, background=background)
        output_bounds = _bounding_box(first.output, background=output_background)
        if input_bounds is None or output_bounds is None:
            continue
        row_delta = output_bounds[0] - input_bounds[0]
        column_delta = output_bounds[2] - input_bounds[2]
        if row_delta or column_delta:
            programs.append(
                Program(
                    op="translate",
                    args={
                        "rows": row_delta,
                        "columns": column_delta,
                        "background": background,
                    },
                )
            )
    return programs


def _split_overlay_programs(task: ArcTask) -> list[Program]:
    first = task.train[0]
    if first.output is None:
        return []
    grid = first.input
    output_colors = sorted({value for row in first.output for value in row})
    backgrounds = sorted({_background(pair.input) for pair in task.train})
    programs: list[Program] = []
    for axis in ("horizontal", "vertical"):
        if axis == "horizontal":
            separator_values = (
                {backgrounds[0]}
                if len(grid) % 2 == 0
                else set(grid[len(grid) // 2])
            )
        else:
            separator_values = (
                {backgrounds[0]}
                if len(grid[0]) % 2 == 0
                else {row[len(grid[0]) // 2] for row in grid}
            )
        if len(separator_values) != 1:
            continue
        separator_color = next(iter(separator_values))
        for background in backgrounds:
            for mode in ("union", "intersection", "xor", "neither"):
                for output_color in output_colors:
                    if output_color != background:
                        programs.append(
                            Program(
                                op="split_overlay",
                                args={
                                    "axis": axis,
                                    "mode": mode,
                                    "background": background,
                                    "separator_color": separator_color,
                                    "output_color": output_color,
                                },
                            )
                        )
    return programs


def _pattern_mosaic_programs(task: ArcTask) -> list[Program]:
    first = task.train[0]
    if first.output is None:
        return []
    background = _background(first.input)
    symbol = _crop(first.input, background=background)
    symbol_rows, symbol_columns = len(symbol), len(symbol[0])
    output_rows, output_columns = len(first.output), len(first.output[0])
    output_colors = sorted({value for row in first.output for value in row})
    if (
        not (2 <= symbol_rows <= 3 and 2 <= symbol_columns <= 3)
        or len({value for row in first.input for value in row}) != 2
        or len(output_colors) != 3
        or len(first.input) % symbol_rows
        or len(first.input[0]) % symbol_columns
        or output_rows < len(first.input)
        or output_columns < len(first.input[0])
    ):
        return []
    return [
        Program(
            op="pattern_mosaic",
            args={
                "background": background,
                "empty_color": empty_color,
                "base_color": base_color,
                "marker_color": marker_color,
                "row_order": list(row_order),
                "column_order": list(column_order),
                "output_rows": output_rows,
                "output_columns": output_columns,
            },
        )
        for row_order in permutations(range(symbol_rows))
        for column_order in permutations(range(symbol_columns))
        for empty_color, base_color, marker_color in permutations(output_colors)
    ]


def _base_programs(task: ArcTask) -> list[Program]:
    programs = [
        Program(op="identity"),
        Program(op="repair_periodic_panels"),
        Program(op="select_unique_panel_per_band"),
        Program(op="deduplicate_repeated_half"),
        Program(op="fold_quadrants_overlap"),
        Program(op="mark_most_frequent_above_separator"),
        Program(op="pack_points_serpentine"),
        Program(op="expand_seed_to_width"),
        Program(op="summarize_nonbackground_count"),
        Program(op="fill_with_most_frequent_color"),
        Program(op="frequency_histogram_columns"),
        Program(op="move_center_block_to_corners"),
        Program(op="complete_periodic_pattern", args={"mode": "tile"}),
        Program(op="complete_periodic_pattern", args={"mode": "diagonal_sum"}),
        Program(op="complete_periodic_pattern", args={"mode": "diagonal_difference"}),
        Program(op="complete_bbox_symmetry"),
        Program(
            op="compose",
            steps=[
                Program(
                    op="concat",
                    args={
                        "axis": "horizontal",
                        "transform": "flip_horizontal",
                        "order": "input_first",
                    },
                ),
                Program(
                    op="concat",
                    args={
                        "axis": "vertical",
                        "transform": "flip_vertical",
                        "order": "input_first",
                    },
                ),
            ],
        ),
        Program(op="rotate", args={"turns": 1}),
        Program(op="rotate", args={"turns": 2}),
        Program(op="rotate", args={"turns": 3}),
        Program(op="flip", args={"axis": "horizontal"}),
        Program(op="flip", args={"axis": "vertical"}),
        Program(op="transpose"),
    ]
    rigid_programs = [
        program for program in programs if program.op in {"rotate", "flip", "transpose"}
    ]
    first_output_colors = sorted(
        {value for row in (task.train[0].output or []) for value in row}
    )
    programs.extend(
        Program(
            op="keep_most_frequent_colors",
            args={"keep_count": keep_count, "output_color": output_color},
        )
        for keep_count in (1, 2, 3)
        for output_color in first_output_colors
    )
    programs.extend(
        Program(op="fill_between_row_markers", args={"fill_color": output_color})
        for output_color in first_output_colors
    )
    programs.extend(
        Program(op="mark_closed_square_corners", args={"output_color": output_color})
        for output_color in first_output_colors
    )
    input_colors = sorted({value for pair in task.train for row in pair.input for value in row})
    programs.extend(
        Program(op="recolor_singleton_components", args={"background": background})
        for background in input_colors
    )
    extraction_programs: list[Program] = []
    backgrounds = sorted({_background(pair.input) for pair in task.train})
    for background in backgrounds:
        extraction_programs.append(Program(op="crop_background", args={"background": background}))
        for connectivity in (4, 8):
            for multicolor in (False, True):
                for criterion in ("largest", "smallest", "widest", "tallest"):
                    extraction_programs.append(
                        Program(
                            op="select_component",
                            args={
                                "background": background,
                                "criterion": criterion,
                                "connectivity": connectivity,
                                "multicolor": multicolor,
                            },
                        )
                    )
        programs.extend(
            Program(
                op="complete_symmetry",
                args={"axis": axis, "background": background},
            )
            for axis in ("horizontal", "vertical", "rotational")
        )
        first_colors = sorted(
            {value for row in task.train[0].input for value in row if value != background}
        )
        programs.extend(
            Program(
                op="move_color_to_contact",
                args={
                    "moving_color": moving_color,
                    "target_color": target_color,
                    "background": background,
                },
            )
            for moving_color in first_colors
            for target_color in first_colors
            if moving_color != target_color
        )
    programs.extend(extraction_programs)
    programs.extend(
        Program(op="compose", steps=[extraction, rigid])
        for extraction in extraction_programs
        for rigid in rigid_programs
    )
    programs.extend(_translation_programs(task))
    programs.extend(_split_overlay_programs(task))
    programs.extend(_pattern_mosaic_programs(task))
    programs.extend(
        Program(
            op="concat",
            args={"axis": axis, "transform": transform, "order": order},
        )
        for axis in ("horizontal", "vertical")
        for transform in (
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_horizontal",
            "flip_vertical",
            "transpose",
        )
        for order in ("input_first", "transformed_first")
    )
    first_input, first_output = task.train[0].input, task.train[0].output
    if first_output is not None:
        input_height, input_width = len(first_input), len(first_input[0])
        output_height, output_width = len(first_output), len(first_output[0])
        if output_height % input_height == 0 and output_width % input_width == 0:
            row_factor, column_factor = output_height // input_height, output_width // input_width
            if row_factor > 1 or column_factor > 1:
                programs.extend(
                    [
                        Program(op="scale", args={"rows": row_factor, "columns": column_factor}),
                        Program(op="tile", args={"rows": row_factor, "columns": column_factor}),
                    ]
                )
        if output_height >= input_height and output_width >= input_width:
            for top in range(output_height - input_height + 1):
                for left in range(output_width - input_width + 1):
                    bottom = output_height - input_height - top
                    right = output_width - input_width - left
                    for color in sorted({cell for row in first_output for cell in row}):
                        programs.append(
                            Program(
                                op="pad",
                                args={
                                    "top": top,
                                    "bottom": bottom,
                                    "left": left,
                                    "right": right,
                                    "color": color,
                                },
                            )
                        )
    return programs


def search_programs(task: ArcTask, *, limit: int = 16) -> list[Candidate]:
    """Search compact single-step and geometry+recolor programs."""
    programs = _base_programs(task)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    expected = [pair.output or [] for pair in task.train]
    expected_first_shape = (len(expected[0]), len(expected[0][0]))

    def add_if_exact(program: Program, transformed: list[Grid]) -> None:
        key = _program_key(program)
        if key in seen:
            return
        seen.add(key)
        if transformed != expected:
            return
        try:
            predictions = [apply_program(program, pair.input) for pair in task.test]
        except (ProgramError, ValueError):
            return
        candidates.append(
            Candidate(
                candidate_id=f"det-{len(candidates) + 1}",
                source="deterministic",
                hypothesis=program.op,
                program=program,
                predictions=predictions,
                verified=True,
                exact_train_pairs=len(task.train),
                train_pairs=len(task.train),
                train_cell_accuracy=1.0,
                train_balanced_accuracy=1.0,
                score=100.0 - program.complexity(),
            )
        )

    for base in programs:
        try:
            transformed = [apply_program(base, pair.input) for pair in task.train]
        except (ProgramError, ValueError):
            continue
        first_shape = (len(transformed[0]), len(transformed[0][0]))
        if first_shape != expected_first_shape:
            continue
        add_if_exact(base, transformed)
        mapping = _infer_recolor(transformed, expected)
        if mapping and any(source != target for source, target in mapping.items()):
            recolor = Program(op="recolor", args={"mapping": mapping})
            combined = (
                recolor if base.op == "identity" else Program(op="compose", steps=[base, recolor])
            )
            recolored = [
                [[mapping.get(value, value) for value in row] for row in grid]
                for grid in transformed
            ]
            add_if_exact(combined, recolored)
    return sorted(candidates, key=lambda item: (-item.score, item.candidate_id))[:limit]


def program_family(program: Program) -> str:
    ops = [step.op for step in program.steps] if program.op == "compose" else [program.op]
    if any(op in {"select_component", "crop_background"} for op in ops):
        return "object-selection"
    if any(op in {"scale", "tile", "pad", "concat"} for op in ops):
        return "size-and-repetition"
    if "pattern_mosaic" in ops:
        return "symbolic-composition"
    if "repair_periodic_panels" in ops:
        return "pattern-repair"
    if "select_unique_panel_per_band" in ops:
        return "panel-selection"
    if "deduplicate_repeated_half" in ops:
        return "repetition-compression"
    if "fold_quadrants_overlap" in ops:
        return "logical-composition"
    if "mark_most_frequent_above_separator" in ops:
        return "counting-and-selection"
    if "keep_most_frequent_colors" in ops:
        return "color-mapping"
    if "pack_points_serpentine" in ops:
        return "object-arrangement"
    if "fill_between_row_markers" in ops:
        return "relational-drawing"
    if "expand_seed_to_width" in ops:
        return "size-and-repetition"
    if "summarize_nonbackground_count" in ops:
        return "counting-and-selection"
    if "fill_with_most_frequent_color" in ops:
        return "counting-and-selection"
    if "frequency_histogram_columns" in ops:
        return "counting-and-selection"
    if "recolor_singleton_components" in ops:
        return "object-selection"
    if "move_center_block_to_corners" in ops:
        return "relational-placement"
    if "complete_periodic_pattern" in ops:
        return "pattern-completion"
    if "complete_bbox_symmetry" in ops:
        return "spatial-transform"
    if "mark_closed_square_corners" in ops:
        return "relational-drawing"
    if "split_overlay" in ops:
        return "logical-composition"
    if "move_color_to_contact" in ops:
        return "relational-placement"
    if "recolor" in ops:
        return "color-mapping"
    if any(op in {"rotate", "flip", "transpose", "translate", "complete_symmetry"} for op in ops):
        return "spatial-transform"
    return "identity"
