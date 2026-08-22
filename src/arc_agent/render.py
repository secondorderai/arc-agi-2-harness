from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from arc_agent.models import Grid

ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]


def render_grid(grid: Grid, output: str | Path, *, cell_size: int = 24) -> Path:
    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width * cell_size + 1, height * cell_size + 1), "#333333")
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            left, top = column * cell_size, row * cell_size
            draw.rectangle(
                (left + 1, top + 1, left + cell_size - 1, top + cell_size - 1),
                fill=ARC_COLORS[value],
            )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
