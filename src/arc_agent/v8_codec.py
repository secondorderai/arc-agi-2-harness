"""Lossless symbolic wire encoding; canonical states and teacher targets stay unchanged."""

from __future__ import annotations

import copy

from arc_agent.v5_symbolic import SymbolicModel

CELL_ENCODINGS = {"named_fields", "triples_v1"}
TRIPLES_FORMAT = "symbolic_cell_triples_v1"
TRIPLES_INSTRUCTION = (
    " For this response, encode each Entity.cells entry as exactly [row,column,color],"
    " three integers in that order, instead of an object with named fields. For example,"
    " cells:[[0,1,2]] means row 0, column 1, observed color 2."
    " Existing evidence or prior states may use named cell fields; their meaning is unchanged."
    " Keep all other symbolic fields and references unchanged in format."
    " Select representative observations; the raw grids remain authoritative and accessible."
)


def pack_symbolic_cells(state: dict) -> dict:
    SymbolicModel.model_validate(state)
    result = copy.deepcopy(state)
    for entity in result["entities"]:
        entity["cells"] = [[c["row"], c["column"], c["color"]] for c in entity["cells"]]
    return result


def unpack_symbolic_cells(state: dict) -> dict:
    result = copy.deepcopy(state)
    if not isinstance(result, dict) or not isinstance(result.get("entities"), list):
        raise ValueError("packed state requires entities")
    for entity in result["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("cells"), list):
            raise ValueError("packed entity requires cells")
        cells = []
        for cell in entity["cells"]:
            if (
                not isinstance(cell, list)
                or len(cell) != 3
                or any(type(value) is not int for value in cell)
            ):
                raise ValueError("packed cells require exactly three integers")
            cells.append(dict(zip(("row", "column", "color"), cell, strict=True)))
        entity["cells"] = cells
    SymbolicModel.model_validate(result)
    return result


def packed_symbolic_schema(schema: dict) -> dict:
    """Replace only cell records; preserve exact row/column/color and collection bounds."""
    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and set(node.get("properties", {})) == {
                "row",
                "column",
                "color",
            }:
                fields = node["properties"]
                node.clear()
                node.update(
                    type="array",
                    prefixItems=[fields[key] for key in ("row", "column", "color")],
                    minItems=3,
                    maxItems=3,
                )
            else:
                for value in node.values():
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result
