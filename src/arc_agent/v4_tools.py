"""Small, deterministic tools; no generated Python or arbitrary process execution."""

from __future__ import annotations

import json
import subprocess
import sys

from arc_agent.dsl import _ALLOWED_ARGS, apply_program
from arc_agent.models import ArcTask, Program, validate_grid
from arc_agent.object_ir import extract_scene

CORE_OPS = {
    "identity",
    "rotate",
    "flip",
    "transpose",
    "recolor",
    "crop_background",
    "scale",
    "tile",
    "pad",
    "select_component",
    "translate",
    "concat",
    "complete_symmetry",
    "compose",
}


def program_catalog() -> dict:
    return {name: sorted(_ALLOWED_ARGS[name]) for name in sorted(CORE_OPS)}


def validate_program(payload: dict) -> Program:
    if len(json.dumps(payload)) > 24000:
        raise ValueError("program too large")
    program = Program.model_validate(payload)
    count = 0

    def visit(node, depth=0):
        nonlocal count
        count += 1
        if count > 16 or depth > 8 or node.op not in CORE_OPS:
            raise ValueError("undefined operator or oversized program")
        if set(node.args) - _ALLOWED_ARGS[node.op]:
            raise ValueError("undefined program argument")
        if node.steps and node.op != "compose":
            raise ValueError("only compose may contain steps")
        if node.op == "compose" and not node.steps:
            raise ValueError("empty composition")
        for argument, value in node.args.items():
            if isinstance(value, bool) and argument != "multicolor":
                raise ValueError("boolean not allowed for a numeric argument")
            if type(value) is int and abs(value) > 30:
                raise ValueError("integer argument outside bounded grid domain")
            if not isinstance(value, (str, int, dict, bool)):
                raise ValueError("unsupported argument value")
            if isinstance(value, dict) and (
                len(value) > 10
                or any(
                    str(k) not in "0123456789"
                    or len(str(k)) != 1
                    or type(v) is not int
                    or not 0 <= v <= 9
                    for k, v in value.items()
                )
            ):
                raise ValueError("invalid color mapping")
        for child in node.steps:
            visit(child, depth + 1)

    visit(program)
    return program


def execute_program(task: ArcTask, payload: dict) -> dict:
    program = validate_program(payload)
    predicted = [apply_program(program, pair.input) for pair in task.train]
    exact = [actual == pair.output for actual, pair in zip(predicted, task.train, strict=True)]
    tests = [apply_program(program, pair.input) for pair in task.test]
    for grid in predicted + tests:
        validate_grid(grid)
    return {
        "exact_train_pairs": sum(exact),
        "train_pairs": len(exact),
        "verified": all(exact),
        "train_predictions": predicted,
        "test_predictions": tests,
        "complexity": program.complexity(),
    }


def run_program(task: ArcTask, payload: dict, timeout: float = 3) -> dict:
    validate_program(payload)
    process = subprocess.run(
        [sys.executable, "-m", "arc_agent.v4_tools"],
        input=json.dumps({"task": task.model_dump(mode="json"), "program": payload}),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if process.returncode:
        raise ValueError(process.stderr[-1000:] or "program execution failed")
    result = json.loads(process.stdout)
    if "error" in result:
        raise ValueError(result["error"])
    return result


def task_grids(task: ArcTask) -> dict:
    grids = {}
    for index, pair in enumerate(task.train):
        grids[f"train_{index}_input"] = pair.input
        grids[f"train_{index}_output"] = pair.output
    for index, pair in enumerate(task.test):
        grids[f"test_{index}_input"] = pair.input
    return grids


def tool_schema() -> list[dict]:
    definitions = [
        (
            "read_grid",
            "Read rows of an authoritative grid. Use listed grid IDs.",
            {
                "grid_id": {"type": "string"},
                "start_row": {"type": "integer"},
                "row_count": {"type": "integer"},
            },
            ["grid_id"],
        ),
        (
            "inspect_scene",
            "Extract objects deterministically; interpretation may be revised.",
            {
                "grid_id": {"type": "string"},
                "connectivity": {"type": "integer", "enum": [4, 8]},
                "multicolor": {"type": "boolean"},
            },
            ["grid_id"],
        ),
        (
            "run_program",
            "Execute a bounded DSL program on demonstrations and test inputs.",
            {"program": {"type": "object"}},
            ["program"],
        ),
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        for name, description, properties, required in definitions
    ]


def invoke_tool(task: ArcTask, call: dict, *, timeout: float = 3) -> dict:
    name, args = call["name"], call["arguments"]
    if name == "run_program":
        if set(args) != {"program"}:
            raise ValueError("run_program requires exactly program")
        return run_program(task, args["program"], timeout=timeout)
    if name not in {"read_grid", "inspect_scene"}:
        raise ValueError("undefined tool")
    allowed = (
        {"grid_id", "start_row", "row_count"}
        if name == "read_grid"
        else {"grid_id", "connectivity", "multicolor"}
    )
    if set(args) - allowed:
        raise ValueError("undefined tool argument")
    grid = task_grids(task)[args["grid_id"]]
    if name == "read_grid":
        start, count = args.get("start_row", 0), args.get("row_count", 10)
        if (
            type(start) is not int
            or type(count) is not int
            or not 0 <= start < len(grid)
            or not 1 <= count <= 10
        ):
            raise ValueError("read_grid requires valid start_row and 1..10 rows")
        return {
            "grid_id": args["grid_id"],
            "shape": [len(grid), len(grid[0])],
            "start_row": start,
            "rows": ["".join(map(str, r)) for r in grid[start : start + count]],
            "next_row": start + count if start + count < len(grid) else None,
        }
    if args.get("connectivity", 4) not in {4, 8} or type(args.get("multicolor", False)) is not bool:
        raise ValueError("invalid segmentation options")
    scene = extract_scene(
        grid, connectivity=args.get("connectivity", 4), multicolor=args.get("multicolor", False)
    )
    objects = [{"bbox": obj.bbox.as_tuple, "area": obj.area} for obj in scene.objects]
    return {
        "objects": objects[:24],
        "object_count": len(objects),
        "truncated": len(objects) > 24,
        "note": "Object extraction is one interpretation, not ground-truth semantics.",
    }


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(
            json.dumps(execute_program(ArcTask.model_validate(request["task"]), request["program"]))
        )
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
