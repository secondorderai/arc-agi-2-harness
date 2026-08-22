from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from arc_agent.models import Grid, validate_grid
from arc_agent.sandbox import (
    ALLOWED_METHODS,
    DISALLOWED_NODES,
    SAFE_BUILTINS,
    UnsafeProgram,
)


def _scalar_count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return sum(_scalar_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_scalar_count(key) + _scalar_count(item) for key, item in value.items())
    return 1


def _grid_key(value: object) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for raw_row in value:
        if not isinstance(raw_row, (list, tuple)) or not raw_row:
            return None
        if any(
            not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9
            for cell in raw_row
        ):
            return None
        row = tuple(raw_row)
        width = len(row) if width is None else width
        if len(row) != width:
            return None
        rows.append(row)
    return tuple(rows)


def canonicalize_source(source: str) -> tuple[str, int]:
    tree = ast.parse(source)
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    complexity = sum(1 for _ in ast.walk(tree))
    return canonical, complexity


def validate_induction_source(
    source: str,
    *,
    forbidden_grids: Iterable[Grid] = (),
    forbidden_identifiers: Iterable[str] = (),
    max_source_chars: int = 20_000,
    max_literal_scalars: int = 64,
) -> None:
    if len(source) > max_source_chars:
        raise UnsafeProgram("program is too large")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeProgram(str(exc)) from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    solve_functions = [node for node in functions if node.name == "solve"]
    if len(solve_functions) != 1:
        raise UnsafeProgram("program must define exactly one solve(train, grid) function")
    args = solve_functions[0].args
    if len(args.args) != 2 or args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
        raise UnsafeProgram("solve must accept exactly two positional arguments")
    if [argument.arg for argument in args.args] != ["train", "grid"]:
        raise UnsafeProgram("solve arguments must be named train and grid")

    function_names = {node.name for node in functions}
    forbidden = {_grid_key(grid) for grid in forbidden_grids}
    forbidden.discard(None)
    forbidden_text = {value for value in forbidden_identifiers if value}
    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODES):
            raise UnsafeProgram(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeProgram("dunder names are not allowed")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr not in ALLOWED_METHODS
        ):
            raise UnsafeProgram(f"method is not allowed: {node.attr}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in SAFE_BUILTINS | function_names
        ):
            raise UnsafeProgram(f"call is not allowed: {node.func.id}")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(value in node.value for value in forbidden_text)
        ):
            raise UnsafeProgram("task identifier is embedded in the program")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            try:
                literal = ast.literal_eval(node)
            except (ValueError, TypeError):
                literal = None
            if literal is not None:
                if _scalar_count(literal) > max_literal_scalars:
                    raise UnsafeProgram("literal collection is too large")
                if _grid_key(literal) in forbidden:
                    raise UnsafeProgram("source task grid is embedded in the program")
        if isinstance(node, ast.Compare):
            expressions = [node.left, *node.comparators]
            has_task_value = any(
                isinstance(item, ast.Name) and item.id in {"train", "grid"}
                for expression in expressions
                for item in ast.walk(expression)
            )
            has_literal_collection = any(
                isinstance(expression, (ast.List, ast.Tuple, ast.Set, ast.Dict))
                for expression in expressions
            )
            if has_task_value and has_literal_collection:
                raise UnsafeProgram("task inputs may not be compared with literal collections")


_INDUCTION_RUNNER = r"""
import json, resource, sys
payload = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
try:
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
except (ValueError, OSError):
    pass
safe = {name: getattr(__builtins__, name) for name in (
    'abs','all','any','bool','dict','enumerate','float','int','len','list','map','max','min',
    'range','reversed','set','sorted','sum','tuple','zip'
)}
namespace = {'__builtins__': safe}
exec(compile(payload['source'], '<candidate>', 'exec'), namespace, namespace)
train = [(item[0], item[1]) for item in payload['train']]
result = namespace['solve'](train, payload['grid'])
sys.stdout.write(json.dumps(result))
"""


@dataclass(frozen=True)
class InductionSandboxResult:
    grid: Grid | None
    error: str | None


def run_induction_program(
    source: str,
    train: Sequence[tuple[Grid, Grid]],
    grid: Grid,
    *,
    timeout_seconds: float = 3.0,
    validate_source: bool = True,
    forbidden_grids: Iterable[Grid] = (),
    forbidden_identifiers: Iterable[str] = (),
    max_source_chars: int = 20_000,
    max_literal_scalars: int = 64,
) -> InductionSandboxResult:
    if validate_source:
        try:
            validate_induction_source(
                source,
                forbidden_grids=forbidden_grids,
                forbidden_identifiers=forbidden_identifiers,
                max_source_chars=max_source_chars,
                max_literal_scalars=max_literal_scalars,
            )
        except UnsafeProgram as exc:
            return InductionSandboxResult(grid=None, error=str(exc))
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _INDUCTION_RUNNER],
            input=json.dumps({"source": source, "train": train, "grid": grid}),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return InductionSandboxResult(grid=None, error="program timed out")
    if completed.returncode != 0:
        return InductionSandboxResult(
            grid=None, error=completed.stderr.strip()[-1_000:] or "program failed"
        )
    try:
        result = json.loads(completed.stdout)
        validate_grid(result)
        return InductionSandboxResult(grid=result, error=None)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return InductionSandboxResult(grid=None, error=f"invalid program output: {exc}")
