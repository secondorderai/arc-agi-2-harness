from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass

from arc_agent.models import Grid, validate_grid

DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Lambda,
)
ALLOWED_METHODS = {
    "add",
    "append",
    "copy",
    "count",
    "discard",
    "extend",
    "get",
    "index",
    "insert",
    "intersection",
    "items",
    "keys",
    "pop",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "union",
    "update",
    "values",
}

SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "set",
    "sorted",
    "sum",
    "tuple",
    "zip",
}


class UnsafeProgram(ValueError):
    pass


def validate_python_source(source: str) -> None:
    if len(source) > 20_000:
        raise UnsafeProgram("program is too large")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeProgram(str(exc)) from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    solve_functions = [node for node in functions if node.name == "solve"]
    if len(solve_functions) != 1:
        raise UnsafeProgram("program must define exactly one solve(grid) function")
    if len(solve_functions[0].args.args) != 1:
        raise UnsafeProgram("solve must accept exactly one argument")
    function_names = {node.name for node in functions}
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


_RUNNER = r"""
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
result = namespace['solve'](payload['grid'])
sys.stdout.write(json.dumps(result))
"""


@dataclass(frozen=True)
class SandboxResult:
    grid: Grid | None
    error: str | None


def run_python_program(source: str, grid: Grid, *, timeout_seconds: float = 3.0) -> SandboxResult:
    try:
        validate_python_source(source)
    except UnsafeProgram as exc:
        return SandboxResult(grid=None, error=str(exc))
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _RUNNER],
            input=json.dumps({"source": source, "grid": grid}),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(grid=None, error="program timed out")
    if completed.returncode != 0:
        return SandboxResult(grid=None, error=completed.stderr.strip()[-1000:] or "program failed")
    try:
        result = json.loads(completed.stdout)
        validate_grid(result)
        return SandboxResult(grid=result, error=None)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return SandboxResult(grid=None, error=f"invalid program output: {exc}")
