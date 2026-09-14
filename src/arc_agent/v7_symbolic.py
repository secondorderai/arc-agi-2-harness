"""Structured symbolic SFT with a separate, label-isolated Python witness.

Bindings establish static links to reachable functions, not semantic equivalence or
dynamic execution order. Neither the witness nor the withheld labels are SFT targets.
"""

from __future__ import annotations

import ast
import json

from pydantic import Field

from arc_agent.models import ArcTask
from arc_agent.sandbox import ALLOWED_METHODS, SAFE_BUILTINS
from arc_agent.v2_sandbox import run_induction_program, validate_induction_source
from arc_agent.v4_config import content_hash
from arc_agent.v4_tools import task_grids
from arc_agent.v5_symbolic import StrictModel, SymbolicModel, validate_symbols


class ActionBinding(StrictModel):
    action_id: str = Field(min_length=1, max_length=64)
    function: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")


class PythonWitness(StrictModel):
    hypothesis_id: str = Field(min_length=1, max_length=64)
    python_source: str = Field(min_length=1, max_length=20000)
    action_bindings: list[ActionBinding] = Field(min_length=1, max_length=16)


class BridgedArtifact(StrictModel):
    symbolic: SymbolicModel
    witness: PythonWitness | None


def output_schema() -> dict:
    schema = BridgedArtifact.model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return schema


WITNESS_CONTRACT = f"""Return a SymbolicModel and an optional separate Python witness.
The witness has hypothesis_id, python_source, action_bindings. It defines solve(train, grid)
returning an integer list-of-lists grid. train contains visible (input_grid, output_grid) pairs
ONLY, including when the checker evaluates an unseen input. Never rely on its withheld output.
For each ordered_actions definition ID, provide one action_binding (action_id, function), in
that order. The named top-level function must be statically reachable by calls from solve;
solve itself is allowed. Bindings link artifacts, not proofs of free-form meaning or order.
Use general loops, conditionals, comprehensions, arithmetic and top-level helper functions.
No imports, classes, lambdas, exceptions, global/nonlocal, with, async, decorators, default
arguments, dunder access, IO, eval/exec, reflection, third-party packages or grid lookup tables.
Allowed builtins: {", ".join(sorted(SAFE_BUILTINS))}.
Allowed methods: {", ".join(sorted(ALLOWED_METHODS))}.
Derive task-dependent values from train/grid. Do not embed observed grids, task identifiers
or large literal collections. Only function definitions are allowed at module scope.
The witness is verification evidence only; never put its source or final predictions inside
symbolic fields. If no credible witness is possible, use null and retain explicit uncertainty.
"""


def validate_bindings(artifact: BridgedArtifact, visible: ArcTask) -> None:
    witness = artifact.witness
    if witness is None:
        raise ValueError("no executable witness; grounding-only research artifact")
    hypothesis = next(
        (h for h in artifact.symbolic.hypotheses if h.id == witness.hypothesis_id), None
    )
    if hypothesis is None or hypothesis.status != "proposed":
        raise ValueError("witness must reference a proposed hypothesis")
    if [b.action_id for b in witness.action_bindings] != hypothesis.ordered_actions:
        raise ValueError("bindings must match all ordered actions in order")
    source = witness.python_source
    validate_induction_source(
        source,
        forbidden_grids=task_grids(visible).values(),
        forbidden_identifiers=[visible.task_id],
    )
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if len(functions) != len(tree.body):
        raise ValueError("only uniquely named function definitions allowed at module scope")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (
            node not in tree.body
            or node.decorator_list
            or node.args.defaults
            or any(v is not None for v in node.args.kw_defaults)
            or node.returns is not None
            or any(
                a.annotation is not None
                for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            )
        ):
            raise ValueError(
                "helpers must be top-level without decorators, defaults or annotations"
            )
    reachable, pending = set(), ["solve"]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(
            node.func.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
        )
    if any(binding.function not in reachable for binding in witness.action_bindings):
        raise ValueError("action binding references an undefined or unreachable function")


def verify_artifact(artifact: BridgedArtifact, visible: ArcTask, heldout: ArcTask) -> dict:
    """Check demos, then the sealed demo; never give its output to solve(train, grid)."""
    result = {
        "grounding_valid": False,
        "visible_verified": False,
        "heldout_verified": False,
        "accepted": False,
        "free_form_semantics_verified": False,
        "action_order_execution_verified": False,
        "witness_backend": "bounded_python",
        "binding_validation": None,
    }
    try:
        if any(p.output is not None for p in visible.test):
            raise ValueError("visible task must not contain test labels")
        if len(heldout.train) != 1 or heldout.train[0].output is None:
            raise ValueError("exactly one labelled demonstration must be withheld")
        validate_symbols(artifact.symbolic, visible)
        result["grounding_valid"] = True
        validate_bindings(artifact, visible)
        witness = artifact.witness
        result["binding_validation"] = "static_reachable_action_functions"
        result["witness_hash"] = content_hash(witness.model_dump(mode="json"))
        # No hidden pair is ever in this list. Each sandbox process gets a fresh JSON copy.
        train = [(p.input, p.output) for p in visible.train]
        predictions, failures = [], []
        for index, pair in enumerate(visible.train):
            executed = run_induction_program(
                witness.python_source, train, pair.input, validate_source=False
            )
            predictions.append(executed.grid)
            if executed.error or executed.grid != pair.output:
                failures.append(
                    {
                        "case": f"train_{index}",
                        "actual": executed.grid,
                        "expected": pair.output,
                        "error": executed.error,
                    }
                )
        result["visible_verified"] = bool(train) and not failures
        result["visible_feedback"] = {
            "exact_train_pairs": len(train) - len(failures),
            "train_pairs": len(train),
            "verified": result["visible_verified"],
            "train_predictions": predictions,
            "failures": failures,
        }
        if result["visible_verified"]:
            sealed = heldout.train[0]
            executed = run_induction_program(
                witness.python_source, train, sealed.input, validate_source=False
            )
            result["heldout_verified"] = executed.error is None and executed.grid == sealed.output
            result["accepted"] = result["heldout_verified"]
            # Sealed outcomes are audit-only. The collector must stop, never repair from them.
            result["heldout_execution_error"] = executed.error
    except (ValueError, TypeError, KeyError, TimeoutError) as exc:
        result["error"] = str(exc)[:1200]
    return result


class PythonTrainingContract:
    artifact_type = BridgedArtifact
    verify = staticmethod(verify_artifact)

    @staticmethod
    def initial_prompt(messages):
        return json.dumps(
            {"evidence": json.loads(messages[-1]["content"]), "contract": WITNESS_CONTRACT},
            separators=(",", ":"),
        )
