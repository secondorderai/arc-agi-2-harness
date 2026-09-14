"""Nanbeige-only explicit symbolic state followed by a separate executable witness."""

from __future__ import annotations

import copy
import json

from pydantic import Field

from arc_agent.models import ArcTask
from arc_agent.sandbox import ALLOWED_METHODS, SAFE_BUILTINS
from arc_agent.v4_adaptive import AdaptiveSession
from arc_agent.v4_experiment import decode_grid
from arc_agent.v4_reasoner import ContextOverflow, LocalReasoner
from arc_agent.v4_state import blind_task
from arc_agent.v4_tools import task_grids
from arc_agent.v5_pipeline import student_prompt
from arc_agent.v5_symbolic import StrictModel, SymbolicModel
from arc_agent.v8_codec import (
    CELL_ENCODINGS,
    TRIPLES_FORMAT,
    TRIPLES_INSTRUCTION,
    packed_symbolic_schema,
)
from arc_agent.v8_grammar import (
    BOUNDED_WS_STYLE,
    COMPACT_STYLES,
    IDENTIFIER_STYLES,
    REFERENCE_STYLES,
    bounded_json_whitespace,
    compact_symbolic_grammar,
    converter_path,
    format_binding,
    identifier_schema,
    rectangular_direct_grammar,
)
from arc_agent.v8_reference_repair import prepare_reference_repair

GRID_ENCODING = "Each grid is a list of digit-row strings; row/column coordinates are zero-based."


def compact_symbolic_prompt(messages: list[dict]) -> list[dict]:
    """An invertible evidence encoding, not model compaction or evidence truncation."""
    result = copy.deepcopy(messages)
    evidence = json.loads(result[-1]["content"])
    original = evidence["grids"]
    compact = {key: ["".join(map(str, row)) for row in grid] for key, grid in original.items()}
    if {key: decode_grid(grid) for key, grid in compact.items()} != original:
        raise ValueError("grid encoding is not lossless")
    evidence.update(grids=compact, grid_encoding=GRID_ENCODING)
    result[0]["content"] += (
        " Be concise; select representative entities, not whole-grid transcriptions."
        " Entity.grid must equal exactly ONE existing key from grids, such as train_0_input;"
        " never combine keys with slashes or invent test output grids. Each cell belongs to"
        " that single grid at its actual zero-based row and column, with its observed color."
        " An entity may contain just one representative cell. Entity IDs name observations;"
        " definition IDs name operations or concepts. Definition.depends_on contains only"
        " other definition IDs, never entity/grid IDs; use [] for a primitive concept."
        " Hypothesis.concepts and ordered_actions contain only defined definition IDs,"
        " not prose instructions. Relation.subject and object are existing entity IDs."
        " Relations may be []; include only relations whose exact cells support them."
        " Do not transcribe or predict a test output anywhere in the symbolic state."
    )
    result[-1]["content"] = json.dumps(evidence, separators=(",", ":"))
    return result


def symbolic_prompt(task: ArcTask, previous=None, feedback=None) -> list[dict]:
    return compact_symbolic_prompt(student_prompt(blind_task(task), previous, feedback))


def direct_prompt(task: ArcTask) -> list[dict]:
    evidence = json.loads(symbolic_prompt(task)[-1]["content"])
    return [
        {
            "role": "system",
            "content": "Infer an ARC transformation from all demonstrations. "
            "Return final JSON with predictions: one {attempt_1,attempt_2} object per test input. "
            'Each attempt is a grid of digit-row strings; attempt_2 may be "same_as_1". '
            "No tools or external knowledge. Keep reasoning concise and finish the JSON.",
        },
        {"role": "user", "content": json.dumps(evidence, separators=(",", ":"))},
    ]


WITNESS_INSTRUCTION = f"""Implement the supplied provisional ARC symbolic model as a general Python
function solve(train, grid), returning an integer list-of-lists output grid. Check the rule against
all visible demonstrations. train contains (input_grid, output_grid) pairs.
Test outputs are unknown.
Return final JSON with exactly one field, python_source, containing complete executable code.
The symbolic model is a hypothesis, not an authority. Address visible verifier counterexamples.
No imports, classes, lambdas, exceptions, global/nonlocal, with, async, dunder access, IO, eval/exec
or reflection. Helpers must be top-level. No task-ID/observed-grid lookup or large literal arrays.
Builtins: {", ".join(sorted(SAFE_BUILTINS))}. Methods: {", ".join(sorted(ALLOWED_METHODS))}.
Use loops, comprehensions and small helpers. Keep the implementation concise, not pseudocode.
"""


def witness_prompt(task: ArcTask, symbolic: dict, previous_source=None, feedback=None):
    evidence = json.loads(symbolic_prompt(task)[-1]["content"])
    evidence["symbolic_model"] = symbolic
    if previous_source:
        evidence["previous_witness"] = previous_source
    if feedback:
        evidence["visible_verifier_feedback"] = feedback
    instruction = WITNESS_INSTRUCTION
    if feedback and feedback.get("repair_scope") == "witness_only":
        instruction += (
            "\nThe previous witness failed static code admission, before its rule was tested."
            " Repair its Python implementation using the exact compiler errors below;"
            " retain the supplied provisional symbolic model. Return the entire corrected"
            " function, not a patch. Never emit raise, assert, try/except, or other forbidden"
            " syntax, even in a branch you expect not to execute. The verifier checks"
            " demonstrations externally; do not add exception-based self-checks."
            " This code-only retry does not establish that the symbolic hypothesis is true.\n"
        )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(evidence, separators=(",", ":"))},
    ]


PYTHON_GRAMMAR = r"""root ::= "</think>" ws "{" ws "\"python_source\"" ws ":" ws string ws "}" ws
ws ::= [ \t\n\r]*
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
"""


class ProgramResponse(StrictModel):
    python_source: str = Field(min_length=1, max_length=20000)


def grounded_schema(task):
    """Constrain observable names and bounds, without supplying a transformation rule."""
    schema = SymbolicModel.model_json_schema()
    variants = []
    for name, grid in task_grids(blind_task(task)).items():
        entity = copy.deepcopy(schema["$defs"]["Entity"])
        entity["properties"]["grid"] = {"type": "string", "const": name}
        cell = copy.deepcopy(schema["$defs"]["Cell"])
        cell["properties"]["row"]["maximum"] = len(grid) - 1
        cell["properties"]["column"]["maximum"] = len(grid[0]) - 1
        entity["properties"]["cells"]["items"] = cell
        variants.append(entity)
    schema["$defs"]["Entity"] = {"oneOf": variants}
    return schema


class StudentReasoner(LocalReasoner):
    structured_style = "legacy"
    symbolic_cell_encoding = "named_fields"

    def __init__(
        self, config, manifest, *, structured_style="legacy", symbolic_cell_encoding="named_fields"
    ):
        if symbolic_cell_encoding not in CELL_ENCODINGS:
            raise ValueError("unknown symbolic cell encoding")
        self.symbolic_cell_encoding = symbolic_cell_encoding
        format_binding(structured_style, manifest)
        self.structured_style = structured_style
        self.grammar_converter = (
            converter_path(manifest) if structured_style in COMPACT_STYLES else None
        )
        super().__init__(config, manifest)

    def prepare(self, messages, tools, *, seed, task):
        return self.prepare_stage(messages, seed=seed, task=task, stage=tools[0]["stage"])

    def prepare_stage(self, messages, *, seed: int, task: ArcTask, stage: str):
        if stage not in {"direct", "symbolic", "witness"}:
            raise ValueError("unknown student stage")
        if stage != "symbolic":
            prepared = super().prepare(messages, [], seed=seed, task=task)
            if prepared["n_predict"] < self.config.max_new_tokens:
                raise ContextOverflow("full answer allowance does not fit; no silent reduction")
            if stage == "witness":
                prepared["grammar"] = PYTHON_GRAMMAR
                prepared["reasoning_budget_tokens"] = min(384, self.config.max_reasoning_tokens)
            elif self.structured_style in COMPACT_STYLES:
                prepared["grammar"] = rectangular_direct_grammar(len(task.test))
            if self.structured_style == BOUNDED_WS_STYLE:
                prepared["grammar"] = bounded_json_whitespace(prepared["grammar"])
            prepared["_stage"] = stage
            return prepared
        # Symbolic SFT supervises explicit state, not hidden reasoning. Match its native
        # non-thinking completion boundary exactly, including the empty native think span.
        repair = (
            prepare_reference_repair(messages)
            if self.structured_style in REFERENCE_STYLES
            else None
        )
        if repair:
            messages = repair["messages"]
        elif self.structured_style in IDENTIFIER_STYLES:
            messages = copy.deepcopy(messages)
            messages[0]["content"] += (
                " Names are compact identifiers, not sentences. In depends_on, concepts and"
                " ordered_actions, copy the exact ID of a definition you include."
                ' For example, a definition named "d1" is referenced as ["d1"],'
                ' never ["apply d1"] or ["apply_d1"]. Put explanations in statement and'
                " provisional_meaning. Use each entity cell coordinate at most once."
            )
        response_format = {}
        if not repair and self.symbolic_cell_encoding == "triples_v1":
            messages = copy.deepcopy(messages)
            messages[0]["content"] += TRIPLES_INSTRUCTION
            response_format = {"_response_format": TRIPLES_FORMAT}
        prompt = self.template.render(messages, thinking=False)
        response = self.client.post("/tokenize", json={"content": prompt, "add_special": False})
        response.raise_for_status()
        tokens = response.json()["tokens"]
        if len(tokens) + self.config.max_new_tokens + 8 > self.config.context_tokens:
            raise ContextOverflow("symbolic evidence and full answer allowance exceed context")
        schema = repair["schema"] if repair else grounded_schema(task)
        if not repair and self.structured_style in IDENTIFIER_STYLES:
            schema = identifier_schema(schema)
        if response_format:
            schema = packed_symbolic_schema(schema)
        constraint = {"json_schema": schema}
        if self.structured_style in COMPACT_STYLES:
            constraint = {"grammar": compact_symbolic_grammar(schema, self.grammar_converter)}
        return {
            "prompt": tokens,
            "n_predict": self.config.max_new_tokens,
            **constraint,
            "temperature": self.config.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "seed": seed,
            "stream": False,
            "cache_prompt": False,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "id_slot": 0,
            "_rendered_prompt": prompt,
            "_prompt_tokens": len(tokens),
            "_stage": stage,
            **(repair["metadata"] if repair else response_format),
        }

    def complete(self, prepared, *, timeout):
        response = super().complete(prepared, timeout=timeout)
        if prepared["_stage"] == "symbolic":
            response["reasoning"] = ""
            response["final"] = response["raw"]["content"].replace("<|im_end|>", "").strip()
        return response


class StudentSession(AdaptiveSession):
    """Reuse the measured one-process context expansion and cumulative swap guard."""

    def __init__(
        self, *args, structured_style="legacy", symbolic_cell_encoding="named_fields", **kwargs
    ):
        if symbolic_cell_encoding not in CELL_ENCODINGS:
            raise ValueError("unknown symbolic cell encoding")
        self.symbolic_cell_encoding = symbolic_cell_encoding
        self.structured_style = structured_style
        super().__init__(*args, **kwargs)
        format_binding(structured_style, self.manifest)

    def _start(self, context):
        super()._start(context)
        cfg = self.reasoner.config
        self.reasoner.close()
        self.reasoner = StudentReasoner(
            cfg,
            self.manifest,
            structured_style=self.structured_style,
            symbolic_cell_encoding=self.symbolic_cell_encoding,
        )

    def prepare_stage(self, messages, *, seed, task, stage):
        # Internal stage routing only; this is never serialized as a model tool.
        return super().prepare(messages, [{"stage": stage}], seed=seed, task=task)
