"""Opt-in formatting constraints, independent of ARC solutions or teacher labels."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import ModuleType

from arc_agent.v4_grammar import final_grammar

CONVERTER_SHA256 = "ee451dc460aa31185226e58988626f64e75ab735169fa3e484fcf16889475ae3"
BOUNDED_WS_STYLE = "reference_repair_bounded_ws"
REFERENCE_STYLES = {"reference_repair", BOUNDED_WS_STYLE}
IDENTIFIER_STYLES = {"compact_identifiers", *REFERENCE_STYLES}
COMPACT_STYLES = {"compact_rectangular", *IDENTIFIER_STYLES}
STYLES = {"legacy", *COMPACT_STYLES}
IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$"


def bounded_json_whitespace(grammar: str) -> str:
    """Limit structural JSON spacing, never text inside JSON strings or reasoning.

    A forced reasoning boundary must not leave an unlimited whitespace-only path
    through the remaining generation allowance. Historical styles are unchanged.
    """
    old = r"ws ::= [ \t\n\r]*"
    lines = grammar.splitlines()
    if lines.count(old) != 1:
        raise ValueError("expected exactly one original JSON whitespace rule")
    return "\n".join(r"ws ::= [ \t\n\r]{0,1}" if line == old else line for line in lines) + "\n"


def identifier_schema(schema: dict) -> dict:
    """Give symbolic names/reference slots a lexical type, without resolving any link.

    Definitions and all semantic text remain model-generated. This restricts name spelling,
    not the transformation vocabulary; undefined names still fail the grounded verifier.
    """
    schema = copy.deepcopy(schema)

    def visit(value):
        if isinstance(value, dict):
            properties = value.get("properties", {})
            for name in ("id", "subject", "object"):
                if name in properties:
                    properties[name]["pattern"] = IDENTIFIER_PATTERN
            for name in ("depends_on", "concepts", "ordered_actions"):
                if name in properties:
                    properties[name]["items"]["pattern"] = IDENTIFIER_PATTERN
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    return schema


def rectangular_direct_grammar(test_count: int) -> str:
    """Each grid chooses a width once; all its rows must have that width."""
    grammar = final_grammar("direct", test_count, [])
    old = 'grid ::= "[" ws row (ws "," ws row){0,29} ws "]"'
    if grammar.count(old) != 1:
        raise ValueError("direct grammar contract changed")
    alternatives = "grid ::= " + " | ".join(f"grid-{width}" for width in range(1, 31))
    rules = []
    for width in range(1, 31):
        rules.append(f'row-{width} ::= "\\"" [0-9]{{{width}}} "\\""')
        rules.append(f'grid-{width} ::= "[" ws row-{width} (ws "," ws row-{width}){{0,29}} ws "]"')
    return grammar.replace(old, alternatives) + "\n".join(rules) + "\n"


def converter_path(manifest: dict) -> Path:
    return (
        Path(manifest["files"]["server"]["path"]).parents[2] / "examples/json_schema_to_grammar.py"
    )


def verified_converter_bytes(path: Path) -> bytes:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != CONVERTER_SHA256:
        raise ValueError("symbolic grammar converter differs from the pinned authors' revision")
    return source


def format_binding(style: str, manifest: dict) -> dict:
    if style not in STYLES:
        raise ValueError("unknown student structured style")
    binding = {"structured_style": style}
    if style in COMPACT_STYLES:
        verified_converter_bytes(converter_path(manifest))
        binding["grammar_converter_sha256"] = CONVERTER_SHA256
    return binding


def compact_symbolic_grammar(schema: dict, path: Path) -> str:
    """Compile the unchanged schema with the pinned converter, then remove JSON spacing.

    The upstream converter implements a subset of JSON Schema. Semantic admission must
    still use the canonical Pydantic model and grounded verifier; this is not a proof.
    """

    def check_refs(value):
        if isinstance(value, dict):
            if "$ref" in value and (
                not isinstance(value["$ref"], str) or not value["$ref"].startswith("#/")
            ):
                raise ValueError("only local symbolic schema references are allowed")
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(schema)
    # Execute exactly the bytes we verified, not a second read through an import loader.
    source = verified_converter_bytes(path)
    module = ModuleType("nanbeige_pinned_schema_converter")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    converter = module.SchemaConverter(
        prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False
    )
    schema = copy.deepcopy(schema)
    converter.resolve_refs(schema, "local-symbolic")
    converter.visit(schema, "")
    grammar = converter.format_grammar()
    old = "space ::= " + module.SPACE_RULE
    if grammar.splitlines().count(old) != 1:
        raise ValueError("symbolic grammar spacing contract changed")
    return (
        "\n".join('space ::= ""' if line == old else line for line in grammar.splitlines()) + "\n"
    )
