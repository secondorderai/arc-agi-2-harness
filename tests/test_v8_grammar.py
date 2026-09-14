"""Real pinned-runtime GBNF checks; no model loading or label-dependent grammars."""

import copy
import json
import re
import subprocess
import tarfile
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_v7 import artifact as artifact
from test_v7 import task as task

from arc_agent.v4_config import V4Config
from arc_agent.v4_reasoner import ContextOverflow, NativeTemplate
from arc_agent.v4_state import blind_task
from arc_agent.v8_cli import archive_source, matching_smoke
from arc_agent.v8_config import V8Config
from arc_agent.v8_grammar import (
    BOUNDED_WS_STYLE,
    CONVERTER_SHA256,
    IDENTIFIER_PATTERN,
    bounded_json_whitespace,
    compact_symbolic_grammar,
    converter_path,
    format_binding,
    identifier_schema,
    rectangular_direct_grammar,
)
from arc_agent.v8_reasoner import (
    PYTHON_GRAMMAR,
    StudentReasoner,
    direct_prompt,
    grounded_schema,
    symbolic_prompt,
    witness_prompt,
)

CONVERTER = Path(".runtime/nanbeige/llama.cpp/examples/json_schema_to_grammar.py")
VALIDATOR = Path(".runtime/nanbeige/grammar-tools/bin/test-gbnf-validator")


@pytest.fixture
def accepts(tmp_path):
    if not VALIDATOR.is_file():
        pytest.fail("build the pinned runtime test-gbnf-validator before grammar verification")

    def check(grammar, value):
        (tmp_path / "grammar.gbnf").write_text(grammar)
        (tmp_path / "input.txt").write_text(value)
        result = subprocess.run(
            [str(VALIDATOR.resolve()), str(tmp_path / "grammar.gbnf"), str(tmp_path / "input.txt")],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        # The upstream validator exits zero for BOTH acceptance and rejection.
        assert "Failed to initialize" not in result.stdout
        check.last_result = result.stdout
        return "Input string is valid according to the grammar." in result.stdout

    return check


def direct_text(first, second="same_as_1", count=1):
    return "</think>" + json.dumps(
        {"predictions": [{"attempt_1": first, "attempt_2": second}] * count}
    )


@pytest.mark.parametrize("width", [1, 2, 15, 30])
@pytest.mark.parametrize("height", [1, 2, 30])
def test_rectangular_grids_preserve_all_dimensions(accepts, width, height):
    assert accepts(rectangular_direct_grammar(1), direct_text(["7" * width] * height))


@pytest.mark.parametrize(
    "grid",
    [[], [""], ["11", "1"], ["1", "111"], ["0"] * 31, ["1" * 31], ["-1"], ["a"], [[1, 2]], ["1 2"]],
)
def test_malformed_grids_rejected_by_native_grammar(accepts, grid):
    assert not accepts(rectangular_direct_grammar(1), direct_text(grid))


def test_two_guesses_can_have_different_shapes_but_each_is_rectangular(accepts):
    assert accepts(rectangular_direct_grammar(1), direct_text(["12"], ["3", "4"]))
    assert not accepts(rectangular_direct_grammar(1), direct_text(["12"], ["3", "44"]))
    assert not accepts(rectangular_direct_grammar(1), direct_text("same_as_1"))


@pytest.mark.parametrize("count", [1, 2, 4, 100])
def test_exact_test_input_count_and_native_boundary(accepts, count):
    grammar = rectangular_direct_grammar(count)
    assert accepts(grammar, direct_text(["0"], count=count))
    assert not accepts(grammar, direct_text(["0"], count=count + 1))
    assert not accepts(grammar, direct_text(["0"], count=count).removeprefix("</think>"))


@pytest.mark.parametrize("count", [0, 101])
def test_test_count_bounds(count):
    with pytest.raises(ValueError):
        rectangular_direct_grammar(count)


@pytest.mark.parametrize("spaces", ["  ", "\n\n", "\n" * 1025, "\t" * 1025])
def test_bounded_structural_whitespace_blocks_runaway_suffixes(accepts, spaces):
    old = rectangular_direct_grammar(1)
    new = bounded_json_whitespace(old)
    value = direct_text(["12", "34"])
    expanded = value.replace("</think>", "</think>" + spaces, 1)
    assert accepts(old, expanded)
    assert not accepts(new, expanded)
    assert accepts(new, value)
    assert accepts(new, value.replace("</think>", "</think>\n", 1))


def test_bounded_witness_whitespace_preserves_code_contents(accepts):
    source = 'def solve(grid):\n    text = "a  b"\n    return grid\n'
    value = "</think>" + json.dumps({"python_source": source})
    grammar = bounded_json_whitespace(PYTHON_GRAMMAR)
    assert accepts(grammar, value)
    assert not accepts(grammar, value.replace("</think>", "</think>" + "\n" * 1025, 1))
    assert json.loads(value.removeprefix("</think>"))["python_source"] == source


@pytest.mark.parametrize("grammar", ['root ::= "a"', "ws ::= [ \\t\\n\\r]*\n" * 2])
def test_bounded_whitespace_rejects_changed_grammar_contract(grammar):
    with pytest.raises(ValueError, match="exactly one"):
        bounded_json_whitespace(grammar)


def test_compact_symbolic_json_preserves_schema_and_string_contents(accepts, task, artifact):
    schema = grounded_schema(task)
    original = copy.deepcopy(schema)
    grammar = compact_symbolic_grammar(schema, CONVERTER)
    assert schema == original
    value = artifact.symbolic.model_dump(mode="json")
    # The upstream converter places required fields before optional/defaulted fields.
    # This is JSON key order only; no field values or requiredness are changed.
    value["version"] = value.pop("version")
    compact = json.dumps(value, separators=(",", ":"))
    assert accepts(grammar, compact), accepts.last_result
    assert not accepts(grammar, json.dumps(value, indent=2))
    assert not accepts(grammar, " " + compact)
    assert not accepts(grammar, compact + "\n")
    assert not accepts(grammar, "</think>" + compact)
    # Removing structural whitespace cannot erase meaningful whitespace/escapes in strings.
    simple = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    simple_grammar = compact_symbolic_grammar(simple, CONVERTER)
    assert accepts(simple_grammar, json.dumps({"text": 'a b\n"quoted"'}, separators=(",", ":")))
    assert not accepts(simple_grammar, '{"text":"a","extra":1}')
    assert not accepts(simple_grammar, "{}")


def test_symbolic_bounds_remain_visible_evidence_only(accepts, task, artifact):
    schema = grounded_schema(task)
    grammar = compact_symbolic_grammar(schema, CONVERTER)
    value = artifact.symbolic.model_dump(mode="json")
    value["version"] = value.pop("version")
    assert accepts(grammar, json.dumps(value, separators=(",", ":"))), accepts.last_result
    value["entities"][0]["grid"] = "test_0_output"
    assert not accepts(grammar, json.dumps(value, separators=(",", ":")))
    task.test[0].output = [[9] * 30] * 30
    assert grammar == compact_symbolic_grammar(grounded_schema(task), CONVERTER)


@pytest.mark.parametrize(
    "ref", ["https://example.com/schema", "file:///tmp/schema", "other.json", 2]
)
def test_no_external_schema_fetch(ref, tmp_path):
    with pytest.raises(ValueError, match="only local"):
        compact_symbolic_grammar({"$ref": ref}, tmp_path / "not-imported.py")


def test_converter_pin_checked_before_import_even_after_prior_use(tmp_path):
    path = tmp_path / "converter.py"
    path.write_bytes(CONVERTER.read_bytes())
    compact_symbolic_grammar({"type": "string"}, path)
    path.write_text('raise RuntimeError("untrusted code must not execute")')
    with pytest.raises(ValueError, match="pinned authors"):
        compact_symbolic_grammar({"type": "string"}, path)


def test_style_default_opt_in_and_smoke_binding():
    assert V8Config().structured_style == "legacy"
    with pytest.raises(ValidationError):
        V8Config(structured_style="frontier")
    manifest = json.loads(Path(".runtime/nanbeige/manifest.json").read_text())
    assert converter_path(manifest).resolve() == CONVERTER.resolve()
    binding = format_binding("compact_rectangular", manifest)
    assert binding["grammar_converter_sha256"] == CONVERTER_SHA256
    report = {**binding, "passed": True, "context_tokens": 4096}
    assert matching_smoke(report, binding, 4096)
    assert not matching_smoke(report, binding, 8192)
    assert not matching_smoke({**report, "structured_style": "legacy"}, binding, 4096)
    assert not matching_smoke({**report, "grammar_converter_sha256": "altered"}, binding, 4096)
    assert not matching_smoke({**report, "passed": False}, binding, 4096)
    assert format_binding("legacy", {}) == {"structured_style": "legacy"}
    with pytest.raises(ValueError, match="unknown"):
        format_binding("frontier", {})


def test_source_archive_includes_converter_and_preserves_original(tmp_path):
    archive_source(
        tmp_path,
        Path("configs/v8-nanbeige-format-baseline.yaml"),
        Path("configs/v8-nanbeige-local.yaml"),
        grammar_converter=CONVERTER,
    )
    original = (tmp_path / "source.tar.gz").read_bytes()
    with tarfile.open(tmp_path / "source.tar.gz") as archive:
        assert (
            archive.extractfile("runtime-tools/json_schema_to_grammar.py").read()
            == CONVERTER.read_bytes()
        )
    archive_source(tmp_path, Path("missing.yaml"), Path("missing-runtime.yaml"))
    assert (tmp_path / "source.tar.gz").read_bytes() == original


@pytest.mark.parametrize("style", ["compact_rectangular", BOUNDED_WS_STYLE])
@pytest.mark.parametrize("output_tokens", [2048, 4096])
def test_new_style_native_stage_contracts_and_full_token_cap(task, style, output_tokens):
    reasoner = object.__new__(StudentReasoner)
    reasoner.structured_style = style
    reasoner.grammar_converter = CONVERTER
    reasoner.config = V4Config(context_tokens=12288, max_new_tokens=output_tokens)
    reasoner.template = NativeTemplate(Path(".runtime/nanbeige/model-hf/tokenizer_config.json"))
    length = [100]
    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"tokens": [1] * length[0]})
        ),
    )
    try:
        direct = reasoner.prepare_stage(
            direct_prompt(task), seed=42, task=blind_task(task), stage="direct"
        )
        expected_direct = rectangular_direct_grammar(len(task.test))
        if style == BOUNDED_WS_STYLE:
            expected_direct = bounded_json_whitespace(expected_direct)
        assert direct["grammar"] == expected_direct
        assert direct["grammar_lazy"] is True and direct["grammar_triggers"] == [
            {"type": 1, "value": "</think>"}
        ]
        symbolic = reasoner.prepare_stage(
            symbolic_prompt(task), seed=42, task=blind_task(task), stage="symbolic"
        )
        assert (
            "grammar" in symbolic
            and "json_schema" not in symbolic
            and "grammar_lazy" not in symbolic
        )
        assert "</think>" in symbolic["_rendered_prompt"]
        witness = reasoner.prepare_stage(
            witness_prompt(task, {}), seed=42, task=blind_task(task), stage="witness"
        )
        expected_witness = (
            bounded_json_whitespace(PYTHON_GRAMMAR) if style == BOUNDED_WS_STYLE else PYTHON_GRAMMAR
        )
        assert witness["grammar"] == expected_witness and witness["reasoning_budget_tokens"] == 384
        assert all(p["n_predict"] == output_tokens for p in [direct, symbolic, witness])
        assert direct["reasoning_budget_tokens"] == 1024
        assert all(
            p["seed"] == 42 and p["cache_prompt"] is False for p in [direct, symbolic, witness]
        )
        length[0] = 11000
        for stage in ["direct", "symbolic", "witness"]:
            with pytest.raises(ContextOverflow):
                reasoner.prepare_stage(
                    symbolic_prompt(task), seed=42, task=blind_task(task), stage=stage
                )
    finally:
        reasoner.close()


@pytest.mark.parametrize(
    "slot", ["id", "subject", "object", "depends_on", "concepts", "ordered_actions"]
)
def test_identifier_slots_reject_prose_but_preserve_names(accepts, slot):
    is_list = slot in {"depends_on", "concepts", "ordered_actions"}
    field = (
        {"type": "array", "items": {"type": "string"}, "minItems": 1}
        if is_list
        else {"type": "string"}
    )
    schema = {
        "type": "object",
        "properties": {slot: field},
        "required": [slot],
        "additionalProperties": False,
    }
    original = copy.deepcopy(schema)
    grammar = compact_symbolic_grammar(identifier_schema(schema), CONVERTER)
    assert schema == original
    for name in ["d1", "copy_identity", "relation:2", "name.with-hyphen", "x" * 64]:
        assert accepts(
            grammar, json.dumps({slot: [name] if is_list else name}, separators=(",", ":"))
        ), accepts.last_result
    for name in ["apply d1", "x\ny", "", "x" * 65, "1d", "é"]:
        assert not accepts(
            grammar, json.dumps({slot: [name] if is_list else name}, separators=(",", ":"))
        )


def test_lexical_identifiers_do_not_certify_links_or_change_meaning(accepts, task, artifact):
    from arc_agent.v5_symbolic import SymbolicModel, validate_symbols

    original = grounded_schema(task)
    schema = identifier_schema(original)
    assert "pattern" not in original["$defs"]["Definition"]["properties"]["id"]
    assert (
        schema["$defs"]["Definition"]["properties"]["provisional_meaning"]
        == original["$defs"]["Definition"]["properties"]["provisional_meaning"]
    )
    value = artifact.symbolic.model_dump(mode="json")
    value["version"] = value.pop("version")
    value["hypotheses"][0]["ordered_actions"] = ["undefined_name"]
    grammar = compact_symbolic_grammar(schema, CONVERTER)
    assert accepts(grammar, json.dumps(value, separators=(",", ":"))), accepts.last_result
    with pytest.raises(ValueError, match="undefined symbolic concept"):
        validate_symbols(SymbolicModel.model_validate(value), task)
    value["hypotheses"][0]["ordered_actions"] = ["apply preserve"]
    assert not accepts(grammar, json.dumps(value, separators=(",", ":")))


def test_admitted_training_target_identifiers_need_no_rewrite():
    path = Path("runs/v8/student-data-pilot-final/symbolic-sft.json")
    original = path.read_bytes()
    names = []

    def visit(value):
        if isinstance(value, dict):
            names.extend(value[k] for k in ["id", "subject", "object"] if k in value)
            for key in ["depends_on", "concepts", "ordered_actions"]:
                names.extend(value.get(key, []))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for row in json.loads(original):
        for message in row["completion"]:
            visit(json.loads(message["content"]))
    assert len(names) >= 435
    assert all(re.fullmatch(IDENTIFIER_PATTERN, name) for name in names)
    assert path.read_bytes() == original


def test_identifier_prompt_is_isolated_and_style_bound(task):
    manifest = json.loads(Path(".runtime/nanbeige/manifest.json").read_text())
    reasoner = StudentReasoner(V4Config(), manifest, structured_style="compact_identifiers")
    reasoner.client.close()
    reasoner.client = httpx.Client(
        base_url="http://127.0.0.1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"tokens": [1] * 100})
        ),
    )
    prompt = symbolic_prompt(task)
    original = copy.deepcopy(prompt)
    try:
        prepared = reasoner.prepare_stage(prompt, seed=42, task=blind_task(task), stage="symbolic")
        assert "copy the exact ID" in prepared["_rendered_prompt"]
        assert "No task" not in prepared["_rendered_prompt"]  # no witness prompt substituted
        assert prepared["n_predict"] == 2048 and "grammar" in prepared
        assert prompt == original
        binding = format_binding("compact_identifiers", manifest)
        assert binding["grammar_converter_sha256"] == CONVERTER_SHA256
        report = {
            **format_binding("compact_rectangular", manifest),
            "passed": True,
            "context_tokens": 4096,
        }
        assert not matching_smoke(report, binding, 4096)
    finally:
        reasoner.close()
