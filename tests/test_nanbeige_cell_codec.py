"""Prototype only: exact symbolic round trips and native grammar preservation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_v7 import artifact as artifact
from test_v7 import task as task
from test_v8_grammar import CONVERTER
from test_v8_grammar import accepts as accepts

from arc_agent.v8_grammar import compact_symbolic_grammar
from arc_agent.v8_reasoner import grounded_schema

SPEC = importlib.util.spec_from_file_location(
    "cell_codec", Path(__file__).parents[1] / "scripts/probe_nanbeige_cell_codec.py"
)
codec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codec)


def test_full_state_round_trip_preserves_order_meanings_and_uncertainty(artifact):
    state = artifact.symbolic.model_dump(mode="json")
    state["entities"][0]["cells"] = [
        {"row": 29, "column": 29, "color": 9},
        {"row": 0, "column": 0, "color": 0},
        {"row": 2, "column": 3, "color": 4},
    ]
    state["hypotheses"][0]["counterevidence"] = ["Not proved.  Keep spacing.\n中文"]
    original = copy.deepcopy(state)
    packed = codec.packed_state(state)
    assert state == original
    assert packed["entities"][0]["cells"] == [[29, 29, 9], [0, 0, 0], [2, 3, 4]]
    packed_original = copy.deepcopy(packed)
    assert codec.unpacked_state(packed) == original
    assert packed == packed_original


@pytest.mark.parametrize(
    "cell",
    [
        [1, 2],
        [1, 2, 3, 4],
        [True, 2, 3],
        [1.0, 2, 3],
        [1, 2, 10],
        [30, 0, 0],
        {"row": 1, "column": 2, "color": 3},
    ],
)
def test_malformed_or_mixed_cell_encoding_rejected(artifact, cell):
    packed = codec.packed_state(artifact.symbolic.model_dump(mode="json"))
    packed["entities"][0]["cells"] = [cell]
    with pytest.raises(ValueError):
        codec.unpacked_state(packed)


def test_prefix_probe_does_not_complete_json_or_rewrite_quoted_prose():
    prefix = '{"entities":[{"cells":[{"row":2,"column":3,"color":4},{"row":'
    rewritten, count = codec.rewrite_complete_cell_objects(prefix)
    assert count == 1 and rewritten == '{"entities":[{"cells":[[2,3,4],{"row":'
    with pytest.raises(ValueError):
        json.loads(rewritten)
    prose = json.dumps({"text": '{"row":2,"column":3,"color":4}'}, separators=(",", ":"))
    assert codec.rewrite_complete_cell_objects(prose) == (prose, 0)


def test_native_packed_grammar_keeps_exact_cell_shape_and_visible_bounds(accepts, task, artifact):
    schema = grounded_schema(task)
    original = copy.deepcopy(schema)
    packed_schema = codec.packed_schema(schema)
    assert schema == original
    grammar = compact_symbolic_grammar(packed_schema, CONVERTER)
    packed = codec.packed_state(artifact.symbolic.model_dump(mode="json"))
    packed["version"] = packed.pop("version")
    assert accepts(grammar, json.dumps(packed, separators=(",", ":"))), accepts.last_result
    for cell in ([0, 0], [0, 0, 1, 2], [0, 0, 10], [1, 0, 1], [0, 2, 1]):
        bad = copy.deepcopy(packed)
        bad["entities"][0]["cells"] = [cell]
        assert not accepts(grammar, json.dumps(bad, separators=(",", ":")))
    task.test[0].output = [[9] * 30] * 30
    assert codec.packed_schema(grounded_schema(task)) == packed_schema
