"""Standalone observation probe; no active solver or training protocol changes."""

import copy
import importlib.util
from pathlib import Path

import pytest
from test_v7 import task as task

from arc_agent.v4_config import content_hash

SPEC = importlib.util.spec_from_file_location(
    "scene_catalog_probe", Path(__file__).parents[1] / "scripts/probe_nanbeige_scene_catalog.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_all_objects_are_pageable_and_exactly_resolvable():
    grid = [[int((r + c) % 2 == 0) for c in range(15)] for r in range(15)]
    catalog = probe.catalog_for_grid(grid, "g", background=0)
    original = copy.deepcopy(catalog)
    reference = content_hash(catalog)
    result = probe.measure(catalog, len)
    assert result["object_count"] > 24 and result["legacy_view_omitted_objects"] > 0
    assert len(result["page_tokens"]) > 1 and result["all_objects_retrievable"]
    first = probe.catalog_page(catalog, reference)
    assert len(first["objects"]) == 24 and first["next_offset"] == 24
    resolved = probe.resolve_object(catalog, reference, first["objects"][0]["id"])
    assert resolved["cells"] == [[0, 0, 1]]
    resolved["cells"][0][2] = 9
    assert catalog == original


def test_segmentation_and_grid_changes_invalidate_old_reference():
    catalog = probe.catalog_for_grid([[1, 0], [0, 2]], "g", background=0)
    reference = content_hash(catalog)
    alternate = probe.catalog_for_grid(
        [[1, 0], [0, 2]], "g", background=0, connectivity=8, multicolor=True
    )
    assert len(catalog["objects"]) == 2 and len(alternate["objects"]) == 1
    with pytest.raises(ValueError, match="stale"):
        probe.catalog_page(alternate, reference)
    catalog["grid"][0][0] = 3
    with pytest.raises(ValueError, match="stale"):
        probe.resolve_object(catalog, reference, "o0")


def test_task_catalogs_ignore_all_test_labels(task):
    before = probe.task_catalogs(task)
    task.test[0].output = [[9] * 30] * 30
    assert probe.task_catalogs(task) == before
    assert not any(c["grid_id"].startswith("test") and "output" in c["grid_id"] for c in before)


def test_uniform_grid_has_empty_catalog_and_lossless_background():
    catalog = probe.catalog_for_grid([[3, 3], [3, 3]], "g")
    result = probe.measure(catalog, len)
    assert result["object_count"] == 0 and result["exact_grid_reconstruction_verified"]
    assert probe.catalog_page(catalog, content_hash(catalog))["next_offset"] is None
    with pytest.raises(ValueError, match="undefined"):
        probe.resolve_object(catalog, content_hash(catalog), "o0")


@pytest.mark.parametrize("offset,size", [(True, 1), (0, True), (-1, 1), (2, 1), (0, 0), (0, 25)])
def test_invalid_pages_rejected(offset, size):
    catalog = probe.catalog_for_grid([[0, 1]], "g", background=0)
    with pytest.raises(ValueError):
        probe.catalog_page(catalog, content_hash(catalog), offset=offset, size=size)
