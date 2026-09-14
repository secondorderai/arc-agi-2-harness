"""Numeric adapter fixtures only; never official student training or approval."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "adapter_payload", ROOT / "scripts/validate_nanbeige_adapter.py"
)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
PREFIX = "base_model.model.model.layers.0.self_attn.q_proj"
A, B = PREFIX + ".lora_A.weight", PREFIX + ".lora_B.weight"
SHAPES = {A: [2, 3], B: [4, 2]}


def fixture_file(tmp_path, *, dtype=torch.float32, mutate=lambda tensors: None):
    tensors = {key: torch.ones(shape, dtype=dtype) for key, shape in SHAPES.items()}
    mutate(tensors)
    path = tmp_path / "fixture.safetensors"
    save_file(tensors, path)
    return path, validator.provenance.checked_file(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_supported_float_payloads(tmp_path, dtype):
    path, sha = fixture_file(tmp_path, dtype=dtype)
    result = validator.inspect_tensors(path, sha, SHAPES)
    assert result["finite_values_verified"] and result["tensor_count"] == 2
    assert result["parameters"] == 14 and result["nonzero_factor_pairs"] == 1
    assert result["effective_delta_verified"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_payloads_rejected(tmp_path, bad):
    path, sha = fixture_file(tmp_path, mutate=lambda tensors: tensors[B].fill_(bad))
    with pytest.raises(ValueError, match="nonfinite"):
        validator.inspect_tensors(path, sha, SHAPES)


@pytest.mark.parametrize("key", [A, B])
def test_noop_factors_rejected(tmp_path, key):
    path, sha = fixture_file(tmp_path, mutate=lambda tensors: tensors[key].zero_())
    with pytest.raises(ValueError, match="no potentially nonzero"):
        validator.inspect_tensors(path, sha, SHAPES)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "integer"])
def test_wrong_inventory_rejected_before_numeric_scan(tmp_path, change):
    def mutate(tensors):
        if change == "missing":
            del tensors[B]
        elif change == "extra":
            tensors["foreign.weight"] = torch.ones(2)
        elif change == "shape":
            tensors[B] = torch.ones(5, 2)
        else:
            tensors[B] = tensors[B].to(torch.int32)

    path, sha = fixture_file(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="tensor names|shape or dtype"):
        validator.inspect_tensors(path, sha, SHAPES)


def test_hash_mismatch_and_symlink_rejected(tmp_path):
    path, sha = fixture_file(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        validator.inspect_tensors(path, "0" * 64, SHAPES)
    link = tmp_path / "linked.safetensors"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular evidence"):
        validator.inspect_tensors(link, sha, SHAPES)


@pytest.mark.parametrize("variant", ["use_dora", "use_rslora", "fan_in_fan_out", "lora_bias"])
def test_nonstandard_scaling_or_weights_rejected(variant):
    config = {
        "base_model_name_or_path": validator.provenance.job.MODEL,
        "revision": validator.provenance.job.REVISION,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": validator.provenance.job.TARGETS,
    }
    validator.check_adapter_configuration(config)
    config[variant] = True
    with pytest.raises(ValueError, match="unsupported adapter inference variant"):
        validator.check_adapter_configuration(config)


def test_provenance_failure_precedes_any_tensor_read(monkeypatch):
    def fail(*args):
        raise ValueError("missing real training provenance")

    def forbidden(*args):
        raise AssertionError("tensor payload read before provenance")

    monkeypatch.setattr(validator.provenance, "audit", fail)
    monkeypatch.setattr(validator, "inspect_tensors", forbidden)
    with pytest.raises(ValueError, match="missing real training"):
        validator.validate(*(["unused"] * 8))


def test_config_is_pinned_and_inventory_covers_shared_physical_layers(tmp_path):
    config = ROOT / ".runtime/nanbeige/model-hf/config.json"
    expected = validator.expected_shapes(config)
    assert len(expected) == 308
    assert all("layers.22." not in key for key in expected)
    assert expected[PREFIX + ".lora_A.weight"] == [16, 3072]
    assert expected[PREFIX + ".lora_B.weight"] == [6144, 16]
    altered = tmp_path / "config.json"
    altered.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        validator.expected_shapes(altered)


def test_official_inventory_matches_pinned_peft_meta_model(tmp_path):
    # Uses the existing isolated trainer environment; zero official weight allocation.
    script = r"""
import sys
sys.path.insert(0, "scripts")
import validate_nanbeige_adapter as checker
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from pathlib import Path
base = Path(".runtime/nanbeige/model-hf")
for name, digest in checker.provenance.preflight.MODEL_CODE.items():
    checker.provenance.checked_file(base / name, digest)
cfg = AutoConfig.from_pretrained(base, trust_remote_code=True, local_files_only=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=checker.provenance.job.TARGETS))
state = get_peft_model_state_dict(model)
assert all(value.is_meta for value in state.values())
actual = {name: list(value.shape) for name, value in state.items()}
assert actual == checker.expected_shapes(base / "config.json")
assert model.base_model.model.model._get_num_loops() == 2
print("308 tensor shapes matched; 22 physical layers, two shared loops; no base weights loaded")
"""
    process = subprocess.run(
        [str(ROOT / ".runtime/nanbeige-trainer/bin/python"), "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_MODULES_CACHE": str(tmp_path / "hf-modules"),
        },
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "308 tensor shapes matched" in process.stdout
