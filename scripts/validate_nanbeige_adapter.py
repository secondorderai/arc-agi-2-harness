"""Inspect a provenance-checked Nanbeige LoRA payload before future conversion.

No base-model weights, pickle/optimizer loading, training, conversion or network.
Passing numeric checks does not prove an effective weight delta, inference or promotion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "trained_provenance", Path(__file__).with_name("audit_nanbeige_trained_checkpoint.py")
)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def expected_shapes(model_config):
    expected_hash = provenance.preflight.MODEL_CODE["config.json"]
    provenance.checked_file(model_config, expected_hash)
    config = provenance.job.read_json(model_config)
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    queries = config["num_attention_heads"] * config["head_dim"]
    keys = config["num_key_value_heads"] * config["head_dim"]
    modules = {
        "self_attn.q_proj": (hidden, queries),
        "self_attn.k_proj": (hidden, keys),
        "self_attn.v_proj": (hidden, keys),
        "self_attn.o_proj": (queries, hidden),
        "mlp.gate_proj": (hidden, intermediate),
        "mlp.up_proj": (hidden, intermediate),
        "mlp.down_proj": (intermediate, hidden),
    }
    shapes = {}
    for layer in range(config["num_hidden_layers"]):
        for module, (inputs, outputs) in modules.items():
            name = f"base_model.model.model.layers.{layer}.{module}"
            shapes[name + ".lora_A.weight"] = [16, inputs]
            shapes[name + ".lora_B.weight"] = [outputs, 16]
    if len(shapes) != 308 or sum(math.prod(s) for s in shapes.values()) != 23969792:
        raise ValueError("official shared-layer adapter inventory changed")
    return shapes


def check_adapter_configuration(config):
    provenance.job.validate_adapter_config(config)
    # The planned plain-LoRA converter uses alpha/r, no activation gating or DoRA.
    for name in (
        "use_rslora",
        "use_dora",
        "use_qalora",
        "fan_in_fan_out",
        "lora_bias",
        "alora_invocation_tokens",
        "layer_replication",
        "target_parameters",
        "trainable_token_indices",
        "arrow_config",
        "megatron_config",
    ):
        if config.get(name):
            raise ValueError(f"unsupported adapter inference variant: {name}")


def inspect_tensors(path, sha256, shapes):
    """Numeric inspection primitive; custom fixture shapes confer no model provenance."""
    import torch
    from safetensors import safe_open

    provenance.checked_file(path, sha256)
    if not shapes or any(
        len(s) != 2 or any(type(n) is not int or n <= 0 for n in s) for s in shapes.values()
    ):
        raise ValueError("require positive matrix shapes")
    counts, dtypes, total = {}, set(), 0
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != set(shapes):
                raise ValueError("adapter tensor names differ from expected physical layers")
            # Validate all metadata before reading any numeric tensor payload.
            for name, shape in shapes.items():
                value = tensors.get_slice(name)
                if value.get_shape() != shape or value.get_dtype() not in {"F32", "F16", "BF16"}:
                    raise ValueError(f"adapter tensor shape or dtype differs: {name}")
            for name, shape in shapes.items():
                value = tensors.get_slice(name)
                dtypes.add(value.get_dtype())
                rows = max(1, 65536 // shape[1])
                nonzero = 0
                for start in range(0, shape[0], rows):
                    block = value[start : min(start + rows, shape[0]), :]
                    if not torch.isfinite(block).all().item():
                        raise ValueError(f"adapter contains nonfinite values: {name}")
                    nonzero += torch.count_nonzero(block).item()
                counts[name] = int(nonzero)
                total += math.prod(shape)
    finally:
        torch.set_num_threads(threads)
    provenance.checked_file(path, sha256)
    nonzero_pairs = sum(
        count > 0 and counts.get(name.replace(".lora_A.weight", ".lora_B.weight"), 0) > 0
        for name, count in counts.items()
        if name.endswith(".lora_A.weight")
    )
    if not nonzero_pairs:
        raise ValueError("adapter has no potentially nonzero A/B pair; cannot improve base weights")
    return {
        "tensor_count": len(shapes),
        "parameters": total,
        "dtypes": sorted(dtypes),
        "finite_values_verified": True,
        "nonzero_factor_pairs": nonzero_pairs,
        "effective_delta_verified": False,
        "adapter_sha256": sha256,
        "tensor_shapes_hash": provenance.job.digest(shapes),
    }


def validate(
    bundle,
    bundle_hash,
    report,
    report_hash,
    checkpoint,
    compatibility,
    compatibility_hash,
    model_config,
):
    # Keep all existing completion, permission and frozen-lineage checks before tensors.
    evidence = provenance.audit(
        bundle, bundle_hash, report, report_hash, checkpoint, compatibility, compatibility_hash
    )
    checkpoint = Path(checkpoint)
    check_adapter_configuration(provenance.job.read_json(checkpoint / "adapter_config.json"))
    numeric = inspect_tensors(
        checkpoint / "adapter_model.safetensors",
        evidence["adapter_sha256"],
        expected_shapes(model_config),
    )
    # Recheck mutable local files rather than trusting only the initial provenance pass.
    if (
        provenance.audit(
            bundle, bundle_hash, report, report_hash, checkpoint, compatibility, compatibility_hash
        )
        != evidence
    ):
        raise ValueError("training evidence changed during tensor inspection")
    return {
        "scope": "stored training provenance plus adapter tensor inventory and finite values",
        "training_evidence": evidence,
        "tensor_evidence": numeric,
        "validator_sha256": provenance.checked_file(__file__),
        "tensor_payload_validated": True,
        "base_weights_loaded": False,
        "model_inference_performed": False,
        "conversion_performed": False,
        "inference_ready": False,
        "promotion_proven": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "report", "checkpoint", "compatibility"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        if name != "checkpoint":
            parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path(".runtime/nanbeige/model-hf/config.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.bundle,
        args.bundle_sha256,
        args.report,
        args.report_sha256,
        args.checkpoint,
        args.compatibility,
        args.compatibility_sha256,
        args.model_config,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
