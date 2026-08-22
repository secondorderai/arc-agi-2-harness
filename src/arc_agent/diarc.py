from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

LAYER_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def convert_peft_key(key: str) -> tuple[str, bool] | None:
    if ".base_layer." in key:
        return None
    if key.startswith("base_model.model.model."):
        converted = "model." + key.removeprefix("base_model.model.model.")
    elif key.startswith("base_model.model.lm_head."):
        converted = "lm_head." + key.removeprefix("base_model.model.lm_head.")
    else:
        raise ValueError(f"unsupported DiARC adapter key: {key}")

    replacements = {
        ".lora_A.weight": ".lora_a",
        ".lora_B.weight": ".lora_b",
        ".lora_embedding_A": ".lora_a",
        ".lora_embedding_B": ".lora_b",
    }
    for suffix, replacement in replacements.items():
        if converted.endswith(suffix):
            return converted.removesuffix(suffix) + replacement, True
    raise ValueError(f"unsupported DiARC LoRA tensor: {key}")


def convert_diarc_adapter(source: Path, output: Path) -> tuple[int, int]:
    peft_config = json.loads((source / "adapter_config.json").read_text())
    rank = int(peft_config["r"])
    alpha = float(peft_config["lora_alpha"])
    converted: dict[str, np.ndarray] = {}
    skipped = 0
    # Loading the whole file through safetensors.numpy fails on the two frozen
    # BF16 base tensors even though they are not part of the LoRA adapter. Open
    # lazily so those tensors can be ignored before NumPy tries to decode them.
    with safe_open(source / "adapter_model.safetensors", framework="numpy") as tensors:
        keys = tensors.keys()
        for key in keys:
            mapping = convert_peft_key(key)
            if mapping is None:
                skipped += 1
                continue
            tensor = tensors.get_tensor(key)
            target, transpose = mapping
            value = tensor.T if transpose else tensor
            if value.dtype == np.float32:
                value = value.astype(np.float16)
            if target in converted:
                raise ValueError(f"duplicate converted adapter key: {target}")
            converted[target] = value

    expected = len(LAYER_KEYS) * 36 * 2 + 4
    if len(converted) != expected:
        raise ValueError(f"expected {expected} LoRA tensors, converted {len(converted)}")
    output.mkdir(parents=True, exist_ok=True)
    save_file(converted, output / "adapters.safetensors")
    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": 36,
        "lora_parameters": {
            "rank": rank,
            "dropout": 0.0,
            "scale": alpha / rank,
            "keys": [*LAYER_KEYS, "model.embed_tokens", "lm_head"],
        },
        "source": "yyxdnmd/DiARC-adapters/qwen3-4b/arc-agi-2",
    }
    (output / "adapter_config.json").write_text(json.dumps(mlx_config, indent=2) + "\n")
    return len(converted), skipped
