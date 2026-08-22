import numpy as np

from arc_agent.diarc import convert_peft_key


def test_convert_peft_layer_keys_and_shapes() -> None:
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    converted = convert_peft_key(key)
    assert converted == ("model.layers.0.self_attn.q_proj.lora_a", True)
    tensor = np.zeros((256, 2560), dtype=np.float32)
    assert tensor.T.shape == (2560, 256)


def test_convert_peft_embedding_and_head_keys() -> None:
    assert convert_peft_key(
        "base_model.model.model.embed_tokens.lora_embedding_B"
    ) == ("model.embed_tokens.lora_b", True)
    assert convert_peft_key("base_model.model.lm_head.lora_B.weight") == (
        "lm_head.lora_b",
        True,
    )
    assert convert_peft_key("base_model.model.lm_head.base_layer.weight") is None
