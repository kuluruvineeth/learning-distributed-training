import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from dtrain.model import LlamaConfig

RENAMES = [
    (r"^model\.", ""),
    (r"^layers\.", "decoder_layers."),
    (r"^embed_tokens", "embedding"),
    (r"\.self_attn\.", ".attention."),
    (r"\.o_proj\.", ".out_proj."),
    (r"^lm_head", "final_proj"),
    (r"^norm\.", "final_norm."),
]


def hf_to_dtrain(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for name, tensor in state_dict.items():
        for pattern, replacement in RENAMES:
            name = re.sub(pattern, replacement, name)
        converted[name] = tensor
    return converted


def load_config(path: Path) -> LlamaConfig:
    hf = json.loads((Path(path) / "config.json").read_text())
    rope = hf.get("rope_parameters") or {}
    return LlamaConfig(
        vocab_size=hf["vocab_size"],
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["intermediate_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf["num_key_value_heads"],
        max_position_embeddings=hf["max_position_embeddings"],
        rms_norm_eps=hf["rms_norm_eps"],
        rope_theta=hf.get("rope_theta", rope.get("rope_theta")),
    )


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    weights = hf_to_dtrain(load_file(Path(path) / "model.safetensors"))
    weights.setdefault("final_proj.weight", weights["embedding.weight"])
    return {name: tensor.float() for name, tensor in weights.items()}
