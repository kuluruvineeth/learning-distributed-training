from dtrain.model import LlamaConfig


def num_params(config: LlamaConfig) -> int:
    h, kv = config.hidden_size, config.num_key_value_heads * config.head_dim
    attention = 2 * h * h + 2 * h * kv
    mlp = 3 * h * config.intermediate_size
    per_layer = attention + mlp + 2 * h
    return config.num_hidden_layers * per_layer + 2 * config.vocab_size * h + h


def training_bytes(
    params: int, mixed_precision: bool = True, fp32_grad_accumulation: bool = False
) -> dict[str, int]:
    if not mixed_precision:
        return {"weights": 4 * params, "gradients": 4 * params, "optimizer": 8 * params}
    out = {
        "weights": 2 * params,
        "gradients": 2 * params,
        "master weights": 4 * params,
        "optimizer": 8 * params,
    }
    if fp32_grad_accumulation:
        out["fp32 gradients"] = 4 * params
    return out


ZERO_SHARDS = {
    1: {"master weights", "optimizer"},
    2: {"master weights", "optimizer", "gradients", "fp32 gradients"},
    3: {"master weights", "optimizer", "gradients", "fp32 gradients", "weights"},
}


def zero_bytes(params: int, stage: int, world: int) -> dict[str, float]:
    sharded = ZERO_SHARDS.get(stage, set())
    full = training_bytes(params, fp32_grad_accumulation=True)
    return {name: size / world if name in sharded else size for name, size in full.items()}


def activation_bytes(config: LlamaConfig, seq_length: int, micro_batch_size: int) -> int:
    h, a = config.hidden_size, config.num_attention_heads
    per_layer = seq_length * micro_batch_size * h * (34 + 5 * a * seq_length / h)
    return int(config.num_hidden_layers * per_layer)
