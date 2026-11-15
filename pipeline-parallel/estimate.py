from dtrain.memory import training_bytes
from dtrain.model import LlamaConfig
from dtrain.pipeline_parallel import split_layers

LLAMA3_8B = LlamaConfig(
    vocab_size=128_256,
    hidden_size=4096,
    intermediate_size=14_336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
)
SEQ, BYTES = 8192, 2


def stage_params(config: LlamaConfig, world: int) -> list[int]:
    h, kv = config.hidden_size, config.num_key_value_heads * config.head_dim
    layer = 2 * h * h + 2 * h * kv + 3 * h * config.intermediate_size + 2 * h
    table = config.vocab_size * h
    out = []
    for rank, layers in enumerate(split_layers(config.num_hidden_layers, world)):
        params = len(layers) * layer
        params += table if rank == 0 else 0
        params += table + h if rank == world - 1 else 0
        out.append(params)
    return out


def main():
    config = LLAMA3_8B
    per_param = sum(training_bytes(1, fp32_grad_accumulation=True).values())
    print("Llama 3 8B: training state per stage, 20 bytes per parameter")
    for world in (2, 4, 8):
        sizes = [p * per_param / 1e9 for p in stage_params(config, world)]
        print(f"  pp {world}: " + "  ".join(f"{s:5.1f}" for s in sizes) + "  GB")
    print("  the first stage carries the embedding, the last the output projection\n")

    activation = SEQ * config.hidden_size * BYTES
    print(f"between two stages, per micro-batch of one {SEQ:,}-token sequence:")
    print(f"  {activation / 1e6:.1f} MB forward, {activation / 1e6:.1f} MB of gradient back")
    print("  point to point, only to the neighbours\n")

    print("bubble: share of time a stage sits idle, (p - 1) / (m + p - 1)")
    print(f"  {'micro-batches':>14}" + "".join(f"{m:>7}" for m in (4, 8, 16, 32, 64)))
    for p in (2, 4, 8):
        row = "".join(f"{(p - 1) / (m + p - 1):>7.0%}" for m in (4, 8, 16, 32, 64))
        print(f"  {f'pp {p}':>14}{row}")


if __name__ == "__main__":
    main()
