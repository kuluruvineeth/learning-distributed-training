from dtrain.memory import num_params, zero_bytes
from dtrain.model import LlamaConfig

GPU = 80e9
MODELS = {
    "Llama 3 8B": LlamaConfig(
        vocab_size=128_256,
        hidden_size=4096,
        intermediate_size=14_336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "Llama 3 70B": LlamaConfig(
        vocab_size=128_256,
        hidden_size=8192,
        intermediate_size=28_672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
    ),
}
STAGES = ["data parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"]


def main():
    for name, config in MODELS.items():
        params = num_params(config)
        print(f"{name}: {params:,} parameters, training state per GPU")
        print(f"  {'':>8} " + "".join(f"{s:>14}" for s in STAGES))
        for world in (8, 64):
            row = [sum(zero_bytes(params, stage, world).values()) for stage in range(4)]
            cells = "".join(f"{size / 1e9:>10,.1f} GB{' ' if size <= GPU else '*'}" for size in row)
            print(f"  {world:>3} GPUs {cells}")
        print()
    print("* more than one 80 GB GPU holds, before a single activation")

    params = num_params(MODELS["Llama 3 8B"])
    print("\nLlama 3 8B on 8 GPUs, per GPU")
    print(f"  {'':<16}" + "".join(f"{s:>14}" for s in STAGES))
    for part in zero_bytes(params, 0, 8):
        cells = "".join(f"{zero_bytes(params, s, 8)[part] / 1e9:>11.1f} GB" for s in range(4))
        print(f"  {part:<16}{cells}")


if __name__ == "__main__":
    main()
