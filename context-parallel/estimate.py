from dtrain.flops import train_flops_per_token
from dtrain.model import LlamaConfig

LLAMA3_8B = LlamaConfig(
    vocab_size=128_256,
    hidden_size=4096,
    intermediate_size=14_336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
)
SEQ, BYTES = 131_072, 2
PEAK, MFU = 989e12, 0.4
GPU = 80e9


def main():
    config = LLAMA3_8B
    layers, h = config.num_hidden_layers, config.hidden_size
    activations = layers * SEQ * h * 34
    kv_per_layer = SEQ * 2 * config.num_key_value_heads * config.head_dim * BYTES
    flops = train_flops_per_token(config, SEQ) * SEQ
    print(f"Llama 3 8B, one sequence of {SEQ:,} tokens, bf16, attention recomputed in tiles\n")
    print(f"activations for one sequence: {activations / 1e9:,.0f} GB")
    print(f"keys and values, one layer, whole sequence: {kv_per_layer / 1e6:,.0f} MB\n")
    print(f"{'cp':>4}{'tokens per GPU':>16}{'activations':>14}{'ring, sent':>13}{'compute':>11}")
    for cp in (1, 2, 4, 8, 16):
        sent = 2 * layers * kv_per_layer * (cp - 1) / cp
        seconds = flops / cp / (PEAK * MFU)
        act = activations / cp
        flag = "" if act < GPU else " *"
        row = f"{cp:>4}{SEQ // cp:>16,}{act / 1e9:>11.1f} GB{flag:<2}{sent / 1e9:>8.1f} GB"
        print(f"{row}{seconds:>9.1f} s")
    print("\n* more than an 80 GB GPU holds, before weights and optimizer state")
    print("ring: keys and values forward, their gradients back, (cp - 1)/cp of each per layer")


if __name__ == "__main__":
    main()
