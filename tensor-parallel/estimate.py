from dtrain.flops import train_flops_per_token
from dtrain.memory import num_params, training_bytes
from dtrain.model import LlamaConfig

LLAMA3_8B = LlamaConfig(
    vocab_size=128_256,
    hidden_size=4096,
    intermediate_size=14_336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
)
SEQ, BYTES = 8192, 2
PEAK, MFU = 989e12, 0.4
LINKS = {"NVLink, inside a node": 450e9, "InfiniBand, across nodes": 50e9}


def main():
    config = LLAMA3_8B
    params = num_params(config)
    norms = (2 * config.num_hidden_layers + 1) * config.hidden_size
    state = sum(training_bytes(params, fp32_grad_accumulation=True).values()) / params
    print(f"Llama 3 8B, one sequence of {SEQ:,} tokens per step, bf16 activations\n")
    print(f"{'tp':>4}{'state per GPU':>16}{'sent per GPU':>16}{'compute per GPU':>18}")
    rows = []
    for tp in (1, 2, 4, 8):
        per_gpu = (params - norms) / tp + norms
        activation = SEQ * config.hidden_size * BYTES
        all_reduces = 4 * config.num_hidden_layers
        sent = all_reduces * activation * 2 * (tp - 1) / tp
        flops = train_flops_per_token(config, SEQ) * SEQ / tp
        seconds = flops / (PEAK * MFU)
        rows.append((tp, sent, seconds))
        gb, ms = per_gpu * state / 1e9, seconds * 1e3
        print(f"{tp:>4}{gb:>13.1f} GB{sent / 1e9:>13.1f} GB{ms:>15.0f} ms")
    count, shape = 4 * config.num_hidden_layers, f"{SEQ:,} × {config.hidden_size:,}"
    print(f"\n{count} all-reduces per step, each {shape}")
    print("in the middle of every layer: the next matmul waits for them\n")
    tp, sent, seconds = rows[-1]
    for name, bandwidth in LINKS.items():
        wait = sent / bandwidth
        talk, work = wait * 1e3, seconds * 1e3
        print(f"tp {tp} over {name}: {talk:.0f} ms of communication, {work:.0f} ms of compute")


if __name__ == "__main__":
    main()
