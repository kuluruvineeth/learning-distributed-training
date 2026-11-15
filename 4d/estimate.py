from dtrain.memory import zero_bytes

LLAMA31_405B = 405_853_388_800
LAYOUTS = [
    ("pre-training, 8,192 GPUs", dict(tp=8, cp=1, pp=16, dp=64)),
    ("pre-training, 16,384 GPUs", dict(tp=8, cp=1, pp=16, dp=128)),
    ("long context, 16,384 GPUs", dict(tp=8, cp=16, pp=16, dp=8)),
]


def main():
    print("Llama 3.1 405B layouts from the Llama 3 paper, training state per GPU")
    print("20 bytes per parameter, optimizer state sharded across cp × dp (ZeRO-1)\n")
    print(f"{'':<28}{'tp':>4}{'cp':>4}{'pp':>4}{'dp':>5}{'GPUs':>8}{'weights':>14}{'state':>10}")
    for name, s in LAYOUTS:
        gpus = s["tp"] * s["cp"] * s["pp"] * s["dp"]
        local = LLAMA31_405B / (s["tp"] * s["pp"])
        state = sum(zero_bytes(local, 1, s["cp"] * s["dp"]).values())
        dims = f"{s['tp']:>4}{s['cp']:>4}{s['pp']:>4}{s['dp']:>5}"
        print(f"{name:<28}{dims}{gpus:>8,}{local / 1e9:>11.2f} B{state / 1e9:>7.1f} GB")
    print("\nunsplit, the same model needs 8,117 GB on one GPU")


if __name__ == "__main__":
    main()
