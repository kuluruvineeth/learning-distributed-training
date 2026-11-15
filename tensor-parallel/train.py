import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.data import micro_batches
from dtrain.model import Llama, LlamaConfig
from dtrain.tensor_parallel import TensorParallel, apply_tensor_parallel
from dtrain.train import train_step

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=8)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    tp = TensorParallel()
    model = apply_tensor_parallel(Llama(LlamaConfig(max_position_embeddings=args.seq)), tp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = micro_batches(tokens, args.seq, args.mbs)
    tokens_per_step = args.mbs * args.grad_acc * args.seq

    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        batch = args.mbs * args.grad_acc
        print(f"tp {tp.world} · {params:,} parameters per rank · global batch {batch}")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        sent = sum(tp.sent.values())
        loss = train_step(model, batches, args.grad_acc)
        optimizer.step()
        sent = sum(tp.sent.values()) - sent
        if rank == 0 and (step == 1 or step % 25 == 0):
            rate = step * tokens_per_step / (time.perf_counter() - start)
            kb = sent / 1e3
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s  {kb:,.0f} KB sent")
    if rank == 0:
        calls = ", ".join(f"{n} {c // args.steps}" for n, c in tp.calls.items())
        print(f"collectives per step: {calls}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
