import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.data import micro_batches
from dtrain.data_parallel import average_across
from dtrain.model import Llama, LlamaConfig
from dtrain.train import train_step
from dtrain.zero import Zero

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    model = Zero(
        Llama(LlamaConfig(max_position_embeddings=args.seq)),
        args.stage,
        lambda params: torch.optim.AdamW(params, lr=args.lr),
    )
    batches = micro_batches(tokens, args.seq, args.mbs, rank, world)
    tokens_per_step = args.mbs * args.grad_acc * world * args.seq

    if rank == 0:
        batch = args.mbs * args.grad_acc * world
        shape = f"mbs {args.mbs} × grad_acc {args.grad_acc} × dp {world}"
        print(f"ZeRO stage {args.stage} · global batch {batch} = {shape}")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.zero_grad()
        sent = sum(model.sent.values())
        loss = average_across(train_step(model, batches, args.grad_acc))
        model.step()
        sent = sum(model.sent.values()) - sent
        if rank == 0 and (step == 1 or step % 25 == 0):
            rate = step * tokens_per_step / (time.perf_counter() - start)
            kb = sent / 1e3
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s  {kb:,.0f} KB sent")

    state = model.state_bytes()
    if rank == 0:
        print(f"\nper rank, after {args.steps} steps")
        for name, size in state.items():
            print(f"  {name:<10} {size / 1e3:8,.1f} KB")
        print(f"  {'total':<10} {sum(state.values()) / 1e3:8,.1f} KB")
        calls = ", ".join(f"{n} {c // args.steps}" for n, c in model.calls.items() if c)
        print(f"  collectives per step: {calls}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
