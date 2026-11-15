import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.context_parallel import ContextParallel, apply_context_parallel, split_batches
from dtrain.data import micro_batches
from dtrain.data_parallel import average_across
from dtrain.model import Llama, LlamaConfig
from dtrain.train import train_step
from dtrain.zero import Zero

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["contiguous", "zigzag"], default="zigzag")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    cp = ContextParallel(layout=args.layout)
    model = apply_context_parallel(Llama(LlamaConfig(max_position_embeddings=args.seq)), cp)
    model = Zero(model, 0, lambda params: torch.optim.AdamW(params, lr=args.lr))
    batches = split_batches(micro_batches(tokens, args.seq, args.mbs), cp)

    if rank == 0:
        local = args.seq // cp.world
        print(f"cp {cp.world}, {args.layout} · {local} of {args.seq} tokens per sequence per rank")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.zero_grad()
        ring, grads, scores = cp.sent, sum(model.sent.values()), cp.scores
        loss = average_across(train_step(model, batches, grad_acc=1))
        model.step()
        ring, grads, scores = cp.sent - ring, sum(model.sent.values()) - grads, cp.scores - scores
        if rank == 0 and (step == 1 or step % 25 == 0):
            rate = step * args.mbs * args.seq / (time.perf_counter() - start)
            sent = f"ring {ring / 1e3:,.0f} KB, gradients {grads / 1e3:,.0f} KB"
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s  {sent}")
    counts = [None] * cp.world
    dist.all_gather_object(counts, scores)
    if rank == 0:
        print(f"query-key scores computed per rank, last step: {counts}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
