import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.data import micro_batches
from dtrain.data_parallel import DataParallelBucket, DataParallelNaive, average_across
from dtrain.model import Llama, LlamaConfig
from dtrain.train import train_step

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["naive", "bucket"], default="bucket")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bucket-mb", type=float, default=0.25)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    model = Llama(LlamaConfig(max_position_embeddings=args.seq))
    if args.impl == "naive":
        model = DataParallelNaive(model)
    else:
        model = DataParallelBucket(model, bucket_cap_mb=args.bucket_mb)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = micro_batches(tokens, args.seq, args.mbs, rank, world)
    tokens_per_step = args.mbs * args.grad_acc * world * args.seq

    if rank == 0:
        batch = args.mbs * args.grad_acc * world
        shape = f"mbs {args.mbs} × grad_acc {args.grad_acc} × dp {world}"
        print(f"{args.impl} · global batch {batch} = {shape}")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        calls = model.all_reduce_calls
        loss = average_across(train_step(model, batches, args.grad_acc))
        optimizer.step()
        if args.impl == "bucket":
            model.reset()
        if rank == 0 and (step == 1 or step % 25 == 0):
            rate = step * tokens_per_step / (time.perf_counter() - start)
            calls = model.all_reduce_calls - calls
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s  {calls} all-reduces")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
