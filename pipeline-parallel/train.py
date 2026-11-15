import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.data import micro_batches
from dtrain.model import Llama, LlamaConfig
from dtrain.pipeline_parallel import (
    PipelineParallel,
    PipelineStage,
    afab,
    one_f_one_b,
    schedule_stats,
)

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"
SCHEDULES = {"afab": afab, "1f1b": one_f_one_b}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=list(SCHEDULES), default="1f1b")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--grad-acc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    config = LlamaConfig(max_position_embeddings=args.seq, num_hidden_layers=args.layers)
    pp = PipelineParallel()
    stage = PipelineStage(Llama(config), pp)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr)
    batches = micro_batches(tokens, args.seq, args.mbs)
    shape = (args.mbs, args.seq, config.hidden_size)
    run = SCHEDULES[args.schedule]

    params = sum(p.numel() for p in stage.parameters())
    print(f"stage {pp.rank}: layers {list(stage.layers)}, {params:,} parameters", flush=True)
    dist.barrier()
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        sent = pp.sent
        loss = run(stage, batches, args.grad_acc, shape)
        optimizer.step()
        if step == 1:
            first = list(pp.log)
        if pp.last and (step == 1 or step % 25 == 0):
            rate = step * args.mbs * args.grad_acc * args.seq / (time.perf_counter() - start)
            kb = (pp.sent - sent) / 1e3
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s  {kb:,.0f} KB sent")

    logs = [None] * pp.world
    dist.all_gather_object(logs, first)
    if pp.last:
        stats = schedule_stats(logs)
        print(f"\n{args.schedule}, {pp.world} stages, {args.grad_acc} micro-batches, first step")
        for rank, log in enumerate(logs):
            print(f"  stage {rank}: {' '.join(log)}")
        print(f"  bubble, with backward = 2 × forward: {stats['bubble']:.0%}")
        print(f"  micro-batches held at once, per stage: {stats['in_flight']}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
