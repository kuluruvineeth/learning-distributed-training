import argparse
import csv
import dataclasses
import gc
import resource
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.data import micro_batches
from dtrain.data_parallel import average_across
from dtrain.grid import Grid, grid_batches, split_model
from dtrain.hf import load_config, load_weights
from dtrain.model import Llama
from dtrain.pipeline_parallel import one_f_one_b
from dtrain.tensor_parallel import full
from dtrain.train import train_step
from dtrain.zero import Zero

CACHE = Path(__file__).resolve().parent / ".cache"
MODEL = CACHE / "SmolLM2-135M"


def pretrained(seq: int) -> Llama:
    config = dataclasses.replace(load_config(MODEL), max_position_embeddings=seq)
    model = Llama(config)
    model.load_state_dict(load_weights(MODEL), strict=True)
    return model


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def single_device(args, tokens, log):
    torch.set_num_threads(args.threads)
    model = pretrained(args.seq)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = micro_batches(tokens, args.seq, args.mbs)
    print(f"1 process · {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = train_step(model, batches, args.grad_acc)
        optimizer.step()
        log(step, loss, time.perf_counter() - start)
    torch.save(model.state_dict(), args.out / "weights.pt")
    print(f"peak memory {peak_mb():,.0f} MB")


def four_d(args, tokens, log):
    torch.set_num_threads(1)
    grid = Grid(dp=args.dp, pp=args.pp, cp=args.cp, tp=args.tp)
    for turn in range(dist.get_world_size()):
        if turn == grid.rank:
            stage, tp, cp = split_model(pretrained(args.seq), grid)
            gc.collect()
        dist.barrier()
    zero = Zero(stage, args.zero, lambda p: torch.optim.AdamW(p, lr=args.lr), grid.groups["cp_dp"])
    batches = grid_batches(tokens, args.seq, args.mbs, grid, cp)
    shape = (args.mbs, args.seq // args.cp, stage.hidden)
    reporter = grid.coords == {"dp": 0, "pp": args.pp - 1, "cp": 0, "tp": 0}
    if grid.rank == 0:
        sizes = " × ".join(f"{d} {n}" for d, n in grid.sizes.items())
        print(f"{dist.get_world_size()} processes = {sizes} · ZeRO-{args.zero}", flush=True)
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        zero.zero_grad()
        loss = one_f_one_b(stage, batches, args.grad_acc, shape)
        loss = average_across(loss, grid.groups["cp_dp"])
        zero.step()
        if reporter:
            log(step, loss, time.perf_counter() - start)

    weights = {
        name: full(p, p.detach(), tp)
        for name, p in zip(stage.names(), stage.parameters(), strict=True)
    }
    if grid.coords["dp"] == grid.coords["cp"] == grid.coords["tp"] == 0:
        torch.save(weights, args.out / f"stage{grid.coords['pp']}.pt")
    peaks = [None] * dist.get_world_size()
    dist.all_gather_object(peaks, peak_mb())
    if grid.rank == 0:
        print(f"peak memory per process {max(peaks):,.0f} MB, all {sum(peaks):,.0f} MB")


def main():
    parser = argparse.ArgumentParser()
    for dim in ("dp", "pp", "cp", "tp"):
        parser.add_argument(f"--{dim}", type=int, default=1)
    parser.add_argument("--zero", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--mbs", type=int, default=8)
    parser.add_argument("--grad-acc", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    args.out = CACHE / "runs" / args.name
    args.out.mkdir(parents=True, exist_ok=True)

    dist.init_process_group("gloo")
    tokens = torch.load(CACHE / "tinystories.pt").long()
    path = args.out / "loss.csv"

    def log(step, loss, seconds):
        with open(path, "w" if step == 1 else "a", newline="") as f:
            writer = csv.writer(f)
            if step == 1:
                writer.writerow(("step", "loss", "seconds"))
            writer.writerow((step, f"{loss:.6f}", f"{seconds:.2f}"))
        if step == 1 or step % 10 == 0:
            print(f"step {step:>4}  loss {loss:.4f}  {seconds:5.2f} s/step", flush=True)

    if dist.get_world_size() == 1:
        single_device(args, tokens, log)
    else:
        four_d(args, tokens, log)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
