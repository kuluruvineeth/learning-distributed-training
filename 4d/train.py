import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dtrain.data_parallel import average_across
from dtrain.grid import Grid, grid_batches, parallelize
from dtrain.model import Llama, LlamaConfig
from dtrain.pipeline_parallel import one_f_one_b

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"


def main():
    parser = argparse.ArgumentParser()
    for dim in ("dp", "pp", "cp", "tp"):
        parser.add_argument(f"--{dim}", type=int, default=1)
    parser.add_argument("--zero", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--grad-acc", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    grid = Grid(dp=args.dp, pp=args.pp, cp=args.cp, tp=args.tp)
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    config = LlamaConfig(max_position_embeddings=args.seq)
    stage, zero, tp, cp = parallelize(
        Llama(config), grid, args.zero, lambda params: torch.optim.AdamW(params, lr=args.lr)
    )
    batches = grid_batches(tokens, args.seq, args.mbs, grid, cp)
    shape = (args.mbs, args.seq // args.cp, config.hidden_size)
    reporter = grid.coords == {"dp": 0, "pp": args.pp - 1, "cp": 0, "tp": 0}

    rows = [None] * dist.get_world_size()
    params = sum(p.numel() for p in stage.parameters())
    dist.all_gather_object(rows, (grid.rank, grid.coords, grid.members, params))
    if grid.rank == 0:
        sizes = " × ".join(f"{d} {n}" for d, n in grid.sizes.items())
        batch = args.mbs * args.grad_acc * args.dp
        print(f"{dist.get_world_size()} GPUs = {sizes} · ZeRO-{args.zero} · global batch {batch}")
        print(
            f"{'GPU':>4}  {'dp pp cp tp':<12}{'tp group':<11}{'pp group':<11}"
            f"{'cp group':<11}{'dp group':<11}{'parameters':>11}"
        )
        for rank, c, m, n in rows:
            coords = " ".join(f"{c[d]:>2}" for d in ("dp", "pp", "cp", "tp"))
            groups = "".join(f"{str(m[g]):<11}" for g in ("tp", "pp", "cp", "dp"))
            print(f"{rank:>4}  {coords:<12}{groups}{n:>11,}")
        print()

    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        zero.zero_grad()
        loss = one_f_one_b(stage, batches, args.grad_acc, shape)
        loss = average_across(loss, grid.groups["cp_dp"])
        zero.step()
        if reporter and (step == 1 or step % 25 == 0):
            seen = step * args.mbs * args.grad_acc * args.dp * args.seq
            rate = seen / (time.perf_counter() - start)
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s", flush=True)

    dist.barrier()
    if reporter:
        sent = {
            "tp": sum(tp.sent.values()),
            "cp": cp.sent,
            "pp": stage.pp.sent,
            "dp": sum(zero.sent.values()),
        }
        per_step = ", ".join(f"{k} {v / args.steps / 1e3:,.0f} KB" for k, v in sent.items())
        print(f"sent per step by this GPU: {per_step}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
