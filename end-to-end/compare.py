import csv
import dataclasses
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dtrain.hf import load_config, load_weights
from dtrain.model import Llama

CACHE = Path(__file__).resolve().parent / ".cache"
MODEL = CACHE / "SmolLM2-135M"
RUNS = CACHE / "runs"
PROMPT = "Tom wanted to fly his kite, but"


def read(name):
    with open(RUNS / name / "loss.csv") as f:
        rows = list(csv.DictReader(f))
    return [float(r["loss"]) for r in rows], [float(r["seconds"]) for r in rows]


def model(weights):
    config = dataclasses.replace(load_config(MODEL), max_position_embeddings=128)
    out = Llama(config).eval()
    out.load_state_dict(weights, strict=True)
    return out


@torch.no_grad()
def generate(llama, tokenizer, tokens=40):
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    for _ in range(tokens):
        ids = torch.cat([ids, llama(ids)[:, -1:].argmax(-1)], dim=1)
    return tokenizer.decode(ids[0])


def main():
    single, single_s = read("single")
    grid, grid_s = read("parallel")
    print(f"{'step':>5}{'1 process':>12}{'12 processes':>15}{'difference':>13}")
    for step in [1, *range(10, len(single) + 1, 10)]:
        a, b = single[step - 1], grid[step - 1]
        print(f"{step:>5}{a:>12.4f}{b:>15.4f}{abs(a - b):>13.1e}")
    worst = max(abs(a - b) for a, b in zip(single, grid, strict=True))
    print(f"largest loss difference over {len(single)} steps: {worst:.1e}")
    print(
        f"seconds per step after step 1: 1 process {sum(single_s[1:]) / (len(single_s) - 1):.1f}, "
        f"12 processes {sum(grid_s[1:]) / (len(grid_s) - 1):.1f}\n"
    )

    trained = torch.load(RUNS / "single" / "weights.pt")
    merged = {}
    for stage in sorted((RUNS / "parallel").glob("stage*.pt")):
        merged.update(torch.load(stage))
    diffs = {name: (trained[name] - merged[name]).abs().max().item() for name in trained}
    name = max(diffs, key=diffs.get)
    print(f"{len(merged)} tensors gathered from the 12 processes")
    print(f"largest weight difference after training: {diffs[name]:.1e} ({name})\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    for label, weights in [
        ("SmolLM2-135M before training", load_weights(MODEL)),
        ("after 100 steps, 1 process", trained),
        ("after 100 steps, 12 processes", merged),
    ]:
        print(f"{label}:\n  {generate(model(weights), tokenizer)!r}\n")


if __name__ == "__main__":
    main()
