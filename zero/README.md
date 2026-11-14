# ZeRO

Data parallelism keeps a full copy of the training state on every rank. Llama 3 8B needs 160.6 GB of it ([why distributed training](../why-distributed/README.md)), more than an 80 GB H100 holds, and adding GPUs doesn't help because every one of them stores the same 160.6 GB. What if each rank kept only a slice?

After the all-reduce, every rank holds the same gradient and runs the same optimizer step on the same numbers. Adam is elementwise: the update of weight i needs only weight i, its gradient and its two moments. So with N ranks, rank r can update 1/N of the weights and keep optimizer state only for those. That's ZeRO ([Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054)), in three stages:

| stage | each rank keeps 1/N of | communication per step |
|---|---|---|
| ZeRO-1 | optimizer state | reduce-scatter gradients, all-gather weights |
| ZeRO-2 | + gradients | same, but the reduce-scatter runs during backward |
| ZeRO-3 | + weights | all-gather each layer before forward and before backward, reduce-scatter gradients |

The wrapper is `Zero(module, stage, optimizer)` in `src/dtrain/zero.py`. Stage 0 is plain data parallelism with the same structure, so all four stages can be measured the same way. Everything runs on CPU with Gloo. No GPU is needed.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run python zero/memory.py
for s in 0 1 2 3; do
  uv run torchrun --nproc-per-node=4 --master-port=29500 zero/train.py --stage $s --steps 100
done
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`.

4 ranks, global batch 8, AdamW, fp32. Every stage prints the single-device loss at every logged step: 5.886, 3.538, 2.757, 2.590, 2.290.

| per rank | data parallel | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| weights | 494.8 KB | 494.8 KB | 494.8 KB | 123.7 KB |
| gradients | 494.8 KB | 494.8 KB | 123.7 KB | 123.7 KB |
| optimizer | 989.7 KB | 247.4 KB | 247.4 KB | 247.4 KB |
| total | 1,979.4 KB | 1,237.1 KB | 866.0 KB | 494.8 KB |
| sent per step | 742 KB | 742 KB | 742 KB | 1,113 KB |

## Units and slices

The model is split into units: the embedding, each decoder layer, the final norm and the final projection. Each unit's weights are packed into one flat buffer, padded to a multiple of N, and rank r owns the r-th contiguous slice. The optimizer only ever sees the slices. For stages 1 and 2 each slice is a view into the flat buffer, so updating the slice updates the model.

## ZeRO-1: optimizer state

1. Backward runs as in data parallelism, each rank on its own micro-batch.
2. Reduce-scatter the gradients: rank r gets the summed gradient for its slice only.
3. Rank r runs the optimizer on its slice.
4. All-gather the updated slices, so every rank has every weight again.

A reduce-scatter followed by an all-gather is how [ring all-reduce](../ring-all-reduce/README.md) works, so ZeRO-1 sends the same bytes as data parallelism and does 1/N of the optimizer work.

## ZeRO-2: gradients

After the reduce-scatter, a rank only needs the gradient for its own slice. ZeRO-2 reduce-scatters each unit's gradients as soon as backward finishes that unit and keeps only its slice. Same bytes as ZeRO-1.

The catch is gradient accumulation. ZeRO-1 adds micro-batch gradients into its full buffer and communicates once per step. ZeRO-2 has no full buffer, so it reduce-scatters after every micro-batch. With `--mbs 1 --grad-acc 2`, ZeRO-1 still sends 742 KB per step and ZeRO-2 sends 1,113 KB: 10 reduce-scatters instead of 5.

## ZeRO-3: weights

Each rank keeps 1/N of the weights too. Before a unit runs, its weights are all-gathered, then freed right after. Backward needs them again, so they're all-gathered a second time before the gradients are reduce-scattered. Two all-gathers and one reduce-scatter: 1.5× the bytes of data parallelism. PyTorch FSDP's `FULL_SHARD` is this stage, and `SHARD_GRAD_OP` is ZeRO-2.

Left out on purpose: this version gathers a unit only when it's needed. Real implementations prefetch the next unit while the current one computes, and train in bf16 with fp32 master weights. Everything here is fp32.

## What it saves

`zero/memory.py` prints the training state per GPU at 20 bytes per parameter, before activations:

| | GPUs | data parallel | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| Llama 3 8B | 8 | 160.6 GB | 76.3 GB | 34.1 GB | 20.1 GB |
| Llama 3 8B | 64 | 160.6 GB | 65.7 GB | 18.3 GB | 2.5 GB |
| Llama 3 70B | 8 | 1,411.1 GB | 670.3 GB | 299.9 GB | 176.4 GB |
| Llama 3 70B | 64 | 1,411.1 GB | 577.7 GB | 161.0 GB | 22.0 GB |

ZeRO-1 and ZeRO-2 stop shrinking once the unsharded part dominates. ZeRO-1 always keeps 8 bytes per parameter (bf16 weights, bf16 gradients, fp32 gradients) and ZeRO-2 keeps 2. Only ZeRO-3 keeps dividing.

## Tests

```bash
uv run pytest -q tests/test_zero.py
```

- Every stage, on 2 and 4 ranks, with and without accumulation, starting from different random weights on every rank, matches a single-device run after 3 steps.
- With AdamW, ZeRO-1, 2 and 3 produce the same weights as data parallelism bit for bit. With accumulation, ZeRO-2 and 3 add the micro-batches in a different order, (a₁+b₁)+(a₂+b₂) instead of (a₁+a₂)+(b₁+b₂), so they match to float tolerance instead.
- The measured per-rank state and bytes sent match the formulas.

## Speed

On a laptop every stage runs slower than plain data parallelism. Each stage replaces one all-reduce per unit with two calls, and with 124K parameters each step is mostly the fixed cost per call. ZeRO trades messages for memory.

## What it doesn't fix

- Activations. ZeRO splits the training state, not what forward keeps for backward ([single-device baseline](../single-device/README.md)).
- One unit still has to fit: ZeRO-3 gathers a whole layer's weights before it runs.
- The global batch still grows with every rank added.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/zero/zero.excalidraw) to follow the diagrams.
