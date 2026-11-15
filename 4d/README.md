# 4D parallelism

Each split so far fixes one limit and leaves the others. Data parallelism adds throughput but not memory. Tensor parallelism splits layers but stays inside a node. Pipeline parallelism crosses nodes but needs many micro-batches. Context parallelism splits long sequences but not the weights. Can all four run at once, on the same model, with every collective reaching exactly the right GPUs?

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run pytest -q tests/test_grid.py
uv run torchrun --nproc-per-node=16 --master-port=29500 4d/train.py --dp 2 --pp 2 --cp 2 --tp 2 --steps 100
uv run python 4d/estimate.py
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`. Everything runs on CPU with Gloo. No GPU is needed.

## The grid

Give every GPU one coordinate per dimension. With 16 GPUs = dp 2 × pp 2 × cp 2 × tp 2, the GPU numbers are laid out tp fastest, then cp, then pp, then dp. A group is the set of GPUs that agree on every coordinate except one:

```bash
uv run torchrun --nproc-per-node=16 --master-port=29500 4d/train.py --dp 2 --pp 2 --cp 2 --tp 2 --steps 100
```

```
16 GPUs = dp 2 × pp 2 × cp 2 × tp 2 · ZeRO-1 · global batch 8
 GPU  dp pp cp tp tp group   pp group   cp group   dp group    parameters
   0   0  0  0  0 [0, 1]     [0, 4]     [0, 2]     [0, 8]          30,976
   1   0  0  0  1 [0, 1]     [1, 5]     [1, 3]     [1, 9]          30,976
   2   0  0  1  0 [2, 3]     [2, 6]     [0, 2]     [2, 10]         30,976
   ...
   5   0  1  0  1 [4, 5]     [1, 5]     [5, 7]     [5, 13]         31,040
   ...
  15   1  1  1  1 [14, 15]   [11, 15]   [13, 15]   [7, 15]         31,040
```

GPU 5 is in four groups at once: it all-reduces with 4 inside its layers (tp), receives activations from 1 (pp), passes keys and values with 7 (cp), and shares its weights with 13 (dp). Each collective names its group; an all-reduce on the wrong group mixes numbers that should never meet.

tp is fastest so tensor-parallel partners are neighbouring GPU numbers, on the same node and its NVLink. dp is slowest, so it spans nodes. This is the order Llama 3 uses ([TP, CP, PP, DP], Llama 3 paper, Figure 5): in their 16-GPU example GPU 0 shares tp with 1, cp with 2, pp with 4 and dp with 8, the same as above.

## One more group: cp × dp

Context-parallel GPUs hold the same weights and see different tokens, exactly like data-parallel GPUs see different sequences. So gradients are averaged across cp × dp together: GPU 0 averages with 2, 8 and 10.

## Code

`src/dtrain/grid.py`

- `Grid(dp, pp, cp, tp)`: coordinates, groups and members.
- `parallelize(model, grid, zero_stage, optimizer)` wraps the model.
- `grid_batches` hands out rows by dp rank and tokens by cp rank.

`parallelize` applies the wrappers in order, each with its own group: tensor parallel on `tp`, context parallel on `cp`, the pipeline stage on `pp`, then ZeRO on `cp_dp`. Nothing in the earlier code changed: each wrapper already took a process group and defaulted to all GPUs.

## Tests

```bash
uv run pytest -q tests/test_grid.py
```

- The groups for dp 2 × pp 2 × tp 2 are the ones worked out by hand: GPU 5 has tp [4, 5], pp [5, 7], dp [1, 5].
- dp 2 × pp 2 × tp 2 with ZeRO-1, pp 2 × cp 2 × tp 2, dp 2 × cp 2 × tp 2 with ZeRO-2, and all four at 2 on 16 processes: after 3 steps of SGD with momentum, every weight shard matches single device.
- Averaging gradients over dp alone instead of cp × dp, or swapping the tp and dp axes, fails every configuration where it matters.

The 16-process run prints the single-device loss:

```
step    1  loss 5.886  3,071 tokens/s
step   25  loss 3.538  5,317 tokens/s
step   50  loss 2.757  5,305 tokens/s
step   75  loss 2.590  5,293 tokens/s
step  100  loss 2.290  5,322 tokens/s
sent per step by this GPU: tp 229 KB, cp 33 KB, pp 33 KB, dp 186 KB
```

Global batch 8 = mbs 2 × grad_acc 2 × dp 2, the same 8 sequences per step as the single-device run with mbs 8. 16 processes on one laptop, so slow: every collective here would be a real link on a cluster.

## At scale

```bash
uv run python 4d/estimate.py
```

```
Llama 3.1 405B layouts from the Llama 3 paper, training state per GPU
20 bytes per parameter, optimizer state sharded across cp × dp (ZeRO-1)

                              tp  cp  pp   dp    GPUs       weights     state
pre-training, 8,192 GPUs       8   1  16   64   8,192       3.17 B   26.0 GB
pre-training, 16,384 GPUs      8   1  16  128  16,384       3.17 B   25.7 GB
long context, 16,384 GPUs      8  16  16    8  16,384       3.17 B   25.7 GB

unsplit, the same model needs 8,117 GB on one GPU
```

The layouts are Table 4 of the [Llama 3 paper](https://arxiv.org/abs/2407.21783), which reports 38 to 43% MFU for them. Meta's dp is FSDP, which shards more than ZeRO-1; the state column is our ZeRO-1 estimate, before activations. For long context, cp goes from 1 to 16 and dp from 128 to 8: the same GPUs, the tokens of each sequence split instead of the batch.

## What's left

Every piece is built and tested against one device, on a toy model.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/4d-parallelism/4d-parallelism.excalidraw) to follow the diagrams.
