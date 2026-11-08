# Data parallelism

One device processes tokens too slowly. What if every rank keeps a full copy of the model, reads different rows, and averages the gradients before the optimizer step?

1. Rank 0's weights are broadcast, so every copy starts identical.
2. Rank r reads every N-th window: `micro_batches(tokens, seq, mbs, rank, world)`.
3. Forward and backward run locally.
4. Gradients are all-reduced and divided by N.
5. Every rank takes the same optimizer step, so the copies stay identical.

The training loop is the one from the [single-device baseline](../single-device/README.md). The wrapper in `src/dtrain/data_parallel.py` does the communication. Everything runs on CPU with Gloo. No GPU is needed.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run python single-device/train.py --steps 100
uv run torchrun --nproc-per-node=4 --master-port=29500 data-parallel/train.py --impl naive --steps 100
uv run torchrun --nproc-per-node=4 --master-port=29500 data-parallel/train.py --impl bucket --steps 100
uv run torchrun --nproc-per-node=4 --master-port=29500 data-parallel/train.py --mbs 1 --grad-acc 2 --steps 100
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`.

Global batch 8 every time:

| step | one device, mbs 8 | 4 ranks × mbs 2, naive | 4 ranks × mbs 2, bucketed | 4 ranks × mbs 1 × grad_acc 2 |
|---|---|---|---|---|
| 1 | 5.886 | 5.886 | 5.886 | 5.886 |
| 25 | 3.538 | 3.538 | 3.538 | 3.538 |
| 50 | 2.757 | 2.757 | 2.757 | 2.757 |
| 100 | 2.290 | 2.290 | 2.290 | 2.290 |

## Naive

A hook on every parameter all-reduces its gradient as soon as backward produces it, and waits. That's one blocking all-reduce per parameter tensor: 21 per step for this model, each paying the full cost of a collective.

## Bucketed

Gradients are copied into flat fp32 buffers of up to `--bucket-mb`. When a bucket's last gradient arrives, an asynchronous all-reduce starts and backward keeps going. At the end of backward the wrapper waits for every bucket and copies the results into `p.grad`. With the default 0.25 MiB cap that's 2 all-reduces per step instead of 21.

Buckets are filled from the output end, the order gradients arrive in during backward.

## Gradient accumulation

Only the last micro-batch needs to communicate. `train_step` turns off `require_backward_grad_sync` for the others, so gradients add up locally and are all-reduced once. With 2 micro-batches, naive stays at 21 all-reduces per step instead of 42, and bucketed stays at 2 instead of 4.

## Tests

```bash
uv run pytest -q tests/test_data_parallel.py
```

Naive and bucketed, on 2 and 4 ranks, with and without accumulation. Every rank starts from different random weights. After each of 3 steps, the gradients and weights match a single-device run on the same global batch.

## Speed

The script prints tokens per second. On a laptop the numbers move a lot between runs, but bucketed is consistently several times faster than naive. Four ranks still aren't faster than one process here: the model has 124K parameters, so each step is milliseconds of compute against a fixed cost per collective. Data parallelism pays off when a step's compute is much larger than its communication.

## What it costs

- Every rank stores the full training state: weights, gradients and optimizer. Adding ranks makes training faster, not smaller.
- Every step all-reduces every gradient: 2(N − 1)/N × the gradient size per rank, from [ring all-reduce](../ring-all-reduce/README.md).
- The global batch grows with N (`mbs × grad_acc × dp`). Past some size, a bigger batch stops helping training, which limits how far data parallelism alone scales.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/data-parallelism/data-parallelism.excalidraw) to follow the diagrams.
