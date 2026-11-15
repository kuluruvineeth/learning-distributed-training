# Tensor parallelism

Data parallelism and [ZeRO](../zero/README.md) still run every layer whole on every rank. ZeRO-3 gathers a layer's full weights right before it runs, and the activations inside the layer are full size. What if one layer is too big for one GPU? Tensor parallelism splits the layer itself: each rank holds a slice of every weight matrix and computes a slice of every activation.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run pytest -q tests/test_tensor_parallel.py
uv run torchrun --nproc-per-node=2 --master-port=29500 tensor-parallel/train.py --steps 100
uv run torchrun --nproc-per-node=2 --master-port=29500 tensor-parallel/activations.py
uv run python tensor-parallel/estimate.py
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`. Everything runs on CPU with Gloo. No GPU is needed.

## Two ways to split a matmul

`Y = X · W`, with W split across 2 GPUs:

- **By columns**: `W = [W₁ W₂]`, so `Y = [X·W₁, X·W₂]`. Each GPU needs the full X and produces half the columns of Y. No communication.
- **By rows**: `W = [W₁; W₂]`, so `Y = X₁·W₁ + X₂·W₂`. Each GPU needs half of X's columns and produces a partial Y. The partial results are summed with an all-reduce.

The second needs its input split by columns, which is exactly what the first produces. So column then row costs one all-reduce, after the second matmul (Shoeybi et al. [2019](https://arxiv.org/abs/1909.08053)):

| block | column parallel | row parallel |
|---|---|---|
| attention | q_proj, k_proj, v_proj: each GPU gets whole heads | out_proj |
| MLP | gate_proj, up_proj: each GPU gets a slice of the 14,336 | down_proj |

The SiLU and the elementwise product work on the slice each GPU already has. Attention heads are independent, so each GPU runs attention for its own heads. With grouped-query attention, the key/value heads have to split evenly too: Llama 3 8B has 8, so 8 GPUs at most.

Backward mirrors forward. Every GPU needs the full gradient of a block's input, but each one only computed the part flowing through its own slice, so the input gradient is all-reduced: one all-reduce per block in forward, one in backward. The embedding is split by vocabulary rows: each GPU looks up the tokens in its range, writes zeros for the rest, and an all-reduce adds the pieces. The output projection is split by vocabulary columns and the logits are all-gathered for the loss.

## Code

`src/dtrain/tensor_parallel.py`: `apply_tensor_parallel(model, TensorParallel())` replaces the layers of an existing model and halves its head counts. The model code doesn't change.

- `Copy`: identity forward, all-reduce backward.
- `Reduce`: all-reduce forward, identity backward.
- `Gather`: all-gather forward, slice backward.
- `ColumnParallelLinear`, `RowParallelLinear` and `VocabParallelEmbedding` use them.

`Copy` sits once on each norm's output, as in Shoeybi et al., not on the input of every column-parallel linear: q, k and v share one all-reduce of their input gradient, so backward makes 2 all-reduces per layer instead of 5. The slices are cut from an existing model, so the tensor-parallel model starts from exactly the single-device weights.

## Tests

```bash
uv run pytest -q tests/test_tensor_parallel.py
```

- Logits and every gradient match the single-device model at 2 ranks, and at 4 ranks on a config with 4 key/value heads.
- 3 training steps with SGD and momentum match single-device weights (`rtol=1e-5`).
- One forward and backward makes exactly 10 all-reduces and 1 all-gather, and each rank holds 62,016 of the 123,712 parameters.
- With either all-reduce removed, or the embedding's zeroing removed, the tests fail.

```bash
uv run torchrun --nproc-per-node=2 --master-port=29500 tensor-parallel/train.py --steps 100
```

```
tp 2 · 62,016 parameters per rank · global batch 8
step    1  loss 5.886  22,542 tokens/s  1,573 KB sent
step   25  loss 3.538  58,639 tokens/s  1,573 KB sent
step   50  loss 2.757  59,723 tokens/s  1,573 KB sent
step   75  loss 2.590  58,513 tokens/s  1,573 KB sent
step  100  loss 2.290  57,667 tokens/s  1,573 KB sent
collectives per step: all_reduce 10, all_gather 1
```

The single-device loss, step for step. Both ranks see the same data: tensor parallel splits the model, not the batch.

## What it splits, and what it doesn't

```bash
uv run torchrun --nproc-per-node=2 --master-port=29500 tensor-parallel/activations.py
```

```
bytes autograd keeps for backward, seq 1024, 4 layers
                                1 device      tp 2, per rank
hidden size, 64 wide             8.13 MB             7.08 MB
logits, 256 wide                 1.05 MB             1.05 MB
inside attention and MLP        14.66 MB             7.95 MB
total                           23.83 MB            16.08 MB
```

Inside attention and the MLP, activations halve. Everything at the hidden size stays full on every rank: the residual stream, the norms, the input to every column-parallel matmul. So do the gathered logits. Sequence parallelism splits those along the sequence; it isn't implemented here.

## The cost

```bash
uv run python tensor-parallel/estimate.py
```

```
Llama 3 8B, one sequence of 8,192 tokens per step, bf16 activations

  tp   state per GPU    sent per GPU   compute per GPU
   1        160.6 GB          0.0 GB           1199 ms
   2         80.3 GB          8.6 GB            600 ms
   4         40.2 GB         12.9 GB            300 ms
   8         20.1 GB         15.0 GB            150 ms

128 all-reduces per step, each 8,192 × 4,096
in the middle of every layer: the next matmul waits for them

tp 8 over NVLink, inside a node: 33 ms of communication, 150 ms of compute
tp 8 over InfiniBand, across nodes: 301 ms of communication, 150 ms of compute
```

Data parallel's traffic scales with the parameters and can overlap the backward pass. Tensor parallel's scales with the activations and sits between matmuls: nothing can start until it's done. Over NVLink that is affordable, across nodes it costs more than the compute. That's why tensor parallel stays inside one node, usually at 8 GPUs at most. On this laptop it's slower than one process for the same reason: 10 blocking all-reduces per step on a model that computes for milliseconds.

The estimate uses peak link bandwidth (450 GB/s per direction for H100 NVLink, 50 GB/s for one 400 Gb/s InfiniBand port), 40% MFU, and leaves out the embedding and output all-reduces.

## What it doesn't fix

- Activations at the hidden size stay replicated (sequence parallelism).
- The split is capped by the number of heads and by the size of one node.
- A model too deep for one node's worth of tensor parallelism still doesn't fit. Pipeline parallelism splits it by layers, across nodes.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/tensor-parallelism/tensor-parallelism.excalidraw) to follow the diagrams.
