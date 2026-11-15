# Context parallelism

Activations grow with the sequence, and nothing so far splits the sequence. Data parallelism, ZeRO and pipeline parallelism all process whole sequences on every GPU. Tensor parallelism splits the inside of a layer but keeps every hidden-size activation full ([tensor parallelism](../tensor-parallel/README.md)). One 131,072-token sequence through Llama 3 8B keeps about 584 GB of activations for backward, even with attention recomputed in tiles. What if each GPU took only a chunk of the tokens? That's context parallelism.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run pytest -q tests/test_context_parallel.py
uv run torchrun --nproc-per-node=4 --master-port=29500 context-parallel/train.py --layout zigzag --steps 100
uv run torchrun --nproc-per-node=4 --master-port=29500 context-parallel/activations.py
uv run python context-parallel/estimate.py
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`. Everything runs on CPU with Gloo. No GPU is needed.

## Everything but attention is per token

Embedding, norms, every matmul, the MLP, the loss: each works on one token at a time. Give a GPU a quarter of the tokens and it does a quarter of that work with a quarter of the activations, no communication needed. Attention is the exception: a token attends to every earlier token, including those on other GPUs.

## Ring attention

Each GPU keeps its queries and passes its keys and values around a ring. After N − 1 hops every GPU has seen every chunk. Attention over one chunk of keys gives a partial result and a log-sum-exp; the partial results merge exactly, the way FlashAttention merges tiles:

```
lse = log(exp(lse₁) + exp(lse₂))
out = exp(lse₁ − lse)·out₁ + exp(lse₂ − lse)·out₂
```

With a causal mask, a chunk only attends to chunks at or before it. Chunks after it are skipped.

## Balance

Split the sequence into 4 contiguous chunks and GPU 3 attends to 4 chunks while GPU 0 attends to 1. Split it into 8 chunks instead and give GPU r chunks r and 7 − r, one early and one late, and every GPU does the same work. Llama 3 balances its context parallelism this way.

## Code

`src/dtrain/context_parallel.py`: `apply_context_parallel(model, ContextParallel(layout="zigzag"))` swaps each layer's attention for `ring_attention` and keeps the RoPE angles of this GPU's positions. `cp.split` cuts inputs and targets the same way. The gradients are averaged across the context-parallel GPUs like data parallel, here with `Zero(model, stage=0)`.

- `ring_attention` is one autograd function. Forward makes N − 1 hops; backward sends the key and value gradients back N − 1 hops, adding each GPU's share on the way. Each block of attention is plain PyTorch, recomputed in backward with `checkpoint`, so autograd derives the rest. The backward ring has to run on every GPU even when a GPU didn't use the later chunks, or its neighbours wait forever: a version with one autograd step per hop deadlocks with contiguous chunks.
- The hops happen before attention starts, so every GPU ends up holding the whole sequence's keys and values, like Llama 3's all-gather variant. A production ring computes each block as it arrives, overlapping the next hop, and keeps two blocks.
- `ContextParallel(layout="zigzag")` balances the work; `layout="contiguous"` is there to compare.

## Tests

```bash
uv run pytest -q tests/test_context_parallel.py
```

- Logits and every gradient match the single-device model, 2 and 4 GPUs, contiguous and zigzag.
- 3 training steps with SGD and momentum match single-device weights.
- Work per GPU: contiguous chunks give GPU r a share proportional to r + 1, zigzag gives all four the same.
- Without the causal mask, without the shifted positions, or without the gradients passed back around the ring, the tests fail.

```bash
uv run torchrun --nproc-per-node=4 --master-port=29500 context-parallel/train.py --layout zigzag --steps 100
```

```
cp 4, zigzag · 16 of 64 tokens per sequence per rank
step    1  loss 5.886  3,984 tokens/s  ring 393 KB, gradients 742 KB
step   25  loss 3.538  6,113 tokens/s  ring 393 KB, gradients 742 KB
step   50  loss 2.757  5,537 tokens/s  ring 393 KB, gradients 742 KB
step   75  loss 2.590  5,767 tokens/s  ring 393 KB, gradients 742 KB
step  100  loss 2.290  5,572 tokens/s  ring 393 KB, gradients 742 KB
query-key scores computed per rank, last step: [1152, 1152, 1152, 1152]
```

The single-device loss at every step. With `--layout contiguous` the same losses, and `[512, 1024, 1536, 2048]` query-key scores per rank. On this laptop zigzag runs slower: twice as many blocks, each a quarter the size, and at this size the per-block overhead outweighs the balance. The balance pays off when blocks are large.

```bash
uv run torchrun --nproc-per-node=4 --master-port=29500 context-parallel/activations.py
```

```
bytes autograd keeps for backward, seq 1024, 4 layers
                               1 device  cp 4, zigzag:     rank 0     rank 1     rank 2     rank 3
hidden size, 64 wide            8.13 MB                   2.03 MB    2.03 MB    2.03 MB    2.03 MB
logits, 256 wide                1.05 MB                   0.27 MB    0.27 MB    0.27 MB    0.27 MB
inside attention and MLP       14.66 MB                   5.51 MB    5.51 MB    5.51 MB    5.51 MB
```

The hidden-size activations and the logits drop to exactly a quarter: tensor parallelism couldn't split those. The rest is a little over a quarter because every GPU keeps the whole sequence's keys and values.

## At scale

```bash
uv run python context-parallel/estimate.py
```

```
Llama 3 8B, one sequence of 131,072 tokens, bf16, attention recomputed in tiles

activations for one sequence: 584 GB
keys and values, one layer, whole sequence: 537 MB

  cp  tokens per GPU   activations   ring, sent    compute
   1         131,072      584.1 GB *     0.0 GB     83.2 s
   2          65,536      292.1 GB *    17.2 GB     41.6 s
   4          32,768      146.0 GB *    25.8 GB     20.8 s
   8          16,384       73.0 GB      30.1 GB     10.4 s
  16           8,192       36.5 GB      32.2 GB      5.2 s

* more than an 80 GB GPU holds, before weights and optimizer state
ring: keys and values forward, their gradients back, (cp - 1)/cp of each per layer
```

32.2 GB against 5.2 s of compute: even at 50 GB/s across nodes the ring needs about 0.6 s, and each hop overlaps the attention on the previous chunk. With grouped-query attention the keys and values are a quarter of the hidden size, which keeps the ring cheap. The activation figure is the per-layer `34 · seq · h` bytes of Korthikanti et al., an estimate for Llama.

## What it doesn't fix

- Attention still costs seq² FLOPs in total; context parallelism shares them, it doesn't remove them.
- Every GPU holds the whole model: context parallelism splits activations, not weights. It combines with the other dimensions.

All four dimensions are built: data, tensor, pipeline and context.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/context-parallelism/context-parallelism.excalidraw) to follow the diagrams.
