# Parallelism degrees

We now have three ways to split training: [data](../data-parallel/README.md), [tensor](../tensor-parallel/README.md) and [pipeline](../pipeline-parallel/README.md) parallelism. Each run so far gave its one split every process. What does dp 2 × tp 8 × pp 4 mean on 64 GPUs?

There is no new code here, only the numbers.

## The rule: multiply

Each degree is how many GPUs share one job. The product is the GPU count:

```
GPUs = dp × tp × pp × cp
```

Lay 8 GPUs out as dp 2 × tp 4 and every GPU gets one coordinate per number. A row is one whole copy of the model; a column is one slice, the same in every copy. GPU 6 = (dp 1, tp 2): the second copy, slice 2 of every layer.

## What each number splits

| | splits | each GPU holds | each GPU sees | talks | over |
|---|---|---|---|---|---|
| dp | the batch | the whole model (or 1/N of its state, with ZeRO) | 1/N of the sequences | gradients, once per step | any link, across nodes |
| tp | every layer | 1/N of every weight matrix | the same tokens as its partners | activations, inside every layer | NVLink, one node |
| pp | the stack of layers | 1/N of the layers, whole | the same tokens, stage after stage | activations, to its neighbours | across nodes |
| cp | each sequence | the whole model | 1/N of every sequence's tokens | keys and values, in a ring | usually one node |

ZeRO is not a degree: it's a setting inside the dp group. Expert parallelism only exists in mixture-of-experts models and isn't built here.

## Why one split isn't enough

Llama 3 8B on 64 GPUs of 80 GB, one kind of split at a time:

- dp 64: 160.6 GB of training state on every GPU. Doesn't fit.
- tp 64: 8 key/value heads to share, so tp 8 at most.
- pp 64: 32 layers to cut, so pp 32 at most.

So a cluster mixes them, and the more a split talks, the closer its GPUs sit. tp 8 on one step of 8,192 tokens: 150 ms of compute, 33 ms of communication over NVLink, 301 ms over InfiniBand. tp stays inside a node, pp crosses nodes with one message per micro-batch, dp syncs gradients once per step while backward runs. (Peak link bandwidth and 40% of peak compute: an estimate, not a run.)

## 64 GPUs = dp 2 × tp 8 × pp 4

pp 4 cuts the 32 layers into 4 stages of 8, one node each. tp 8 splits each stage's layers across that node's 8 GPUs. dp 2 makes a second copy of those 32 GPUs, each copy with half the batch. Each GPU holds 1 / (tp × pp) = 1/32 of the weights; dp makes copies, it doesn't divide.

Training state per GPU, Llama 3 8B, 20 bytes per parameter, before activations:

| layout | per GPU |
|---|---|
| dp 64 | 160.6 GB |
| dp 8 × tp 8 | 20.1 GB |
| dp 2 × tp 8 × pp 4 | 5.7 GB (heaviest stage) |
| dp 64 + ZeRO-3 | 2.5 GB |

State isn't the whole bill. ZeRO-3 gathers every whole layer twice a step on every GPU, ZeRO splits state but not activations, and dp 64 means a global batch of at least 64 sequences.

## The code already takes a group

`DataParallelNaive`, `TensorParallel`, `PipelineParallel` and `Zero` each take a process group, and `None` means every process. That's why each run's degree was the whole world. Give each wrapper a smaller group and the numbers multiply.

The three runs on 4, 2 and 2 processes split the same model three ways and all reach loss 2.290 at step 100.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/parallelism-degrees/parallelism-degrees.excalidraw) to follow the diagrams.
