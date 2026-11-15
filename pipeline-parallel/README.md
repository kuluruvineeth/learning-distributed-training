# Pipeline parallelism

Tensor parallelism has to stay inside one node. What if the model doesn't fit in one node? Llama 3 70B needs 1,411 GB of training state ([why distributed training](../why-distributed/README.md)): 176 GB per GPU even split 8 ways inside a node. The model has to cross nodes, and the links between nodes are slow. Pipeline parallelism cuts the model by depth instead: GPU 0 gets the first layers, GPU 1 the next ones, and so on. Each stage only talks to its neighbours, one activation at a time, point to point. That works across nodes.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run pytest -q tests/test_pipeline_parallel.py
uv run torchrun --nproc-per-node=2 --master-port=29500 pipeline-parallel/train.py --schedule 1f1b --steps 100
uv run torchrun --nproc-per-node=4 --master-port=29500 pipeline-parallel/train.py \
  --schedule 1f1b --layers 4 --mbs 1 --grad-acc 8 --steps 1
uv run python pipeline-parallel/estimate.py
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`. Everything runs on CPU with Gloo. No GPU is needed.

## The bubble

With one batch, the stages take turns: stage 1 waits while stage 0 runs forward, stage 0 waits while the others run, and then the same again backward. Only one GPU works at a time.

Split the batch into m micro-batches and the stages overlap: while stage 1 works on micro-batch 0, stage 0 starts micro-batch 1. The pipeline still has to fill and drain. With p stages, each stage sits idle for a share of

```
(p − 1) / (m + p − 1)
```

of the step: the bubble. More micro-batches, smaller bubble. The micro-batches are gradient accumulation ([single-device baseline](../single-device/README.md)): their gradients add up before one optimizer step.

## Two schedules

- **All forward, all backward (AFAB)**: run every micro-batch forward, then every one backward. Simple, but each stage holds the activations of all m micro-batches at once.
- **One forward, one backward (1F1B)**: stage r runs p − r − 1 forwards to fill the pipeline, then alternates one forward and one backward, then drains. Same bubble, but stage r holds at most p − r micro-batches.

## Code

`src/dtrain/pipeline_parallel.py`

- `PipelineStage(model, PipelineParallel())` keeps this stage's layers, the embedding on the first stage, and the norm and output projection on the last.
- `split_layers` decides which layers each stage gets.
- `afab` and `one_f_one_b` are the two schedules.

Each micro-batch's loss is divided by the number of micro-batches, as in single-device gradient accumulation, so the gradients match one device exactly. Sends are `isend` and waited on at the end of the step, so two neighbours sending to each other at the same moment can't block each other. Every rank reads the same data stream: the first stage uses the inputs, the last stage the targets.

## Tests

```bash
uv run pytest -q tests/test_pipeline_parallel.py
```

- 2 and 4 stages, both schedules: after each of 3 SGD-with-momentum steps, every gradient and weight equals a single-device run with the same micro-batches, bit for bit. Same micro-batches, same arithmetic per layer, same order of accumulation: only where the layers run changes.
- The schedules' order, bubble and memory match the formulas. Without the loss divided by the number of micro-batches, or with the gradient not sent back, the tests fail.

```bash
uv run torchrun --nproc-per-node=2 --master-port=29500 pipeline-parallel/train.py --schedule 1f1b --steps 100
```

```
stage 0: layers [0], 61,824 parameters
stage 1: layers [1], 61,888 parameters
step    1  loss 5.886  61,512 tokens/s  131 KB sent
step   25  loss 3.538  74,555 tokens/s  131 KB sent
step   50  loss 2.757  74,509 tokens/s  131 KB sent
step   75  loss 2.590  75,767 tokens/s  131 KB sent
step  100  loss 2.290  76,702 tokens/s  131 KB sent
```

The single-device loss with `--mbs 2 --grad-acc 4`, step for step, and the same with `--schedule afab`. 131 KB per step: 4 micro-batches of activations forward from stage 0, 4 of gradients back from stage 1.

Each run also prints its first step's real order. 4 stages, 8 micro-batches:

```bash
uv run torchrun --nproc-per-node=4 --master-port=29500 pipeline-parallel/train.py \
  --schedule 1f1b --layers 4 --mbs 1 --grad-acc 8 --steps 1
```

```
1f1b, 4 stages, 8 micro-batches, first step
  stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
  stage 1: F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
  stage 2: F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
  stage 3: F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
  bubble, with backward = 2 × forward: 27%
  micro-batches held at once, per stage: [4, 3, 2, 1]
```

With `--schedule afab`: the same 27% bubble, and 8 micro-batches held on every stage. The bubble comes from replaying the recorded order with forward = 1 and backward = 2 time units; it matches (p − 1) / (m + p − 1) = 3 / 11.

## At scale

```bash
uv run python pipeline-parallel/estimate.py
```

```
Llama 3 8B: training state per stage, 20 bytes per parameter
  pp 2:  80.3   80.3  GB
  pp 4:  45.4   34.9   34.9   45.4  GB
  pp 8:  28.0   17.4   17.4   17.4   17.4   17.4   17.4   28.0  GB
  the first stage carries the embedding, the last the output projection

between two stages, per micro-batch of one 8,192-token sequence:
  67.1 MB forward, 67.1 MB of gradient back
  point to point, only to the neighbours

bubble: share of time a stage sits idle, (p - 1) / (m + p - 1)
   micro-batches      4      8     16     32     64
            pp 2    20%    11%     6%     3%     2%
            pp 4    43%    27%    16%     9%     4%
            pp 8    64%    47%    30%    18%    10%
```

Compare tensor parallelism: 128 all-reduces per step in the middle of every layer. Here it's one message per micro-batch per boundary, and the next stage can start on it right away.

## What it doesn't fix

- The bubble only shrinks with many micro-batches, which means a big global batch. Interleaved stages and zero-bubble schedules shrink it further; not built here.
- The stages aren't balanced: the embedding and output projection make the first and last heavier.
- Every stage still processes whole sequences, so very long sequences still don't fit.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/pipeline-parallelism/pipeline-parallelism.excalidraw) to follow the diagrams.
