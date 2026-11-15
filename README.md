# Learning distributed training

I'm learning how models train across GPUs. This repo is where I work through the questions with code, diagrams, and small experiments.

- [Why distributed training?](why-distributed/README.md): build a model, count its parameters, and work out training memory and compute.
- [How GPUs communicate](collectives/README.md): follow tensors across four processes, then see what breaks when one process skips a call.
- [Ring all-reduce](ring-all-reduce/README.md): build all-reduce from send and receive, compare it with PyTorch, and see why two correct sums can differ.
- [Single-device baseline](single-device/README.md): train a small model on one device and check that gradient accumulation and recomputation give the same loss.
- [Data parallelism](data-parallel/README.md): train on four ranks with different data, then cut 21 gradient messages per step down to 2.
- [ZeRO](zero/README.md): split the optimizer state, gradients and weights across ranks, and check that each rank holds a quarter while the loss stays the same.
- [Tensor parallelism](tensor-parallel/README.md): split each layer's weights across two ranks, check the loss matches one device, and see why it stays inside one node.
- [Pipeline parallelism](pipeline-parallel/README.md): cut the model by depth, compare two schedules, and measure the bubble.

## Run the code

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/kuluruvineeth/learning-distributed-training.git
cd learning-distributed-training
git switch pipeline-parallelism
uv sync --locked --python 3.12
uv run pytest -q
uv run torchrun --nproc-per-node=4 --master-port=29500 collectives/collectives.py
uv run torchrun --nproc-per-node=4 --master-port=29500 ring-all-reduce/ring_all_reduce.py
uv run python single-device/train.py
uv run torchrun --nproc-per-node=4 --master-port=29500 data-parallel/train.py
uv run torchrun --nproc-per-node=4 --master-port=29500 zero/train.py --stage 3
uv run torchrun --nproc-per-node=2 --master-port=29500 tensor-parallel/train.py
uv run torchrun --nproc-per-node=2 --master-port=29500 pipeline-parallel/train.py --schedule 1f1b
```

These examples run on CPU. The [collectives notes](collectives/README.md) include the macOS setup and the intentional deadlock example. The [ring notes](ring-all-reduce/README.md) cover the communication cost and the floating-point example. The [baseline notes](single-device/README.md) cover the three training runs and the memory measurements. The [data parallelism notes](data-parallel/README.md) compare naive and bucketed gradient sync. The [ZeRO notes](zero/README.md) cover the three stages and what each one saves. The [tensor parallelism notes](tensor-parallel/README.md) cover the column and row splits and the cost at scale. The [pipeline parallelism notes](pipeline-parallel/README.md) cover the bubble, the two schedules and the memory per stage. The [model notes](why-distributed/README.md) cover the parameter, memory, and compute calculations.

## Diagrams

Download a board and open it in [Excalidraw](https://excalidraw.com):

- [Why distributed training?](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/why-distributed-training/why-distributed-training.excalidraw)
- [How GPUs communicate](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/collectives/collectives.excalidraw)
- [Ring all-reduce](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/ring-all-reduce/ring-all-reduce.excalidraw)
- [Single-device baseline](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/single-device-baseline/single-device-baseline.excalidraw)
- [Data parallelism](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/data-parallelism/data-parallelism.excalidraw)
- [ZeRO](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/zero/zero.excalidraw)
- [Tensor parallelism](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/tensor-parallelism/tensor-parallelism.excalidraw)
- [Pipeline parallelism](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/pipeline-parallelism/pipeline-parallelism.excalidraw)

The commits follow the build. `main` contains the work I've shared so far. Topic branches and tags keep the matching snapshots: [`why-distributed-training`](https://github.com/kuluruvineeth/learning-distributed-training/tree/why-distributed-training), [`collectives`](https://github.com/kuluruvineeth/learning-distributed-training/tree/collectives), [`ring-all-reduce`](https://github.com/kuluruvineeth/learning-distributed-training/tree/ring-all-reduce), [`single-device-baseline`](https://github.com/kuluruvineeth/learning-distributed-training/tree/single-device-baseline), [`data-parallelism`](https://github.com/kuluruvineeth/learning-distributed-training/tree/data-parallelism), [`zero`](https://github.com/kuluruvineeth/learning-distributed-training/tree/zero), [`tensor-parallelism`](https://github.com/kuluruvineeth/learning-distributed-training/tree/tensor-parallelism) and [`pipeline-parallelism`](https://github.com/kuluruvineeth/learning-distributed-training/tree/pipeline-parallelism).

## License

[Apache-2.0](LICENSE).
