# Learning distributed training

I'm learning how models train across GPUs. This repo is where I work through the questions with code, diagrams, and small experiments.

- [Why distributed training?](why-distributed/README.md): build a model, count its parameters, and work out training memory and compute.
- [How GPUs communicate](collectives/README.md): follow tensors across four processes, then see what breaks when one process skips a call.
- [Ring all-reduce](ring-all-reduce/README.md): build all-reduce from send and receive, compare it with PyTorch, and see why two correct sums can differ.
- [Single-device baseline](single-device/README.md): train a small model on one device and check that gradient accumulation and recomputation give the same loss.
- [Data parallelism](data-parallel/README.md): train on four ranks with different data, then cut 21 gradient messages per step down to 2.

## Run the code

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/kuluruvineeth/learning-distributed-training.git
cd learning-distributed-training
git switch data-parallelism
uv sync --locked --python 3.12
uv run pytest -q
uv run torchrun --nproc-per-node=4 --master-port=29500 collectives/collectives.py
uv run torchrun --nproc-per-node=4 --master-port=29500 ring-all-reduce/ring_all_reduce.py
uv run python single-device/train.py
uv run torchrun --nproc-per-node=4 --master-port=29500 data-parallel/train.py
```

These examples run on CPU. The [collectives notes](collectives/README.md) include the macOS setup and the intentional deadlock example. The [ring notes](ring-all-reduce/README.md) cover the communication cost and the floating-point example. The [baseline notes](single-device/README.md) cover the three training runs and the memory measurements. The [data parallelism notes](data-parallel/README.md) compare naive and bucketed gradient sync. The [model notes](why-distributed/README.md) cover the parameter, memory, and compute calculations.

## Diagrams

Download a board and open it in [Excalidraw](https://excalidraw.com):

- [Why distributed training?](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/why-distributed-training/why-distributed-training.excalidraw)
- [How GPUs communicate](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/collectives/collectives.excalidraw)
- [Ring all-reduce](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/ring-all-reduce/ring-all-reduce.excalidraw)
- [Single-device baseline](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/single-device-baseline/single-device-baseline.excalidraw)
- [Data parallelism](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/data-parallelism/data-parallelism.excalidraw)

The commits follow the build. `main` contains the work I've shared so far. Topic branches and tags keep the matching snapshots: [`why-distributed-training`](https://github.com/kuluruvineeth/learning-distributed-training/tree/why-distributed-training), [`collectives`](https://github.com/kuluruvineeth/learning-distributed-training/tree/collectives), [`ring-all-reduce`](https://github.com/kuluruvineeth/learning-distributed-training/tree/ring-all-reduce), [`single-device-baseline`](https://github.com/kuluruvineeth/learning-distributed-training/tree/single-device-baseline) and [`data-parallelism`](https://github.com/kuluruvineeth/learning-distributed-training/tree/data-parallelism).

## License

[Apache-2.0](LICENSE).
