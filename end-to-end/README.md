# End to end

Everything so far was tested on a 124K-parameter toy. Does any of it hold on a real model? Here an open model and open data are trained twice with the same code, once on one process and once split across 12 with data, pipeline and tensor parallelism and ZeRO-1, and compared step by step.

- **Model**: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), a Llama: 30 layers, hidden 576, 9 query heads sharing 3 key/value heads, vocabulary 49,152. Apache-2.0.
- **Data**: the validation split of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), 27,630 stories, 5,468,541 tokens with SmolLM2's tokenizer.
- **Training**: 100 steps of AdamW at 1e-4, 16 sequences of 128 tokens per step, fp32 on CPU.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run python end-to-end/prepare.py
uv run torchrun --nproc-per-node=1 --master-port=29500 end-to-end/train.py --name single
uv run torchrun --nproc-per-node=12 --master-port=29500 end-to-end/train.py \
  --dp 2 --pp 2 --tp 3 --mbs 4 --name parallel
uv run python end-to-end/compare.py
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`. Everything runs on CPU with Gloo. No GPU is needed, but on an 8-core laptop the two runs take about 15 and 35 minutes, and the 12 processes together peak at about 19 GB of memory.

`prepare.py` downloads `SmolLM2-135M` and `TinyStoriesV2-GPT4-valid.txt` from the Hugging Face Hub into `end-to-end/.cache/` on first run, then tokenizes the stories. Global batch 16 both times: `--mbs 8 --grad-acc 2` on one process, `dp 2 × mbs 4 × grad_acc 2` on twelve.

## The model is the real one

`src/dtrain/hf.py: load_config, load_weights` read the Hugging Face checkpoint into our `Llama` unchanged. On "Once upon a time, there was a little girl named Lily. She loved to", our logits and `transformers`' differ by at most 1.7e-4 (logits around 20), and both predict " play".

SmolLM2 ties its output projection to its embedding. Both runs untie them, copying the embedding into `final_proj` and training the two separately, so the pipeline's first and last stages don't have to share a weight. The comparison is between our two runs, so this is the same in both.

## The parallel run

12 processes = dp 2 × pp 2 × tp 3, ZeRO-1 over the two data-parallel copies:

- tp 3, because the 3 key/value heads have to split evenly.
- pp 2: layers 0–14 with the embedding, layers 15–29 with the norm and output projection.
- dp 2: two copies of that 6-process model, each with half the batch.

That is the shape of Llama 3's pre-training layout (tp, pp, dp, with cp 1), on one laptop. Context parallel and all four dimensions together are proved on the toy model in [4d](../4d/README.md) on 16 processes.

## Result

```
 step   1 process   12 processes   difference
    1      2.4757         2.4757      0.0e+00
   10      1.9416         1.9416      3.0e-06
   20      2.0280         2.0281      9.0e-06
   30      2.0780         2.0780      1.4e-05
   40      1.8861         1.8861      6.0e-06
   50      1.8393         1.8393      2.0e-06
   60      1.7994         1.7994      1.0e-06
   70      1.8119         1.8119      9.0e-06
   80      1.8682         1.8682      1.2e-05
   90      1.8262         1.8262      8.0e-06
  100      2.0187         2.0187      1.4e-05
largest loss difference over 100 steps: 3.5e-05
seconds per step after step 1: 1 process 8.9, 12 processes 21.3

273 tensors gathered from the 12 processes
largest weight difference after training: 1.5e-03 (final_proj.weight)
```

The same loss to four decimals at every step. The loss moves up and down between steps because each step sees different stories; both runs move together.

The weights agree to 7.4e-4 relative to each tensor's largest value. They aren't equal because the two runs add the same numbers in a different order: tensor parallel sums partial products across GPUs, data parallel averages two half-batches. Adam then divides every update by the running size of its gradient, so a rounding difference of 1e-7 can move an element by a fraction of a learning-rate step, and over 100 steps the worst element ends up about 15 steps of 1e-4 apart. The largest differences are on tokens the run did see; rows of the embedding the data never touched match exactly.

What the model says after "Tom wanted to fly his kite, but":

```
before training:
  he was afraid of the wind. "I don't know what to do," he said. "I don't know what to do." "I don't know what to do
after 100 steps, 1 process:
  he was scared. He thought it was a big wind and he did not want to fly. He said, "No, I want to fly!" His mom said, "No, you can
after 100 steps, 12 processes:
  he was scared. He thought it was a big wind and he did not want to fly. He said, "No, I want to fly!" His mom said, "No, you can
```

Before: SmolLM2 loops. After 100 steps on TinyStories it writes a short story, and both runs write the same one, word for word.

## Speed, honestly

On one 8-core laptop, 12 processes take 21.3 s per step against 8.9 s for one. They share the same cores, and every collective goes through the operating system. On a cluster each process would be a GPU of its own, and the same code would run on NCCL. The point of this run is the right answer, not the speed.

## 24 processes didn't fit

All four dimensions on SmolLM2 means dp 2 × pp 2 × cp 2 × tp 3 = 24 processes. Twice that run started and matched the single-process loss at every step it logged, then aborted inside gloo's network transport (`Unexpected opcode` in `gloo/transport/uv/pair.cc`): once after more than 60 steps, once after 4. The laptop was already 5.7 GB into swap from other programs, and 24 processes took it to 13 GB. The same 24-process layout on a 2-layer SmolLM2 ran 40 steps clean in every combination of dimensions, and the point-to-point patterns alone ran 300 rounds clean, so the failure only appeared under heavy swapping. 12 processes fit in memory and ran clean.

## Everything, checked against one device

| | proved against one device |
|---|---|
| [collectives](../collectives/README.md), [ring-all-reduce](../ring-all-reduce/README.md) | ring all-reduce bit-exact against `torch.distributed` |
| [data-parallel](../data-parallel/README.md) | naive and bucketed, with and without accumulation |
| [zero](../zero/README.md) | stages 1–3, bit-exact against data parallel |
| [tensor-parallel](../tensor-parallel/README.md) | columns, rows, heads, vocabulary |
| [pipeline-parallel](../pipeline-parallel/README.md) | AFAB and 1F1B, bit-exact |
| [context-parallel](../context-parallel/README.md) | ring attention, zigzag |
| [4d](../4d/README.md) | all four at once, 16 processes |
| end-to-end | SmolLM2-135M, 12 processes, the same loss and the same story |

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/end-to-end/end-to-end.excalidraw) to follow the diagrams.
