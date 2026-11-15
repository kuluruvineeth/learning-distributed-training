from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

CACHE = Path(__file__).resolve().parent / ".cache"
MODEL = CACHE / "SmolLM2-135M"
STORIES = CACHE / "TinyStoriesV2-GPT4-valid.txt"


def ensure_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (MODEL / "config.json").is_file():
        snapshot_download("HuggingFaceTB/SmolLM2-135M", local_dir=MODEL)
    if not STORIES.is_file():
        hf_hub_download(
            "roneneldan/TinyStories",
            STORIES.name,
            repo_type="dataset",
            local_dir=CACHE,
        )


def main():
    ensure_cache()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = STORIES.read_text()
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    ids = []
    for chunk in tokenizer(stories)["input_ids"]:
        ids += chunk + [tokenizer.eos_token_id]
    tokens = torch.tensor(ids, dtype=torch.int32)
    torch.save(tokens, CACHE / "tinystories.pt")
    print(
        f"{len(stories):,} stories, {tokens.numel():,} tokens, vocabulary {tokenizer.vocab_size:,}"
    )


if __name__ == "__main__":
    main()
