from pathlib import Path

import torch
from transformers import AutoTokenizer

CACHE = Path(__file__).resolve().parent / ".cache"
MODEL = CACHE / "SmolLM2-135M"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = (CACHE / "TinyStoriesV2-GPT4-valid.txt").read_text()
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
