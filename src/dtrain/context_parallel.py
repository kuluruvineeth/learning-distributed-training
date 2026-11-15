import math

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


class ContextParallel:
    def __init__(self, group: dist.ProcessGroup | None = None, layout: str = "contiguous"):
        if layout not in ("contiguous", "zigzag"):
            raise ValueError("layout must be contiguous or zigzag")
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.layout = layout
        self.next = self._global((self.rank + 1) % self.world)
        self.prev = self._global((self.rank - 1) % self.world)
        self.sent = 0
        self.scores = 0

    def _global(self, rank: int) -> int:
        return dist.get_global_rank(self.group, rank) if self.group is not None else rank

    def chunks(self, rank: int) -> list[int]:
        if self.layout == "contiguous":
            return [rank]
        return [rank, 2 * self.world - 1 - rank]

    @property
    def num_chunks(self) -> int:
        return self.world * (1 if self.layout == "contiguous" else 2)

    def split(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if x.size(dim) % self.num_chunks:
            raise ValueError(f"length {x.size(dim)} does not split into {self.num_chunks} chunks")
        pieces = x.chunk(self.num_chunks, dim=dim)
        return torch.cat([pieces[c] for c in self.chunks(self.rank)], dim=dim)

    def exchange(self, x: torch.Tensor, to: int, src: int) -> torch.Tensor:
        x = x.contiguous()
        sending = dist.isend(x, to, group=self.group)
        out = torch.empty_like(x)
        dist.recv(out, src, group=self.group)
        sending.wait()
        self.sent += x.numel() * x.element_size()
        return out


def split_batches(batches, cp: ContextParallel):
    for inputs, targets in batches:
        yield cp.split(inputs), cp.split(targets)


class RingGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cp):
        ctx.cp = cp
        seen = [x]
        for _ in range(cp.world - 1):
            seen.append(cp.exchange(seen[-1], cp.next, cp.prev))
        return torch.stack(seen)

    @staticmethod
    def backward(ctx, grad):
        cp = ctx.cp
        total = grad[-1]
        for step in range(cp.world - 2, -1, -1):
            total = grad[step] + cp.exchange(total, cp.prev, cp.next)
        return total, None


def attention_block(q, k, v, causal):
    k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        future = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    return torch.exp(scores - lse) @ v, lse


def ring_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp: ContextParallel
) -> torch.Tensor:
    mine = cp.chunks(cp.rank)
    qs = q.chunk(len(mine), dim=-2)
    parts = [[] for _ in mine]
    ks, vs = RingGather.apply(k, cp), RingGather.apply(v, cp)
    for step in range(cp.world):
        source = (cp.rank - step) % cp.world
        kbs, vbs = ks[step].chunk(len(mine), dim=-2), vs[step].chunk(len(mine), dim=-2)
        for i, (a, qa) in enumerate(zip(mine, qs, strict=True)):
            for b, kb, vb in zip(cp.chunks(source), kbs, vbs, strict=True):
                if b <= a:
                    block = checkpoint(attention_block, qa, kb, vb, b == a, use_reentrant=False)
                    parts[i].append(block)
                    cp.scores += qa.size(-2) * kb.size(-2)
    out = []
    for blocks in parts:
        lse = torch.logsumexp(torch.stack([b[1] for b in blocks]), dim=0)
        out.append(sum(torch.exp(b[1] - lse) * b[0] for b in blocks))
    return torch.cat(out, dim=-2)
