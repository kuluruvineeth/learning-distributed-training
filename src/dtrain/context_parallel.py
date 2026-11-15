import torch
import torch.distributed as dist


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
