import torch
import torch.distributed as dist


class PipelineParallel:
    def __init__(self, group: dist.ProcessGroup | None = None):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.first, self.last = self.rank == 0, self.rank == self.world - 1
        self.prev = (
            dist.get_global_rank(group, self.rank - 1)
            if group and not self.first
            else self.rank - 1
        )
        self.next = (
            dist.get_global_rank(group, self.rank + 1) if group and not self.last else self.rank + 1
        )
        self.sent = 0
        self.log = []
        self.sending = []

    def send(self, x: torch.Tensor, dst: int) -> None:
        x = x.contiguous()
        self.sending.append((dist.isend(x, dst, group=self.group), x))
        self.sent += x.numel() * x.element_size()

    def flush(self) -> None:
        for work, _ in self.sending:
            work.wait()
        self.sending.clear()

    def recv(self, shape: tuple, src: int) -> torch.Tensor:
        x = torch.empty(shape)
        dist.recv(x, src, group=self.group)
        return x
