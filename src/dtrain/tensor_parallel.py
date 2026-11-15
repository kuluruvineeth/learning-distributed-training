import torch
import torch.distributed as dist


class TensorParallel:
    def __init__(self, group: dist.ProcessGroup | None = None):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.sent = {"all_reduce": 0, "all_gather": 0}
        self.calls = dict.fromkeys(self.sent, 0)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        self.sent["all_reduce"] += x.numel() * x.element_size() * 2 * (self.world - 1) // self.world
        self.calls["all_reduce"] += 1
        return x

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        parts = [torch.empty_like(x) for _ in range(self.world)]
        dist.all_gather(parts, x.contiguous(), group=self.group)
        out = torch.cat(parts, dim=dim)
        self.sent["all_gather"] += out.numel() * out.element_size() * (self.world - 1) // self.world
        self.calls["all_gather"] += 1
        return out
