import torch
import torch.nn as nn


def units(module: nn.Module) -> list[nn.Module]:
    out = []
    for child in module.children():
        if isinstance(child, nn.ModuleList):
            out.extend(child)
        elif any(p.requires_grad for p in child.parameters()):
            out.append(child)
    return out


class Unit:
    def __init__(self, module: nn.Module, rank: int, world: int):
        self.module = module
        self.params = [p for p in module.parameters() if p.requires_grad]
        self.shapes = [p.shape for p in self.params]
        self.offset, self.numel = {}, 0
        for p in self.params:
            self.offset[p] = self.numel
            self.numel += p.numel()
        self.size = -(-self.numel // world)
        self.own = slice(rank * self.size, (rank + 1) * self.size)
        self.world = world
        self.ready = set()
        self.flat = None
        self.grad = None
        self.shard = None

    def flatten(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        flat = torch.zeros(self.size * self.world)
        torch.cat([t.reshape(-1) for t in tensors], out=flat[: self.numel])
        return flat

    def bind(self, flat: torch.Tensor) -> None:
        offset = 0
        for p, shape in zip(self.params, self.shapes, strict=True):
            n = shape.numel()
            p.data = flat[offset : offset + n].view(shape)
            offset += n

    def free(self) -> None:
        for p in self.params:
            p.data = torch.empty(0)
