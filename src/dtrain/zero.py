from collections.abc import Callable

import torch
import torch.distributed as dist
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


class Zero(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        stage: int,
        make_optimizer: Callable[[list[nn.Parameter]], torch.optim.Optimizer],
        group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        if stage not in (0, 1, 2, 3):
            raise ValueError("stage must be 0, 1, 2 or 3")
        self.module = module
        self.stage = stage
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.sent = {"all_reduce": 0, "reduce_scatter": 0, "all_gather": 0}
        self.calls = dict.fromkeys(self.sent, 0)
        self.units = [Unit(m, self.rank, self.world) for m in units(module)]
        self.unit_of = {p: u for u in self.units for p in u.params}
        src = dist.get_global_rank(group, 0) if group is not None else 0
        for u in self.units:
            flat = u.flatten([p.detach() for p in u.params])
            dist.broadcast(flat, src=src, group=group)
            if stage == 3:
                u.shard = nn.Parameter(flat[u.own].clone())
                u.free()
                u.module.register_forward_pre_hook(lambda m, args, u=u: self._gather(u))
                u.module.register_forward_hook(
                    lambda m, args, out, u=u: self._after_forward(u, out)
                )
            else:
                u.flat = flat
                u.bind(flat)
                u.shard = nn.Parameter(flat if stage == 0 else flat[u.own])
            if stage < 2:
                u.grad = torch.zeros_like(flat)
        for p in self.unit_of:
            p.register_post_accumulate_grad_hook(self._on_grad)
        self.optimizer = make_optimizer([u.shard for u in self.units])

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
