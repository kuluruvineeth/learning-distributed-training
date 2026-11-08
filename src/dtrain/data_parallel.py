import contextlib

import torch.distributed as dist
import torch.nn as nn


def broadcast_parameters(module: nn.Module, group: dist.ProcessGroup | None) -> None:
    src = dist.get_global_rank(group, 0) if group is not None else 0
    for p in module.parameters():
        dist.broadcast(p.data, src=src, group=group)


class DataParallelNaive(nn.Module):
    def __init__(self, module: nn.Module, group: dist.ProcessGroup | None = None):
        super().__init__()
        self.module = module
        self.group = group
        self.world = dist.get_world_size(group)
        self.require_backward_grad_sync = True
        self.all_reduce_calls = 0
        broadcast_parameters(module, group)
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._all_reduce_grad)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _all_reduce_grad(self, p: nn.Parameter) -> None:
        if self.require_backward_grad_sync:
            dist.all_reduce(p.grad, group=self.group)
            p.grad /= self.world
            self.all_reduce_calls += 1

    @contextlib.contextmanager
    def no_sync(self):
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = True
