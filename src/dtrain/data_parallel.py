import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable


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


class Bucket:
    def __init__(self, params: list[nn.Parameter], grads: torch.Tensor, group, world: int):
        self.params = set(params)
        self.ready = set()
        self.grads = grads
        self.group = group
        self.world = world
        self.handle = None

    def mark_ready(self, p: nn.Parameter) -> bool:
        self.ready.add(p)
        if len(self.ready) < len(self.params):
            return False
        self.grads /= self.world
        self.handle = dist.all_reduce(self.grads, group=self.group, async_op=True)
        return True

    def wait(self) -> None:
        self.handle.wait()
        self.handle = None
        self.ready.clear()

    def reset(self) -> None:
        self.handle = None
        self.ready.clear()
        self.grads.zero_()


class DataParallelBucket(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        bucket_cap_mb: float = 25,
        group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        self.module = module
        self.group = group
        self.require_backward_grad_sync = True
        self.all_reduce_calls = 0
        self._callback_queued = False
        self._handed_out = set()
        broadcast_parameters(module, group)
        params = [p for p in module.parameters() if p.requires_grad]
        self.buckets, self.bucket_of = self._build_buckets(params, bucket_cap_mb)
        for p in params:
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _build_buckets(self, params: list[nn.Parameter], bucket_cap_mb: float):
        capacity = int(bucket_cap_mb * 1024 * 1024 / 4)
        groups, sizes = [[]], [0]
        for p in reversed(params):
            if sizes[-1] and sizes[-1] + p.numel() > capacity:
                groups.append([])
                sizes.append(0)
            groups[-1].append(p)
            sizes[-1] += p.numel()
        world = dist.get_world_size(self.group)
        buckets, bucket_of = [], {}
        for group_params, size in zip(groups, sizes, strict=True):
            flat = torch.zeros(size, dtype=torch.float32)
            offset = 0
            for p in group_params:
                p.main_grad = flat[offset : offset + p.numel()].view_as(p)
                offset += p.numel()
            bucket = Bucket(group_params, flat, self.group, world)
            buckets.append(bucket)
            bucket_of.update({p: bucket for p in group_params})
        return buckets, bucket_of

    def _on_grad_ready(self, p: nn.Parameter) -> None:
        if p in self._handed_out:
            p.main_grad.copy_(p.grad)
            self._handed_out.discard(p)
        else:
            p.main_grad.add_(p.grad)
        p.grad = None
        if not self.require_backward_grad_sync:
            return
        if not self._callback_queued:
            Variable._execution_engine.queue_callback(self._finish_sync)
            self._callback_queued = True
        if self.bucket_of[p].mark_ready(p):
            self.all_reduce_calls += 1

    def _finish_sync(self) -> None:
        for bucket in self.buckets:
            bucket.wait()
        for bucket in self.buckets:
            for p in bucket.params:
                p.grad = p.main_grad.to(p.dtype)
                self._handed_out.add(p)
        self._callback_queued = False

    def reset(self) -> None:
        self._handed_out.clear()
        for bucket in self.buckets:
            bucket.reset()

    @contextlib.contextmanager
    def no_sync(self):
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = True


def average_across(value: float, group: dist.ProcessGroup | None = None) -> float:
    t = torch.tensor([value])
    dist.all_reduce(t, group=group)
    return t.item() / dist.get_world_size(group)
