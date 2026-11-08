import torch.distributed as dist
import torch.nn as nn


def broadcast_parameters(module: nn.Module, group: dist.ProcessGroup | None) -> None:
    src = dist.get_global_rank(group, 0) if group is not None else 0
    for p in module.parameters():
        dist.broadcast(p.data, src=src, group=group)
