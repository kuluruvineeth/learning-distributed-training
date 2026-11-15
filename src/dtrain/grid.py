import math
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from dtrain.context_parallel import ContextParallel, apply_context_parallel, split_batches
from dtrain.data import micro_batches
from dtrain.pipeline_parallel import PipelineParallel, PipelineStage
from dtrain.tensor_parallel import TensorParallel, apply_tensor_parallel
from dtrain.zero import Zero

DIMS = ("dp", "pp", "cp", "tp")
GROUPS = {"dp": ("dp",), "pp": ("pp",), "cp": ("cp",), "tp": ("tp",), "cp_dp": ("dp", "cp")}


class Grid:
    def __init__(self, dp: int = 1, pp: int = 1, cp: int = 1, tp: int = 1):
        self.sizes = dict(zip(DIMS, (dp, pp, cp, tp), strict=True))
        world = dist.get_world_size()
        if math.prod(self.sizes.values()) != world:
            raise ValueError(
                f"dp × pp × cp × tp = {math.prod(self.sizes.values())}, but {world} GPUs"
            )
        self.rank = dist.get_rank()
        self.layout = torch.arange(world).view(dp, pp, cp, tp)
        self.coords = dict(zip(DIMS, (self.layout == self.rank).nonzero()[0].tolist(), strict=True))
        self.groups, self.members = {}, {}
        for name, dims in GROUPS.items():
            rows = self.rows(dims)
            self.groups[name] = dist.new_subgroups_by_enumeration(rows)[0]
            self.members[name] = next(row for row in rows if self.rank in row)

    def rows(self, dims: tuple[str, ...]) -> list[list[int]]:
        varying = [DIMS.index(d) for d in dims]
        fixed = [i for i in range(len(DIMS)) if i not in varying]
        size = math.prod(self.layout.size(i) for i in varying)
        return self.layout.permute(*fixed, *varying).reshape(-1, size).tolist()


def split_model(model: nn.Module, grid: Grid):
    tp = TensorParallel(grid.groups["tp"])
    apply_tensor_parallel(model, tp)
    cp = ContextParallel(grid.groups["cp"], layout="zigzag")
    apply_context_parallel(model, cp)
    return PipelineStage(model, PipelineParallel(grid.groups["pp"])), tp, cp


def parallelize(
    model: nn.Module,
    grid: Grid,
    zero_stage: int,
    make_optimizer: Callable[[list[nn.Parameter]], torch.optim.Optimizer],
):
    stage, tp, cp = split_model(model, grid)
    zero = Zero(stage, zero_stage, make_optimizer, grid.groups["cp_dp"])
    return stage, zero, tp, cp


def grid_batches(tokens: torch.Tensor, seq: int, mbs: int, grid: Grid, cp: ContextParallel):
    dp, rank = grid.sizes["dp"], grid.coords["dp"]
    return split_batches(micro_batches(tokens, seq, mbs, rank, dp), cp)
