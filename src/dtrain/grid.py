import math

import torch
import torch.distributed as dist

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
