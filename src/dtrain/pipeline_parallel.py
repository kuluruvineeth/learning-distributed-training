import torch
import torch.distributed as dist
import torch.nn as nn


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


def split_layers(num_layers: int, world: int) -> list[range]:
    sizes = [num_layers // world + (1 if i < num_layers % world else 0) for i in range(world)]
    starts = [sum(sizes[:i]) for i in range(world)]
    return [range(start, start + size) for start, size in zip(starts, sizes, strict=True)]


class PipelineStage(nn.Module):
    def __init__(self, model: nn.Module, pp: PipelineParallel):
        super().__init__()
        self.pp = pp
        self.layers = split_layers(len(model.decoder_layers), pp.world)[pp.rank]
        self.embedding = model.embedding if pp.first else None
        self.decoder_layers = nn.ModuleList(model.decoder_layers[i] for i in self.layers)
        self.final_norm = model.final_norm if pp.last else None
        self.final_proj = model.final_proj if pp.last else None
        self.register_buffer("cos", model.cos, persistent=False)
        self.register_buffer("sin", model.sin, persistent=False)
        self.hidden = model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, self.cos, self.sin)
        if self.final_proj is not None:
            x = self.final_proj(self.final_norm(x))
        return x

    def names(self) -> list[str]:
        out = []
        if self.embedding is not None:
            out.append("embedding.weight")
        for i, layer in zip(self.layers, self.decoder_layers, strict=True):
            out += [f"decoder_layers.{i}.{name}" for name, _ in layer.named_parameters()]
        if self.final_proj is not None:
            out += ["final_norm.weight", "final_proj.weight"]
        return out
