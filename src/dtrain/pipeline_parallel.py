from collections.abc import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn

from dtrain.train import loss_fn


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


def _forward(stage, batches, grad_acc, shape):
    pp = stage.pp
    inputs, targets = next(batches)
    x = inputs if pp.first else pp.recv(shape, pp.prev).requires_grad_()
    out = stage(x)
    loss = None
    if pp.last:
        loss = loss_fn(out, targets) / grad_acc
    else:
        pp.send(out.detach(), pp.next)
    return x, out, loss


def _backward(stage, x, out, loss, shape):
    pp = stage.pp
    if pp.last:
        loss.backward()
    else:
        torch.autograd.backward(out, pp.recv(shape, pp.next))
    if not pp.first:
        pp.send(x.grad, pp.prev)


def afab(stage: PipelineStage, batches: Iterator, grad_acc: int, shape: tuple) -> float:
    pp, saved, total = stage.pp, [], 0.0
    for i in range(grad_acc):
        x, out, loss = _forward(stage, batches, grad_acc, shape)
        pp.log.append(f"F{i}")
        saved.append((x, out, loss))
        total += loss.item() if loss is not None else 0.0
    for i in range(grad_acc):
        _backward(stage, *saved.pop(0), shape)
        pp.log.append(f"B{i}")
    pp.flush()
    return total


def one_f_one_b(stage: PipelineStage, batches: Iterator, grad_acc: int, shape: tuple) -> float:
    pp, saved, total = stage.pp, [], 0.0
    warmup = min(pp.world - pp.rank - 1, grad_acc)
    forwards = backwards = 0

    def forward():
        nonlocal forwards, total
        x, out, loss = _forward(stage, batches, grad_acc, shape)
        pp.log.append(f"F{forwards}")
        forwards += 1
        saved.append((x, out, loss))
        total += loss.item() if loss is not None else 0.0

    def backward():
        nonlocal backwards
        _backward(stage, *saved.pop(0), shape)
        pp.log.append(f"B{backwards}")
        backwards += 1

    for _ in range(warmup):
        forward()
    for _ in range(grad_acc - warmup):
        forward()
        backward()
    for _ in range(warmup):
        backward()
    pp.flush()
    return total


def schedule_stats(logs: list[list[str]], forward: float = 1.0, backward: float = 2.0) -> dict:
    world = len(logs)
    done = {}
    clock = [0.0] * world
    busy = [0.0] * world
    pending = [list(log) for log in logs]
    while any(pending):
        progressed = False
        for r in range(world):
            if not pending[r]:
                continue
            op = pending[r][0]
            kind, i = op[0], int(op[1:])
            need = (r - 1, f"F{i}") if kind == "F" else (r + 1, f"B{i}")
            if kind == "F" and r == 0 or kind == "B" and r == world - 1:
                ready = 0.0
            elif need in done:
                ready = done[need]
            else:
                continue
            cost = forward if kind == "F" else backward
            clock[r] = max(clock[r], ready) + cost
            busy[r] += cost
            done[(r, op)] = clock[r]
            pending[r].pop(0)
            progressed = True
        if not progressed:
            raise ValueError("schedule deadlocks")
    span = max(clock)
    in_flight = []
    for log in logs:
        live = peak = 0
        for op in log:
            live += 1 if op[0] == "F" else -1
            peak = max(peak, live)
        in_flight.append(peak)
    return {"bubble": 1 - sum(busy) / (world * span), "in_flight": in_flight}
