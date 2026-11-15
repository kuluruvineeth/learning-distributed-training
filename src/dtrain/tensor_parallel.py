import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


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


class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return x

    @staticmethod
    def backward(ctx, grad):
        return ctx.tp.all_reduce(grad.clone()), None


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        return tp.all_reduce(x.clone())

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return tp.all_gather(x, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        tp = ctx.tp
        return grad.chunk(tp.world, dim=-1)[tp.rank].contiguous(), None


def shard(weight: torch.Tensor, dim: int, tp: TensorParallel) -> nn.Parameter:
    if weight.size(dim) % tp.world:
        raise ValueError(f"size {weight.size(dim)} does not split across {tp.world} ranks")
    piece = weight.detach().chunk(tp.world, dim=dim)[tp.rank].clone()
    param = nn.Parameter(piece)
    param.shard_dim = dim
    return param


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, tp: TensorParallel, gather_output: bool = False):
        super().__init__()
        self.weight = shard(linear.weight, 0, tp)
        self.tp = tp
        self.gather_output = gather_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        return Gather.apply(y, self.tp) if self.gather_output else y


class RowParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, tp: TensorParallel):
        super().__init__()
        self.weight = shard(linear.weight, 1, tp)
        self.tp = tp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Reduce.apply(F.linear(x, self.weight), self.tp)


class VocabParallelEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, tp: TensorParallel):
        super().__init__()
        self.weight = shard(embedding.weight, 0, tp)
        self.rows = self.weight.size(0)
        self.start = tp.rank * self.rows
        self.tp = tp

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        outside = (ids < self.start) | (ids >= self.start + self.rows)
        local = (ids - self.start).masked_fill(outside, 0)
        out = F.embedding(local, self.weight).masked_fill(outside.unsqueeze(-1), 0.0)
        return Reduce.apply(out, self.tp)


def apply_tensor_parallel(model: nn.Module, tp: TensorParallel) -> nn.Module:
    def copy_output(module, args, out):
        return Copy.apply(out, tp)

    for layer in model.decoder_layers:
        attention, mlp = layer.attention, layer.mlp
        if attention.num_heads % tp.world or attention.num_kv_heads % tp.world:
            raise ValueError("attention and key/value heads must split evenly across ranks")
        attention.q_proj = ColumnParallelLinear(attention.q_proj, tp)
        attention.k_proj = ColumnParallelLinear(attention.k_proj, tp)
        attention.v_proj = ColumnParallelLinear(attention.v_proj, tp)
        attention.out_proj = RowParallelLinear(attention.out_proj, tp)
        attention.num_heads //= tp.world
        attention.num_kv_heads //= tp.world
        mlp.up_proj = ColumnParallelLinear(mlp.up_proj, tp)
        mlp.gate_proj = ColumnParallelLinear(mlp.gate_proj, tp)
        mlp.down_proj = RowParallelLinear(mlp.down_proj, tp)
        layer.input_layernorm.register_forward_hook(copy_output)
        layer.post_attention_layernorm.register_forward_hook(copy_output)
    model.embedding = VocabParallelEmbedding(model.embedding, tp)
    model.final_norm.register_forward_hook(copy_output)
    model.final_proj = ColumnParallelLinear(model.final_proj, tp, gather_output=True)
    return model


@torch.no_grad()
def full(param: nn.Parameter, tensor: torch.Tensor, tp: TensorParallel) -> torch.Tensor:
    dim = getattr(param, "shard_dim", None)
    if dim is None:
        return tensor.clone()
    parts = [torch.empty_like(tensor) for _ in range(tp.world)]
    dist.all_gather(parts, tensor.contiguous(), group=tp.group)
    return torch.cat(parts, dim=dim)
