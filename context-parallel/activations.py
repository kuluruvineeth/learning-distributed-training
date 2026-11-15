import torch
import torch.distributed as dist

from dtrain.context_parallel import ContextParallel, apply_context_parallel
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn


def saved(model, inputs, targets):
    sizes = {}

    def pack(t):
        sizes[t.untyped_storage().data_ptr()] = (t.untyped_storage().nbytes(), tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = loss_fn(model(inputs), targets)
    loss.backward()
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    return [v for ptr, v in sizes.items() if ptr not in params]


def by_width(items, config):
    names = {config.hidden_size: "hidden size, 64 wide", config.vocab_size: "logits, 256 wide"}
    out = dict.fromkeys([*names.values(), "inside attention and MLP"], 0)
    for size, shape in items:
        out[names.get(shape[-1] if shape else None, "inside attention and MLP")] += size
    return out


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    seq = 1024
    config = LlamaConfig(max_position_embeddings=seq, num_hidden_layers=4)
    ids = torch.randint(0, 256, (1, seq + 1), generator=torch.Generator().manual_seed(0))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    torch.manual_seed(0)
    single = by_width(saved(Llama(config), inputs, targets), config)
    torch.manual_seed(0)
    cp = ContextParallel(layout="zigzag")
    model = apply_context_parallel(Llama(config), cp)
    split = by_width(saved(model, cp.split(inputs), cp.split(targets)), config)
    rows = [None] * cp.world
    dist.all_gather_object(rows, split)
    if rank == 0:
        print(f"bytes autograd keeps for backward, seq {seq}, {config.num_hidden_layers} layers")
        header = "".join(f"{f'rank {r}':>11}" for r in range(cp.world))
        print(f"{'':<28}{'1 device':>11}  cp {cp.world}, zigzag:{header}")
        for name in single:
            cells = "".join(f"{row[name] / 1e6:>8.2f} MB" for row in rows)
            print(f"{name:<28}{single[name] / 1e6:>8.2f} MB{'':>15}{cells}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
