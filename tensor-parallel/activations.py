import torch
import torch.distributed as dist

from dtrain.model import Llama, LlamaConfig
from dtrain.tensor_parallel import TensorParallel, apply_tensor_parallel
from dtrain.train import loss_fn


def saved(model, ids):
    sizes = {}

    def pack(t):
        sizes[t.untyped_storage().data_ptr()] = (t.untyped_storage().nbytes(), tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = loss_fn(model(ids[:, :-1]), ids[:, 1:])
    loss.backward()
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    return [v for ptr, v in sizes.items() if ptr not in params]


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    seq = 1024
    config = LlamaConfig(max_position_embeddings=seq, num_hidden_layers=4)
    ids = torch.randint(
        0, config.vocab_size, (1, seq + 1), generator=torch.Generator().manual_seed(0)
    )
    torch.manual_seed(0)
    single = saved(Llama(config), ids)
    torch.manual_seed(0)
    tp = TensorParallel()
    split = saved(apply_tensor_parallel(Llama(config), tp), ids)
    if rank == 0:
        widths = {config.hidden_size: "hidden size, 64 wide", config.vocab_size: "logits, 256 wide"}

        def by_width(items):
            out = dict.fromkeys([*widths.values(), "inside attention and MLP"], 0)
            for size, shape in items:
                out[widths.get(shape[-1] if shape else None, "inside attention and MLP")] += size
            return out

        a, b = by_width(single), by_width(split)
        print(f"bytes autograd keeps for backward, seq {seq}, {config.num_hidden_layers} layers")
        print(f"{'':<28}{'1 device':>12}{f'tp {tp.world}, per rank':>20}")
        for name in a:
            print(f"{name:<28}{a[name] / 1e6:>9.2f} MB{b[name] / 1e6:>17.2f} MB")
        print(f"{'total':<28}{sum(a.values()) / 1e6:>9.2f} MB{sum(b.values()) / 1e6:>17.2f} MB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
