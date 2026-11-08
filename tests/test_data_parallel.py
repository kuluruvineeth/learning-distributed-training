import pytest
import torch

from dtrain.data import micro_batches
from dtrain.data_parallel import DataParallelBucket, DataParallelNaive
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn, train_step
from tests.harness import run_distributed

CONFIG = LlamaConfig(max_position_embeddings=32)
SEQ, GLOBAL_BATCH, STEPS = 32, 8, 3


def wrap(model, impl):
    if impl == "naive":
        return DataParallelNaive(model)
    return DataParallelBucket(model, bucket_cap_mb=0.01)


def _matches_single_device(rank, world, impl, grad_acc):
    tokens = torch.randint(
        0, CONFIG.vocab_size, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4)
    )
    torch.manual_seed(0)
    reference = Llama(CONFIG)
    torch.manual_seed(100 + rank)
    local = Llama(CONFIG)
    if rank == 0:
        local.load_state_dict(reference.state_dict())
    model = wrap(local, impl)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_batches = micro_batches(tokens, SEQ, GLOBAL_BATCH)
    batches = micro_batches(tokens, SEQ, GLOBAL_BATCH // world // grad_acc, rank, world)
    for _ in range(STEPS):
        ref_opt.zero_grad()
        opt.zero_grad()
        train_step(reference, ref_batches, grad_acc=1)
        train_step(model, batches, grad_acc=grad_acc)
        for (name, a), (_, b) in zip(
            reference.named_parameters(), model.module.named_parameters(), strict=True
        ):
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-6, msg=name)
        ref_opt.step()
        opt.step()
        if impl == "bucket":
            model.reset()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), model.module.named_parameters(), strict=True
    ):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("grad_acc", [1, 2], ids=lambda g: f"acc{g}")
@pytest.mark.parametrize("world", [2, 4], ids=lambda w: f"N{w}")
@pytest.mark.parametrize("impl", ["naive", "bucket"])
def test_data_parallel_matches_single_device(impl, world, grad_acc):
    run_distributed(world, _matches_single_device, impl, grad_acc)


def _sync_every_micro_batch(rank, world, impl):
    tokens = torch.randint(
        0, CONFIG.vocab_size, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4)
    )
    torch.manual_seed(0)
    reference = Llama(CONFIG)
    local = Llama(CONFIG)
    local.load_state_dict(reference.state_dict())
    model = wrap(local, impl)
    train_step(reference, micro_batches(tokens, SEQ, GLOBAL_BATCH), grad_acc=1)
    batches = micro_batches(tokens, SEQ, GLOBAL_BATCH // world // 2, rank, world)
    for _ in range(2):
        inputs, targets = next(batches)
        (loss_fn(model(inputs), targets) / 2).backward()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), model.module.named_parameters(), strict=True
    ):
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("impl", ["naive", "bucket"])
def test_sync_on_every_micro_batch(impl):
    run_distributed(2, _sync_every_micro_batch, impl)
