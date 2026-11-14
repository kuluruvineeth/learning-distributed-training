import pytest
import torch
import torch.multiprocessing as mp

from dtrain.data import micro_batches
from dtrain.memory import zero_bytes
from dtrain.model import Llama, LlamaConfig
from dtrain.train import train_step
from dtrain.zero import Zero
from tests.harness import run_distributed

CONFIG = LlamaConfig(max_position_embeddings=32)
SEQ, GLOBAL_BATCH, STEPS = 32, 8, 3


def stream():
    return torch.randint(
        0, CONFIG.vocab_size, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4)
    )


def momentum(params):
    return torch.optim.SGD(params, lr=0.1, momentum=0.9)


def adamw(params):
    return torch.optim.AdamW(params, lr=1e-3)


def train(rank, world, stage, grad_acc, optimizer, steps=STEPS):
    torch.manual_seed(100 + rank)
    local = Llama(CONFIG)
    if rank == 0:
        torch.manual_seed(0)
        local.load_state_dict(Llama(CONFIG).state_dict())
    model = Zero(local, stage, optimizer)
    batches = micro_batches(stream(), SEQ, GLOBAL_BATCH // world // grad_acc, rank, world)
    for _ in range(steps):
        model.zero_grad()
        train_step(model, batches, grad_acc)
        model.step()
    return model


def _matches_single_device(rank, world, stage, grad_acc):
    torch.manual_seed(0)
    reference = Llama(CONFIG)
    optimizer = momentum(reference.parameters())
    batches = micro_batches(stream(), SEQ, GLOBAL_BATCH)
    for _ in range(STEPS):
        optimizer.zero_grad()
        train_step(reference, batches, grad_acc=1)
        optimizer.step()
    model = train(rank, world, stage, grad_acc, momentum)
    for (name, a), b in zip(reference.named_parameters(), model.full_parameters(), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("grad_acc", [1, 2], ids=lambda g: f"acc{g}")
@pytest.mark.parametrize("world", [2, 4], ids=lambda w: f"N{w}")
@pytest.mark.parametrize("stage", [0, 1, 2, 3], ids=lambda s: f"stage{s}")
def test_zero_matches_single_device(stage, world, grad_acc):
    run_distributed(world, _matches_single_device, stage, grad_acc)


def _matches_data_parallel(rank, world, stage, grad_acc):
    base = train(rank, world, 0, grad_acc, adamw).full_parameters()
    sharded = train(rank, world, stage, grad_acc, adamw).full_parameters()
    for a, b in zip(base, sharded, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    ("stage", "grad_acc"),
    [(1, 1), (2, 1), (3, 1), (1, 2)],
    ids=["stage1", "stage2", "stage3", "stage1-acc2"],
)
def test_zero_is_bit_identical_to_data_parallel(stage, grad_acc):
    run_distributed(4, _matches_data_parallel, stage, grad_acc)


def _state_and_traffic(rank, world, results):
    for stage in range(4):
        model = train(rank, world, stage, 1, adamw, steps=1)
        if rank == 0:
            results[stage] = (model.state_bytes(), dict(model.sent), dict(model.calls))


def test_zero_state_and_traffic():
    with mp.Manager() as manager:
        results = manager.dict()
        run_distributed(4, _state_and_traffic, results)
        results = dict(results)
    params = sum(p.numel() for p in Llama(CONFIG).parameters())
    expected = {0: (4, 4, 8), 1: (4, 4, 2), 2: (4, 1, 2), 3: (1, 1, 2)}
    for stage, (weights, grads, optimizer) in expected.items():
        state = results[stage][0]
        assert state == {
            "weights": weights * params,
            "gradients": grads * params,
            "optimizer": optimizer * params,
        }
    sent = {stage: sum(results[stage][1].values()) for stage in range(4)}
    assert sent[0] == sent[1] == sent[2] == 2 * 3 * 4 * params // 4
    assert sent[3] == 3 * sent[0] // 2


def _accumulation_traffic(rank, world, results):
    for stage in (1, 2):
        model = train(rank, world, stage, 2, adamw, steps=1)
        if rank == 0:
            results[stage] = model.calls["reduce_scatter"]


def test_zero2_reduce_scatters_every_micro_batch():
    with mp.Manager() as manager:
        results = manager.dict()
        run_distributed(2, _accumulation_traffic, results)
        assert results[2] == 2 * results[1]


def test_zero_bytes_for_llama3_8b():
    params = 8_030_261_248
    totals = {stage: sum(zero_bytes(params, stage, 8).values()) / 1e9 for stage in range(4)}
    assert round(totals[0], 1) == 160.6
    assert round(totals[1], 1) == 76.3
    assert round(totals[2], 1) == 34.1
    assert round(totals[3], 1) == 20.1
