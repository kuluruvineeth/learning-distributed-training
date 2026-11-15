import pytest
import torch
import torch.multiprocessing as mp

from dtrain.data import micro_batches
from dtrain.model import Llama, LlamaConfig
from dtrain.pipeline_parallel import (
    PipelineParallel,
    PipelineStage,
    afab,
    one_f_one_b,
    schedule_stats,
    split_layers,
)
from dtrain.train import train_step
from tests.harness import run_distributed

SEQ, MBS, GRAD_ACC, STEPS = 32, 2, 4, 3
SCHEDULES = {"afab": afab, "1f1b": one_f_one_b}


def config(world):
    return LlamaConfig(max_position_embeddings=SEQ, num_hidden_layers=max(2, world))


def stream():
    return torch.randint(0, 256, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4))


def momentum(params):
    return torch.optim.SGD(params, lr=0.1, momentum=0.9)


def _matches_single_device(rank, world, schedule):
    torch.manual_seed(0)
    reference = Llama(config(world))
    torch.manual_seed(0)
    pp = PipelineParallel()
    stage = PipelineStage(Llama(config(world)), pp)
    ref_opt, opt = momentum(reference.parameters()), momentum(stage.parameters())
    ref_batches = micro_batches(stream(), SEQ, MBS)
    batches = micro_batches(stream(), SEQ, MBS)
    shape = (MBS, SEQ, reference.config.hidden_size)
    expected = dict(reference.named_parameters())
    for _ in range(STEPS):
        ref_opt.zero_grad()
        opt.zero_grad()
        train_step(reference, ref_batches, GRAD_ACC)
        SCHEDULES[schedule](stage, batches, GRAD_ACC, shape)
        for name, p in zip(stage.names(), stage.parameters(), strict=True):
            assert torch.equal(p.grad, expected[name].grad), name
        ref_opt.step()
        opt.step()
    for name, p in zip(stage.names(), stage.parameters(), strict=True):
        assert torch.equal(p, expected[name]), name


@pytest.mark.parametrize("schedule", ["afab", "1f1b"])
@pytest.mark.parametrize("world", [2, 4], ids=lambda w: f"N{w}")
def test_pipeline_is_bit_identical_to_single_device(world, schedule):
    run_distributed(world, _matches_single_device, schedule)


def _record(rank, world, schedule, grad_acc, results):
    torch.manual_seed(0)
    pp = PipelineParallel()
    stage = PipelineStage(Llama(config(world)), pp)
    batches = micro_batches(stream(), SEQ, 1)
    SCHEDULES[schedule](stage, batches, grad_acc, (1, SEQ, stage.hidden))
    results[rank] = list(pp.log)


def logs(world, schedule, grad_acc):
    with mp.Manager() as manager:
        results = manager.dict()
        run_distributed(world, _record, schedule, grad_acc, results)
        return [results[r] for r in range(world)]


def test_schedules_order_bubble_and_memory():
    p, m = 4, 8
    afab_logs, ofob_logs = logs(p, "afab", m), logs(p, "1f1b", m)
    assert afab_logs[0] == [f"F{i}" for i in range(m)] + [f"B{i}" for i in range(m)]
    assert ofob_logs[0][:4] == ["F0", "F1", "F2", "F3"]
    assert ofob_logs[-1][:4] == ["F0", "B0", "F1", "B1"]
    for log in (afab_logs, ofob_logs):
        assert schedule_stats(log)["bubble"] == pytest.approx((p - 1) / (m + p - 1))
    assert schedule_stats(afab_logs)["in_flight"] == [m] * p
    assert schedule_stats(ofob_logs)["in_flight"] == [p - r for r in range(p)]


def test_split_layers():
    assert [list(r) for r in split_layers(32, 4)] == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    assert [len(r) for r in split_layers(5, 2)] == [3, 2]
