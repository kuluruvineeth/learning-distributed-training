import math

import pytest
import torch
import torch.multiprocessing as mp

from dtrain.data import micro_batches
from dtrain.grid import Grid, grid_batches, parallelize
from dtrain.model import Llama, LlamaConfig
from dtrain.pipeline_parallel import one_f_one_b
from dtrain.tensor_parallel import full
from dtrain.train import train_step
from tests.harness import run_distributed

SEQ, MBS, GRAD_ACC, STEPS = 32, 1, 2, 3
CONFIG = LlamaConfig(max_position_embeddings=SEQ)


def stream():
    return torch.randint(0, 256, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4))


def momentum(params):
    return torch.optim.SGD(params, lr=0.1, momentum=0.9)


def _groups(rank, world, results):
    grid = Grid(dp=2, pp=2, tp=2)
    results[rank] = (grid.coords, grid.members)


def test_groups_for_dp2_pp2_tp2():
    with mp.Manager() as manager:
        results = manager.dict()
        run_distributed(8, _groups, results)
        coords, members = results[5]
    assert coords == {"dp": 1, "pp": 0, "cp": 0, "tp": 1}
    assert members["tp"] == [4, 5]
    assert members["pp"] == [5, 7]
    assert members["dp"] == [1, 5]


def _matches_single_device(rank, world, sizes, zero_stage):
    grid = Grid(**sizes)
    torch.manual_seed(0)
    reference = Llama(CONFIG)
    ref_opt = momentum(reference.parameters())
    torch.manual_seed(0)
    stage, zero, tp, cp = parallelize(Llama(CONFIG), grid, zero_stage, momentum)
    ref_batches = micro_batches(stream(), SEQ, MBS * grid.sizes["dp"])
    batches = grid_batches(stream(), SEQ, MBS, grid, cp)
    shape = (MBS, SEQ // grid.sizes["cp"], CONFIG.hidden_size)
    for _ in range(STEPS):
        ref_opt.zero_grad()
        zero.zero_grad()
        train_step(reference, ref_batches, GRAD_ACC)
        one_f_one_b(stage, batches, GRAD_ACC, shape)
        ref_opt.step()
        zero.step()
    expected = dict(reference.named_parameters())
    for name, p in zip(stage.names(), stage.parameters(), strict=True):
        torch.testing.assert_close(
            full(p, p.detach(), tp), expected[name], rtol=1e-5, atol=1e-6, msg=name
        )


@pytest.mark.parametrize(
    ("sizes", "zero_stage"),
    [
        ({"dp": 2, "pp": 2, "tp": 2}, 1),
        ({"pp": 2, "cp": 2, "tp": 2}, 0),
        ({"dp": 2, "cp": 2, "tp": 2}, 2),
        ({"dp": 2, "pp": 2, "cp": 2, "tp": 2}, 1),
    ],
    ids=["dp2-pp2-tp2", "pp2-cp2-tp2", "dp2-cp2-tp2", "dp2-pp2-cp2-tp2"],
)
def test_4d_matches_single_device(sizes, zero_stage):
    run_distributed(math.prod(sizes.values()), _matches_single_device, sizes, zero_stage)
