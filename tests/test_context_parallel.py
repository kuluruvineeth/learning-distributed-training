import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dtrain.context_parallel import ContextParallel, apply_context_parallel, split_batches
from dtrain.data import micro_batches
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn, train_step
from dtrain.zero import Zero
from tests.harness import run_distributed

SEQ = 32
CONFIG = LlamaConfig(max_position_embeddings=SEQ)


def stream():
    return torch.randint(0, 256, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4))


def models(layout):
    torch.manual_seed(0)
    reference = Llama(CONFIG)
    torch.manual_seed(0)
    cp = ContextParallel(layout=layout)
    return reference, apply_context_parallel(Llama(CONFIG), cp), cp


def _forward_backward_match(rank, world, layout):
    reference, model, cp = models(layout)
    ids = stream()[: 2 * SEQ + 2].view(2, SEQ + 1)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits_ref = reference(inputs)
    logits = model(cp.split(inputs))
    torch.testing.assert_close(logits, cp.split(logits_ref), rtol=1e-5, atol=1e-5)
    loss_fn(logits_ref, targets).backward()
    (loss_fn(logits, cp.split(targets)) / world).backward()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), model.named_parameters(), strict=True
    ):
        dist.all_reduce(b.grad)
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("world", [2, 4], ids=lambda w: f"N{w}")
def test_logits_and_gradients_match_single_device(world, layout):
    run_distributed(world, _forward_backward_match, layout)


def _training_matches(rank, world):
    reference, model, cp = models("zigzag")
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    model = Zero(model, 0, lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))
    ref_batches = micro_batches(stream(), SEQ, 4)
    batches = split_batches(micro_batches(stream(), SEQ, 4), cp)
    for _ in range(3):
        ref_opt.zero_grad()
        model.zero_grad()
        train_step(reference, ref_batches, grad_acc=1)
        train_step(model, batches, grad_acc=1)
        ref_opt.step()
        model.step()
    for (name, a), b in zip(reference.named_parameters(), model.full_parameters(), strict=True):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6, msg=name)


def test_training_matches_single_device():
    run_distributed(2, _training_matches)


def _scores(rank, world, layout, results):
    _, model, cp = models(layout)
    model(cp.split(stream()[:SEQ].view(1, SEQ)))
    results[rank] = cp.scores


def scores(layout):
    with mp.Manager() as manager:
        results = manager.dict()
        run_distributed(4, _scores, layout, results)
        return [results[r] for r in range(4)]


def test_zigzag_balances_the_work():
    layers, chunk = CONFIG.num_hidden_layers, SEQ // 4
    assert scores("contiguous") == [layers * (r + 1) * chunk**2 for r in range(4)]
    assert scores("zigzag") == [layers * 9 * (chunk // 2) ** 2] * 4
