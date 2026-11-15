import pytest
import torch
import torch.multiprocessing as mp

from dtrain.data import micro_batches
from dtrain.model import Llama, LlamaConfig
from dtrain.tensor_parallel import TensorParallel, apply_tensor_parallel, full
from dtrain.train import loss_fn, train_step
from tests.harness import run_distributed

CONFIG = LlamaConfig(max_position_embeddings=32)
MHA = LlamaConfig(max_position_embeddings=32, num_key_value_heads=4)
SEQ, BATCH, STEPS = 32, 8, 3


def stream():
    return torch.randint(0, 256, (64 * SEQ + 1,), generator=torch.Generator().manual_seed(4))


def models(config):
    torch.manual_seed(0)
    reference = Llama(config)
    torch.manual_seed(0)
    tp = TensorParallel()
    return reference, apply_tensor_parallel(Llama(config), tp), tp


def _forward_backward_match(rank, world, config):
    reference, model, tp = models(config)
    ids = stream()[: 2 * SEQ + 2].view(2, SEQ + 1)
    logits_ref = reference(ids[:, :-1])
    logits = model(ids[:, :-1])
    torch.testing.assert_close(logits, logits_ref, rtol=1e-5, atol=1e-5)
    loss_fn(logits_ref, ids[:, 1:]).backward()
    loss_fn(logits, ids[:, 1:]).backward()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), model.named_parameters(), strict=True
    ):
        torch.testing.assert_close(full(b, b.grad, tp), a.grad, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize(("world", "config"), [(2, CONFIG), (4, MHA)], ids=["N2-gqa", "N4-mha"])
def test_logits_and_gradients_match_single_device(world, config):
    run_distributed(world, _forward_backward_match, config)


def _training_matches(rank, world):
    reference, model, tp = models(CONFIG)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    ref_batches = micro_batches(stream(), SEQ, BATCH)
    batches = micro_batches(stream(), SEQ, BATCH // 2)
    for _ in range(STEPS):
        ref_opt.zero_grad()
        opt.zero_grad()
        train_step(reference, ref_batches, grad_acc=1)
        train_step(model, batches, grad_acc=2)
        ref_opt.step()
        opt.step()
    for (name, a), (_, b) in zip(
        reference.named_parameters(), model.named_parameters(), strict=True
    ):
        torch.testing.assert_close(full(b, b.detach(), tp), a, rtol=1e-5, atol=1e-6, msg=name)


def test_training_matches_single_device():
    run_distributed(2, _training_matches)


def _traffic(rank, world, results):
    _, model, tp = models(CONFIG)
    ids = stream()[: 2 * SEQ + 2].view(2, SEQ + 1)
    loss_fn(model(ids[:, :-1]), ids[:, 1:]).backward()
    if rank == 0:
        results.update(calls=dict(tp.calls), params=sum(p.numel() for p in model.parameters()))


def test_communication_and_parameters_per_rank():
    with mp.Manager() as manager:
        results = manager.dict()
        run_distributed(2, _traffic, results)
        results = dict(results)
    layers = CONFIG.num_hidden_layers
    assert results["calls"] == {"all_reduce": 1 + 2 * layers + 2 * layers + 1, "all_gather": 1}
    replicated = 2 * layers * CONFIG.hidden_size + CONFIG.hidden_size
    total = sum(p.numel() for p in Llama(CONFIG).parameters())
    assert results["params"] == (total - replicated) // 2 + replicated


def _rejects_uneven_heads(rank, world):
    with pytest.raises(ValueError):
        apply_tensor_parallel(Llama(CONFIG), TensorParallel())


def test_rejects_heads_that_do_not_split():
    run_distributed(4, _rejects_uneven_heads)
