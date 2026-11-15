import pytest
import torch

from dtrain.hf import hf_to_dtrain
from dtrain.memory import num_params
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn

CONFIG = LlamaConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=172,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=32,
)


def hf_pair():
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.LlamaConfig(
        vocab_size=CONFIG.vocab_size,
        hidden_size=CONFIG.hidden_size,
        intermediate_size=CONFIG.intermediate_size,
        num_hidden_layers=CONFIG.num_hidden_layers,
        num_attention_heads=CONFIG.num_attention_heads,
        num_key_value_heads=CONFIG.num_key_value_heads,
        head_dim=CONFIG.head_dim,
        max_position_embeddings=CONFIG.max_position_embeddings,
        rms_norm_eps=CONFIG.rms_norm_eps,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    hf = transformers.LlamaForCausalLM(hf_config).eval()
    ours = Llama(CONFIG).eval()
    ours.load_state_dict(hf_to_dtrain(hf.state_dict()), strict=True)
    return hf, ours


def test_logits_match_transformers():
    hf, ours = hf_pair()
    ids = torch.randint(0, CONFIG.vocab_size, (2, 32), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(ours(ids), hf(ids).logits, rtol=1e-5, atol=1e-5)


def test_gradients_match_transformers():
    hf, ours = hf_pair()
    ids = torch.randint(0, CONFIG.vocab_size, (2, 33), generator=torch.Generator().manual_seed(2))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    loss_fn(ours(inputs), targets).backward()
    loss_fn(hf(inputs).logits, targets).backward()
    hf_grads = hf_to_dtrain({n: p.grad for n, p in hf.named_parameters()})
    for name, p in ours.named_parameters():
        torch.testing.assert_close(p.grad, hf_grads[name], rtol=1e-5, atol=1e-6)


def test_param_count_formula():
    assert num_params(CONFIG) == sum(p.numel() for p in Llama(CONFIG).parameters())


def test_llama3_8b_param_count():
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    assert num_params(config) == 8_030_261_248


def test_recompute_gives_same_gradients():
    torch.manual_seed(0)
    plain = Llama(CONFIG)
    recomputed = Llama(CONFIG, recompute=True)
    recomputed.load_state_dict(plain.state_dict())
    ids = torch.randint(0, CONFIG.vocab_size, (2, 33), generator=torch.Generator().manual_seed(3))
    for model in (plain, recomputed):
        loss_fn(model(ids[:, :-1]), ids[:, 1:]).backward()
    for (name, a), (_, b) in zip(
        plain.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


def test_load_a_tied_hugging_face_checkpoint(tmp_path):
    transformers = pytest.importorskip("transformers")
    from dtrain.hf import load_config, load_weights

    hf_config = transformers.LlamaConfig(
        vocab_size=CONFIG.vocab_size,
        hidden_size=CONFIG.hidden_size,
        intermediate_size=CONFIG.intermediate_size,
        num_hidden_layers=CONFIG.num_hidden_layers,
        num_attention_heads=CONFIG.num_attention_heads,
        num_key_value_heads=CONFIG.num_key_value_heads,
        max_position_embeddings=CONFIG.max_position_embeddings,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    hf = transformers.LlamaForCausalLM(hf_config).eval()
    hf.save_pretrained(tmp_path)
    config = load_config(tmp_path)
    assert config.rope_theta == 100000.0
    ours = Llama(config).eval()
    ours.load_state_dict(load_weights(tmp_path), strict=True)
    ids = torch.randint(0, CONFIG.vocab_size, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(ours(ids), hf(ids).logits, rtol=1e-5, atol=1e-5)
