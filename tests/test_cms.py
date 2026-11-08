import copy

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm_sleep.cms import cms_chains, inject_cms, set_dream


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).float().eval()


def test_injection_is_identity():
    base = tiny_model()
    m = copy.deepcopy(base)
    inject_cms(m, layers=[1, 2], r=8, slots=4, periods=[1, 2, 4])
    x = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        diff = (base(x).logits - m(x).logits).abs().max()
    assert diff < 1e-5


def test_only_cms_trainable_and_masked_experts_get_no_grad():
    m = tiny_model()
    inject_cms(m, layers=[1], r=8, slots=4, periods=[1, 2])
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(".chain." in n for n in trainable)
    blk = cms_chains(m)[0].blocks[1]
    blk.unlock()
    with torch.no_grad():
        for b in cms_chains(m)[0].blocks:
            b.base.B.normal_(0, 0.1)
            for e in b.experts:
                e.B.normal_(0, 0.1)
    m(torch.randint(0, 128, (1, 8))).logits.sum().backward()
    assert blk.experts[0].A.grad is not None
    for e in blk.experts[1:]:
        assert e.A.grad is None and e.B.grad is None


def test_unlocking_zero_init_expert_keeps_output():
    m = tiny_model()
    inject_cms(m, layers=[1], r=8, slots=4, periods=[1, 2])
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        before = m(x).logits
        cms_chains(m)[0].blocks[1].unlock()
        after = m(x).logits
    assert (before - after).abs().max() < 1e-6


def test_dream_routing_changes_output():
    m = tiny_model()
    inject_cms(m, layers=[1], r=8, slots=4, periods=[1, 2], top_k=1)
    blk = cms_chains(m)[0].blocks[1]
    for _ in range(4):
        blk.unlock()
    with torch.no_grad():
        for e in blk.experts:
            e.B.normal_(0, 0.5)
        blk.router.weight.normal_(0, 1)
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        a = m(x).logits
        set_dream(m, True)
        b = m(x).logits
    assert (a - b).abs().max() > 1e-4


def test_slot_exhaustion_and_reset():
    m = tiny_model()
    inject_cms(m, layers=[0], r=8, slots=2, periods=[1])
    blk = cms_chains(m)[0].blocks[0]
    blk.unlock(); blk.unlock()
    try:
        blk.unlock()
        assert False
    except RuntimeError:
        pass
    blk.reset_experts()
    assert len(blk.free_slots) == 2
