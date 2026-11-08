import torch

from llm_sleep.cms import cms_chains, inject_cms
from llm_sleep.dream import DreamConfig, DreamTask, TrialLoRA, dream, rest_em, select
from llm_sleep.schedule import LevelSchedule
from test_cms import tiny_model


class Tok:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        ids = [1 + (ord(c) % 120) for c in text[:20]]

        class R:
            input_ids = torch.tensor([ids])
        return R()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def setup():
    m = tiny_model()
    inject_cms(m, layers=[1], r=8, slots=4, periods=[1, 8])
    blk = cms_chains(m)[0].blocks[1]
    blk.unlock(); blk.unlock()
    with torch.no_grad():
        for e in blk.experts:
            e.B.normal_(0, 0.1)
    cfg = DreamConfig(n_dreams=4, top_k=1, n_random=1, dream_tokens=8, trial_steps=3, trial_lr=1e-2,
                      lora_r=4, lora_alpha=8)
    lora = TrialLoRA(m, cfg)
    torch.manual_seed(0)
    target = torch.randint(1, 128, (1, 12))

    def metric(model):
        return -model(target, labels=target).loss.item()

    return m, cfg, lora, DreamTask("some passage", metric)


def test_trial_lora_is_inert_outside_trials():
    m, cfg, lora, task = setup()
    x = torch.randint(1, 128, (1, 10))
    with torch.no_grad():
        a = m(x).logits
    out = dream(m, Tok(), lora, task, cfg)
    with torch.no_grad():
        b = m(x).logits
    assert (a - b).abs().max() < 1e-6
    assert len(out["results"]) == 2
    assert all(r["reward"] in (0, 1) for r in out["results"])
    assert not any(p.requires_grad for p in lora.params)


def test_select_topk_plus_random():
    idx = select([0.1, 0.9, 0.5, 0.3], k=2, b=1)
    assert idx[:2] == [1, 2] and idx[2] in (0, 3)


def test_rest_em_updates_fast_memory():
    m, cfg, lora, task = setup()
    sched = LevelSchedule(m, lr=0.1)
    seq = torch.randint(1, 128, (1, 16))
    before = cms_chains(m)[0].blocks[0].base.B.detach().clone()
    rest_em(sched, m, [seq], prompt_len=4)
    assert not torch.equal(before, cms_chains(m)[0].blocks[0].base.B)
