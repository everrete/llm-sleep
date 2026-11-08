import copy

import torch

from llm_sleep.cms import cms_chains, inject_cms
from llm_sleep.consolidate import SleepConfig, consolidate
from llm_sleep.schedule import LevelSchedule
from test_cms import tiny_model


class Tok:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def nll(m, x):
    with torch.no_grad():
        return m(x, labels=x).loss.item()


def setup(seed=0, steps=7, periods=(1, 8)):
    m = tiny_model()
    inject_cms(m, layers=[1, 2], r=16, slots=4, periods=list(periods))
    torch.manual_seed(seed)
    x = torch.randint(1, 128, (1, 24))
    sched = LevelSchedule(m, lr=0.3)
    for _ in range(steps):
        sched.wake_step(m(x, labels=x).loss)
    return m, x, sched


def test_schedule_periods():
    m, x, sched = setup()
    assert sched.step_i == 7
    assert sched.due(8) == [0, 1] and sched.due(4) == [0]


def test_prospective_does_not_apply():
    m, x, sched = setup()
    m(x, labels=x).loss.backward()
    blk = sched.level_blocks(0)[0]
    before = blk.base.B.detach().clone()
    pro = sched.prospective(0)
    assert torch.equal(blk.base.B, before)
    assert not torch.equal(pro[(id(blk), "B")], before)


def test_consolidation_protects_against_overwrite():
    torch.manual_seed(1)
    m, a, sched = setup(steps=40, periods=(1, 64))
    torch.manual_seed(2)
    b = torch.randint(1, 128, (1, 24))
    learned_a = nll(m, a)
    for _ in range(20):
        m(b, labels=b).loss.backward()
    control = copy.deepcopy(m)
    csched = LevelSchedule(control, lr=sched.lr)
    for p, q in zip(sched.level_params(0), csched.level_params(0)):
        q.grad = None if p.grad is None else p.grad.clone()
    csched.optims[0].step()
    forgot_a = nll(control, a)

    cfg = SleepConfig(steps=150, lr=3e-2, lam=0.0, alpha=1.0, gen_tokens=16, teacher_samples=16)
    out = consolidate(m, Tok(), sched, 0, [a[:, :8]], cfg)
    assert out and out["new_slots"]
    sched.optims[0].step()
    kept_a = nll(m, a)
    assert forgot_a - learned_a > 0.3
    assert kept_a < forgot_a - 0.5 * (forgot_a - learned_a)


def test_imitation_rl_path_runs():
    torch.manual_seed(3)
    m, a, sched = setup(steps=8, periods=(1, 64))
    m(a, labels=a).loss.backward()
    cfg = SleepConfig(steps=4, lr=1e-2, lam=0.5, alpha=0.5, gen_tokens=8, lti_samples=3, teacher_samples=2)
    out = consolidate(m, Tok(), sched, 0, [a[:, :6]], cfg)
    rewards = [r for _, r in out["history"]]
    assert len(rewards) == 4 and all(0.0 <= r <= 1.0 for r in rewards)
