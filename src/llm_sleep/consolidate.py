import contextlib
import random
from dataclasses import dataclass

import torch

from .ks import gjsd, imitation_reward, seq_logprob


@dataclass
class SleepConfig:
    steps: int = 100
    lr: float = 5e-4
    lam: float = 0.5
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.0
    z0: float = 0.5
    gen_tokens: int = 64
    lti_samples: int = 2
    temperature: float = 1.0
    teacher_samples: int = 4


def _snapshot(blocks):
    return [{k: v.detach().clone() for k, v in b.state_dict().items()} for b in blocks]


def _load(blocks, states):
    with torch.no_grad():
        for b, s in zip(blocks, states):
            b.load_state_dict(s)


@contextlib.contextmanager
def _as_state(blocks, states):
    cur = _snapshot(blocks)
    _load(blocks, states)
    try:
        yield
    finally:
        _load(blocks, cur)


def _generate(model, tok, prompt_ids, n_new, temperature):
    out = model.generate(prompt_ids, max_new_tokens=n_new, do_sample=True, temperature=temperature,
                         top_p=1.0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return out


def consolidate(model, tok, sched, sender: int, prompts: list[torch.Tensor], cfg: SleepConfig,
                judge=None, log=None):
    """Knowledge seeding of level `sender` into a new expert at level sender+1 (paper Sec. 3.2-3.3).
    Leaves sender base at pre-update values with grads intact so the caller's optimizer step commits it."""
    recv = sender + 1
    if recv >= sched.n_levels or not prompts:
        return None
    s_blocks = sched.level_blocks(sender)
    r_blocks = sched.level_blocks(recv)
    blocks = s_blocks + r_blocks
    grads = {id(p): p.grad.detach().clone() for b in s_blocks for p in b.base.parameters() if p.grad is not None}
    teacher = _snapshot(blocks)

    prosp = sched.prospective(sender)
    with torch.no_grad():
        for b in s_blocks:
            for n, p in b.base.named_parameters():
                p.copy_(prosp[(id(b), n)])
            b.reset_experts()
    new = [b.unlock() for b in r_blocks]

    train = []
    hooks = []
    for b, i in zip(r_blocks, new):
        train += list(b.experts[i].parameters())
        train.append(b.router.weight)
        row = torch.zeros_like(b.router.weight)
        row[i] = 1
        hooks.append(b.router.weight.register_hook(lambda g, row=row: g * row))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in train:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train, lr=cfg.lr)

    dev = next(model.parameters()).device
    with torch.no_grad(), _as_state(blocks, teacher):
        tdata = [[_generate(model, tok, p.to(dev), cfg.gen_tokens, cfg.temperature) for _ in range(cfg.teacher_samples)]
                 for p in prompts]

    hist = []
    for step in range(cfg.steps):
        k = random.randrange(len(prompts))
        prompt = prompts[k].to(dev)
        plen = prompt.shape[1]
        if random.random() < cfg.lam:
            with torch.no_grad():
                seq = _generate(model, tok, prompt, cfg.gen_tokens, cfg.temperature)
        else:
            seq = random.choice(tdata[k])
        with torch.no_grad(), _as_state(blocks, teacher):
            t_logits = model(seq).logits[:, :-1]
        s_logits = model(seq).logits[:, :-1]
        mask = torch.zeros(seq.shape[0], seq.shape[1] - 1, device=dev)
        mask[:, plen - 1:] = 1
        d_loss = gjsd(s_logits, t_logits, mask, cfg.beta, cfg.temperature)

        rl_loss = torch.zeros((), device=dev)
        if cfg.alpha < 1:
            ref = random.choice(tdata[k])[0]
            cut = random.randint(plen, max(plen, ref.shape[0] - 2))
            pre = ref[:cut].unsqueeze(0)
            ref_txt = tok.decode(ref[cut:], skip_special_tokens=True)
            with torch.no_grad():
                outs = [_generate(model, tok, pre, max(1, ref.shape[0] - cut), cfg.temperature)
                        for _ in range(cfg.lti_samples)]
            rewards = torch.tensor([imitation_reward(tok.decode(o[0, cut:], skip_special_tokens=True),
                                                     ref_txt, cfg.gamma, cfg.z0, judge) for o in outs], device=dev)
            adv = rewards - rewards.mean()
            for o, a in zip(outs, adv):
                if a == 0:
                    continue
                lg = model(o).logits[:, :-1]
                m = torch.zeros(1, o.shape[1] - 1, device=dev)
                m[:, cut - 1:] = 1
                rl_loss = rl_loss - a * seq_logprob(lg, o[:, 1:], m).squeeze() / len(outs)
            hist_r = rewards.mean().item()
        else:
            hist_r = float("nan")

        loss = cfg.alpha * d_loss + (1 - cfg.alpha) * rl_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append((d_loss.item(), hist_r))
        if log:
            log(step, d_loss.item(), hist_r)

    for h in hooks:
        h.remove()
    student_recv = _snapshot(r_blocks)
    _load(s_blocks, teacher[: len(s_blocks)])
    with torch.no_grad():
        for b in s_blocks:
            b.reset_experts()
    _load(r_blocks, student_recv)
    for b in s_blocks:
        for p in b.parameters():
            p.grad = grads.get(id(p))
    for p in model.parameters():
        p.requires_grad_(False)
    for l in range(sched.n_levels):
        for p in sched.level_params(l):
            p.requires_grad_(True)
    return {"new_slots": new, "history": hist}
