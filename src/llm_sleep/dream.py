import random
from dataclasses import dataclass, field
from typing import Callable

import torch
from peft import LoraConfig, inject_adapter_in_model

from .cms import set_dream

DREAM_PROMPT = "{context}\n\nList several implications and restatements of the passage above.\n"


@dataclass
class DreamConfig:
    n_dreams: int = 8
    top_k: int = 2
    n_random: int = 1
    dream_tokens: int = 128
    temperature: float = 1.0
    trial_steps: int = 10
    trial_lr: float = 1e-4
    lora_r: int = 64
    lora_alpha: int = 128
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    prompt: str = DREAM_PROMPT


@dataclass
class DreamTask:
    context: str
    metric: Callable[[object], float]


class TrialLoRA:
    """Disposable LoRA used to test one dream at a time (paper Sec. 3.4)."""

    def __init__(self, model, cfg: DreamConfig, name="dream_trial"):
        lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.lora_targets, lora_dropout=0.0)
        prior = {n: p.requires_grad for n, p in model.named_parameters()}
        inject_adapter_in_model(lc, model, adapter_name=name)
        for n, p in model.named_parameters():
            if n in prior:
                p.requires_grad_(prior[n])
        self.model = model
        self.params = [p for n, p in model.named_parameters() if f".{name}." in n or n.endswith(f".{name}")]
        self.a = [p for n, p in model.named_parameters() if "lora_A" in n and name in n]
        self.b = [p for n, p in model.named_parameters() if "lora_B" in n and name in n]
        self.reset()
        self.trainable(False)

    @torch.no_grad()
    def reset(self):
        for p in self.a:
            torch.nn.init.kaiming_uniform_(p, a=5**0.5)
        for p in self.b:
            p.zero_()

    def trainable(self, on: bool):
        for p in self.params:
            p.requires_grad_(on)


def _sft_loss(model, ids, prompt_len):
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    return model(ids, labels=labels).loss


def generate_dreams(model, tok, task: DreamTask, cfg: DreamConfig):
    dev = next(model.parameters()).device
    prompt = tok(cfg.prompt.format(context=task.context), return_tensors="pt").input_ids.to(dev)
    set_dream(model, True)
    try:
        with torch.no_grad():
            outs = [model.generate(prompt, max_new_tokens=cfg.dream_tokens, do_sample=True,
                                   temperature=cfg.temperature, top_p=1.0,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    for _ in range(cfg.n_dreams)]
    finally:
        set_dream(model, False)
    return prompt, outs


def importance(model, lora: TrialLoRA, seq, prompt_len) -> float:
    lora.trainable(True)
    for p in lora.params:
        p.grad = None
    _sft_loss(model, seq, prompt_len).backward()
    sq = sum(float(p.grad.pow(2).sum()) for p in lora.params if p.grad is not None)
    for p in lora.params:
        p.grad = None
    lora.trainable(False)
    return sq**0.5


def select(scores: list[float], k: int, b: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top = order[:k]
    rest = order[k:]
    return top + random.sample(rest, min(b, len(rest)))


def trial(model, lora: TrialLoRA, seq, prompt_len, task: DreamTask, base_score: float, cfg: DreamConfig):
    lora.reset()
    lora.trainable(True)
    opt = torch.optim.AdamW(lora.params, lr=cfg.trial_lr)
    for _ in range(cfg.trial_steps):
        opt.zero_grad(set_to_none=True)
        _sft_loss(model, seq, prompt_len).backward()
        opt.step()
    lora.trainable(False)
    with torch.no_grad():
        score = task.metric(model)
    lora.reset()
    return int(score > base_score), score


def dream(model, tok, lora: TrialLoRA, task: DreamTask, cfg: DreamConfig, log=None):
    """One dreaming pass: sample, rank, trial, reward. Returns accepted (prompt_ids, dream_ids) pairs."""
    frozen = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    lora.reset()
    try:
        with torch.no_grad():
            base = task.metric(model)
        prompt, outs = generate_dreams(model, tok, task, cfg)
        plen = prompt.shape[1]
        scores = [importance(model, lora, o, plen) for o in outs]
        chosen = select(scores, cfg.top_k, cfg.n_random)
        accepted, results = [], []
        for i in chosen:
            r, s = trial(model, lora, outs[i], plen, task, base, cfg)
            results.append({"idx": i, "importance": scores[i], "score": s, "reward": r})
            if log:
                log(results[-1])
            if r:
                accepted.append(outs[i])
        return {"base": base, "prompt": prompt, "accepted": accepted, "results": results}
    finally:
        for p in model.parameters():
            p.requires_grad_(frozen.get(id(p), False))


def rest_em(sched, model, accepted_seqs, prompt_len, epochs: int = 1, on_due=None):
    """ReST-EM M-step: learn accepted dreams through the wake path so they enter fast memory."""
    for _ in range(epochs):
        for seq in accepted_seqs:
            sched.wake_step(_sft_loss(model, seq, prompt_len), on_due=on_due)
