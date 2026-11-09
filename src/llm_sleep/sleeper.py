import random

import torch

from .cms import inject_cms
from .consolidate import SleepConfig, consolidate
from .dream import DreamConfig, DreamTask, TrialLoRA, dream, rest_em
from .schedule import LevelSchedule


class Sleeper:
    """Wake/sleep lifecycle around a Hugging Face causal LM."""

    def __init__(self, model, tok, layers, r=64, slots=8, periods=(1, 4, 16, 64, 256), top_k=2,
                 wake_lr=1e-3, wake_optim=torch.optim.SGD, sleep_cfg: SleepConfig | None = None, dream_cfg: DreamConfig | None = None,
                 prompt_len=32, replay=64, sleep_prompts=8, sleep_every=None, log=print):
        self.model, self.tok = model, tok
        inject_cms(model, layers, r=r, slots=slots, periods=periods, top_k=top_k)
        self.sched = LevelSchedule(model, lr=wake_lr, optim=wake_optim)
        self.sleep_cfg = sleep_cfg or SleepConfig()
        self.dream_cfg = dream_cfg or DreamConfig()
        self.lora = None
        self.prompt_len = prompt_len
        self.sleep_every = sleep_every
        self.replay = replay
        self.sleep_prompts = sleep_prompts
        self.memory = []
        self.seen = 0
        self.log = log
        self.sleeps = []

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _on_due(self, level):
        if level + 1 >= self.sched.n_levels:
            return
        if self.sleep_every and self.sched.step_i % self.sleep_every[level]:
            return
        if not all(b.free_slots for b in self.sched.level_blocks(level + 1)):
            self.log(f"level {level + 1} full; skipping consolidation")
            return
        cutoff = self.sched.step_i - self.sched.periods[level]
        old = [p for t, p in self.memory if t <= cutoff]
        prompts = random.sample(old, min(self.sleep_prompts, len(old)))
        out = consolidate(self.model, self.tok, self.sched, level, prompts, self.sleep_cfg)
        if out:
            self._clear_optim_state(level + 1, out["new_slots"])
        self.sleeps.append({"step": self.sched.step_i, "level": level, **(out or {})})
        if out and self.log:
            d = [h[0] for h in out["history"]]
            self.log(f"sleep L{level}->L{level + 1} at step {self.sched.step_i}: "
                     f"distill {d[0]:.4f} -> {d[-1]:.4f}")

    def sleep_now(self, level: int):
        prompts = [p for _, p in self.memory][-self.sleep_prompts:]
        out = consolidate(self.model, self.tok, self.sched, level, prompts, self.sleep_cfg)
        if out:
            self._clear_optim_state(level + 1, out["new_slots"])
            self.sleeps.append({"step": self.sched.step_i, "level": level, **out})
        return out

    def reset(self, chain_state):
        self.model.load_state_dict(chain_state, strict=False)
        for o in self.sched.optims:
            o.state.clear()
            o.zero_grad(set_to_none=True)
        self.sched.step_i = 0
        self.memory, self.seen, self.sleeps = [], 0, []

    def _clear_optim_state(self, level, slots):
        opt = self.sched.optims[level]
        for b, i in zip(self.sched.level_blocks(level), slots):
            for p in b.experts[i].parameters():
                opt.state.pop(p, None)

    def _remember(self, prompt):
        self.seen += 1
        item = (self.sched.step_i, prompt.detach())
        if len(self.memory) < self.replay:
            self.memory.append(item)
        else:
            j = random.randrange(self.seen)
            if j < self.replay:
                self.memory[j] = item

    def wake(self, ids: torch.Tensor, labels: torch.Tensor | None = None, prompt: torch.Tensor | None = None,
             sleep: bool = True):
        ids = ids.to(self.device)
        self.model.train(False)
        loss = self.model(ids, labels=ids if labels is None else labels.to(self.device)).loss
        self._remember(ids[:, : self.prompt_len] if prompt is None else prompt.to(self.device))
        self.sched.wake_step(loss, on_due=self._on_due if sleep else None)
        return loss.item()

    def wake_text(self, text: str):
        return self.wake(self.tok(text, return_tensors="pt").input_ids)

    def dream(self, task: DreamTask, rounds: int = 1):
        if self.lora is None:
            self.lora = TrialLoRA(self.model, self.dream_cfg)
        outs = []
        for _ in range(rounds):
            out = dream(self.model, self.tok, self.lora, task, self.dream_cfg,
                        log=lambda r: self.log(f"dream {r}") if self.log else None)
            if out["accepted"]:
                rest_em(self.sched, self.model, out["accepted"], out["prompt"].shape[1], on_due=self._on_due)
            outs.append(out)
        return outs

    def state_dict(self):
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if ".chain." in n} | {
            n: b.cpu() for n, b in self.model.named_buffers() if ".chain." in n}

    def save(self, path):
        torch.save({"cms": self.state_dict(), "step": self.sched.step_i}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        missing = self.model.load_state_dict(ck["cms"], strict=False)
        self.sched.step_i = ck["step"]
        return missing
