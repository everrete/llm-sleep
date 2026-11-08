import torch

from .cms import CMSBlock, cms_chains


class LevelSchedule:
    """Per-level accumulated updates (paper Eq. 2) with sleep triggers."""

    def __init__(self, model, lr: float = 1e-3, optim=torch.optim.SGD):
        self.model = model
        self.chains = cms_chains(model)
        self.periods = [b.period for b in self.chains[0].blocks]
        self.n_levels = len(self.periods)
        self.lr = lr
        self.optims = [optim(self.level_params(l), lr=lr) for l in range(self.n_levels)]
        self.step_i = 0

    def level_blocks(self, l) -> list[CMSBlock]:
        return [c.blocks[l] for c in self.chains]

    def level_params(self, l):
        return [p for b in self.level_blocks(l) for p in b.parameters()]

    def due(self, step=None) -> list[int]:
        s = self.step_i if step is None else step
        return [l for l, p in enumerate(self.periods) if s % p == 0]

    def wake_step(self, loss, on_due=None):
        """Backward one microbatch; for each due level (slowest first) call on_due(l) then apply its update."""
        loss.backward()
        self.step_i += 1
        due = sorted(self.due(), reverse=True)
        for l in due:
            if on_due is not None:
                on_due(l)
            self.optims[l].step()
            self.optims[l].zero_grad(set_to_none=True)
        return due

    @torch.no_grad()
    def prospective(self, l) -> dict:
        """Parameters level l would have after its pending update, without applying it."""
        out = {}
        for b in self.level_blocks(l):
            for n, p in b.base.named_parameters():
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                out[(id(b), n)] = p - self.lr * g
        return out
