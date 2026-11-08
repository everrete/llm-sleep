import random

import torch
import torch.nn.functional as F
from torch import nn


class LowRankMLP(nn.Module):
    def __init__(self, d: int, r: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(d, r))
        self.B = nn.Parameter(torch.zeros(r, d))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        return F.silu(x @ self.A.to(x.dtype)) @ self.B.to(x.dtype)

    @torch.no_grad()
    def reset(self):
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.B.zero_()


class CMSBlock(nn.Module):
    def __init__(self, d: int, r: int, slots: int, period: int, top_k: int = 2):
        super().__init__()
        self.period = period
        self.top_k = top_k
        self.base = LowRankMLP(d, r)
        self.experts = nn.ModuleList(LowRankMLP(d, r) for _ in range(slots))
        self.router = nn.Linear(d, slots, bias=False)
        nn.init.zeros_(self.router.weight)
        self.register_buffer("active", torch.zeros(slots, dtype=torch.bool))
        self.dream = False

    @property
    def free_slots(self):
        return [i for i, a in enumerate(self.active.tolist()) if not a]

    def forward(self, x):
        out = x + self.base(x)
        idx = self.active.nonzero().flatten().tolist()
        if not idx:
            return out
        logits = self.router(x.to(self.router.weight.dtype))[..., idx]
        k = min(self.top_k, len(idx))
        top_v, top_i = logits.topk(k, dim=-1)
        gates = torch.zeros_like(logits).scatter(-1, top_i, top_v.softmax(-1))
        if self.dream:
            rnd = torch.randint(len(idx), gates.shape[:-1] + (1,), device=x.device)
            extra = torch.zeros_like(gates).scatter(-1, rnd, 1.0 / (k + 1))
            gates = gates * (k / (k + 1)) + extra
        for j, e in enumerate(idx):
            g = gates[..., j : j + 1]
            if g.any():
                out = out + g.to(x.dtype) * self.experts[e](x)
        return out

    @torch.no_grad()
    def unlock(self) -> int:
        free = self.free_slots
        if not free:
            raise RuntimeError("no free expert slots")
        i = free[0]
        self.experts[i].reset()
        self.active[i] = True
        return i

    @torch.no_grad()
    def reset_experts(self):
        for i in self.active.nonzero().flatten().tolist():
            self.experts[i].reset()
        self.active.zero_()
        self.router.weight.zero_()


class CMSChain(nn.Module):
    def __init__(self, d: int, r: int, slots: int, periods: list[int], top_k: int = 2):
        super().__init__()
        assert all(b % a == 0 for a, b in zip(periods, periods[1:])), "periods must nest"
        self.blocks = nn.ModuleList(CMSBlock(d, r, slots, p, top_k) for p in periods)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class CMSWrappedMLP(nn.Module):
    def __init__(self, mlp: nn.Module, chain: CMSChain):
        super().__init__()
        self.mlp = mlp
        self.chain = chain

    def forward(self, x, *args, **kwargs):
        return self.mlp(x, *args, **kwargs) + (self.chain(x) - x)


def _decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find decoder layers")


def inject_cms(model, layers, r=64, slots=8, periods=(1, 4, 16, 64, 256), top_k=2):
    for p in model.parameters():
        p.requires_grad_(False)
    dl = _decoder_layers(model)
    d = model.config.hidden_size
    dev = next(model.parameters()).device
    chains = []
    for li in layers:
        layer = dl[li]
        chain = CMSChain(d, r, slots, list(periods), top_k).to(device=dev, dtype=torch.float32)
        layer.mlp = CMSWrappedMLP(layer.mlp, chain)
        chains.append(chain)
    return chains


def cms_chains(model):
    return [m for m in model.modules() if isinstance(m, CMSChain)]


def set_dream(model, on: bool):
    for c in cms_chains(model):
        for b in c.blocks:
            b.dream = on


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
