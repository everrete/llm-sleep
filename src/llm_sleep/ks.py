import math

import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein


def gjsd(student_logits, teacher_logits, mask, beta=0.5, temperature=1.0):
    s = F.log_softmax(student_logits.float() / temperature, -1)
    t = F.log_softmax(teacher_logits.float() / temperature, -1)
    if beta == 0:
        per = (t.exp() * (t - s)).sum(-1)
    elif beta == 1:
        per = (s.exp() * (s - t)).sum(-1)
    else:
        m = torch.logsumexp(torch.stack([s + math.log(1 - beta), t + math.log(beta)]), 0)
        per = beta * (t.exp() * (t - m)).sum(-1) + (1 - beta) * (s.exp() * (s - m)).sum(-1)
    mask = mask.float()
    return (per * mask).sum() / mask.sum().clamp(min=1)


def lev_reward(pred: str, ref: str, z0: float = 0.5) -> float:
    if not pred and not ref:
        return 1.0
    z = Levenshtein.distance(pred, ref) / max(len(pred), len(ref))
    return 1.0 - z if z <= z0 else 0.0


def imitation_reward(pred: str, ref: str, gamma: float, z0: float, judge=None) -> float:
    sem = float(judge(pred, ref)) if (judge is not None and gamma > 0) else 0.0
    return gamma * sem + (1 - gamma) * lev_reward(pred, ref, z0)


def seq_logprob(logits, labels, mask):
    lp = torch.gather(F.log_softmax(logits.float(), -1), -1, labels.unsqueeze(-1)).squeeze(-1)
    return (lp * mask.float()).sum(-1)
