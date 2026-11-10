import argparse
import json
import random
import re
import string
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_sleep.consolidate import SleepConfig
from llm_sleep.dream import DreamConfig, DreamTask
from llm_sleep.sleeper import Sleeper

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["base", "ft", "dream", "sleep"], required=True)
ap.add_argument("--mode", choices=["paper", "transfer"], default="paper")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--passages", type=int, default=20)
ap.add_argument("--dev_q", type=int, default=2)
ap.add_argument("--min_test", type=int, default=3)
ap.add_argument("--wake_epochs", type=int, default=5)
ap.add_argument("--wake_lr", type=float, default=1e-3)
ap.add_argument("--n_dreams", type=int, default=8)
ap.add_argument("--dream_rounds", type=int, default=1)
ap.add_argument("--dream_tokens", type=int, default=96)
ap.add_argument("--sleep_steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sequential", action="store_true")
ap.add_argument("--layers", default="mid", help="mid | all | every:N")
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--slots", type=int, default=16)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="runs")
a = ap.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)

ds = load_dataset("rajpurkar/squad", split="validation")
by = {}
for ex in ds:
    by.setdefault(ex["context"], []).append(ex)
ctxs = [c for c, qs in by.items() if len(qs) >= a.dev_q + a.min_test]
random.shuffle(ctxs)
ctxs = ctxs[: a.passages]

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
n = model.config.num_hidden_layers
if a.layers == "mid":
    layers = [n // 2, n - 4]
elif a.layers == "all":
    layers = list(range(n))
else:
    layers = list(range(0, n, int(a.layers.split(":")[1])))
s = Sleeper(model, tok, layers=layers, r=a.rank, slots=a.slots, periods=(1, 8, 64), wake_lr=a.wake_lr,
            wake_optim=torch.optim.Adam, sleep_prompts=64, log=None,
            sleep_cfg=SleepConfig(steps=a.sleep_steps, gen_tokens=64, teacher_samples=4, lti_samples=2, alpha=0.5,
                                  mode=a.mode),
            dream_cfg=DreamConfig(n_dreams=a.n_dreams, top_k=2, n_random=1, dream_tokens=a.dream_tokens,
                                  trial_steps=10, trial_lr=1e-4))
init = s.state_dict()


def norm(t):
    t = t.lower()
    t = "".join(ch for ch in t if ch not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


def qprompt(q):
    return f"Question: {q}\nAnswer:"


@torch.no_grad()
def qa_acc(qs):
    hits = 0
    for ex in qs:
        ids = tok(qprompt(ex["question"]), return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        ans = norm(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0])
        hits += int(any(norm(g) and norm(g) in ans for g in ex["answers"]["text"]))
    return hits / len(qs)


@torch.no_grad()
def answer_nll(qs):
    tot = 0.0
    for ex in qs:
        p = tok(qprompt(ex["question"]), return_tensors="pt").input_ids
        full = tok(qprompt(ex["question"]) + " " + ex["answers"]["text"][0], return_tensors="pt").input_ids.cuda()
        lab = full.clone()
        lab[:, : p.shape[1]] = -100
        tot += model(full, labels=lab).loss.item()
    return tot / len(qs)


rows = []
t0 = time.time()
for k, ctx in enumerate(ctxs):
    if not a.sequential:
        s.reset(init)
    qs = by[ctx]
    dev, test = qs[: a.dev_q], qs[a.dev_q:]
    row = {"passage": k, "base": qa_acc(test)}
    if a.method != "base":
        ids = tok(ctx, return_tensors="pt").input_ids
        for _ in range(a.wake_epochs):
            s.wake(ids, sleep=False)
        row["after_wake"] = qa_acc(test)
        if a.method in ("dream", "sleep"):
            outs = s.dream(DreamTask(ctx, lambda m: -answer_nll(dev)), rounds=a.dream_rounds)
            row["dreams_accepted"] = sum(len(o["accepted"]) for o in outs)
            row["after_dream"] = qa_acc(test)
        if a.method == "sleep":
            s.sleep_now(0)
            s.sleep_now(1)
            row["after_sleep"] = qa_acc(test)
    if a.sequential:
        row["seen"] = [qa_acc(by[c][a.dev_q:]) for c in ctxs[: k + 1]]
    row["elapsed_s"] = round(time.time() - t0)
    rows.append(row)
    print(json.dumps(row), flush=True)

keys = [k for k in ("base", "after_wake", "after_dream", "after_sleep") if k in rows[0]]
summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
if a.sequential:
    last = rows[-1]["seen"]
    learned = [r["seen"][-1] for r in rows]
    summary["final_avg"] = sum(last) / len(last)
    summary["forgetting"] = sum(learned[j] - last[j] for j in range(len(last) - 1)) / max(1, len(last) - 1)
print("summary", json.dumps(summary))
Path(a.out).mkdir(parents=True, exist_ok=True)
out = Path(a.out) / f"squad-{a.method}-{a.mode}-s{a.seed}-n{a.passages}-t{a.min_test}{'-seq' if a.sequential else ''}{a.tag}.json"
out.write_text(json.dumps({"args": vars(a), "rows": rows, "summary": summary}, indent=1))
