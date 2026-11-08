import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_sleep.consolidate import SleepConfig
from llm_sleep.sleeper import Sleeper

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["sleep", "nosleep"], required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--tasks", type=int, default=5)
ap.add_argument("--per_task", type=int, default=6)
ap.add_argument("--train_per_class", type=int, default=20)
ap.add_argument("--test_per_class", type=int, default=10)
ap.add_argument("--periods", default="8,32,128")
ap.add_argument("--wake_lr", type=float, default=1e-2)
ap.add_argument("--sleep_steps", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs")
ap.add_argument("--opaque", type=int, default=1)
ap.add_argument("--optim", choices=["sgd", "adam"], default="adam")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--only_task0", type=int, default=0)
ap.add_argument("--slots", type=int, default=64)
ap.add_argument("--eval_every", type=int, default=0)
ap.add_argument("--mode", choices=["paper", "transfer"], default="paper")
ap.add_argument("--sleep_every", default="")
a = ap.parse_args()

random.seed(a.seed)
torch.manual_seed(a.seed)
ds = load_dataset("clinc/clinc_oos", "plus")
names = ds["train"].features["intent"].names
oos = names.index("oos")
intents = [i for i in range(len(names)) if i != oos]
random.shuffle(intents)
intents = intents[: a.tasks * a.per_task]
tasks = [intents[i * a.per_task:(i + 1) * a.per_task] for i in range(a.tasks)]


def pick(split, n):
    by = {}
    for ex in ds[split]:
        if ex["intent"] in intents and len(by.setdefault(ex["intent"], [])) < n:
            by[ex["intent"]].append(ex["text"])
    return by


train, test = pick("train", a.train_per_class), pick("test", a.test_per_class)
codes = random.sample(range(100, 1000), len(intents))
label = {i: (f"code {c}" if a.opaque else names[i].replace("_", " ")) for i, c in zip(intents, codes)}

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
n = model.config.num_hidden_layers
s = Sleeper(model, tok, layers=[n // 2, n - 4], r=64, slots=a.slots, periods=tuple(int(x) for x in a.periods.split(",")),
            wake_lr=a.wake_lr, sleep_every=tuple(int(x) for x in a.sleep_every.split(",")) if a.sleep_every else None, wake_optim=torch.optim.Adam if a.optim == "adam" else torch.optim.SGD, log=None,
            sleep_cfg=SleepConfig(steps=a.sleep_steps, gen_tokens=8, teacher_samples=2, lti_samples=2, alpha=0.5, mode=a.mode))


def prompt(text):
    return f"Query: {text}\nIntent:"


def encode(text, intent):
    p = tok(prompt(text), return_tensors="pt").input_ids
    full = tok(prompt(text) + " " + label[intent] + tok.eos_token, return_tensors="pt").input_ids
    lab = full.clone()
    lab[:, : p.shape[1]] = -100
    return full, lab, p


@torch.no_grad()
def evaluate(seen):
    correct = total = 0
    cands = [" " + label[i] + tok.eos_token for i in seen]
    for i in seen:
        for text in test[i]:
            p = prompt(text)
            plen = len(tok(p).input_ids)
            batch = tok([p + c for c in cands], return_tensors="pt", padding=True).to("cuda")
            logits = model(**batch).logits[:, :-1].float()
            tgt = batch.input_ids[:, 1:]
            lp = torch.gather(logits.log_softmax(-1), -1, tgt.unsqueeze(-1)).squeeze(-1)
            m = batch.attention_mask[:, 1:].clone()
            m[:, : plen - 1] = 0
            score = (lp * m).sum(-1) / m.sum(-1)
            correct += int(seen[score.argmax().item()] == i)
            total += 1
    return correct / total


@torch.no_grad()
def per_task_acc(seen_tasks):
    return [evaluate(t) for t in seen_tasks]


res = {"args": vars(a), "zero_shot": None, "after_task": []}
res["zero_shot"] = evaluate(intents)
t0 = time.time()
for ti, task in enumerate(tasks):
    exs = [(x, i) for i in task for x in train[i]] * a.epochs
    random.shuffle(exs)
    for k, (x, i) in enumerate(exs):
        full, lab, p = encode(x, i)
        loss = s.wake(full, labels=lab, prompt=p, sleep=(a.method == "sleep"))
        if a.eval_every and (k + 1) % a.eval_every == 0:
            print(json.dumps({"step": k + 1, "loss": round(loss, 3), "acc": evaluate(task), "sleeps": len(s.sleeps)}), flush=True)
    seen = [i for t in tasks[: ti + 1] for i in t]
    row = {"task": ti, "seen_acc": evaluate(seen), "per_task": per_task_acc(tasks[: ti + 1]),
           "elapsed_s": round(time.time() - t0), "sleeps": len(s.sleeps)}
    res["after_task"].append(row)
    print(json.dumps(row), flush=True)
    if a.only_task0:
        raise SystemExit

final = res["after_task"][-1]["per_task"]
peaks = [max(r["per_task"][k] for r in res["after_task"] if len(r["per_task"]) > k) for k in range(a.tasks)]
res["final_avg"] = sum(final) / len(final)
res["forgetting"] = sum(p - f for p, f in zip(peaks[:-1], final[:-1])) / max(1, a.tasks - 1)
res["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
Path(a.out).mkdir(parents=True, exist_ok=True)
out = Path(a.out) / f"clinc-{a.method}-s{a.seed}-p{a.periods.replace(',', '_')}-{a.optim}{a.wake_lr}-e{a.epochs}-{a.mode}-se{a.sleep_every.replace(',', '_')}.json"
out.write_text(json.dumps(res, indent=1))
print("final_avg", round(res["final_avg"], 3), "forgetting", round(res["forgetting"], 3), "->", out)
