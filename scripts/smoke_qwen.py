import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_sleep.consolidate import SleepConfig
from llm_sleep.sleeper import Sleeper

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
n = model.config.num_hidden_layers
s = Sleeper(model, tok, layers=[n // 2, n - 4], r=64, slots=8, periods=(1, 4, 16), wake_lr=1e-2,
            sleep_cfg=SleepConfig(steps=20, gen_tokens=32, teacher_samples=2, lti_samples=2, alpha=0.5))
fact = "The Velmora Institute was founded in 1987 by the botanist Ilse Krantz in the town of Orbeck."
q = tok(fact, return_tensors="pt").input_ids.cuda()

def nll():
    with torch.no_grad():
        return model(q, labels=q).loss.item()

print("before", round(nll(), 3))
t = time.time()
for i in range(16):
    s.wake(q)
print("after 16 wake steps", round(nll(), 3), f"{time.time() - t:.1f}s",
      "sleeps", [(x["level"], x["step"]) for x in s.sleeps])
print("peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
