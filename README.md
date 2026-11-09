# llm-sleep

An implementation of the Sleep paradigm from *Language Models Need Sleep: Learning to
Self-Modify and Consolidate Memories* (Behrouz, Hashemi, Javanmard, Mirrokni, arXiv 2606.03979),
built to attach to any Hugging Face causal LM. There is no official code; this follows the paper
as written and records every gap it had to fill in `DECISIONS.md`.

## What it does

- **Wake.** A chain of low-rank memory blocks (a Continuum Memory System) is added beside the MLP
  of chosen decoder layers. Each block updates at its own period from accumulated gradients
  (paper Eq. 2). The pretrained weights stay frozen.
- **Sleep, consolidation.** Before a fast block's update lands, a pre-allocated masked expert in
  the next slower block is unlocked and trained by knowledge seeding: generalized JSD distillation
  from the pre-update model (teacher) into the post-update model (student), plus an RL imitation
  term rewarding semantic and Levenshtein similarity to teacher continuations. Then the fast
  block's experts are reset.
- **Sleep, dreaming.** Self-generated data with one random extra expert per router,
  gradient-importance selection, per-dream LoRA trials, ReST-EM.

## Status

| Milestone | State |
|---|---|
| CMS injection, masked experts, dream routing | done, tested |
| Multi-frequency wake schedule | done, tested |
| Knowledge seeding (GJSD + imitation RL) | done, tested on a toy model |
| Dreaming (random-expert dreams, gradient selection, LoRA trials, ReST-EM) | done, tested |
| `Sleeper` wake/sleep lifecycle on a real model | done, smoke-tested on Qwen2.5-0.5B (1.5 GB VRAM) |
| CLINC150 class-incremental benchmark (`scripts/clinc.py`) | done: sleep does not beat no sleep yet, see RESULTS.md |
| SQuAD knowledge incorporation (`scripts/squad.py`) | dreaming +7 points over fine-tuning; sleep runs pending |

On a toy model, after a fast block's update on new data overwrites what it learned earlier,
one consolidation keeps roughly 60-70% of the earlier learning; without sleep it keeps none.

## Setup

```
uv sync
uv run pytest -q
```

On NixOS, CUDA needs `LD_LIBRARY_PATH=/run/opengl-driver/lib`. Tested on an RTX 5060 (8 GB),
torch 2.11 + cu128.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_sleep.sleeper import Sleeper

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="bfloat16").cuda()
s = Sleeper(model, tok, layers=[12, 20], periods=(8, 32, 128))
s.wake_text("Something worth remembering.")   # sleeps automatically at level boundaries
```
