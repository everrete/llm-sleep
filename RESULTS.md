# Results

## CLINC150 class-incremental, Qwen2.5-0.5B, 2026-11-08

5 tasks x 6 intents, opaque labels, 20 train / 10 test per intent, 3 epochs per task, Adam 1e-3,
CMS periods 1,8,64 in layers 12 and 20, sleep every 120 / 360 steps, 150 sleep steps, seed 0.
`final_avg` is mean accuracy over the 5 tasks after the last one (chance 1/6 within a task);
`forgetting` is mean drop from each earlier task's best.

| method | task-0 acc right after learning | final_avg | forgetting | wall time |
|---|---|---|---|---|
| no sleep | 0.98 | **0.52** | 0.59 | 1.5 min |
| sleep, paper mode | 0.72 | 0.37 | 0.55 | 15 min |
| sleep, transfer mode | 0.38 | 0.21 | 0.11 (nothing learned to forget) | 15 min |
| sleep, paper mode, thorough (all 64 replay prompts, 4 teacher samples, 400 steps) | 0.67 | 0.34 | 0.64 | 43 min |
| sleep, transfer mode, thorough | 0.17 | 0.18 | 0.22 | 45 min |

Neither sleep mode beats no sleep here, and making each sleep 8x more thorough did not change that. Paper mode forgets about as much and learns less;
transfer mode loses most of what the fast block knew at each reset. One seed, small scale.
The first guess (sleep starved of data) is ruled out by the thorough rows. Paper mode still
forgets as much as no sleep while learning each task less well; transfer mode cannot carry a
task's mapping across the fast-block reset even with 400 distillation steps over all prompts.

## SQuAD knowledge incorporation, Qwen2.5-0.5B, 2026-11-08

Single-passage setting: for each SQuAD validation passage the memory chain is reset, the model
reads the passage (20 wake passes, Adam 3e-3), then is asked the passage's questions with no
context. 2 questions per passage are used only for the dream reward; the rest are the test.
Score: normalized gold answer contained in the greedy answer. 12 passages, seed 0.

| stage | accuracy |
|---|---|
| base model, no passage | 0.006 |
| after reading the passage (fine-tuning only) | 0.175 |
| after one round of dreaming (8 dreams, top 2 + 1 random, LoRA trials, ReST-EM into fast memory) | **0.248** |

Dreaming adds about 7 points over fine-tuning alone, the same direction as the paper's Table 3,
where dreaming carries most of the gain.

### Sleep after dreaming, first 6 passages

Same setup, then `sleep_now(0)` and `sleep_now(1)` (200 distillation steps each). Dream results
vary between runs because dream sampling is random.

| run | after wake | after dream | after sleep |
|---|---|---|---|
| dream only (first 6 of the 12-passage run) | 0.167 | 0.278 | - |
| sleep, `paper` mode | 0.167 | 0.278 | **0.222** |
| sleep, `transfer` mode | 0.167 | 0.111 | 0.000 |

`paper` mode keeps most of what dreaming added (5 of 6 passages unchanged; passage 0 dropped from
0.33 to 0). `transfer` mode wipes everything, which matches its CLINC result: copying the fast
block into a fresh expert and resetting the fast block loses the knowledge. Each passage has
only 1 to 3 test questions, so these numbers are noisy.

### Sequential passages (no reset), 10 passages, 2026-11-09

Memory is kept across passages. After each passage every passage seen so far is re-tested.
`final_avg` is accuracy over all 10 passages at the end; `forgetting` is the average drop from
just-learned to final for passages 1-9.

| run | just learned (avg) | final_avg | forgetting |
|---|---|---|---|
| fine-tuning only | 0.276 | 0.048 | 0.254 |
| dreaming | 0.198 | 0.000 | 0.220 |
| dreaming + `paper` sleep | 0.176 | 0.021 | **0.172** |

All three forget nearly everything earlier. Sleep forgets the least and is the only run that
kept passage 6 (14 test questions) to the end (0.21 vs 0 for dreaming), but most passages have
1-3 test questions, so the gaps are within noise. Needs passages with many questions.

### Sequential passages, passages with at least 8 test questions, 2026-11-10

| run | just learned (avg) | final_avg | forgetting |
|---|---|---|---|
| fine-tuning only | 0.123 | 0.022 | 0.112 |
| dreaming | 0.080 | 0.004 | 0.085 |
| dreaming + `paper` sleep | 0.125 | **0.029** | 0.106 |

With more questions per passage the 0.5B model barely learns a passage in the first place (about
12% right after reading), so every method ends near zero and the differences are noise. At this
model size the benchmark has a floor effect; it can't separate the methods.

Qwen2.5-1.5B, fine-tuning only, same 10 passages: just learned 0.158, final_avg 0.029,
forgetting 0.142. The bigger model learns a bit more but still forgets almost all of it.

### Capacity probe, 0.5B, fine-tuning only, 6 sequential passages (min 8 test questions)

| memory placement | rank | lr | epochs | just learned |
|---|---|---|---|---|
| 2 layers (default) | 64 | 3e-3 | 20 | 0.083 |
| all 24 layers | 64 | 3e-3 | 20 | 0.069 |
| all 24 layers | 64 | 1e-3 | 20 | **0.172** |
| all 24 layers | 256 | 1e-3 | 20 | 0.021 |
| all 24 layers | 64 | 1e-3 | 40 | 0.161 |

More memory capacity does not fix it: the best setting learns 17% of a passage's answers. The
limit is closed-book QA from one raw passage on a 0.5B model, not the size of the memory.
