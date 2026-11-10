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
