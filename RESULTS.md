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
