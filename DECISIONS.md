# Decisions where the paper is underspecified

1. **Chain placement.** The paper does not say how CMS attaches to a pretrained Transformer.
   The chain runs in parallel with the MLP: `out = mlp(x) + (chain(x) - x)`, fed the same
   normalized input. Feeding it the MLP output instead gave it near-zero activations and it
   could not learn.
2. **Block form.** Each block is `x + base(x) + sum_k gate_k * expert_k(x)`, base and experts
   being low-rank SiLU MLPs (`A: d x r`, `B: r x d`, B zero-initialized so new pieces start as
   identity). Router is linear, top-k softmax over unlocked slots only.
3. **What wake updates.** A level's base, router and active experts all update at that level's
   period. Plain SGD, so the "prospective" update is `p - lr * grad`.
4. **Simultaneous boundaries.** When several levels are due, the slowest consolidates first, so
   an expert seeded into level l is never wiped by l's own consolidation in the same step.
5. **Last level.** The slowest level has nowhere to consolidate into; its update applies directly.
6. **Teacher prompts.** The paper samples a dataset from the teacher but does not say from what
   prompts. Prompts are prefixes of recent wake inputs; several teacher samples per prompt.
7. **Router during seeding.** Only the new slot's router row is trained (gradient-masked);
   everything else is frozen, matching "only update the expanded parameters".
8. **Semantic reward.** Pluggable judge callable; default gamma = 0 (Levenshtein only) until a
   judge is chosen.
9. **Dream routing.** One uniformly random unlocked expert is mixed in at weight 1/(k+1).
10. **Dream trials.** One disposable LoRA (q_proj, v_proj) is injected once and reset (B = 0)
    between trials, so it is inert outside a trial. Importance score is the L2 norm of the SFT
    gradient over that LoRA.
11. **Where accepted dreams go.** ReST-EM's update step trains on accepted dreams through the wake
    path, so they enter fast memory and are consolidated by later sleeps like any other input.
12. **Full last level.** Slots are pre-allocated (the paper's masking trick). When the receiving
    level has no free slot, that consolidation is skipped and logged.
13. **CLINC labels.** Intents are mapped to opaque codes ("code 417") so the base model cannot
    guess them from the name; descriptive names gave 58% zero-shot and hid forgetting.
14. **Sleep prompts come from older inputs.** Read literally, the student (post-update) is
    distilled toward the teacher (pre-update). Practised on the inputs being learned right now,
    that trains the new expert to undo the newest learning; on CLINC the model stayed at chance.
    Sleep prompts are a reservoir sample of past wake inputs older than the level's period.
15. **Two consolidation modes.** `paper`: as written (protect pre-update behaviour, commit the
    fast update). `transfer`: commit the fast update, copy the whole fast block into the new
    slow expert, then reset the fast block (the "synaptic pruning" of the blog summary).
16. **Update period and sleep interval are separate.** The paper ties sleep to each level's
    update boundary, with periods counted in tokens. Counted in examples, a period long enough
    to make sleep affordable leaves the fast block with almost no updates (3 per CLINC task, no
    learning). `sleep_every` sets how many steps pass between sleeps of a level; the fast level
    can still update every step.
