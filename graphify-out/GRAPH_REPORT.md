# Graph Report - llm-sleep  (2026-11-08)

## Corpus Check
- 15 files · ~4,468 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 125 nodes · 255 edges · 12 communities (8 shown, 3 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d83e9674`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- CMSBlock
- dream.py
- consolidate.py
- cms.py
- Sleeper
- LevelSchedule
- setup
- DECISIONS.md
- llm-sleep
- clinc.py
- Tensor

## God Nodes (most connected - your core abstractions)
1. `Sleeper` - 14 edges
2. `consolidate()` - 14 edges
3. `LevelSchedule` - 13 edges
4. `TrialLoRA` - 12 edges
5. `inject_cms()` - 12 edges
6. `dream()` - 11 edges
7. `setup()` - 10 edges
8. `CMSBlock` - 10 edges
9. `cms_chains()` - 10 edges
10. `DreamConfig` - 9 edges

## Surprising Connections (you probably didn't know these)
- `setup()` --uses--> `DreamConfig`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/dream.py
- `setup()` --uses--> `DreamTask`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/dream.py
- `setup()` --uses--> `TrialLoRA`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/dream.py
- `setup()` --calls--> `cms_chains()`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/cms.py
- `setup()` --calls--> `inject_cms()`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/cms.py

## Import Cycles
- None detected.

## Communities (12 total, 3 thin omitted)

### Community 0 - "CMSBlock"
Cohesion: 0.16
Nodes (5): Module, CMSBlock, CMSChain, LowRankMLP, no_grad

### Community 1 - "dream.py"
Cohesion: 0.24
Nodes (14): dream(), DreamConfig, DreamTask, generate_dreams(), importance(), no_grad, One dreaming pass: sample, rank, trial, reward. Returns accepted (prompt_ids,…, ReST-EM M-step: learn accepted dreams through the wake path so they enter fast… (+6 more)

### Community 2 - "consolidate.py"
Cohesion: 0.14
Nodes (21): _as_state(), consolidate(), _generate(), _load(), Tensor, Knowledge seeding of level `sender` into a new expert at level sender+1 (paper…, SleepConfig, _snapshot() (+13 more)

### Community 3 - "cms.py"
Cohesion: 0.30
Nodes (11): cms_chains(), CMSWrappedMLP, _decoder_layers(), inject_cms(), set_dream(), test_dream_routing_changes_output(), test_injection_is_identity(), test_only_cms_trainable_and_masked_experts_get_no_grad() (+3 more)

### Community 4 - "Sleeper"
Cohesion: 0.22
Nodes (4): SleepConfig, Wake/sleep lifecycle around a Hugging Face causal LM., Sleeper, Tensor

### Community 5 - "LevelSchedule"
Cohesion: 0.25
Nodes (5): LevelSchedule, no_grad, Backward one microbatch; for each due level (slowest first) call on_due(l) then…, Parameters level l would have after its pending update, without applying it., Per-level accumulated updates (paper Eq. 2) with sleep triggers.

### Community 6 - "setup"
Cohesion: 0.31
Nodes (5): setup(), test_rest_em_updates_fast_memory(), test_select_topk_plus_random(), test_trial_lora_is_inert_outside_trials(), Tok

### Community 10 - "clinc.py"
Cohesion: 0.48
Nodes (5): encode(), evaluate(), per_task_acc(), prompt(), no_grad

## Knowledge Gaps
- **2 isolated node(s):** `Decisions where the paper is underspecified`, `llm-sleep`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 34 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LevelSchedule` connect `LevelSchedule` to `CMSBlock`, `consolidate.py`, `Sleeper`, `setup`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `Sleeper` connect `Sleeper` to `dream.py`, `consolidate.py`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Why does `CMSBlock` connect `CMSBlock` to `cms.py`, `LevelSchedule`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Sleeper` (e.g. with `DreamConfig` and `DreamTask`) actually correct?**
  _`Sleeper` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `LevelSchedule` (e.g. with `CMSBlock` and `setup()`) actually correct?**
  _`LevelSchedule` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `TrialLoRA` (e.g. with `Sleeper` and `setup()`) actually correct?**
  _`TrialLoRA` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Decisions where the paper is underspecified`, `llm-sleep` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._