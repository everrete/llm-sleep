# Graph Report - llm-sleep  (2026-11-08)

## Corpus Check
- 14 files · ~3,772 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 116 nodes · 251 edges · 10 communities (7 shown, 2 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d135f4bc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cms.py
- sleeper.py
- consolidate
- inject_cms
- Sleeper
- LevelSchedule
- test_consolidate.py
- DECISIONS.md
- llm-sleep

## God Nodes (most connected - your core abstractions)
1. `Sleeper` - 16 edges
2. `consolidate()` - 15 edges
3. `LevelSchedule` - 15 edges
4. `inject_cms()` - 13 edges
5. `TrialLoRA` - 12 edges
6. `dream()` - 11 edges
7. `CMSBlock` - 10 edges
8. `cms_chains()` - 10 edges
9. `tiny_model()` - 10 edges
10. `setup()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_consolidation_protects_against_overwrite()` --uses--> `SleepConfig`  [INFERRED]
  tests/test_consolidate.py → src/llm_sleep/consolidate.py
- `test_imitation_rl_path_runs()` --uses--> `SleepConfig`  [INFERRED]
  tests/test_consolidate.py → src/llm_sleep/consolidate.py
- `setup()` --uses--> `DreamConfig`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/dream.py
- `setup()` --uses--> `DreamTask`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/dream.py
- `setup()` --uses--> `TrialLoRA`  [INFERRED]
  tests/test_dream.py → src/llm_sleep/dream.py

## Import Cycles
- None detected.

## Communities (10 total, 2 thin omitted)

### Community 0 - "cms.py"
Cohesion: 0.13
Nodes (7): Module, CMSBlock, CMSChain, CMSWrappedMLP, _decoder_layers(), LowRankMLP, no_grad

### Community 1 - "sleeper.py"
Cohesion: 0.23
Nodes (15): set_dream(), dream(), DreamConfig, DreamTask, generate_dreams(), importance(), no_grad, One dreaming pass: sample, rank, trial, reward. Returns accepted (prompt_ids,… (+7 more)

### Community 2 - "consolidate"
Cohesion: 0.21
Nodes (13): _as_state(), consolidate(), _generate(), _load(), Tensor, Knowledge seeding of level `sender` into a new expert at level sender+1 (paper…, _snapshot(), gjsd() (+5 more)

### Community 3 - "inject_cms"
Cohesion: 0.24
Nodes (13): cms_chains(), inject_cms(), test_dream_routing_changes_output(), test_injection_is_identity(), test_only_cms_trainable_and_masked_experts_get_no_grad(), test_slot_exhaustion_and_reset(), test_unlocking_zero_init_expert_keeps_output(), tiny_model() (+5 more)

### Community 4 - "Sleeper"
Cohesion: 0.21
Nodes (4): SleepConfig, Tensor, Wake/sleep lifecycle around a Hugging Face causal LM., Sleeper

### Community 5 - "LevelSchedule"
Cohesion: 0.25
Nodes (5): LevelSchedule, no_grad, Backward one microbatch; for each due level (slowest first) call on_due(l) then…, Parameters level l would have after its pending update, without applying it., Per-level accumulated updates (paper Eq. 2) with sleep triggers.

### Community 6 - "test_consolidate.py"
Cohesion: 0.42
Nodes (7): nll(), setup(), test_consolidation_protects_against_overwrite(), test_imitation_rl_path_runs(), test_prospective_does_not_apply(), test_schedule_periods(), Tok

## Knowledge Gaps
- **2 isolated node(s):** `llm-sleep`, `Decisions where the paper is underspecified`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 31 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LevelSchedule` connect `LevelSchedule` to `cms.py`, `sleeper.py`, `inject_cms`, `Sleeper`, `test_consolidate.py`?**
  _High betweenness centrality (0.197) - this node is a cross-community bridge._
- **Why does `Sleeper` connect `Sleeper` to `sleeper.py`, `LevelSchedule`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `CMSBlock` connect `cms.py` to `LevelSchedule`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Sleeper` (e.g. with `SleepConfig` and `DreamConfig`) actually correct?**
  _`Sleeper` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `LevelSchedule` (e.g. with `CMSBlock` and `Sleeper`) actually correct?**
  _`LevelSchedule` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `TrialLoRA` (e.g. with `Sleeper` and `setup()`) actually correct?**
  _`TrialLoRA` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `llm-sleep`, `Decisions where the paper is underspecified` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._