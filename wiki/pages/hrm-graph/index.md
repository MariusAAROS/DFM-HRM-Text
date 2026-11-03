# HRM-Graph

Graph-reasoning work in the DFM-HRM-Text fork (W&B projects `HRM-Graph` and `Graph-Positional-Encoding-Exploration`).

* [Graph Reasoning Experiments](graph-reasoning-experiments.md) - GraphQA/KGQA recursion study ported from HRM-Text-for-Graphs: tokenization parity, SM120 attention, configs, smoke runs, HRM-Text-1B fine-tune vs GRL.
* [Graph-Injection Methods and Their Papers](graph-injection-methods.md) - what each graph_inject method run is, the paper behind it, its adaptations and the control to read it against.
* [Unsaturated GraphQA Tasks](graphqa-unsaturated-tasks.md) - which GraphQA subtasks have headroom and why: train/test duplicates, missing max-flow capacities, counting; diagnostic runs.
* [Graph-Structure Injection on GraphQA](graph-structure-injection.md) - graph_inject: hard/soft/embedding-level graph injection methods for HRM-Text-1B and Mimir under one LoRA trainer; verification, run matrix, results.
* [KG / Graph Pretraining Corpus](kg-pretraining-corpus.md) - Tier 1+2 KG and graph-reasoning corpora for Mimir pretraining: sources, triple store with holdout, data_io rows, 6.8B Gemma tokens, verification.
* [GraphQA Chain of Thought](graphqa-cot.md) - graphqa_v2_cot: decomposed traces for every GraphQA task; text + LoRA reaches 0.91-0.97 (v2 text 0.82-0.87); remaining errors are the hidden steps (triangle intersections, augmenting-path choice).
* [GraphQA v3](graphqa-v3.md) - graphqa_v3_cot (built 2026-11-02): v2-CoT without exact train duplicates (12% of train) plus ~1,000 isomorphism-deduplicated large TriangleCounting / MaximumFlow rows; val/test unchanged; evidence and reason for each change.
* [Is Mimir v1.5 Best at Graph Reasoning?](mimir-graph-benchmark-plan.md) - approved plan (2026-11-02) for zero-shot / SFT / RLVR comparisons: claim challenges, hypotheses H1-H4, models (HRM family, IFM looped ablation, Ouro, Qwen3), dataset tiers with Tier 2 cleaning rules and Tier 3 verdict, splits, arms E0-E4, and the VeRL + vLLM RL stack (vLLM 0.27.1 runs hrm_text natively).
