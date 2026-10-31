# HRM-Graph

Graph-reasoning work in the DFM-HRM-Text fork (W&B projects `HRM-Graph` and `Graph-Positional-Encoding-Exploration`).

* [Graph Reasoning Experiments](graph-reasoning-experiments.md) - GraphQA/KGQA recursion study ported from HRM-Text-for-Graphs: tokenization parity, SM120 attention, configs, smoke runs, HRM-Text-1B fine-tune vs GRL.
* [Graph-Injection Methods and Their Papers](graph-injection-methods.md) - what each graph_inject method run is, the paper behind it, its adaptations and the control to read it against.
* [Unsaturated GraphQA Tasks](graphqa-unsaturated-tasks.md) - which GraphQA subtasks have headroom and why: train/test duplicates, missing max-flow capacities, counting; diagnostic runs.
* [Graph-Structure Injection on GraphQA](graph-structure-injection.md) - graph_inject: hard/soft/embedding-level graph injection methods for HRM-Text-1B and Mimir under one LoRA trainer; verification, run matrix, results.
* [KG / Graph Pretraining Corpus](kg-pretraining-corpus.md) - Tier 1+2 KG and graph-reasoning corpora for Mimir pretraining: sources, triple store with holdout, data_io rows, 6.8B Gemma tokens, verification.
* [GraphQA Chain of Thought](graphqa-cot.md) - graphqa_v2_cot: decomposed traces for every GraphQA task; text + LoRA reaches 0.91-0.97 (v2 text 0.82-0.87); remaining errors are the hidden steps (triangle intersections, augmenting-path choice).
