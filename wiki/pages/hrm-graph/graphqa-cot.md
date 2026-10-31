---
type: Runbook
title: GraphQA Chain of Thought
description: "graphqa_v2_cot, a GraphQA version whose every target is a decomposed trace (copies, lookups, single additions) ending in 'Answer: <gold>'; trace design, budget fallbacks, eval changes, and the 3-model text + LoRA results against v2."
tags: [graphs, graphqa, chain-of-thought, counting, max-flow, hrm-text-1b, mimir]
status: stable
last_updated: 2026-10-30
confidence: high
---
# GraphQA Chain of Thought

The v2 diagnostics ([Unsaturated GraphQA Tasks](/pages/hrm-graph/graphqa-unsaturated-tasks.md)) showed that the
unsaturated GraphQA tasks fail on computation, not on structure, and that a chain of thought helps only if each step
is one a single pass already does reliably. `graphqa_v2_cot` gives every task such a trace and trains `text` + LoRA on
the 3 pretrained models.

- Code: [`graph_inject/cot.py`](../../../graph_inject/cot.py) (one trace function per task, `cot_response`),
  builder [`scripts/graph_inject/build_graphqa_cot.py`](../../../scripts/graph_inject/build_graphqa_cot.py),
  test `test_graphqa_cot_traces` in `tests/test_graph_inject.py`.
- Data: `data/graphs/graphqa/hrm-text-grl-v2-cot/standard/{train,val,test}.jsonl`: same rows, order and prompts as
  v2; `response` is the trace; added fields `cot_detail` and `cot_gold_replaced`. v2 is untouched.
- Config: `graph_inject/configs/datasets/graphqa_v2_cot.yaml`; runs in
  `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject_graphqa_v2_cot/<model>/text_s0`.
- W&B: project `Graph-Positional-Encoding-Exploration`, group `graphqa-v2-cot-inject`.

## Design (2026-10-29)

**Trace formats.** One step per line; the last line is `Answer: <gold>`. Every step is a copy, a lookup, `+1` or one
`a + b`/`a - b`. Traces follow the prompt's edge order where the gold does, and increasing node order elsewhere, so
they are deterministic.

| Task | Trace |
|---|---|
| NodeCount | `Count the nodes: 0 → 1, 1 → 2, …` |
| EdgeCount | `Count the edges: (0, 1) → 1, (0, 2) → 2, …` (prompt order, self-loops included) |
| NodeDegree | incident edges with a running count; a self-loop adds 2 |
| EdgeExistence | `Edges with node u: …` → `Neighbors of node u: …` → `Node v is (not) a neighbor of node u.` |
| ConnectedNodes | incident edges → `Neighbors in order of appearance: …` |
| DisconnectedNodes | incident edges → neighbours → `Nodes in alphabetical order: 0, 1, 10, …` → `Check: 0 keep, 1 neighbor, …` → `Kept: …` |
| Reachability, ShortestPath | layered BFS: `Node u (distance d): neighbors … → new: ….`, stops when t is reached or nothing is new |
| CycleCheck | union-find over the edges: `(a, b): node a is in ca, node b is in cb: merge, x, y now in lo.`, stops at the first edge inside one component or a self-loop |
| TriangleCounting | `Neighbors with a larger number:` per node, then for each edge u < v the common larger neighbours w > v with a running total: `(0, 1): 5, 6, 7 → 3, total 3` |
| MaximumFlow | Edmonds-Karp: `Path k: s → … → t, residuals …, bottleneck b, flow F + b = F'. Update: (x→y) r - b = r', (y→x) r + b = r''.`, then `No augmenting path … is left.` |

**Gold semantics the traces reproduce** (checked on every v2 row): EdgeCount counts self-loop tuples; NodeDegree counts
a self-loop twice; DisconnectedNodes is in *string* order (`10, 11, 2`); ConnectedNodes lists neighbours in order of
first appearance, although the question says "alphabetical". 3 ConnectedNodes rows (train 329, 637; test 59) follow
no rule: their gold is replaced by the appearance order (`cot_gold_replaced`). `cot_response` raises on any other
mismatch with the gold.

**Token budget.** prompt + response must be ≤ 4,090 tokens with all three tokenizers (RoPE has 4,096 positions). The
builder picks the most detailed tier that fits (`cot_detail`):
- TriangleCounting: `full` → `edge_counts` (no list of w; 8 train rows) → `node_counts` (5 train rows, complete graphs).
- MaximumFlow: `full` → `forward_updates` (7 train, 1 val, 1 test) → `no_updates` (1 test).
- Longest rows: TriangleCounting 4,080, MaximumFlow 4,018 tokens.

**Steps still hidden:** finding an augmenting path (MaximumFlow), and the intersections in the coarser triangle tiers.

**Eval changes** (`graph_inject/generate.py`, `graph_inject/train.py`, defaults unchanged for other datasets):
- `eval.max_seq_len: 4096` stops a sample at prompt + generated = 4,096, so no trace runs past the RoPE table.
- `greedy_acc` and the set metrics score `final_answer` (the text after `Answer:`). `tf_em` would need the whole trace
  verbatim, so the best epoch is chosen on `eval.select_metric: greedy_acc`.
- Decoding costs ~0.14 s per step at batch 8 (the max: KV cache ~3 GB per slot at 4,096 tokens), so a full val pass
  would take ~30 min per epoch. Epoch selection uses a fixed task-stratified 55-question val subset
  (`eval.val_subset: 55`, 3–7 per task, proportional to the task's share of val); test is complete (324).
- Smoke run lesson: without the 4,096 cap a few MaximumFlow traces loop until `max_generation`, and val outputs are
  only written at the end of the pass, so the first smoke run looked hung for 16 min. Smoke-test CoT with a small
  `eval.limit` and a low `max_generation` first.

## Results: text + LoRA on the 3 models (2026-10-30)

Seed 0, test greedy_acc on the 324 deduplicated v2 questions, best epoch by val greedy_acc on the 55-question subset
(HRM-1B epoch 7, Mimir v1 epoch 6, Mimir v1.5 epoch 7). v2 rows: same prompts, direct answer, from
[Unsaturated GraphQA Tasks](/pages/hrm-graph/graphqa-unsaturated-tasks.md). Each run took ~2.3–2.6 h
(training ~3 min per epoch, the rest val and test generation).

| Model | Data | greedy_acc | greedy_em | EdgeCount | NodeDegree | DisconnectedNodes | MaximumFlow | TriangleCounting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HRM-1B | v2 `text` | 0.824 | 0.824 | 0.39 | 0.83 | 0.88 | 0.59 | 0.26 |
| | v2 best (`text-edge-index`) | 0.852 | 0.852 | 1.00 | 0.83 | 0.97 | 0.47 | 0.22 |
| | **CoT** | **0.910** | 0.870 | 1.00 | 1.00 | 0.97 | 0.56 | **0.56** |
| Mimir v1 | v2 `text` | 0.858 | 0.778 | 0.50 | 0.91 | 0.97 | 0.59 | 0.33 |
| | v2 best (`text-edge-index`) | 0.892 | 0.784 | 1.00 | 0.89 | 1.00 | 0.56 | 0.44 |
| | **CoT** | **0.944** | 0.910 | 1.00 | 1.00 | 1.00 | **0.71** | **0.70** |
| Mimir v1.5 | v2 `text` | 0.867 | 0.784 | 0.56 | 0.89 | 1.00 | 0.68 | 0.30 |
| | v2 best (`text-edge-index`) | 0.904 | 0.796 | 1.00 | 0.94 | 1.00 | 0.71 | 0.37 |
| | **CoT** | **0.966** | 0.932 | 1.00 | 1.00 | 1.00 | **0.85** | **0.78** |

Per-task columns are greedy_acc. `greedy_em` (raw `pred.startswith(gold)`) means different things on the two datasets:
- On CoT it requires the whole trace verbatim, so it is below greedy_acc whenever a step differs but the answer is
  right.
- On v2 Mimir it is lowered by a tokenizer artifact: the NodeCount / EdgeCount golds start with a space (`' 4.'`) that
  the Gemma decode drops. On Mimir v1.5 `text` that is 27 questions, the whole 8.3-pt gap.

Compare models on greedy_acc.

- **CoT adds 8.6 / 8.6 / 9.9 pt over v2 `text`** and 4–6 pt over the best v2 diagnostic. The `text` seed sd on
  GraphQA is ~1.1 pt ([seed variance](/pages/hrm-graph/graph-structure-injection.md#seed-variance-of-text-2026-10-29)).
- **8 tasks are saturated** on every model: EdgeCount and NodeDegree reach 1.00. The only misses outside MaximumFlow and
  TriangleCounting are 2 on HRM-1B (a DisconnectedNodes copy, `17` for `7`; a ShortestPath BFS that skipped the
  "reached" line).
- **Test row 59** (ConnectedNodes, replaced gold) is right against the CoT gold and wrong against the v2 gold on all 3
  models, so against v2's gold every CoT score is 0.3 pt lower (0.907 / 0.941 / 0.963). The v2 `text` runs also miss
  it (they answer `0, 1.`, gold `1, 0.`).
- Caveats: the CoT train set has 38× more target tokens (0.75M vs 0.02M; 1.8× more tokens in total, prompts
  0.91M). CoT selects the epoch on a 55-question val subset with greedy_acc, v2 on full val with tf_em. One seed.

**Where the remaining errors are** (test; first line where the generation departs from the gold trace):

| | HRM-1B | Mimir v1 | Mimir v1.5 |
|---|---:|---:|---:|
| TriangleCounting ≤ 5 / 6–20 / > 20 triangles | 12/12, 2/4, 1/11 | 12/12, 4/4, 3/11 | 12/12, 3/4, 6/11 |
| TriangleCounting first error: intersection / neighbour-list copy | 10 / 2 | 8 / 0 | 5 / 1 |
| MaximumFlow ≤ 3 / 4–8 / > 8 augmenting paths | 18/20, 0/5, 1/9 | 20/20, 2/5, 2/9 | 19/20, 5/5, 5/9 |
| MaximumFlow first error: path choice / update arithmetic / stopped early or late | 11 / 2 / 2 | 6 / 3 / 1 | 2 / 2 / 1 |
| of the wrong paths: uses an edge with residual 0 / valid but longer / valid alternative | 8 / 3 / 0 | 2 / 2 / 2 | 0 / 0 / 2 |
| Runaway traces (no `Answer:` before 4,096 tokens) | 3 MaxFlow, 1 Triangle | 1 MaxFlow | 1 MaxFlow |

- **Every error is in a step that is still hidden.** TriangleCounting fails at the set intersection (common larger
  neighbours), not at copying, counting or summing. MaximumFlow fails mostly at choosing the augmenting path (the BFS
  is not written out). On HRM-1B that path usually uses a saturated edge, i.e. the model does not read the updated
  residuals back.
- **The answer is faithful to the trace:** in every MaximumFlow failure with an `Answer:` line, the answer equals the
  model's own last flow value. Errors come from wrong steps, not from ignoring the trace.
- **Size is the limit:** small instances (≤ 5 triangles, ≤ 3 paths) are ~100% on every model, and accuracy drops with
  the number of steps. The larger model (Mimir v1.5) holds up longest.
- The 2 test rows at coarser MaximumFlow tiers (`forward_updates`, `no_updates`) are wrong on all 3 models (n = 2).

**Next:** write out the hidden steps where the budget allows: the intersection as a per-w check list, and the
augmenting-path BFS over residuals > 0. Also replicate with seeds 1–2 (reseeded data), since one seed and a
55-question val subset select the epoch.
