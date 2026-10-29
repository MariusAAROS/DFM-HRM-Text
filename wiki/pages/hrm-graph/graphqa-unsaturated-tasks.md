---
type: Runbook
title: Unsaturated GraphQA Tasks
description: Which GraphQA subtasks are not saturated under graph_inject (EdgeCount, TriangleCounting, MaximumFlow, NodeDegree), why (train/test duplicates, missing max-flow capacities, counting and multi-term arithmetic), and the diagnostic runs (printed capacities, numbered edges, triangle hint, triangle CoT).
tags: [graphs, graphqa, evaluation, data-quality, counting, chain-of-thought, hrm-text-1b, mimir]
status: stable
last_updated: 2026-10-29
confidence: high
---
# Unsaturated GraphQA Tasks

No structural encoding beats `text` + LoRA on GraphQA
([Graph-Structure Injection](/pages/hrm-graph/graph-structure-injection.md)). This page asks which subtasks still have
headroom, and what the input or the model lacks to saturate them.

- Data: `data/graphs/graphqa/hrm-text-grl/standard` (GRL prompt). The raw split is `data/graphs/graphqa/baseline/standard`.
- Runs analysed: the seed-0 `graph_inject` matrix plus `text` seeds 0–2, in
  `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject/<model>/<method>_s<seed>/eval_test.json`.
- Diagnostic runs: dataset `graphqa_v2`, W&B project `Graph-Positional-Encoding-Exploration`, group
  `graphqa-v2-inject`.

## Diagnosis (2026-10-29)

**Per-task test `greedy_acc`** (mean over the 3 models; `text` = 3 models × 3 seeds):

| Method | ConnN | Cycle | DiscN | EdgeCnt | EdgeEx | MaxFlow | NodeCnt | NodeDeg | Reach | ShortP | Triangle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n (test) | 34 | 36 | 41 | 26 | 34 | 34 | 27 | 40 | 39 | 34 | 40 |
| `text` | 0.97 | 1.00 | 0.97 | 0.65 | 1.00 | 0.17 | 1.00 | 0.91 | 1.00 | 0.98 | 0.56 |
| `text-incident` | 0.97 | 0.99 | 0.98 | 0.45 | 1.00 | 0.21 | 1.00 | 0.98 | 1.00 | 0.99 | 0.53 |
| `text-colors` | 0.97 | 1.00 | 0.98 | 0.64 | 1.00 | 0.16 | 1.00 | 0.90 | 1.00 | 0.99 | 0.53 |
| `node-tokens` | 0.97 | 1.00 | 0.97 | 0.65 | 1.00 | 0.19 | 1.00 | 0.94 | 1.00 | 0.98 | 0.52 |
| `abs-pe` | 0.97 | 1.00 | 0.97 | 0.59 | 1.00 | 0.17 | 1.00 | 0.94 | 1.00 | 0.98 | 0.51 |
| `rel-bias` | 0.97 | 1.00 | 0.97 | 0.64 | 1.00 | 0.15 | 1.00 | 0.91 | 1.00 | 0.99 | 0.53 |
| `graphtoken` | 0.97 | 1.00 | 0.97 | 0.64 | 1.00 | 0.16 | 1.00 | 0.91 | 1.00 | 0.98 | 0.53 |
| `graphtoken-frozen` | 0.93 | 1.00 | 0.72 | 0.60 | 0.95 | 0.12 | 1.00 | 0.75 | 1.00 | 0.92 | 0.47 |

`prompt-tuning`, `abs-id`, `rel-bias-elem` and `fullft-text` are indistinguishable from `text`.

- **Saturated** (≥ 0.97 for every LoRA arm): CycleCheck, EdgeExistence, NodeCount, Reachability, ShortestPath,
  ConnectedNodes, DisconnectedNodes. These are lookups and short traversals.
- **Unsaturated:** EdgeCount, TriangleCounting, MaximumFlow, NodeDegree. All four count or compute over the whole
  graph.

**Train/test duplicates: the GraphQA split is not clean.**
- 61 of 385 test rows, and 78 of 385 val rows, have an instruction identical to a train row, always with the same
  answer.
- The raw `baseline/standard` split has the same counts, so the duplicates come from GraphQA itself, not from our
  preparation. They affect every GraphQA number in this repo.
- MetaQA and KQA Pro have 0 duplicates.
- GraphQA reuses graphs across splits, so the tasks whose question names no node repeat exactly:

| Task | Test n | Identical in train | `text` on duplicates | `text` on clean |
| --- | ---: | ---: | ---: | ---: |
| TriangleCounting | 40 | 13 | 0.99 | **0.35** |
| EdgeCount | 26 | 8 | 1.00 | **0.49** |
| NodeDegree | 40 | 5 | 0.98 | 0.90 |
| CycleCheck | 36 | 17 | 1.00 | 1.00 |
| NodeCount | 27 | 10 | 1.00 | 1.00 |
| ConnectedNodes | 34 | 7 | 1.00 | 0.97 |
| DisconnectedNodes | 41 | 1 | 1.00 | 0.97 |

**Clean-subset accuracy per method** (seed 0; `text` over 3 seeds):

| Method | EdgeCount (18) | Triangle (27) | NodeDegree (35) | MaxFlow (34) |
| --- | ---: | ---: | ---: | ---: |
| `text` | 0.49 | 0.35 | 0.90 | 0.17 |
| `text-incident` | 0.22 | 0.30 | 0.98 | 0.21 |
| `text-colors` | 0.48 | 0.31 | 0.89 | 0.16 |
| `node-tokens` | 0.50 | 0.30 | 0.93 | 0.19 |
| `abs-pe` | 0.41 | 0.27 | 0.93 | 0.17 |
| `rel-bias` | 0.48 | 0.32 | 0.90 | 0.15 |
| `graphtoken` | 0.48 | 0.31 | 0.90 | 0.16 |

No structural method moves a clean unsaturated task. The only real per-task effect is `text-incident` on NodeDegree.

**MaximumFlow: the capacities were dropped upstream.** All 350 MaximumFlow rows (train, val and test) were checked
by recomputing the flow with every capacity set to 1 (`networkx.maximum_flow_value`):

| Check | Result |
| --- | --- |
| Gold equals the unit-capacity flow | 27 / 339 |
| Gold is an integer multiple of it | 214 / 339 |
| Gold / unit-capacity flow | 1.0 to 10.0, 84 distinct values |
| Gold below the unit-capacity flow | 0 |
| No path between source and sink | 11, all with gold 0 |

- The data is consistent with random whole-number capacities from 1 to 10 per edge. For example, a 7-node complete
  graph has gold 36 where unit capacities give 6, and on path graphs the gold is the smallest capacity on the path.
- The capacities are printed in neither our prompts nor the raw split, so the task cannot be answered from the text.
- The model guesses small numbers (most frequent outputs 3, 4, 1, 2). Accuracy is 0.39 when the gold is below 3 and
  0.03 when it is 20 or more.

**Labels of the counting tasks are exact.** EdgeCount (26/26), TriangleCounting (40/40), NodeDegree (40/40) and
NodeCount (27/27) gold equal values recomputed from the prompt. EdgeCount counts self-loop tuples, and NodeDegree
counts a self-loop twice.

**Error analysis** (`text`, 9 runs, test):
- **EdgeCount.**
  - Accuracy is 1.00 below 20 edges, 0.60 at 20–40, and about 0.2 at 40 or more.
  - Wrong answers are estimates: only 10% are off by one, and 61% are too low. For example, gold 117 gets 88–105.
  - Complete graphs are right through the n(n−1)/2 shortcut.
  - `text-incident` hurts (0.22 clean), because each edge is written twice and the model would have to halve the
    total.
- **TriangleCounting.**
  - Graphs with fewer than 20 edges score 0.84, and complete graphs score 0.97 through the C(n, 3) shortcut.
  - Other graphs with 20 or more edges score **0.08**.
  - Per-node triangle information is partly in RWSE (3-step return probabilities), which `abs-pe` and `graphtoken`
    use, but they don't gain.
- **NodeDegree.**
  - 76% of the errors are off by one, and 74% are too low: one of the node's mentions scattered through the edge list
    is missed.
  - `text-incident`, which lists each node's neighbours together, reaches 0.98.
- **Reading.** No unsaturated task lacks *structure*:
  - MaxFlow lacks data (the capacities);
  - EdgeCount and TriangleCounting lack computation: an explicit count, or intermediate steps, which one direct-answer
    pass with no chain of thought can't do over 50–130 edges;
  - NodeDegree lacks locality, which a serialization fixes.

**Re-scored on the 324 clean test questions** (`python -m graph_inject.summarize --test_ids
data/graphs/graphqa/hrm-text-grl-v2/standard/dedup_report.json`, from the saved per-sample results, no retraining).
Overall `greedy_acc` (HRM-1B / Mimir v1 / Mimir v1.5; `text` over 3 seeds):

| Method | Full test (385) | Clean test (324) |
| --- | --- | --- |
| `text` | 0.826 / 0.842 / 0.851 | 0.794 / 0.812 / 0.824 |
| `text-incident` | 0.834 / 0.842 / 0.831 | 0.802 / 0.815 / 0.799 |
| `node-tokens` | 0.821 / 0.849 / 0.855 | 0.787 / 0.821 / 0.827 |
| `abs-pe` | 0.813 / 0.836 / 0.852 | 0.778 / 0.806 / 0.824 |
| `rel-bias` | 0.821 / 0.842 / 0.844 | 0.787 / 0.812 / 0.818 |

- Removing the duplicates lowers every arm by about 3 pt and leaves the ranking unchanged: still no method beyond
  `text`.

## Diagnostic runs (2026-10-29)

**Code.**
- `scripts/graph_inject/build_graphqa_v2.py` builds the dataset.
- `graph_inject/configs/datasets/graphqa_v2.yaml` is its config.
- `configs/methods/text-{edge-index,tri-hint,tri-cot}.yaml` are the three arms:
  - `TextMethod` encodings `edge_index` and `tri_hint`, applied to the `hint_tasks` only;
  - `response_format: tri_cot`, through the new `GraphMethod.rewrite_response` hook.
- `generate.score` compares only the text after `Answer:` when the gold has one; `tf_em` stays on the full target.
- Tests: `test_graphqa_v2_maxflow_and_dedup`, `test_unsaturated_task_diagnostics`.
- Smoke run: all three arms, HRM-1B, 200 rows, 1 epoch.

**Queue:** started 2026-10-29 10:03, log `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject/queue_v2.log`, output
root `graph_inject_graphqa_v2`. 12 runs, about 7–8 h (estimated).

Dataset `graphqa_v2` (`scripts/graph_inject/build_graphqa_v2.py` → `data/graphs/graphqa/hrm-text-grl-v2/standard`):
- **MaximumFlow:** capacities drawn uniformly from 1–10 per edge (fixed seed) and printed as `(u, v, c)`; gold
  recomputed.
- **Dedup:** val/test rows identical to a train row are dropped (test 385 → 324, val 385 → 307). Train is unchanged.

Arms (LoRA, all 3 models, seed 0). Each hint touches only its target task, so the other tasks double as a noise check:

| Arm | Change | Question |
| --- | --- | --- |
| `text` on v2 | baseline | Does MaxFlow rise above the 0.17 guess level once capacities are printed? |
| `text-edge-index` | EdgeCount edges written `e1: (0, 1) e2: …` | Was counting the bottleneck? |
| `text-tri-hint` | TriangleCounting prompt lists the per-node triangle counts | Can it add them up? |
| `text-tri-cot` | TriangleCounting target: per-node counts, sum, ÷3, then `Answer:` | Can it compute the per-node counts itself? |

- A CoT that lists every triangle was rejected: train graphs have up to 969 triangles (about 9k tokens), past the
  4,096-token context. Per-node counts fit in about 150 tokens.
- NodeDegree needs no run, since `text-incident` already answers it.
- Old runs are re-scored on the 324 clean test questions from their saved per-sample outputs.

## Results: printed capacities and numbered edges (2026-10-29)

Test `greedy_acc` on the 324-question v2 test set, seed 0, best epoch on the 307-question v2 val.
- "Old `text`" is the original GraphQA run re-scored on the same 324 questions, averaged over seeds 0–2.
- The old and v2 training sets differ only in the 281 MaximumFlow rows, which now print capacities.

| Model | Arm | Overall | EdgeCount (18) | MaxFlow (34) | Triangle (27) | NodeDegree (35) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HRM-1B | old `text` | 0.794 | 0.39 | 0.17 | 0.32 | 0.83 |
| | v2 `text` | 0.824 | 0.39 | 0.59 | 0.26 | 0.83 |
| | `text-edge-index` | 0.852 | **1.00** | 0.47 | 0.22 | 0.83 |
| Mimir v1 | old `text` | 0.812 | 0.48 | 0.21 | 0.33 | 0.90 |
| | v2 `text` | 0.858 | 0.50 | 0.59 | 0.33 | 0.91 |
| | `text-edge-index` | 0.892 | **1.00** | 0.56 | 0.44 | 0.89 |
| Mimir v1.5 | old `text` | 0.824 | 0.59 | 0.13 | 0.38 | 0.95 |
| | v2 `text` | 0.867 | 0.56 | 0.68 | 0.30 | 0.89 |
| | `text-edge-index` | 0.904 | **1.00** | 0.71 | 0.37 | 0.94 |

**MaximumFlow is answerable once the capacities are printed.**
- It rises from about 0.17 (guessing) to 0.59–0.68, which lifts overall `greedy_acc` by 3.0 / 4.6 / 4.3 pt. The
  other tasks move only by seed-noise amounts.
- The remaining errors are on large flows, where the model estimates: gold 81 is predicted 35–106, gold 68 is
  predicted 44–70. This is the same failure as EdgeCount, now on a flow computation.

**Numbering the edges saturates EdgeCount** (1.00 on all 3 models, from 0.39 / 0.50 / 0.56).
- The count was the bottleneck, not the structure: with the last index printed, the model copies it.
- The tasks that `edge_index` doesn't touch stay within noise. NodeDegree is unchanged. TriangleCounting (27
  questions) and MaxFlow (34) move by a few questions, which is one seed's spread.

The triangle arms are in the next section.

## Results: triangle hint and triangle chain of thought (2026-10-29)

Same conditions as above (v2 test, 324 questions, seed 0). TriangleCounting has 27 clean test questions. The other
columns are tasks these arms don't touch, so they show the noise.

| Model | Arm | Overall | Triangle (27) | EdgeCount (18) | MaxFlow (34) | NodeDegree (35) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HRM-1B | v2 `text` | 0.824 | 0.26 | 0.39 | 0.59 | 0.83 |
| | `text-tri-hint` | 0.843 | **0.52** | 0.39 | 0.50 | 0.80 |
| | `text-tri-cot` | 0.821 | 0.30 | 0.39 | 0.50 | 0.86 |
| Mimir v1 | v2 `text` | 0.858 | 0.33 | 0.50 | 0.59 | 0.91 |
| | `text-tri-hint` | 0.873 | **0.48** | 0.44 | 0.56 | 0.97 |
| | `text-tri-cot` | 0.870 | 0.30 | 0.61 | 0.71 | 0.94 |
| Mimir v1.5 | v2 `text` | 0.867 | 0.30 | 0.56 | 0.68 | 0.89 |
| | `text-tri-hint` | 0.898 | **0.59** | 0.56 | 0.71 | 0.94 |
| | `text-tri-cot` | 0.870 | 0.37 | 0.56 | 0.65 | 0.89 |

Untouched tasks move by up to 4 questions (MaxFlow 0.50–0.71), which is the single-seed noise level.

**TriangleCounting accuracy by true triangle count** (3 models pooled):

| Arm | ≤ 5 triangles (12 q) | 6–20 (4 q) | > 20 (11 q) |
| --- | ---: | ---: | ---: |
| v2 `text` | 0.67 | 0.00 | 0.00 |
| `text-tri-hint` | 0.97 | 0.50 | 0.06 |
| `text-tri-cot` | 0.69 | 0.08 | 0.00 |

**Checking the `text-tri-cot` traces step by step** (HRM-1B / Mimir v1 / Mimir v1.5):
- 1 / 4 / 3 traces are incomplete: Sum, division or `Answer:` is missing.
- **Per-node counts:** 123/318, 115/282 and 123/284 node counts are right, about 40%. All counts are right in 6 / 8 / 10
  of 27 traces, and the answer is then right in 22 of 24.
- **Sum:** when some counts are wrong (21 / 19 / 17 traces), the written `Sum` equals the sum of the model's own counts
  in only 2 / 4 / 6.
- **Division:** `Sum / 3` is consistent in 26/26, 23/23 and 18/24.

**Reading.**
- **Given the per-node counts, the model adds them up only for small graphs.** `tri-hint` gains 15–29 pt on every
  model, but nearly all of the gain is at ≤ 5 triangles (0.67 → 0.97). Above 20 triangles it stays at 0.06. Summing
  about 10–19 numbers and dividing by 3 in one direct-answer pass fails too.
- **`tri-cot` equals `text`.** Neither step it writes out is reliable: a per-node count is a hidden computation (find
  the neighbours, test every pair), right about 40% of the time. The written `Sum` is not the sum of the counts above it.
  The CoT helps only when every step is already right.
- Compare EdgeCount: numbering the edges works because the answer becomes a copy of the last index.
- **For an all-task CoT:** each step must be one that a single pass already does reliably (copy or lookup), not a
  count or a multi-term addition. Candidates, not yet tested:
  - running totals (`2 + 2 = 4, + 0 = 4, ...`) instead of one `Sum`;
  - per node, its neighbour list and the connected pairs instead of a bare count.

  The full triangle list was ruled out earlier for length.
