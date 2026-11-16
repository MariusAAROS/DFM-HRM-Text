---
type: Runbook
title: Mimir Graph Benchmark - Why Trace SFT (E2) Scores Below Answer-Only SFT (E1)
description: "Row-level analysis of the E1 / E2 / E3 test outputs of the Mimir v1.5 graph benchmark: the E1 - E2 macro gap split by cause (KGQA questions whose gold path is missing from the prompt, hop counts absent from the SFT data, CLRS traces made of repeated lines, CLUTRR), an audit of the training traces, why GraphQA is the exception, and what to change before the next trace SFT or GRPO run."
tags: [graphs, mimir, benchmark, sft, cot, traces, grpo, kgqa, clrs, webqsp, cwq, metaqa, kqapro]
status: draft
last_updated: 2026-11-16
confidence: medium
---
# Mimir Graph Benchmark - Why Trace SFT (E2) Scores Below Answer-Only SFT (E1)

In the [results](/pages/hrm-graph/mimir-graph-benchmark-results.md), trace SFT (E2) is below answer-only SFT (E1) on
every dataset but GraphQA (macro 0.583 vs 0.642), and GRPO from E2 (E3) only gets back to E1 (0.639). This page asks
why, from the per-row outputs, and whether to fix the E2 data or run GRPO from E1.

- Code: [`graph_bench/eval/trace_gap_breakdown.py`](../../../graph_bench/eval/trace_gap_breakdown.py) (the tables
  below; arguments: `--names E1,E2,E3` and the arms' `.samples.jsonl` files, the answer-only arm first).
- Relaunch: data [`graph_bench/build_sft_v2.sh`](../../../graph_bench/build_sft_v2.sh), queue
  [`graph_bench/run_sft_v2.sh`](../../../graph_bench/run_sft_v2.sh), log `mimir_bench/sft_v2_queue.log`. W&B project
  `Mimir-Graph-RLVR`, eval groups `mimir-bench-{e1p,e2c,e2p,e3c}`, SFT training in `mimir-bench-sft`.
- W&B report "Dashboard Summary" (project `Mimir-Graph-RLVR`, URL in `mimir_bench/wandb_report_url.txt`): test,
  val, anonymised, never-scored rows, GRPO curves and baselines, one run set per section so test and val runs never
  share a chart. Built by [`graph_bench/eval/wandb_report.py`](../../../graph_bench/eval/wandb_report.py) (needs
  `wandb-workspaces`, see its docstring); new evaluation runs appear by themselves, rerun it to colour new models.
- Data: `mimir_bench/{e1,e2,e3}/mimir15/named[_t3k].samples.jsonl` (same 9,554 rows, same order), bench records in
  `data/graphs/bench/<dataset>/`, SFT file `data/graphs/bench/sft/train_cot.jsonl`.

## Analysis (2026-11-16)

Mimir v1.5 only, one seed, named test sets. A KGQA record has a trace only when its gold path is in the prompt
(`rec.trace`; "unsupported" in WebQSP / CWQ, missing facts in the retrieved MetaQA / KQA Pro files). Both SFT arms
trained only on records with a trace (`--traced-only`).

### The traces are correct; three of their properties cost accuracy

Audit of the train records behind `train_cot.jsonl`:

| Check | Result |
| --- | --- |
| Facts quoted in a KGQA trace are in the prompt | 100% (MetaQA 5,145, WebQSP 8,946, KQA Pro 14,338 quoted facts) |
| Last result of the trace equals the answer | MetaQA 2,588 / 2,588, KQA Pro 3,377 / 3,377, WebQSP 1,968 / 2,004 (the rest: lists cut at 25 items, unions of several paths) |
| CLRS: last state equals the answer | 100% for 5 algorithms, 0-76% for the other 7 (the answer is another output, or one more step) |

So the traces are not wrong. What costs accuracy:

1. **KGQA traces exist only where the gold path is in the prompt, and 37% of the KGQA test rows (1,582 of 4,330) have
   no such path.** E2 runs its program into a dead end and answers "none" (201 of the 639 such MetaQA / KQA Pro
   retrieved rows; E1 never does) or a wrong entity. E1 guesses and is right 0.46.
2. **Compositions absent from the SFT data.** MetaQA trains on 1-2 hops and tests 3 hops: E2 writes only 2 hops on 36
   of 200 rows. WebQSP is 76% 1-hop: on CWQ rows with a path, 38 of E2's 102 errors stop one hop early. On CLUTRR E2
   applies the MetaQA "Hop" template and answers an entity or a relation on the path.
3. **CLRS traces print only the output array, which stays unchanged for most steps.** Four algorithms have ≥ 90%
   repeated lines (SCC: median 59 states, 3.8 distinct). E2 learns to repeat the line and loses count: 65% of their
   OOD-size rows never reach an `Answer:` line.

WebQSP rows with a path (E2 0.834 vs E1 0.901): 50 of E2's 118 errors loop while copying long fact lists with Freebase
ids and never answer, 31 take the gold path and miscopy, 37 take another path. The path header is the first line of
the trace and was chosen from the answers (shortest path whose walk gives the gold set), so some are not what the
question says ("who played Stanley Kowalski" → `common.topic.image`).

### The macro gap by cause

Accuracy per group of rows, and each group's share of the macro difference to E1 (sums to the reported −0.059):

| Rows | n | E1 | E2 | E3 | E2 − E1 (macro) | E3 − E1 (macro) |
| --- | --- | --- | --- | --- | --- | --- |
| KGQA, gold path not in the prompt | 1,582 | **0.460** | 0.327 | 0.414 | −0.028 | −0.009 |
| KGQA, path in prompt, composition not in SFT (MetaQA 3-hop, CWQ) | 608 | **0.887** | 0.752 | 0.849 | −0.011 | −0.003 |
| CLUTRR (held out) | 1,000 | **0.192** | 0.118 | 0.154 | −0.011 | −0.005 |
| CLRS, 4 repeated-line algorithms, test / OOD size | 335 / 300 | **0.904 / 0.433** | 0.630 / 0.093 | 0.851 / 0.180 | −0.015 | −0.007 |
| WebQSP, path in prompt | 710 | **0.901** | 0.834 | 0.873 | −0.007 | −0.003 |
| CLRS, other 8 algorithms, OOD size | 600 | 0.102 | 0.043 | **0.142** | −0.003 | +0.002 |
| CLRS, other 8 algorithms, test | 665 | 0.635 | 0.647 | **0.741** | +0.001 | +0.005 |
| MetaQA 1-2 hop and KQA Pro, path in prompt | 1,430 | 0.958 | 0.966 | **0.973** | +0.001 | +0.003 |
| GraphQA (3 files) | 2,324 | 0.804 | 0.895 | **0.904** | +0.013 | +0.014 |

- **Where a valid, informative trace exists for the test question and its shape was trained, E2 is level with E1 or
  above** (last three rows), and E3 is above both.
- **Almost half of the gap (−0.028) is rows where the answer can't be derived from the prompt.** E1's score there is
  guessing, not graph reasoning. When the answer is not in the prompt at all it gets 0.02 (CWQ, 196 rows), 0.10
  (MetaQA, 88) and 0.34 (KQA Pro retrieved, 427: "English", "1", yes / no). For WebQSP / CWQ "unsupported" rows the
  prompt is built from the shortest-path triples that reach the gold answers, so the answer can be picked out without
  the question's constraint (E1 0.887 on the 381 CWQ rows with one answer in the prompt).
- The anonymised sets show the same split: E2 loses on MetaQA 3-hop (0.85 → 0.645) and CWQ, not on MetaQA 1-2 hop or
  KQA Pro.

CLRS by algorithm (share of repeated trace lines in train; test accuracy E1 / E2 / E3):

| Algorithm | Repeated lines | E1 | E2 | E3 |
| --- | --- | --- | --- | --- |
| articulation_points | 0.98 | 0.98 | 0.70 | 0.82 |
| strongly_connected_components | 0.95 | 0.88 | 0.15 | 0.72 |
| dag_shortest_paths | 0.92 | 0.78 | 0.67 | 0.86 |
| bridges | 0.91 | 1.00 | 0.99 | 1.00 |
| dfs | 0.69 | 0.48 | 0.70 | 0.88 |
| topological_sort | 0.60 | 0.65 | 0.73 | 0.87 |
| bfs, bellman_ford, dijkstra, mst_prim, mst_kruskal, floyd_warshall | 0.00-0.63 | 0.33-0.96 | within ±0.10 of E1 | |

dfs and topological_sort, whose state changes at most steps (8 distinct states), are where the trace helps.

### Why GraphQA is the exception

- Its traces were written for the task ([GraphQA CoT](/pages/hrm-graph/graphqa-cot.md)): every line is a step of the
  computation, not a copy of the output.
- The answer needs that computation: E1 → E2 EdgeCount 0.50 → 1.00, TriangleCounting 0.37 → 0.81, NodeDegree
  0.86 → 1.00. KGQA with the path in the prompt is a 1-2 hop lookup that E1 already does at 0.93-0.97 in one pass.
- Test and train come from the same generator with the same 11 tasks, and the graph is always complete.

### What GRPO repaired (E3)

Fewer loops (WebQSP rows without an `Answer:` line 7.0% → 1.5%, CLRS repeated-line OOD 65% → 25%), the missing hop
(CWQ rows with a path 0.706 → 0.833), SCC 0.15 → 0.72. On rows without a path it says "none" less often (201 → 124)
and guesses more: there the reward pays for an answer the trace does not support.

### Decisions this supports

- **Report KGQA with and without the gold path in the prompt.** The pooled number mixes reasoning with guessing and
  is what puts E1 ahead.
- **Fix the E2 data before another trace SFT or GRPO run**, by expected gain:
  1. CLRS: collapse repeated states (or print the algorithm's working state, or answer-only for the four
     algorithms). Up to +0.015 macro.
  2. KGQA rows without a path: train on them with a target that ends in a best answer (822 WebQSP train rows were
     dropped by `--traced-only`; MetaQA / KQA Pro have no retrieved train file). Up to +0.028, if the model then
     guesses as well as E1.
  3. 3-hop MetaQA and more 2-hop traces, unless 3-hop is kept as a held-out composition.
  4. WebQSP: shorter fact lists in the trace (the loops are on long CVT lists).
- **GRPO from E1 is not expected to fix what E1 lacks.** E1's deficit is the computation tasks (EdgeCount,
  TriangleCounting, MaximumFlow, dfs), and GRPO from the base (E3b) learned short answers, not traces. It can only
  sharpen what E1 already samples; E1's pass@16 has not been measured and would bound that gain.

Not checked: the three baselines' outputs (their E2 drop has the same shape per split), a second seed, whether a
mixed target (traces for GraphQA / CLRS, answer-only for KGQA) keeps both behaviours in one model.

## Existing reports re-scored by what the prompt allows (2026-11-16)

KGQA test rows (MetaQA, KQA Pro, WebQSP, CWQ; 4,330) in three groups: the gold path is in the facts; no path, but
every answer is in the facts; not every answer is in the facts (nothing to derive, a right answer is a guess). Full
output per model: `mimir_bench/rescore_20261116/<model>.txt`.

| Mimir v1.5 | n | E0 | E1 | E2 | E3 | E3b | E4 GRPO | E4 SFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Path in facts | 2,748 | 0.343 | **0.928** | 0.884 | 0.920 | 0.870 | 0.900 | 0.891 |
| No path, answers in facts | 718 | 0.164 | **0.790** | 0.578 | 0.734 | 0.696 | 0.677 | 0.717 |
| Not all answers in facts | 864 | 0.045 | 0.185 | 0.118 | 0.148 | 0.208 | 0.157 | **0.211** |

| Baselines, E1 / E2 | Qwen2.5-7B | OLMo-3-7B | Qwen3.5-4B |
| --- | --- | --- | --- |
| Path in facts | 0.880 / 0.863 | 0.869 / 0.831 | 0.908 / 0.862 |
| No path, answers in facts | 0.818 / 0.620 | 0.765 / 0.510 | 0.808 / 0.549 |
| Not all answers in facts | 0.231 / 0.130 | 0.175 / 0.120 | 0.171 / 0.116 |
| E2 − E1 macro, total | −0.048 | −0.039 | −0.047 |
| - of which rows without a path | −0.031 | −0.032 | −0.032 |
| - of which the 4 repeated-line CLRS algorithms | −0.010 | −0.004 | −0.009 |

- **The baselines' E2 drop has the same origin as Mimir's**: two thirds or more of it is the rows without a path. On
  MetaQA 1-2 hop and KQA Pro with the path present their E2 is level with E1 (+0.001 to +0.004 macro).
- **Mimir's lead after SFT holds on the reasoning rows**: E1 with the path in the facts 0.928 vs 0.908 / 0.880 / 0.869.
- The answer-only arms (E1, E3b, E4 SFT) lead where nothing can be derived (0.19-0.21 vs 0.12-0.16): guessing.

## Relaunch: design and launch (2026-11-16)

Aim: remove the two data causes (CLRS repeated lines, no training on questions without a path), measure each fix
alone, then redo the RL step from the repaired model. Decisions (user, 2026-11-16): both E2 variants, one GRPO run
from the better one, Mimir only.

| Arm | From | Data | Compared with | Question |
| --- | --- | --- | --- | --- |
| **E2c** | base | E2's 17,916 rows; CLRS traces with consecutive equal states merged (`export_hrm --collapse-repeats`) | E2, E1 | Was the CLRS loss the trace format? |
| **E1p** | base | E1's rows + the 523 WebQSP train questions without a trace whose answers are all in the prompt (`--answerable-untraced`) | E1 | Control for E2p: same questions |
| **E2p** | base | E2c's rows + the same 523 questions, answer line only | E2c, E1p | Can one model trace where a path exists and answer directly where none does? |
| **E3c** | E2c or E2p | GRPO, 200 steps, E3's parquet and settings | E3, E1 / E1p | Do fixed traces + RL beat answer-only SFT? |
| E1 val pass@16 | E1 | no training | | Bound on what GRPO from E1 could gain |

- **E3c's start model, rule fixed before any result:** E2p if its macro val pass@1 (5 val sets) is ≥ 0.01 above E2c's,
  else E2c (`mimir_bench/e3c_base.txt`).
- **E3c = E3's settings**: accuracy reward, group filter on accuracy, `REF_KL=False` set explicitly
  (`run_grpo_graph.sh` turns it on for any name other than `e3_grpo_e2`).
- **Risk E2p measures:** the E4 SFT control stopped writing traces everywhere. Check: share of generations with a
  trace on WebQSP rows that have a path (`trace_gap_breakdown`, E2: 1.000) and their accuracy against E2c.
- **GRPO from E1 is not scheduled.** If E1's val pass@16 − pass@1 is under ~0.03 macro it has nothing to sharpen.
- Unchanged on purpose: test files, MetaQA 3-hop held out, retrieved MetaQA / KQA Pro eval-only, the WebQSP trace
  format (its copy loops are left to GRPO), baselines not retrained, one seed.

**Data** (`build_sft_v2.sh`, all checks passed):
- The exporter arguments of E1 / E2 were not recorded. Recovered: inputs in the order GraphQA, MetaQA, KQA Pro,
  WebQSP, CLRS, with `--prompt instruct --traced-only --cap 3700 --seed 0 --extra sft/replay.jsonl`; this reproduces
  `train_answer.jsonl` byte for byte.
- `train_cot_c.jsonl`: E2's rows in E2's order; 2,674 of the 3,699 CLRS responses changed, CLRS response text
  2.52M → 0.90M characters, every `Answer:` line unchanged.
- `train_answer_p.jsonl` (18,440 rows) and `train_cot_p.jsonl` (18,439): + 523 WebQSP rows (of the 822 without a
  trace; 299 lack an answer in the prompt), none a val / test question, longest 2,254 tokens. Their row order differs
  from E1 / E2's (one shuffle over more rows).
- GRPO's parquet already held 817 WebQSP questions without a path, so E3 trained on them and the SFT arms did not.

**Queue** (launched 2026-11-16 14:15, detached; waits for E5's queue, whose GRPO was at step 142 of 200): smoke
(200 rows, merge, 50-row eval), E1 val pass@16, E1p, E2c, E2p with their evals, start-model choice, E3c and its evals
(named, anon, val pass@16, OSM), breakdown into `mimir_bench/trace_gap_v2.txt`. Estimate ~2.8 days of GPU, of which
E3c ~52 h. Results: not yet. Stopped the same day while still waiting (nothing had run), on the user's hold for the
split audit below, then relaunched at 16:18 with the holdback evals (see the end of the audit). Smoke passed at 16:25.

**W&B host:** the Python that runs `eval_vllm` and VeRL (`/home/ucloud/miniforge3`) authenticates only on
`api.forge.coreweave.com` (401 on `api.wandb.ai`, checked 2026-11-16), so the queue exports `WANDB_BASE_URL` like
`run_e5.sh`. The repo `.venv` resolves to the same host without it.

## Train / val / test audit (2026-11-16)

Checked on the bench records against every file an arm trains on (the five SFT files, `verl/train.parquet`).

| Dataset | Trained on (SFT / GRPO) | Val (model choice, in-training val) | Test | Never trained in E1-E3c |
| --- | --- | --- | --- | --- |
| GraphQA | train 3,648 | val 307 | test 324, test_fresh 1,100, test_large 1,100 | test_large (bigger graphs) |
| CLRS | 3,700 of train's 17,277 | val 1,200 | test 1,200 | test_ood (larger n) |
| MetaQA | train 2,588 (1-2 hop) | val 262 | test 592, test_retrieved 597 | 3-hop questions; retrieved subgraphs |
| KQA Pro | train 3,377 | val 372 | test 563, test_retrieved 578 | retrieved subgraphs |
| WebQSP | 2,004 (E1 / E2), 2,527 (E1p / E2p), 2,788 (GRPO) of train's 2,826 | val 245 | test 1,628 | |
| CWQ | none | (val 3,519 unused) | test 3,531 | whole dataset |
| CLUTRR | none | (val unused) | test 1,048 | whole dataset |
| OSM | none | (val unused) | test 2,000 | whole dataset |

Only E4 and E5 train on CLUTRR, CWQ or OSM; the relaunch arms start from base Mimir, not from them.

- **No prompt is shared** between the trained rows and any val or test file, or between val and test (0 everywhere).
  No test question appears in the 2,600 replay rows. CLRS test / val inputs are absent from train (0 shared).
- **GraphQA test is not graph-disjoint from train:** 290 of its 324 rows use a graph that a training row uses with
  another question, 79 with the same task and other arguments (val: 272 of 307). `test_fresh` (2 of 1,100) and
  `test_large` (0) are clean, and the scores agree (E2 0.960 test, 0.968 fresh; E1 0.858 both): no visible
  inflation. Quote `test_fresh`.
- **CWQ is held out as a dataset, not as questions.** Its questions are built from WebQSP questions, across WebQSP's
  split: 2,131 of the 3,531 test rows extend a WebQSP question that an arm trained on (1,581 for E1 / E2, 1,958 for
  E1p / E2p, 2,125 for GRPO), and 877 have the same answer set as that question. 1,203 extend a WebQSP test question
  and 197 a WebQSP val question: the unexposed subset. No sign of inflation on the scored rows (E1: base trained,
  same answers 0.391; base trained, other answers 0.846; base not trained 0.687), but E1p / E2p raise the exposed
  share, so compare CWQ across arms on the 1,400 unexposed rows too.
- **E5's SFT control is contaminated on WebQSP test** (not the relaunch arms): 928 of the 2,979 CWQ training rows in
  `verl_e5/train.parquet` extend a WebQSP test question (465 of the 1,628 test questions). E5's GRPO arm left CWQ out
  and is not affected.
- **MetaQA, KQA Pro and WebQSP share their knowledge base between train and test** (no shared question; 49-60% of
  test rows have answers that also occur as a train answer). The anonymised sets are the control for memorised entities.
- **Not checkable here:** whether Mimir's pretraining data contains these public sets.
- **The relaunch's fixes were chosen after reading test errors.** Scored rows are a fixed subset (≤ 1,000 per file,
  seed 0). Rows never scored by any arm: CWQ 2,531, WebQSP 628, CLRS test 200, GraphQA fresh / large 100 each;
  MetaQA, KQA Pro and OSM have none.

**Added to the relaunch after the audit (user, 2026-11-16):**
- **CWQ is also reported by its base question** (`trace_gap_breakdown`): rows whose base WebQSP question is in WebQSP
  train (620 of the 1,000 scored) and rows whose base is a WebQSP val / test question (380). On the existing arms the
  two agree (E1 0.668 / 0.687, E2 0.523 / 0.539, E3 0.650 / 0.655).
- **Holdback rows**: `eval_vllm --limit 1000 --complement` scores the rows that every earlier report leaves out
  (3,559: CWQ 2,531, WebQSP 628, CLRS test 200, GraphQA fresh / large 100 each). Seven models: E1, E2, E3 (existing)
  and E1p, E2c, E2p, E3c, reports `<arm>/mimir15/holdback[_t3k].json`, breakdown `mimir_bench/trace_gap_v2_holdback.txt`.
  A fix that only fitted the 1,000 scored rows would not show there. Do not use these rows to design changes.

