---
type: Runbook
title: Mimir Graph Benchmark - Traces or Direct Answers? Diagnosis, Relaunch and Comparison Plan
description: "Does training Mimir v1.5 to write step-by-step traces beat training it to answer directly, once both get the same RL? The page follows the reasoning in order: why trace SFT (E2) scored below answer-only SFT (E1), whether the train / val / test splits can be trusted, what was fixed and why, the relaunched models (E1p, E2c, E2p, E3c, E3e1), the comparison steps and the decision rules fixed before the results, and the results so far."
tags: [graphs, mimir, benchmark, sft, cot, traces, grpo, kgqa, clrs, webqsp, cwq, metaqa, kqapro, leakage]
status: draft
last_updated: 2026-11-24
confidence: medium
---
# Mimir Graph Benchmark - Traces or Direct Answers? Diagnosis, Relaunch and Comparison Plan

**The question.** In the [results](/pages/hrm-graph/mimir-graph-benchmark-results.md), Mimir trained to write a
step-by-step trace before its answer (E2, macro 0.583) scored below Mimir trained to write the answer only (E1,
0.642), and RL on top of the traces (E3, 0.639) only got back to E1. Is that because traces do not help, or because of
how the traces and the scores were built? And with the same RL, which recipe wins?

**How this page is organised:** the diagnosis (sections 1-3), the decisions it led to (4), the models trained to test
them (5), the comparison steps with their decision rules, fixed before the results (6), the results (7), and what
this work does not fix (8). Sections carry the date of the work.

**Status (2026-11-17):** the SFT models are done (steps 1-3 answered). E3c's GRPO is running; E3e1 is queued after it
(steps 4-6 pending, results expected ~2026-11-19 and ~2026-11-21).
**Status (2026-11-24):** all steps answered (section 7, "GRPO models"). Answer-only SFT + GRPO (E3e1) is the better recipe;
traces pay only on GraphQA. Mimir vs the baselines, re-run with an LR sweep and 3 seeds:
[Mimir vs the Baselines](/pages/hrm-graph/mimir-graph-vs-baselines.md).

- Analysis code: [`graph_bench/eval/trace_gap_breakdown.py`](../../../graph_bench/eval/trace_gap_breakdown.py)
  (`--names E1,E2,...` then the models' `.samples.jsonl` files, the answer-only model first).
- Data: [`graph_bench/build_sft_v2.sh`](../../../graph_bench/build_sft_v2.sh). Queues:
  [`graph_bench/run_sft_v2.sh`](../../../graph_bench/run_sft_v2.sh) (log `mimir_bench/sft_v2_queue.log`),
  [`graph_bench/rl/run_e3_from_e1.sh`](../../../graph_bench/rl/run_e3_from_e1.sh) (log `mimir_bench/e3e1_queue.log`).
- Outputs: `mimir_bench/<model>/mimir15/*.json` and `.samples.jsonl`; breakdowns `mimir_bench/trace_gap_v2.txt` (usual
  test rows) and `trace_gap_v2_holdback.txt` (never-scored rows); re-scoring of the older models
  `mimir_bench/rescore_20261116/<model>.txt`.
- W&B: project `Mimir-Graph-RLVR`, eval groups `mimir-bench-<model>`. Report **"Dashboard Summary"** (URL in
  `mimir_bench/wandb_report_url.txt`), built by [`graph_bench/eval/wandb_report.py`](../../../graph_bench/eval/wandb_report.py):
  one run set per section, so test and val runs never share a chart; new runs appear by themselves, rerun it to give
  new models their colour.

## Vocabulary

| Term | Meaning |
| --- | --- |
| E1 | Base Mimir + SFT on the answer line only (`Answer: X.`) |
| E2 | Base Mimir + SFT on a trace, then the answer line. Same questions as E1 |
| E3 | E2 + 200 GRPO steps (reward 1 when the answer is right) |
| E1p | E1 + 523 WebQSP training questions that have no gold path but all their answers in the facts |
| E2c | E2 with clean CLRS traces (consecutive identical lines merged) |
| E2p | E2c + the same 523 questions, answer line only for them |
| E3c | GRPO from E2c (the rule in section 6 chose E2c over E2p), E3's data and settings |
| E3e1 | GRPO from E1, E3's data and settings |
| Path in facts | KGQA test question whose facts contain a path from the question entity to the answer (a trace can be written) |
| No path, answers in facts | No such path, but every answer entity appears in the facts |
| Not all answers in facts | Nothing to derive: a right answer is a guess |
| Never-scored rows (holdback) | The 3,559 test questions outside the fixed 1,000-per-file subset that every evaluation used |
| `_t3k` / `_t2k` | Answer length limit of the evaluation: 3,072 / 2,048 tokens (none: 1,024) |

All numbers: one seed, greedy test answers, accuracy from 0 to 1. Macro = mean over the 7 test datasets.

## 1. Are the E2 traces wrong? (2026-11-16)

The first hypothesis was that GraphQA traces are correct and the other datasets' traces are not. Audit of the training
records behind `train_cot.jsonl`:

| Check | Result |
| --- | --- |
| Facts quoted in a KGQA trace are in the prompt | 100% (MetaQA 5,145, WebQSP 8,946, KQA Pro 14,338 quoted facts) |
| Last result of the trace equals the answer | MetaQA 2,588 / 2,588, KQA Pro 3,377 / 3,377, WebQSP 1,968 / 2,004 (the rest: lists cut at 25 items, unions of several paths) |
| CLRS: last state equals the answer | 100% for 5 algorithms, 0-76% for the other 7 (the answer is another output, or one more step) |

**The traces are correct.** The hypothesis is rejected; the losses come from three properties of the traces and of
the test sets:

1. **Questions without a path.** KGQA traces exist only where the gold path is in the facts, so E1 and E2 trained only
   on such questions, but 37% of the KGQA test questions (1,582 of 4,330) have no such path. E2 runs its method into
   a dead end and answers "none" (201 of the 639 such MetaQA / KQA Pro retrieved rows; E1 never does) or a wrong
   entity. E1 guesses and is right 0.46.
2. **Compositions absent from the SFT data.** MetaQA trains on 1-2 hops and tests 3: E2 writes only 2 hops on 36 of
   200 rows. WebQSP is 76% 1-hop: on CWQ rows with a path, 38 of E2's 102 errors stop one hop early. On CLUTRR E2
   applies the MetaQA "Hop" template and answers an entity on the path.
3. **CLRS traces made of repeated lines.** They print only the output array, which stays unchanged for most steps:
   four algorithms have ≥ 90% repeated lines (strongly-connected-components: median 59 states, 3.8 distinct). E2
   learns to repeat the line and loses count: 65% of their larger-graph rows never reach an `Answer:` line.

Also on WebQSP rows with a path (E2 0.834 vs E1 0.901): 50 of E2's 118 errors loop while copying long fact lists with
Freebase ids, 31 take the gold path and miscopy, 37 take another path. The path that heads a trace was chosen from the
answers (shortest path whose walk gives the gold set), so some do not follow the question ("who played Stanley
Kowalski" → `common.topic.image`).

## 2. Where the E1 - E2 gap comes from (2026-11-16)

Accuracy per group of test rows, and each group's share of the macro difference to E1 (sums to the reported −0.059):

| Rows | n | E1 | E2 | E3 | E2 − E1 (macro) | E3 − E1 (macro) |
| --- | --- | --- | --- | --- | --- | --- |
| KGQA, gold path not in the facts | 1,582 | **0.460** | 0.327 | 0.414 | −0.028 | −0.009 |
| KGQA, path in facts, composition not in SFT (MetaQA 3-hop, CWQ) | 608 | **0.887** | 0.752 | 0.849 | −0.011 | −0.003 |
| CLUTRR (never trained) | 1,000 | **0.192** | 0.118 | 0.154 | −0.011 | −0.005 |
| CLRS, 4 repeated-line algorithms, test / larger graphs | 335 / 300 | **0.904 / 0.433** | 0.630 / 0.093 | 0.851 / 0.180 | −0.015 | −0.007 |
| WebQSP, path in facts | 710 | **0.901** | 0.834 | 0.873 | −0.007 | −0.003 |
| CLRS, other 8 algorithms, larger graphs | 600 | 0.102 | 0.043 | **0.142** | −0.003 | +0.002 |
| CLRS, other 8 algorithms, test | 665 | 0.635 | 0.647 | **0.741** | +0.001 | +0.005 |
| MetaQA 1-2 hop and KQA Pro, path in facts | 1,430 | 0.958 | 0.966 | **0.973** | +0.001 | +0.003 |
| GraphQA (3 files) | 2,324 | 0.804 | 0.895 | **0.904** | +0.013 | +0.014 |

- **Where a trained, informative trace exists, E2 is level with E1 or above** (last three rows), and E3 above both.
- **Almost half of the gap is questions where the answer cannot be derived from the facts.** E1's score there is
  guessing: where the answer is not in the facts at all it gets 0.02 (CWQ), 0.10 (MetaQA) and 0.34 (KQA Pro
  retrieved: "English", "1", yes / no). For WebQSP / CWQ questions without a path, the facts were built from the
  shortest-path triples that reach the gold answers, so the answer can be picked out without the question's
  condition (E1 0.887 on the 381 such CWQ rows with one answer in the facts).

CLRS by algorithm (share of repeated trace lines in training; test accuracy):

| Algorithm | Repeated lines | E1 | E2 | E3 |
| --- | --- | --- | --- | --- |
| articulation_points | 0.98 | 0.98 | 0.70 | 0.82 |
| strongly_connected_components | 0.95 | 0.88 | 0.15 | 0.72 |
| dag_shortest_paths | 0.92 | 0.78 | 0.67 | 0.86 |
| bridges | 0.91 | 1.00 | 0.99 | 1.00 |
| dfs | 0.69 | 0.48 | 0.70 | 0.88 |
| topological_sort | 0.60 | 0.65 | 0.73 | 0.87 |
| bfs, bellman_ford, dijkstra, mst_prim, mst_kruskal, floyd_warshall | 0.00-0.63 | 0.33-0.96 | within ±0.10 of E1 | |

dfs and topological_sort, whose state changes at most steps, are where the trace helps.

**Why GraphQA is the exception.** Its traces were written for the task ([GraphQA CoT](/pages/hrm-graph/graphqa-cot.md)):
every line is a step of the computation, and the answer needs it (E1 → E2: EdgeCount 0.50 → 1.00, TriangleCounting
0.37 → 0.81, NodeDegree 0.86 → 1.00). KGQA with the path in the facts is a 1-2 hop lookup that E1 already does at
0.93-0.97 in one pass. GraphQA test and train come from the same generator, and the graph is always complete.

**What GRPO repaired in E3.** Fewer loops (WebQSP rows without an `Answer:` line 7.0% → 1.5%, CLRS repeated-line
larger graphs 65% → 25%), the missing hop (CWQ rows with a path 0.706 → 0.833), SCC 0.15 → 0.72. Where there is no
path it says "none" less often (201 → 124) and guesses more: the reward pays for an answer the trace does not support.

**Same pattern in the baselines.** Existing reports re-scored by what the facts allow (KGQA, 4,330 rows):

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
| - of which questions without a path | −0.031 | −0.032 | −0.032 |
| - of which the 4 repeated-line CLRS algorithms | −0.010 | −0.004 | −0.009 |

The baselines' E2 drop has the same origin (two thirds or more from questions without a path), and Mimir's lead after
SFT holds on the reasoning rows (E1 with the path in the facts 0.928 vs 0.908 / 0.880 / 0.869).

## 3. Can the splits be trusted? Train / val / test audit (2026-11-16)

Checked against every file a model trains on (the five SFT files, `verl/train.parquet`).

| Dataset | Trained on (SFT / GRPO) | Val | Test | Never trained in E1-E3e1 |
| --- | --- | --- | --- | --- |
| GraphQA | train 3,648 | 307 | test 324, test_fresh 1,100, test_large 1,100 | test_large (bigger graphs) |
| CLRS | 3,700 of 17,277 | 1,200 | test 1,200 | test_ood (larger graphs) |
| MetaQA | train 2,588 (1-2 hop) | 262 | test 592, test_retrieved 597 | 3-hop questions; retrieved subgraphs |
| KQA Pro | train 3,377 | 372 | test 563, test_retrieved 578 | retrieved subgraphs |
| WebQSP | 2,004 (E1 / E2), 2,527 (E1p / E2p), 2,788 (GRPO) of 2,826 | 245 | test 1,628 | |
| CWQ | none | (unused) | test 3,531 | whole dataset |
| CLUTRR | none | (unused) | test 1,048 | whole dataset |
| OSM | none | (unused) | test 2,000 | whole dataset |

Val is used only for GRPO's in-training validation and for the choice of E3c's start model; epochs are fixed (the
last), so nothing is selected on test. Only E4 and E5 train on CLUTRR, CWQ or OSM.

- **No question is shared** between training and any val or test file, or between val and test (0 everywhere). No
  test question appears in the 2,600 replay rows; CLRS test / val inputs are absent from training.
- **GraphQA test reuses training graphs:** 290 of its 324 questions use a graph that a training question uses (79 with
  the same task). `test_fresh` (2 of 1,100) and `test_large` (0) are clean, and scores agree (E2 0.960 test, 0.968
  fresh). Quote `test_fresh`.
- **CWQ is held out as a dataset, not as questions.** Each CWQ question extends a WebQSP question, across WebQSP's
  split: 2,131 of the 3,531 test questions extend one a model trained on (1,581 for E1 / E2, 1,958 for E1p / E2p, 2,125
  for GRPO); 877 have the same answers. 1,400 extend a WebQSP val / test question: the unexposed subset. No sign of
  inflation (E1: base trained, same answers 0.391; base not trained 0.687).
- **E5's SFT control is contaminated on WebQSP test** (not the models of this page): 928 of its CWQ training rows
  extend a WebQSP test question (465 of the 1,628).
- **MetaQA, KQA Pro and WebQSP share their knowledge base between train and test** (no shared question; 49-60% of
  test answers also occur as training answers). The anonymised sets are the control for memorised entities.
- **Not checkable:** whether Mimir's pretraining data contains these public sets.
- **The fixes below were designed after reading test errors** on the fixed 1,000-per-file subset. Never-scored rows:
  CWQ 2,531, WebQSP 628, CLRS test 200, GraphQA fresh / large 100 each; MetaQA, KQA Pro and OSM have none.

Two measures added because of this audit (user, 2026-11-16): CWQ is also reported on its unexposed subset, and every
model is evaluated on the never-scored rows (`eval_vllm --limit 1000 --complement`), used only to confirm results
(section 6, step 6), never to design changes.

## 4. Decisions and their reasons (2026-11-16)

| Decision | Reason |
| --- | --- |
| Remove repeated lines from CLRS traces (E2c) | Cause 3 of section 1; up to +0.015 macro; one change, so E2c vs E2 measures it alone |
| Train on WebQSP questions without a path, answer line only (E1p, E2p) | Cause 1: E2 never saw a question its method cannot solve. 523 of the 822 dropped WebQSP questions have every answer in the facts; the other 299 have nothing to learn from. MetaQA / KQA Pro have no retrieved training file |
| E1p as control of E2p | E2p has 523 more questions than E1; only E1p separates "more questions" from "traces" |
| Report KGQA by question group and CWQ on its unexposed subset | The pooled score mixes reasoning with guessing (section 2) and CWQ exposure differs between models (section 3) |
| Keep unchanged: test files, MetaQA 3-hop held out, retrieved MetaQA / KQA Pro eval-only, WebQSP trace format, one seed | Comparability with E1-E3; 3-hop and CWQ stay tests of unseen compositions; WebQSP copy loops left to GRPO (E3 cut them 7.0% → 1.5%) |
| Baselines not retrained (user) | Their E2 drop has the same origin (section 2); retrain them only if Mimir's picture changes |
| GRPO from E1 (E3e1): first not scheduled, then queued | First reasoning: E1's deficit is the computation tasks, and GRPO from the base (E3b) learned short answers, not traces. **Superseded (2026-11-16, same day)** by E1's val pass@16: gap 0.101 (section 7), above the 0.03 rule, so queued (user) |
| Stop E5 at step 150 (user, 2026-11-16) | Free the GPU; its step-150 checkpoint is kept and resumable ([E5](/pages/hrm-graph/mimir-graph-e5-multidataset-rl.md#grpo-stopped-at-step-152-2026-11-16)) |

## 5. The models (2026-11-16)

| Model | Starts from | Trained on | Answers which question |
| --- | --- | --- | --- |
| **E2c** | base | E2's 17,916 rows, CLRS repeated states merged (`export_hrm --collapse-repeats`) | Was the CLRS loss caused by the trace format? |
| **E1p** | base | E1's rows + 523 WebQSP questions without a path (`--answerable-untraced`) | Control for E2p |
| **E2p** | base | E2c's rows + the same 523 questions, answer line only | Can one model trace where a path exists and answer directly where none does? |
| **E3c** | E2c (rule below) | GRPO, 200 steps, E3's data (`verl/train.parquet`) and settings | Does RL on repaired traces beat RL on the old traces (E3)? |
| **E3e1** | E1 | GRPO, same data, settings and steps as E3c | Does RL on direct answers do as well as RL on traces? |

- **Settings shared by all GRPO runs:** accuracy reward, groups with a constant reward dropped, no reference-KL pass
  (`REF_KL=False`, set explicitly: `run_grpo_graph.sh` turns it on for any run name but `e3_grpo_e2`), 16 questions ×
  8 answers per step, answers up to 2,048 tokens.
- **Evaluations of every new model:** test (`named`, 1,024 tokens for E1p, `named_t3k` otherwise), anonymised test,
  val pass@16 (trace models and GRPO models), never-scored rows; OSM for the GRPO models. E1, E2 and E3 are also
  evaluated on the never-scored rows.

**Data** (`build_sft_v2.sh`, all checks passed):
- E1 / E2's exporter arguments were not recorded. Recovered: inputs GraphQA, MetaQA, KQA Pro, WebQSP, CLRS in that
  order, `--prompt instruct --traced-only --cap 3700 --seed 0 --extra sft/replay.jsonl`; reproduces
  `train_answer.jsonl` byte for byte.
- `train_cot_c.jsonl`: E2's rows in E2's order; 2,674 of 3,699 CLRS responses changed, every `Answer:` line unchanged.
- `train_answer_p.jsonl` (18,440 rows) / `train_cot_p.jsonl` (18,439): + 523 WebQSP rows, none a val / test question,
  longest 2,254 tokens; row order differs from E1 / E2's (one shuffle over more rows).
- GRPO's training file already held 817 WebQSP questions without a path: E3 trained on them, the SFT models did not.

Unique training tokens (one epoch, Mimir tokenizer; question tokens / answer tokens, the latter carry the loss):

| File (model) | Rows | Question | Answer | Total |
| --- | --- | --- | --- | --- |
| `train_answer` (E1) | 17,917 | 13.61M | 1.28M | 14.9M |
| `train_cot` (E2) | 17,916 | 13.61M | 6.23M | 19.8M |
| `train_answer_p` (E1p) | 18,440 | 14.37M | 1.28M | 15.7M |
| `train_cot_c` (E2c) | 17,916 | 13.61M | 4.89M | 18.5M |
| `train_cot_p` (E2p) | 18,439 | 14.37M | 4.89M | 19.3M |

The CLRS fix cuts CLRS answer tokens 2.10M → 0.76M. The 2,600 replay rows are 1.3M of each total (1.01M answer).

**Runs:** queue 1 (`run_sft_v2.sh`) started 2026-11-16 16:18 after a passed smoke test (200 rows, merge, 50-question
eval). The GPU-time estimate was ~2.8 days. Queue 2 (`run_e3_from_e1.sh`, started 17:46) waits for queue 1 to exit.
The Python that runs vLLM and VeRL (`/home/ucloud/miniforge3`, wandb 0.30.0) needs
`WANDB_BASE_URL=https://api.forge.coreweave.com` (401 on `api.wandb.ai`); both queues export it.

## 6. Comparison steps and decision rules

Rules fixed before the results they apply to. Noise: one seed; differences under ~0.02 macro are not conclusive. E1
and E1p differ by 0.008 macro but by 0.088 on CLRS test with identical CLRS rows, so single-file differences can be
noise too: a conclusion needs the macro and the targeted rows to agree.

**Step 1. Does the CLRS fix work?** E2c vs E2 (and E1 as the level to reach).
- Read: the four repeated-line algorithms on test and larger graphs; the other eight algorithms; GraphQA.
- Rule: the four algorithms back to about E1's level, everything else within ±0.02.
- **Answered (2026-11-17): yes** (section 7).

**Step 2. Does training on questions without a path help, and at what cost?** E2p vs E2c; E2p vs E1p on equal data.
- Read: KGQA "no path, answers in facts"; CWQ on its unexposed subset; share of WebQSP questions with a path that still
  get a trace (rule ≥ 0.95, E2: 1.000) and their accuracy (rule within 0.02 of E2c); CLUTRR; macro.
- **Answered (2026-11-17): it helps KGQA but costs CLUTRR entirely** (section 7).

**Step 3. Where do traces stand after SFT alone?** Best trace model (E2c) vs best answer-only model (E1, E1p).
- **Answered (2026-11-17): answer-only is still ahead, by 0.026 macro** (section 7).

**Rule between steps 2 and 4: E3c's start model.** E2p if its macro val pass@1 (5 val sets) is ≥ 0.01 above E2c's,
else E2c. Applied 2026-11-17: E2p 0.8036, E2c 0.7998 → **E2c**.

**Step 4. Does RL on repaired traces beat RL on the old ones?** E3c vs E3.
- Read: macro test; the rows the fixes target, where an effect is large enough to see with one seed: CLRS
  repeated-line algorithms on larger graphs (E3 0.180, E1 0.433) and KGQA "no path, answers in facts" (E3 0.734,
  E1 0.790); val pass@1 and pass@16.
- Rule: "better" only if the targeted rows improve clearly and the macro does not drop; a macro change under 0.02 alone
  proves nothing.

**Step 5. Headline: RL on traces vs RL on direct answers.** E3c vs E3e1, same data and RL budget, both against E1 /
E1p.
- Read: macro test and anonymised macro; KGQA "path in facts" (reasoning over the graph); GraphQA fresh and larger
  graphs (computation); CLRS on larger graphs; CWQ unexposed subset (unseen compositions); CLUTRR (unseen task); val
  pass@1 and pass@16 (headroom left).
- Rules:
  - E3e1 ≥ E3c (within 0.02 macro, and not behind on "path in facts"): answer-only SFT + RL is the better recipe on this
    benchmark; traces are justified only for GraphQA-like computation tasks.
  - E3c above E3e1 by more than 0.02 macro and on "path in facts": traces pay off once RL is applied.
  - Otherwise: report by dataset family (computation / KGQA lookup / unseen compositions), no overall winner.
- Also read: does either beat E1 / E1p (does RL add anything over plain answer-only SFT)?

**Step 6. Confirmation on never-scored questions.** Repeat steps 1, 4 and 5 on the holdback rows
(`trace_gap_v2_holdback.txt`, W&B report section 6). A conclusion stands only if its direction is the same there.

Where to read each step: `trace_gap_v2.txt` (question groups, CLRS per algorithm, CWQ by base question, trace share),
the W&B report (per-file accuracy, val, GRPO curves), and section 7 of this page.

## 7. Results

### E1 val pass@16 (2026-11-16)

Val, 200 questions per dataset, 16 samples at temperature 1, mean of 5 datasets:

| Model | pass@1 | pass@16 | Gap (≈ what RL can still add) |
| --- | --- | --- | --- |
| E1 | 0.803 | 0.904 | **0.101** |
| E2 | 0.781 | 0.905 | 0.124 |
| E3 | 0.871 | 0.928 | 0.057 |

E1 can sample right answers as often as E2 (pass@16 0.904 vs 0.905); its room is mostly CLRS (0.64 → 0.85) and WebQSP
(0.69 → 0.79). Above the 0.03 rule: E3e1 queued.

### SFT models: steps 1-3 (2026-11-17)

| Model | Macro test | Macro anon | Val pass@1 / @16 | CLRS test / larger | WebQSP | CWQ | CLUTRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.642 | 0.814 | 0.803 / 0.904 | 0.725 / 0.212 | 0.781 | 0.675 | 0.192 |
| E1p | **0.650** | **0.817** | | **0.813 / 0.261** | **0.798** | **0.677** | 0.156 |
| E2 | 0.583 | 0.750 | 0.781 / 0.905 | 0.641 / 0.060 | 0.712 | 0.529 | 0.118 |
| E2c | 0.616 | 0.764 | 0.800 / 0.904 | 0.755 / 0.201 | 0.733 | 0.570 | 0.148 |
| E2p | 0.610 | 0.802 | 0.804 / 0.905 | 0.754 / 0.212 | 0.759 | 0.662 | **0.000** |
| E3 (GRPO from E2) | 0.639 | 0.806 | 0.871 / 0.928 | 0.778 / 0.154 | 0.761 | 0.652 | 0.154 |

**Step 1 - the CLRS fix works (E2c − E2 = +0.033 macro).** The four repeated-line algorithms: test 0.630 → 0.940,
larger graphs 0.093 → 0.420, level with E1 (0.904 / 0.433); SCC 0.15 → 0.91. The other eight algorithms and GraphQA
do not move.

**Step 2 - training on questions without a path helps KGQA but costs CLUTRR.**
- E2p vs E2c: "no path, answers in facts" 0.621 → 0.728 (E1 0.790), CWQ 0.570 → 0.662, anonymised macro 0.764 → 0.802.
- E2p still writes a trace on 95.1% of WebQSP questions with a path (rule ≥ 0.95, met) and is right on 0.869 of them
  (E2c 0.855, rule met).
- But CLUTRR 0.148 → 0.000: E2p answers 65% of CLUTRR questions directly with a name or a sentence ("Answer: Katherine
  is Leslie's daughter.") instead of a relation word, as the 7B baselines' E2 did. The answer-only rows taught it to
  answer directly whenever the trace format does not fit. Its macro ends below E2c's (0.610 vs 0.616).

**Step 3 - after SFT alone, answer-only is still ahead.** E1p 0.650 and E1 0.642 vs E2c 0.616: the fixes shrank the
gap from 0.059 to 0.026, larger than the 0.008 observed between E1 and E1p. Remaining gap, E2c vs E1 (macro points):

| Rows | E1 | E2c | Contribution |
| --- | --- | --- | --- |
| KGQA, gold path not in the facts | 0.460 | 0.345 | −0.024 |
| Compositions not in SFT (MetaQA 3-hop, CWQ) | 0.887 | 0.798 | −0.007 |
| CLUTRR | 0.192 | 0.148 | −0.006 |
| WebQSP, path in facts | 0.901 | 0.855 | −0.005 |
| CLRS (all) | | | ≈ 0 |
| GraphQA | 0.804 | 0.905 | **+0.014** |

- **Traces win only on GraphQA** (fresh graphs 0.971 vs 0.858, larger graphs 0.816 vs 0.732). On CLRS the fix brought
  traces back to E1's level, not above it, and E1p is ahead there (0.813 vs 0.755 on test).
- **E1 stays slightly ahead even where a path exists** (KGQA "path in facts": E1 0.928, E2p 0.910, E2c 0.896), from
  unseen compositions and WebQSP, where a trace that stops a hop early or loops costs more than a direct answer.
- **What this implies for steps 4-5:** the traces' value can only show after RL (GRPO took E2 0.583 → 0.639; E3c
  starts 0.033 higher). E1 has as much sampling headroom as E2, so E3e1 may well match or beat E3c.

### Pending (superseded 2026-11-24: both runs finished, see the next section)

- **E3c** (GRPO from E2c): started 2026-11-17 03:16, W&B run `42xn080v`, ~14 min/step; training expected to end
  ~2026-11-19 02:30, its evaluations and the never-scored evaluations of E1 / E2 / E3 ~2026-11-19 09:00.
- **E3e1** (GRPO from E1): starts after that, expected ~2026-11-21.

### GRPO models: steps 4-6 (2026-11-21..23)

Same GRPO for all three (VeRL, 200 steps, 16 × 8 rollouts, accuracy reward), one seed each. Analysis and figures:
`notebooks/mimir_rlvr_v2.ipynb`; CIs are a paired bootstrap over the test rows, stratified by dataset.

| Model | Macro test | Macro anon | KGQA path in facts | GraphQA | CLRS | Never-scored rows | GRPO cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 (no RL) | 0.642 | 0.814 | 0.928 | 0.804 | 0.482 | 0.772 | |
| E3 (from E2) | 0.639 | 0.806 | 0.920 | 0.904 | 0.483 | 0.797 | |
| E3c (from E2c) | 0.650 | 0.810 | 0.924 | 0.904 | 0.532 | 0.809 | 55 GPU h, answers ~380-580 tokens |
| **E3e1 (from E1)** | **0.662** | **0.822** | **0.935** | 0.825 | **0.586** | **0.824** | 27 GPU h, answers ~22-38 tokens |

- **Step 4 - E3c is better than E3, on CLRS only.** +0.011 macro (95% CI +0.006 to +0.017). CLRS repeated-line
  algorithms on larger graphs 0.180 → 0.453 (target met); KGQA "no path, answers in facts" unchanged (0.734 → 0.731).
- **Step 5 - answer-only SFT + RL is the better recipe on this benchmark.** E3e1 − E3c = +0.012 macro (CI +0.006 to
  +0.018), and E3e1 is ahead on "path in facts" (0.935 vs 0.924): the first rule applies. Traces win only on GraphQA
  (0.904 vs 0.825). RL adds over answer-only SFT: E3e1 − E1 = +0.020, E3e1 − E1p = +0.012.
- **Step 6 - confirmed on never-scored rows:** E3e1 0.824 > E3c 0.809 > E3 0.797 > E1 0.772, the same order. On these
  fewer rows the E3e1 − E3c and E3c − E3 intervals include 0.
- **Headroom left (val):** E3e1's pass@16 − pass@1 is 0.046 (E1 0.101), and its pass@16 is below E1's (0.896 vs 0.904):
  GRPO turned half of E1's sampling headroom into greedy accuracy and narrowed its sampling. E3c keeps 0.035.
- E3e1's looping lists are not new (E1 1.0% of rows, E3e1 0.9%).
- **Caveats:** one seed per GRPO model. E3e1 is the best of 8 Mimir variants, picked after seeing test scores, so it is
  not the basis of the comparison with the baselines.

## 8. What this work does not fix

- **The test sets themselves.** 37% of KGQA test questions still have no path to the answer and 20% have the answer
  missing from the facts, so the pooled score keeps rewarding guessing (read the question groups); GraphQA `test`
  reuses training graphs (read `test_fresh`); CWQ is built on WebQSP questions (read the unexposed subset);
  CycleCheck marks a correct "Yes" as wrong for models not trained on GraphQA's wording
  ([results](/pages/hrm-graph/mimir-graph-benchmark-results.md#e3b-grpo-from-base-mimir-and-the-benchmarks-findings-2026-11-13)).
- **The WebQSP trace format** behind the copy loops.
- **Baselines**: not retrained, so "Mimir is the best model after the same SFT" rests on E1 and the old E2 data.
  Superseded (2026-11-24): with a tuned LR and 3 seeds, Qwen3.5-4B E1 ties Mimir E1 (0.643 vs 0.644); see
  [Mimir vs the Baselines](/pages/hrm-graph/mimir-graph-vs-baselines.md).
- **One seed** for every model.

## Change log

- 2026-11-16: analysis, re-scoring, audit, relaunch design, E1 pass@16 and the E3e1 queue, written as dated sections.
- 2026-11-17: SFT results added, then the page reorganised on request into diagnosis → decisions → models →
  comparison steps → results, with the decision rules gathered in section 6. No number changed. Superseded claim kept
  and marked in section 4 (GRPO from E1).
- 2026-11-24: steps 4-6 results (E3c, E3e1, never-scored rows) added in section 7; status line and section 8 updated.
