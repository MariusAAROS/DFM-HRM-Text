---
type: Runbook
title: "Mimir Graph Benchmark E5 - RL on New Datasets from E3"
description: "E5 of the Mimir v1.5 graph benchmark: starting from the E3 model (GRPO after trace SFT), train on datasets Mimir never trained on (OSM, CLUTRR, CWQ) with GRPO and with an SFT control, and measure their test sets, the old test sets and the sets still held out. Design, data, settings, launch, and results as they come."
tags: [graphs, mimir, benchmark, grpo, sft, osm, clutrr, cwq, kgqa]
status: draft
last_updated: 2026-11-16
confidence: medium
---
# Mimir Graph Benchmark E5 - RL on New Datasets from E3

**Question.** If we take the best RL model (E3) and keep training it on graph datasets it has never seen in
training, does it get better on *their* test sets, without getting worse on the old ones? This is not a zero-shot
question any more: the model trains on the new datasets, and is tested on questions it has not seen.

It follows the [benchmark](/pages/hrm-graph/mimir-graph-benchmark.md) (design of E0-E4) and its
[results](/pages/hrm-graph/mimir-graph-benchmark-results.md).

- Code: `graph_bench/rl/run_e5.sh` (queue), `graph_bench/rl/build_e5_data.sh` (data),
  `graph_inject/configs/datasets/bench_e5_answer.yaml` (control).
- Data: `data/graphs/bench/verl_e5/{train,train_grpo,val}.parquet`, `data/graphs/bench/sft/e5_answer.jsonl`.
- Runs: `/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench/{e5_grpo_multi,sft_e5}`; reports in
  `mimir_bench/{e5,e5sft}/mimir15/`; queue log `mimir_bench/e5_queue.log`.
- W&B: project `Mimir-Graph-RLVR`. GRPO training: run `e5_grpo_multi` (no group, as the other GRPO arms). Groups
  `mimir-bench-e5` (GRPO evals, control training) and `mimir-bench-e5sft` (control evals).

## Design (2026-11-13)

**Why this experiment.** E0-E4 showed that the benchmark training does not transfer zero-shot: E3 scores 0.100 on
OSM, 0.154 on CLUTRR, 0.652 on CWQ. So the next question is whether RL can *add* datasets one after another.

**Start model, both arms:** E3, `hf/mimir15-e3_grpo_e2-step200` (trace SFT, then 200 GRPO steps).

**Datasets.**

| Role | Datasets | In training | Test |
| --- | --- | --- | --- |
| New: Mimir never trained on them | OSM (Paris spatial QA), CLUTRR (family relations), CWQ (complex KGQA) | yes | their test sets |
| Old: trained in E2 / E3 | GraphQA, CLRS, MetaQA, KQA Pro, WebQSP | a small replay share | their test sets: do they drop? |
| Still held out | CLRS OOD, GraphQA large (larger graphs than in training) | no | the only transfer probes left |

OSM train, val and test use the same Paris graph but share no question and no innermost sub-question
([OSM build](/pages/hrm-graph/osm-spatial-kgqa.md)). So OSM test measures new questions about a known city, not a
new city.

**Training mixture** (`verl_e5/train.parquet`, 15,627 prompts after dropping those over 2,048 Mimir tokens, the
GRPO prompt limit):

| Dataset | Asked | Kept | Dropped as too long |
| --- | --- | --- | --- |
| OSM | 6,000 | 5,667 | 333 (most at 4 hops: 1,195 kept against ~1,500 at 1-3 hops) |
| CLUTRR | 3,000 | 3,000 | 0 |
| CWQ | 3,000 | 2,979 | 21 |
| GraphQA, CLRS, MetaQA | 800 each | 800 each | 0 |
| KQA Pro / WebQSP | 800 each | 790 / 791 | 10 / 9 |

Test sets are not filtered by length.

**The two arms.**

| | E5 GRPO | E5 SFT control |
| --- | --- | --- |
| What it learns from | a reward on its own sampled answers | the gold answers |
| Settings | E3's: 200 steps, 16 prompts x 8 rollouts, LoRA r32, lr 1e-5, responses up to 2,048 tokens | LoRA r16, lr 2e-4, 1 epoch |
| Rows | ~3,200 prompt groups of the mixture | the same questions, all of them: 15,373 rows |
| Target | - | new datasets: the `Answer:` line only (no traces exist). Old datasets: trace + answer |
| Cost (estimate) | ~47 h | ~2 h |

The control exists because of E4: on CLUTRR, SFT on answers reached 0.916 and GRPO 0.281. Without the control we
could not say whether RL is a good way to add a dataset. Unlike E4's control, this one keeps trace rows for the old
datasets (E4's control stopped writing traces and lost the tasks that need them). 254 old WebQSP rows have no trace
and are not in the control.

**Two changes to GRPO compared with E3**, both because of OSM:
- **Reward = set-F1 for set answers** (`GRAPH_REWARD=f1`; exact match elsewhere). E3 gets the exact OSM set on
  10% of test rows but a set-F1 of 0.383: it often returns part of the answer. An exact-match reward would be 0
  for almost every rollout, and GRPO learns nothing from a group whose rollouts all get the same reward (E3b: CLRS
  stayed at 0.097 for this reason). F1 gives partial credit, so more groups carry a signal.
- **Group filtering on the reward** (`FILTER_METRIC=score`, new variable in `run_grpo_graph.sh`; default `acc` as
  before). Dynamic sampling discards groups with a constant metric. With the old `acc` filter, an OSM group whose F1
  varies but whose exact match is always 0 would be thrown away.
The F1 reward also applies to the set answers of the old KGQA datasets, which E3 trained with exact match.

**Step 0, before any training: is there anything for GRPO to learn from?** E3 answers 200 val questions of each new
dataset 16 times (temperature 1.0, 2,048 tokens; `mimir_bench/e3/mimir15/e5new_pass16_t2k.json`). For each dataset
the queue computes the share of questions whose first 8 answers do not all get the same reward. **Rule: under 20%,
the dataset is left out of the GRPO mixture** (`verl_e5/train_grpo.parquet`) and stays only in the control. The
queue log prints the shares and the decision.

**Order of the queue:** step 0, a 2-step GRPO smoke run (stops the queue if it fails), the control and its evals
(first, because it is cheap), then GRPO and its evals.

**Evals, both arms** (same settings as E3's reports): `named_t3k` (12 test files, up to 1,000 rows each),
`osm_t3k` (all 2,000 OSM test rows), `anon_t3k`, `val_pass16_t2k` (8 val sets, 200 questions x 16 samples).

**How to read the result.** "Before" is E3 with the same reports.

| Look at | Tells |
| --- | --- |
| OSM, CLUTRR, CWQ test, arm − E3 | does training on a dataset improve it, and by how much for RL vs SFT |
| The 5 old datasets' test, arm − E3 | forgetting |
| CLRS OOD and GraphQA large, arm − E3 | does adding datasets help or hurt what is still unseen |
| Median answer length | does the model still write its reasoning |
| pass@16 − pass@1 on val | headroom left for more RL |

Limits: one seed; the arms are not matched (the control sees every row with its gold answer, GRPO about a fifth of
them with a reward only); GRPO's reward change (F1) is confounded with the new data.

## Launch (2026-11-13)

- Step 0 started 19:10 (`mimir_bench/e5_headroom.sh`), detached. The queue (`run_e5.sh`) waits for its report.
- Code changes: `graph_bench/sources/osm.py` now also converts OSM train (20,000 rows; val / test unchanged);
  `export_verl.py` takes a per-file cap (`file.jsonl@N`); `export_hrm.py --ids-from <parquet>` keeps the records of
  a GRPO parquet; `run_grpo_graph.sh` reads `FILTER_METRIC`.
- Results: not yet. They will be added below as dated sections.

## Step 0 outcome and the GRPO crash at step 125 (2026-11-16)

- **Step 0** (E3 on 200 val questions per new dataset, W&B run `l8pt06fx`): share of questions whose first 8
  answers get a non-constant reward: OSM 0.740, CLUTRR 0.315, **CWQ 0.180 → left out of GRPO** (under the 0.20
  rule). GRPO trains on 12,648 prompts (`verl_e5/train_grpo.parquet`); CWQ stays in the control and, for the GRPO
  arm, stays a never-trained test set.
- **Smoke** (2 steps): passed; 104 of 256 trained rollouts had a reward strictly between 0 and 1.
- **Control:** trained in 29 min (2026-11-13 22:06), four reports in `e5sft/mimir15/`. Results to be written up with
  GRPO's.
- **GRPO crashed at step 125** (2026-11-15 20:16, exit 137), W&B run `u59na0zw`. Killed by our own
  `scripts/graphs/mem_guard.sh`, not by the system: the summed RSS of the process group reached 91 GB against the
  90 GB cap during the checkpoint save, while the container used 59 GB of its 128 GB. Summed RSS counts shared pages
  once per process, so it overstates. Earlier arms peaked at 83-88 GB under the same cap; E5 is ~20.7 min/step
  (E3: 14).
  - The step-125 checkpoint was complete (same files and sizes as step 100).
  - Fix: `run_e5.sh` now uses `GRPO_RSS_CAP_GB` (default 105); the 110 GB container cap is unchanged.
  - Resumed 2026-11-16 08:36 from step 125 into the same W&B run. The queue had gone on after the crash and
    printed "E5 queue done" without GRPO evals; its log is `mimir_bench/e5_queue.run1.log`.

## WebQSP test is contaminated for the control (2026-11-16)

CWQ questions are built from WebQSP questions across WebQSP's split. 928 of the 2,979 CWQ rows in
`verl_e5/train.parquet`, which the SFT control trains on, extend a WebQSP test question (465 of the 1,628). Read the
control's WebQSP test score with that in mind; the GRPO arm left CWQ out and is not affected. Audit:
[train / val / test audit](/pages/hrm-graph/mimir-graph-trace-sft-analysis.md#3-can-the-splits-be-trusted-train--val--test-audit-2026-11-16).

## GRPO stopped at step 152 (2026-11-16)

Stopped on the user's decision (16:18) to free the GPU for the trace-SFT relaunch
([design](/pages/hrm-graph/mimir-graph-trace-sft-analysis.md#5-the-models-2026-11-16)). The queue and
the trainer were killed between two checkpoints.

- **Last checkpoint: step 150** (`mimir_bench/e5_grpo_multi/ckpt/global_step_150`, saved 15:36): same files and sizes as
  step 125, the model file passes its archive check, `data.pt` and the optimizer state load.
- No `DONE` marker, no merged model and no GRPO evals. Relaunching `run_e5.sh` (a copy) resumes from step 150 into
  W&B run `u59na0zw` and then evaluates; 50 steps remain (~17 h at 20.7 min/step).
- The control's reports (`e5sft/mimir15/`) are complete and not yet written up.

