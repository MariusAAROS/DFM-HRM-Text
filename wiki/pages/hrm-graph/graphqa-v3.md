---
type: Runbook
title: GraphQA v3
description: "graphqa_v3_cot (built 2026-11-02): graphqa_v2_cot with exact train duplicates removed and extra isomorphism-deduplicated large TriangleCounting / MaximumFlow train rows, val/test unchanged; records the evidence and the reason for each change."
tags: [graphs, graphqa, chain-of-thought, counting, max-flow, mimir, data]
status: stable
last_updated: 2026-11-02
confidence: high
---
# GraphQA v3

With the CoT target ([GraphQA Chain of Thought](/pages/hrm-graph/graphqa-cot.md)), Mimir v1.5 `text` saturates every
GraphQA task except TriangleCounting (test 0.78 on seeds 0 and 1) and MaximumFlow (0.85 / 0.79). v3 asks whether
more, and more diverse, training data on these two tasks closes the gap. Prompts, traces, val and test stay unchanged.

- Built 2026-11-02 (run pending):
  - builder `scripts/graph_inject/build_graphqa_v3.py` (report in `build_report.json` next to the data);
  - data `data/graphs/graphqa/hrm-text-grl-v3-cot/standard`;
  - config `graph_inject/configs/datasets/graphqa_v3_cot.yaml`;
  - out_root `graph_inject_graphqa_v3_cot`, W&B group `graphqa-v3-cot-inject`;
  - test `test_graphqa_v3_data` in `tests/test_graph_inject.py`.

## What motivated it (2026-11-02)

Measured on `graphqa_v2_cot` and the Mimir v1.5 `text` runs:
- **The val TriangleCounting "75%" is 3/4.** The 55-question val subset has 4 TriangleCounting questions. The one miss
  is val row 109 (90 triangles): wrong until epoch 7 (seed 0) or 9 (seed 1), then right. It is not a plateau.
- **Test errors are on large instances** (Mimir v1.5, seed 0):
  - TriangleCounting: ≤ 5 triangles 12/12, 6–20 3/4, > 20 6/11;
  - MaximumFlow: ≤ 3 augmenting paths 19/20, 4–8 5/5, > 8 5/9.
- **No train → val/test leak on these tasks:**
  - v2 already drops val/test rows whose prompt equals a train prompt.
  - Every val/test TriangleCounting graph appears in train, but only under other tasks (GraphQA draws all tasks from
    one graph pool), never as a TriangleCounting question.
  - Up to relabelling, 2 val graphs (0 test) match a train TriangleCounting graph.
  - A leak would push val up, not cap it.
- **Train repeats itself:**
  - 372 of 3,080 train rows (12%) are exact copies of another train row: EdgeCount 87, NodeCount 81,
    TriangleCounting 77, CycleCheck 75, the others ≤ 18.
  - They are mostly fixed-shape graphs (complete, star, path) that the generator draws again and again.
  - MaximumFlow has none, because v2 drew fresh capacities for every row.
- **Few distinct large TriangleCounting examples:**
  - Train has 269 TriangleCounting rows but only 185 distinct graphs up to relabelling.
  - Of the 78 rows with > 20 triangles, 31 are complete graphs (13 distinct); val/test contain no complete graph.
    The 13 rows at a coarser CoT tier are all among these 78.
  - That leaves 47 distinct non-complete graphs with > 20 triangles.
  - Train has 63 MaximumFlow rows with > 8 augmenting paths.

## Design: differences from graphqa_v2_cot (planned 2026-11-02)

| # | Change | v2-CoT | v3 | Why |
|---|---|---|---|---|
| 1 | Exact train duplicates | kept (372 rows) | one copy of each | A copy adds no new example and tilts training toward fixed-shape graphs. |
| 2 | TriangleCounting train rows | 269 (78 with > 20 triangles) | + ~400 with > 20 triangles, + ~100 with 6–20 | The test errors are in these bins. ~400 is about 8× the 47 distinct large graphs: a chosen dose, not a derived one. |
| 3 | MaximumFlow train rows | 281 (63 with > 8 paths) | + ~400 with > 8 paths, + ~100 with 4–8 | Same reason (> 8 paths: 5/9 on Mimir v1.5). |
| 4 | Source of new graphs | — | GraphQA's own generator (`graph_generator_utils`, imported as in `scripts/kg_pretrain/gen_graphqa.py`): er, ba, sbm, sfn, star, path, with fresh seeds. No algorithm fills > 50% of a size bin; a `gen_algorithm` field on each row. | Keeps the benchmark's graph distribution. The cap stops dense ER graphs from filling every large bin. |
| 5 | Complete graphs | 37 train TriangleCounting rows | none among the new rows | Every K_n draw gives the same prompt, and val/test hold none. |
| 6 | Dedup | exact prompt match, val/test against train | Up to relabelling, a new row is dropped if its graph matches a val/test graph (any task), an existing train graph of the same task, or another new row. Existing train rows matching a val/test graph of the same task are dropped (2 TriangleCounting rows, matching val). Superseded (2026-11-02): the build drops 60 rows, 58 of them MaximumFlow, see Build. | No val/test graph is trained on under its own task, even relabelled, and every new row is a new graph. |
| 7 | CoT detail of new rows | builder falls back to coarser tiers | Full detail only. A graph whose full trace exceeds 4,090 tokens with any of the 3 tokenizers is rejected, not coarsened. | All but 3 val/test rows are at full detail. The coarse tiers hide the intersection or the updates, which is where the errors are. |
| 8 | Checkpoints (trainer option) | LoRA of the best epoch only | LoRA of every epoch (54 MB each) | HRM-1B seed 1 picked epoch 2 on a 2-question val spike (test 0.858, against 0.910 for seed 0), and that run can't be re-tested at a later epoch. With every epoch saved, test can also run at epoch 10, which involves no selection. |

**Kept as in v2-CoT, on purpose:**
- **Val and test files are byte-identical** (307 / 324 rows, so the same 55-question subset), so v3 scores compare
  directly with every v2-CoT run.
- **Same prompt template:** GRL `#### Question\n` plus the adjacency encoder. MaximumFlow capacities are uniform 1–10,
  printed as `(u, v, c)`, with the gold computed by networkx, as in `build_graphqa_v2.py`.
- **Same trace formats** (`graph_inject/cot.py`). The intersection stays hidden; exposing it is a separate arm, so a
  v3 gain can be attributed to the data alone.
- **No new rows for the other 9 tasks.** They are at 1.00 on test and serve as a regression check for the new task
  mix.
- **A val/test graph asked under another task in train is kept.** That is GraphQA's design, and it gives away no
  answer.
- **Same recipe and epoch selection:** 10 epochs, LoRA, best val greedy_acc on the 55-question subset, ties to the
  earliest epoch.
  - Full val would be the cleaner selection, but it costs ~35–40 min per epoch against ~7 min: about 0.6–0.7 min per
    TriangleCounting/MaximumFlow val row, and val has 57.
  - The epoch-10 test (#8) gives the selection-free number instead.

**Run plan:**
- Mimir v1.5, `text`, seed 0, queued after the CoT `text` seed runs.
- Baseline: v2-CoT `text` Mimir v1.5, seeds 0 / 1 / 2 = 0.966 / 0.960 / pending; TriangleCounting 0.78 / 0.78;
  MaximumFlow 0.85 / 0.79.
- Report test at the selected epoch (comparable with v2-CoT) and at epoch 10.
- Size (estimate): ~1,000 new rows of ~1,800 tokens (TriangleCounting > 20) to ~2,300 tokens (MaximumFlow > 8 paths).
  - That is +~1.9M tokens: train goes from 1.66M to ~3.4M after dedup, and training from ~3 to ~6 min per epoch.
  - The run takes ~3–3.5 h, plus ~37 min for the epoch-10 test.
- Success criteria:
  - TriangleCounting > 20 triangles and MaximumFlow > 8 paths rise by more than the seed spread;
  - the 9 saturated tasks stay at 1.00.
- A perfect TriangleCounting score is not expected while the intersection is hidden: one wrong line among ~50–100
  breaks the total.

## Build (2026-11-02)

`build_graphqa_v3.py --seed 20261102 --graphs-per-algorithm 4000` takes ~4 min on CPU.
- **val/test:** byte-identical to v2-CoT.
- **Train:** 3,648 rows: 2,648 v2 rows kept plus 1,000 new rows (v2-CoT: 3,080). About 3.1M tokens, against 1.66M
  (estimate from character counts).
- **v2 rows dropped:**
  - 372 exact duplicates;
  - 60 rows isomorphic to a val/test graph of the same task: TriangleCounting 2, MaximumFlow 58.
  - The MaximumFlow matches are small graphs (median 8.5 edges, 41 of 58 with a single augmenting path) that GraphQA
    reuses across splits. Their capacities and s/t differ from the val/test row, so the answer differs; they are
    dropped anyway, to keep the "no val/test graph under its own task" rule exact.
- **New rows:** all 4 quotas filled; size and token stats below.

  | Task | Bin | Rows | Algorithm mix |
  |---|---|---:|---|
  | TriangleCounting | > 20 triangles | 400 | ba 139, sbm 136, er 125 |
  | | 6–20 | 100 | sbm 31, ba 30, sfn 21, er 18 |
  | MaximumFlow | > 8 paths | 400 | sbm 163, er 157, ba 80 |
  | | 4–8 | 100 | ba 36, sbm 31, er 27, sfn 6 |

  - star and path graphs cannot reach these bins: 0 triangles, 1 path. sfn reaches only the lower bins.
  - Rejected among the candidates drawn before the quotas filled:
    - TriangleCounting: 32 too long at full detail, 13 isomorphic to an earlier new row, 3 to a val/test graph,
      8 complete, 8 without edges;
    - MaximumFlow: 31 too long, 7 val/test, 2 new, 1 train, 8 complete, 11 without edges.
- **Size, new rows against test:**
  - TriangleCounting: median 48 triangles (test 12), max 346 (test 171), median 14 nodes (test 11), median 46 edges
    (test 24).
  - MaximumFlow: median 12 paths (test 2), max 23 (test 34), median 54 edges (test 16).
  - Tokens: TriangleCounting mean 1,718, MaximumFlow mean 2,212, max 4,055.
  - The new rows are deliberately larger than test: they target the bins where test fails.
- **Train task mix:**
  - MaximumFlow 723 and TriangleCounting 690 rows, against 200–288 for each other task;
  - 69% of train characters, against 32% in v2-CoT.
- **Trainer options** (`graph_inject/train.py`, `base.yaml` defaults off, on in `graphqa_v3_cot.yaml`):
  - `eval.save_every_epoch` writes `trainable_ep<N>.pt`;
  - `eval.test_last_epoch` writes `eval_test_last.json` and W&B `test_last_greedy_acc`.
  - A 2-epoch smoke run (HRM-1B, 40 rows) wrote both. Its best epoch was the last, so only the copy branch ran, not
    the reload-and-evaluate branch.
- **Prompt format check:** GraphQA's adjacency encoder (`graph_text_encoder.encode_graph`) reproduces 686 of 700 v1
  TriangleCounting/MaximumFlow prompts exactly. The other 14 differ only in edge order, from the node insertion order of
  the check's rebuilt graphs.
