---
type: Runbook
title: Mimir Graph Benchmark - Mimir vs the Baselines (Fair Comparison, Names or Facts, GLM-5.3 Cap)
description: "Is Mimir v1.5 better than the baselines once they get the plan's LR sweep and 3 seeds? Records the fairness runs (Qwen3.5-4B LR sweep, 3 seeds of Mimir E1 and Qwen3.5-4B E1: a tie), the analysis showing Qwen answers from entity names and Mimir from the given facts (anonymised eval), and GLM-5.3's token-cap cut-offs (16k / 32k / 64k, cap-visibility pilot, lenient rescoring)."
tags: [graphs, mimir, benchmark, sft, lora, qwen, glm, kgqa, clrs, clutrr, anonymisation, seeds, lr-sweep]
status: draft
last_updated: 2026-11-24
confidence: high
---
# Mimir Graph Benchmark - Mimir vs the Baselines

A review of the "Mimir after RL is the best model" table (notebook `notebooks/mimir_rlvr_v2.ipynb`, section 12)
found it overclaimed:
- the baselines got no LR sweep and one seed, against the plan ([benchmark plan](/pages/hrm-graph/mimir-graph-benchmark.md));
- GLM-5.3 was cut off by its token cap on a quarter of the rows;
- the best Mimir model (E3e1) is the best of 8 variants, picked after seeing test scores.

This page records the runs that fix it and what they changed. The earlier results are in
[Results and Run Log](/pages/hrm-graph/mimir-graph-benchmark-results.md) and
[Traces or Direct Answers?](/pages/hrm-graph/mimir-graph-trace-sft-analysis.md).

- **Fairness queue** `graph_bench/run_fairness.sh`, outputs under `$ROOT=/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench`:
  - Qwen3.5-4B sweep models in `sft_baselines/qwen35_4b/answer_lr<lr>[_s<seed>]`;
  - Mimir seeds in `sft_answer/mimir/text_s<seed>`;
  - reports in `e1/qwen35_4b_lr<lr>[_s<seed>]/` and `e1/mimir15_s<seed>/`;
  - the chosen LR in `sft_baselines/qwen35_4b/chosen_lr.txt`.
- **Names-vs-facts analysis:** `graph_bench/eval/names_vs_facts.py`.
- **GLM-5.3 token cap:**
  - `graph_bench/eval/run_glm_cap.sh` (`MODE=pilot|cut|full`, `CAP`, `FROM`);
  - `graph_bench/eval/glm_cap_pilot.py` (`prepare`, `compare`, `seed`);
  - outputs in `$ROOT/e0/glm53/` (`named_t32k`, `anon_t32k`, `cap_pilot/verdict.json`).
- **Lenient rescoring:** `graph_bench/eval/lenient_score.py`.
- **W&B:** project `Mimir-Graph-RLVR`, one eval run per report.

## Design: rules fixed before the runs (2026-11-23)

- **Which baseline is tuned.** Only the best E1 baseline, Qwen3.5-4B (0.590 at the default LR). Qwen2.5-7B and OLMo-3-7B
  keep their single untuned run. This stays within the GPU budget (~40 h).
- **LR sweep.** Qwen3.5-4B E1 at 5e-5, 1e-4 and 4e-4, plus the existing 2e-4; otherwise E1's recipe: `sft_hf`, LoRA,
  3 epochs, the same 17,917 answer-only rows.
- **LR choice, written down before any sweep result.** The highest macro **val** accuracy wins; a tie goes to 2e-4.
  - Val: greedy, 200 rows per val file (GraphQA, MetaQA, KQA Pro, WebQSP, CLRS), 1,024 tokens.
  - Test was never used to pick.
- **Seeds 1 and 2** for Mimir E1 (graph_inject `seed=`: data order and LoRA init) and for Qwen3.5-4B E1 at the
  chosen LR (`sft_hf --seed`). Seed 0 is the existing model. Every seed is scored on the same 9,554 named and 1,882
  anonymised test rows (ids asserted).
- **Mimir's LR was not swept** (2e-4, fixed before the benchmark). The comparison below is therefore a tuned
  baseline against an untuned Mimir.
- **GLM-5.3:**
  - Rerun only the cut-off rows at a higher cap, if a pilot shows the cap does not change GLM's behaviour.
    Otherwise rerun every row (see the GLM section).
  - Strict scoring stays the main score. Lenient scoring and the cut-off rate get their own columns.

## Results: the tuned baseline ties Mimir after the same SFT (2026-11-24)

**LR sweep, Qwen3.5-4B E1, val macro (greedy):**

| lr | GraphQA | MetaQA | KQA Pro | WebQSP | CLRS | macro val |
| --- | --- | --- | --- | --- | --- | --- |
| **5e-5 (chosen)** | 0.760 | 0.995 | 0.950 | 0.840 | 0.680 | **0.845** |
| 1e-4 | 0.770 | 0.995 | 0.955 | 0.835 | 0.585 | 0.828 |
| 2e-4 (default, used in every earlier table) | 0.685 | 0.975 | 0.955 | 0.785 | 0.540 | 0.788 |
| 4e-4 | 0.540 | 0.960 | 0.835 | 0.600 | 0.410 | 0.669 |

**Test, mean ± sd over 3 seeds, greedy, macro over the 7 datasets** (Δ: ± is the standard error of the difference):

| | Mimir E1 (lr 2e-4) | Qwen3.5-4B E1 (lr 5e-5) | Δ Mimir − Qwen |
| --- | --- | --- | --- |
| **macro test** | 0.644 ± 0.002 | 0.643 ± 0.008 | **+0.001 ± 0.005** |
| macro lenient | 0.648 ± 0.002 | 0.646 ± 0.008 | +0.002 |
| macro anonymised | 0.828 ± 0.012 | 0.765 ± 0.023 | **+0.063 ± 0.015** |
| KGQA, path in facts | 0.930 ± 0.004 | 0.938 ± 0.002 | −0.008 |
| GraphQA | 0.785 ± 0.016 | 0.735 ± 0.014 | +0.050 |
| CLRS | 0.513 ± 0.026 | 0.393 ± 0.009 | +0.119 |
| MetaQA | 0.850 ± 0.001 | 0.849 ± 0.004 | +0.001 |
| KQA Pro | 0.717 ± 0.012 | 0.717 ± 0.005 | 0.000 |
| WebQSP | 0.783 ± 0.006 | 0.821 ± 0.005 | −0.038 |
| CWQ (never trained) | 0.676 ± 0.003 | 0.693 ± 0.007 | −0.017 |
| CLUTRR (never trained) | 0.185 ± 0.009 | 0.290 ± 0.050 | −0.105 |

GraphQA by task:
- Mimir leads on MaximumFlow (+0.20), NodeDegree (+0.18), DisconnectedNodes (+0.10) and ShortestPath (+0.085).
- The others are level within ±0.04.

What this changes:
- **Superseded:** "Mimir v1.5 is the best model after the same SFT" (+0.052, Results page finding 2). That number
  compared Mimir with Qwen3.5-4B at an untuned LR. Tuning the LR alone gave Qwen +0.058 on test (0.590 → 0.648 at seed 0).
- **Mimir is ahead only after GRPO:** E3e1 0.662, +0.019 over the tuned baseline. That is one seed, E3e1 is the best of
  8 variants picked after seeing test scores, and the baseline got no RL. GRPO on the tuned Qwen3.5-4B E1 (~27 GPU h,
  as E3e1) would show whether the lead comes from Mimir or from GRPO.
- **SFT-seed noise is small on the macro:** Mimir sd 0.002, Qwen 0.008. It is larger per dataset: CLRS 0.026 for Mimir,
  CLUTRR 0.050 for Qwen.
- **Run cost:**
  - a Qwen3.5-4B SFT takes ~4.3 h (5,769 steps, ~2,800 tokens/s, 32 GB);
  - a Mimir SFT ~1.4 h;
  - one test eval (named + anonymised) ~12 min.

## Names or facts? Qwen answers from entity names, Mimir from the given facts (2026-11-24)

**Question.** The tie hides different per-dataset profiles:
- Qwen leads on WebQSP, CWQ and CLUTRR;
- Mimir leads on GraphQA, CLRS and the anonymised sets.

Is Qwen better on less reasoning-heavy rows, and does it lean on language and world knowledge more than Mimir?

**Method.** `graph_bench/eval/names_vs_facts.py`, accuracy averaged over the 3 seeds of each model:
- named KGQA rows grouped by what the facts allow (the groups of the
  [trace analysis](/pages/hrm-graph/mimir-graph-trace-sft-analysis.md));
- hop count and CLUTRR chain length;
- the same questions with names and anonymised, paired by (source, id).

**Named test: where each model leads**

| Rows | Mimir | Qwen | Δ |
| --- | --- | --- | --- |
| MetaQA, path in facts (977) | 0.974 | 0.956 | **+0.018** |
| MetaQA 3-hop (398) | 0.696 | 0.666 | **+0.030** |
| MetaQA, no path but answers in facts (124) | 0.411 | 0.505 | −0.094 |
| WebQSP, no path but answers in facts (175) | 0.802 | 0.859 | −0.057 |
| WebQSP, path in facts (710) | 0.904 | 0.942 | −0.038 |
| CWQ, path in facts (347) | 0.849 | 0.871 | −0.022 |
| CWQ, not all answers in facts (234) | 0.028 | 0.067 | −0.038 |
| KQA Pro, path in facts (714) | 0.937 | 0.943 | −0.006 |
| CLUTRR k3 / k6 / k10 | 0.061 / 0.189 / 0.181 | 0.178 / 0.357 / 0.365 | −0.12 / −0.17 / −0.18 |

- **Qwen's largest gains are where the facts do not give the answer.** There, knowing the entity helps.
- **Mimir leads on MetaQA's multi-hop chains**, which use a closed movie graph.
- **Qwen also leads on WebQSP and CWQ rows whose path is in the facts.** These are natural-language questions about
  real Freebase entities.
- **CLUTRR is reasoning-heavy and Qwen still leads at every chain length**, most at k6-k10. It is reasoning written in
  language (a story, kinship terms), not over a given graph.

**The same 1,882 questions with names and anonymised** (MetaQA, KQA Pro, CWQ test; WebQSP has no anonymised rows):

| Rows | Named: Mimir / Qwen | Anonymised: Mimir / Qwen | Drop: Mimir / Qwen |
| --- | --- | --- | --- |
| **all (1,882)** | 0.840 / 0.845 | 0.795 / 0.731 | **−0.045 / −0.114** |
| MetaQA, path in facts (592) | 0.968 / 0.951 | 0.944 / 0.862 | −0.025 / −0.089 |
| KQA Pro, path in facts (456) | 0.942 / 0.951 | 0.915 / 0.872 | −0.027 / −0.079 |
| CWQ, path in facts (301) | 0.854 / 0.875 | 0.787 / 0.713 | −0.066 / −0.162 |
| CWQ, no path but answers in facts (349) | 0.904 / 0.911 | 0.809 / 0.714 | −0.096 / −0.197 |
| CWQ, not all answers in facts (184) | 0.031 / 0.065 | 0.007 / 0.018 | −0.024 / −0.047 |

- **With names the two models are level on these questions. Anonymised, Qwen loses 2.5 times as much as Mimir.**
- **The gap holds on rows whose facts contain the whole path**, where names are not needed: Qwen's drop is about 3 times
  Mimir's on MetaQA and KQA Pro, and 2.5 times on CWQ. Qwen uses names as a shortcut even when the facts suffice;
  Mimir reads the facts.
- **Conclusion.**
  - Mimir is the better reasoner over a given graph: algorithm tasks, multi-hop MetaQA, anonymised questions.
  - Qwen3.5-4B is better where language and world knowledge help: real-entity questions, CLUTRR.
  - The macro averages these out to a tie. The anonymised macro is the benchmark's cleanest test of graph reasoning.
- **Caveats:**
  - None of the 17,917 SFT rows is anonymised, so ids like `E58` are unfamiliar to both models.
  - Anonymised answers are arbitrary ids that must be copied exactly, so part of Mimir's edge may be symbol copying
    (consistent with its GraphQA / CLRS lead).
  - One baseline only. Qwen2.5-7B and OLMo-3-7B were not seeded or tuned.

## GLM-5.3: token-cap cut-offs, cap-visibility pilot, lenient scoring (2026-11-23/24)

**Problem.** The SDU server applies the cap as `max_completion_tokens`, which covers the hidden reasoning and the answer
together. A row whose reasoning reaches the cap returns no answer and scores 0. At 16k, 2,571 of the 9,554 named rows
(26.9%) were cut off: 2,534 inside the reasoning, 37 while writing the answer.

**Pilot: does the cap change GLM's behaviour?** If GLM could see its budget, a bigger cap would change the rows that
already finished, and every row would need rerunning.
- Method: `glm_cap_pilot prepare` sampled 140 rows that finished at 16k (20 per dataset, seed 0). They were rerun at
  each cap and compared with the original 16k run.
- Rule, fixed before the run: the cap is invisible if, at the higher cap, answer agreement is within 0.03 of a second
  16k run's and the median reasoning length within ±10% of it.

| Rerun | Same answer as original | Median reasoning length vs original | Accuracy (original 0.786) | Cut off |
| --- | --- | --- | --- | --- |
| 16k again (serving noise) | 0.743 | ×1.04 | 0.643 | 25 / 140 |
| 32k | 0.814 | ×1.05 | 0.750 | 11 / 140 |
| 64k | 0.886 | ×0.99 | 0.786 | 0 / 140 |

- **Verdict: the cap is invisible at 32k and at 64k.** Rerunning only the cut-off rows is valid.
- **Agreement rises with the cap only because fewer rerun rows hit it.** Serving at temperature 0 is not deterministic:
  25 of 140 rows that had finished at 16k were cut off when rerun at 16k.
- The prompt never mentions a budget. A "budget" phrase appears in 1 of 140 reasoning traces at 32k and at 64k (0 at
  16k), in the model's own words.

**32k rerun of the cut-off rows** (`MODE=cut CAP=32768 FROM=t16k`). The 16k rows that finished are reused:
`glm_cap_pilot seed` keys them on the final report's rows, last attempt per key.

| | 16k | 32k |
| --- | --- | --- |
| named rows cut off | 2,571 (26.9%) | 1,229 (12.9%) |
| per dataset: GraphQA / MetaQA / KQA Pro / WebQSP / CLRS / CWQ / CLUTRR | 11 / 8 / 18 / 20 / 54 / 34 / 44% | 5 / 3 / 11 / 12 / 22 / 22 / 21% |
| named macro, strict / lenient | 0.572 / 0.625 | 0.650 / 0.713 |
| anonymised rows cut off | 606 (32%) | 471 (25%; anonymised CWQ 53%) |
| anonymised macro | 0.607 | 0.643 |
| completion tokens, named run | 58M | 70M |

- On top of the cut-offs, 141 named rows failed at 32k with `InternalServerError: Invalid request` (CLRS 137, CWQ 4).
  They are scored 0, like the cut-offs, and are rerun at 64k.
- At 32k, GLM's lenient macro (0.713) is above every fine-tuned model under the same rule (E3e1 0.665). Its strict macro
  (0.650) is a lower bound: 12.9% of rows are still cut off.
- **64k rerun of the remaining rows** (`MODE=cut CAP=65536 FROM=t32k`, client timeout 3,600 s) started 2026-11-24.
  It calls 1,370 named rows (1,229 cut off + 141 errors) and 471 anonymised rows. Results to be added here.

**Lenient scoring** (`lenient_score.py`, the same rules for every model; strict stays the main score).
- A row also counts as right if, after format-only fixes, it matches the gold:
  - CLRS: the same integers, whatever the separators;
  - GraphQA: the yes/no word, the node set, or the first integer;
  - KGQA / CLUTRR: markdown and quotes stripped, Y/M/D dates normalised, `|` / `;` sets.
- Effect: GLM +0.063 at 32k (GraphQA 0.805 → 0.947 on answered rows, CLRS 0.547 → 0.820, KQA Pro +0.05, WebQSP / CWQ
  +0.03-0.04). Every fine-tuned model gains only +0.002 to +0.006, so the format losses are GLM's alone.

**What lenient does not fix.** Of the 8,187 rows GLM answered at 32k, 1,238 are still wrong:
- **Real errors (about half):**
  - CLRS lists of the right length with wrong entries (150);
  - EdgeExistence answered "Yes" where no edge exists;
  - CLUTRR in-law mix-ups ("father-in-law" for "father", 79);
  - wrong entities.
- **Abstentions (165):** "Cannot be determined from the given facts" or "unknown", mostly on rows whose facts lack the
  path (KQA Pro 72, MetaQA 44, CWQ 30). The fine-tuned models were trained to always answer.
- **Answer-shape differences (~340):**
  - several answers where the gold has one (MetaQA 91);
  - a sentence instead of the relation asked for (KQA Pro);
  - qualifiers ("Las Vegas, Nevada");
  - CLRS conventions: None / NIL for Prim's root, an edge list for Kruskal's matrix, a different shape (94).
- Counting "the answer contains a gold" as right lifts every model by about +0.03 and changes no ranking (GLM 0.746,
  E3e1 0.691, Mimir E1 0.674, Qwen3.5-4B E1 0.675), so it is not used. The fix would be format examples in GLM's prompt.

## Fails (2026-11-23/24)

- **The plan expected 7,020 reused rows and 2,534 to rerun; the seeding gave 6,983 and 2,571.** The difference is the
  37 rows cut off while writing the answer (`finish == "length"` with non-empty text). The seeding was right; the plan
  counted only empty answers.
- **Report rows have no source field, and `metaqa/test_retrieved` / `kqapro/test_retrieved` reuse the ids of the plain
  test files.** Deriving the source from the id (`metaqa/test/1` → `metaqa/test`) gives `KeyError` or, worse, the wrong
  record. Take the source from E3e1's report, which has one (same rows, same order: assert it), as the notebook and
  `names_vs_facts.py` do.
- **The 32k client timeout (600 s) was close:** requests averaged ~340 s. It is raised to 3,600 s for 64k.
- **The container's memory read 95 GB** (cgroup limit 128 GB) while the queue ran. 92 GB of it was file cache from
  loading models (`memory.stat` `file`), which the kernel can reclaim; the processes used 2 GB. Check `anon` vs `file`
  before stopping jobs.

## Next

- GRPO on the tuned Qwen3.5-4B E1 (E3e1's settings), to decide whether Mimir's lead after RL is Mimir's or GRPO's.
- An LR sweep for Mimir E1 (the comparison above tunes only the baseline).
- GLM-5.3 at 64k (running); then possibly format examples in its prompt for the answer-shape losses.
