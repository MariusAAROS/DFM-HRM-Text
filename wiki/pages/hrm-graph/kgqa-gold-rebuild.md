---
type: Runbook
title: MetaQA and KQA Pro Gold-Only Rebuild
description: Audit of the MetaQA and KQA Pro subgraph prompts (how often the answer and the reasoning path are in the prompt, per seeding arm) and the approved rebuild into gold-only training files with a program-driven KQA Pro extraction.
tags: [graphs, kgqa, metaqa, kqapro, dataset, mimir, benchmark]
status: stable
last_updated: 2026-11-03
confidence: medium
---
# MetaQA and KQA Pro Gold-Only Rebuild

SFT and RL need questions that can be answered, and reasoned through, from the facts in their own prompt. The
MetaQA and KQA Pro prompts built by the GRL generator do not always allow that. This page records the audit of
2026-11-03 and the approved rebuild.

- **Source generators (unchanged):** `/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM/src/datasets/{metaqa,kqapro}/`.
  The output files `data/graphs/{metaqa,kqapro}/hrm-text-grl` and every earlier score on them stay reproducible.
- **Bench converters:** [`graph_bench/sources/metaqa.py`](../../../graph_bench/sources/metaqa.py),
  [`graph_bench/sources/kqapro.py`](../../../graph_bench/sources/kqapro.py). Output goes to `data/graphs/bench/{metaqa,kqapro}/`.
- **Benchmark page:** [Mimir v1.5 Graph-Reasoning Benchmark](/pages/hrm-graph/mimir-graph-benchmark.md).

## Audit (2026-11-03)

**How the generator builds prompts.** Each question is emitted twice, once per seeding arm, and both copies sit in
the same train / val / test file:
- **`gold`.**
  - MetaQA: the question's relation chain is walked from the topic entity. Every topic → answer path (the
    "spine") is always kept, the rest of the 80-triple budget is off-chain distractors, and a row is dropped if an
    answer is missing.
  - KQA Pro: seeds are the program's `Find` entities, entities matched by filter values, and the answer entity.
    The builder takes a 1-hop ball and force-keeps up to 20 facts containing the answer string, in a 60-fact budget.
- **`retrieved`.** A k-hop ball around entity names matched in the question text. It never sees the answer: this is
  the retrieval-noise condition, by design.

**Coverage, all splits** (answer = every gold answer appears in the prompt facts; trace = the bench converter's
step-by-step trace reproduces the gold exactly):

| Arm, task | MetaQA answer / trace | | KQA Pro task | gold_1hop answer / trace | retrieved_2hop answer / trace |
| --- | --- | --- | --- | --- | --- |
| gold 1-hop | 100% / 94.6% | | Basic | 100% / 73% | 63% / 34% |
| gold 2-hop | 100% / 100% | | Comparison | 99.9% / 52% | 60% / 30% |
| gold 3-hop | 100% / 100% | | Logical | 100% / 53% | 69% / 28% |
| retrieved 1-hop | 99.6% / 94.0% | | Multihop | 100% / 29% | 51% / 4.5% |
| retrieved 2-hop | 88.1% / 64.3% | | Count | n/a / 2.6% | n/a / 0.8% |
| retrieved 3-hop | 66.9% / 29.3% | | Qualifier | 73% / 41% | 9% / 27% |
| | | | Verify | n/a / 57% | n/a / 25% |

Count and Verify answers are numbers or yes/no, so "answer in prompt" does not apply. The topic entity is in the
prompt in 96–100% of rows of every arm.

**Findings.**
- **MetaQA `gold` is sound.**
  - All of MetaQA's missing answers come from the `retrieved` arm.
  - The 90 `gold` 1-hop rows without a trace are homonym movies (two "Scent of a Woman", 1974 and 1992, are one node
    of the KB). The walk correctly finds both years, while MetaQA's gold has one.
- **MetaQA train holds 3-hop questions** (2,648 of 7,967), although the plan keeps 3-hop as a within-task OOD split
  that is never trained on.
- **KQA Pro `gold_1hop` guarantees the answer string, not the program's path.** Comparison needs the compared
  attribute of both entities, Count every member of the set, Multihop each link. Only ~50% of the gold rows trace.
- **Rows without a trace are not an answer-presence problem in the gold arms:** the answer is there, the path is not.
  In the retrieved arms, the answer itself is often missing: those questions cannot be answered from the prompt.
- **KQA Pro minimal support fits a prompt for ~90% of questions.** Method:
  - Run the KoPL program on the full KB (3,000 random official-train questions, 0 execution errors).
  - Count the facts each step must show for the entities that survive it. The prompt is closed-world, so a filter
    chain needs only its survivors, an argmax only its winner, and a Count every member.

| Task | Median facts | ≤ 40 | ≤ 60 | ≤ 100 |
| --- | --- | --- | --- | --- |
| Basic | 2 | 100% | 100% | 100% |
| Verify | 3 | 96% | 96% | 96% |
| Qualifier | 3 | 91% | 92% | 92% |
| Count | 6 | 85% | 88% | 92% |
| Comparison | 5 | 83% | 86% | 88% |
| Multihop | 20 | 64% | 66% | 74% |
| Logical | 17 | 59% | 65% | 72% |
| All | | 89% | 90% | 92% |

  This count is an approximation of the closed-world minimum. The rebuild measures the real figure per question.
  KQA Pro prompts take about 20 tokens per fact (median 1,350 tokens for 60 facts).

## Rebuild plan (approved 2026-11-03)

**Layout, per dataset:**
- `train` / `val` / `test.jsonl`: **gold only**, used for SFT, RL and the main eval.
- `test_retrieved.jsonl`: the retrieved arm, an eval condition only, with an `answer_in_prompt` flag per row.

Scores on the gold-only files are **not comparable** with earlier MetaQA / KQA Pro results, which were trained and
tested on the mixed files.

**MetaQA (filtering only):**
- split by arm;
- 3-hop out of train and val (it stays in test as the OOD depth split);
- drop the 90 homonym rows.

**KQA Pro (new program-driven gold extraction, in `graph_bench`; the GRL generator is not modified):**
1. **Spine:** run the program on the full KB and keep the facts each step reads for its surviving entities. State
   derived `instance of` facts explicitly, because FilterConcept matches through the concept hierarchy.
2. **Noise:** fill the budget with near-miss distractors (other facts of the same entities, other members of the
   filtered concept).
3. **Validate (closed world):** re-run the program on the prompt facts alone. Drop and resample any distractor that
   changes the result, for example an extra counted member or a larger rival.
4. **Size cap:** drop questions whose spine does not fit, and count them.
   - Same question sample as GRL: 4,000 train pool, 600 test.
   - First version with simple distractors; richer near-misses only if it turns out too easy.

**Expected:** answer coverage and traces near 100% on every kept question, with about 10% dropped (mostly Logical
and Multihop).

## Build (2026-11-03)

Mixed-arm files before the rebuild: `data/graphs/bench/_mixed_20261103/{metaqa,kqapro}/`.

**MetaQA** (`graph_bench/sources/metaqa.py`, gold-only routing):

| File | Rows | Traced | Content |
| --- | --- | --- | --- |
| train | 2,588 | 100% | gold, 1-hop 1,255 + 2-hop 1,333 |
| val | 262 | 100% | gold, 1-2 hop |
| test | 592 | 100% | gold, 1 / 2 / 3-hop 192 / 200 / 200 |
| test_retrieved | 597 | 64.5% | retrieved arm; 85.3% have every answer in the prompt (`meta.answer_in_prompt`) |

Dropped: 90 homonym rows and 1,466 gold 3-hop train / val rows.

**KQA Pro** (new [`graph_bench/sources/kqapro_gold.py`](../../../graph_bench/sources/kqapro_gold.py), same 4,600 questions
as GRL):
- **Spine.** Computed backwards from the last program step, and every `Find` entity gets at least one anchor fact.
- **Distractors.** Validated in chunks of 5 up to 60 facts; 97 were rejected.
- **Validation on the name-merged KB.** The model reads names, so homonyms are one node (Chicago the city and the
  musical).
- **Empty-set rule.** An empty entity set is rejected (the gold reached through it is luck), except on a branch that
  feeds a non-empty `Or`.

| Task | Kept / questions | Spine > 40 facts | Fails validation | Program fails on the full KB |
| --- | --- | --- | --- | --- |
| Basic | 1,153 / 1,174 (98%) | 0 | 0 | 21 |
| Comparison | 856 / 860 (99.5%) | 2 | 0 | 2 |
| Logical | 318 / 319 (99.7%) | 0 | 0 | 1 |
| Multihop | 108 / 111 (97%) | 0 | 0 | 3 |
| Qualifier | 983 / 1,035 (95%) | 36 | 11 | 5 |
| Verify | 549 / 556 (99%) | 4 | 3 | 0 |
| Count | 345 / 545 (63%) | 65 | 114 | 21 |

- **Count.** 110 Count questions have the answer 0. An empty set gives the prompt no evidence for 0, so the empty-set
  rule drops nearly all of them. Of the non-zero Count questions, 345 / 435 (79%) are kept.
- **Splits.** Train 3,377, val 372, test 563, all 100% traced, 60 facts per prompt. test_retrieved is GRL's 578
  retrieved-arm rows (26% traced).
- **Validation is sensitive.** Removing one spine fact breaks the answer for 100% of Basic / Logical / Multihop /
  Verify spine facts, 84% Comparison, 72% Count and 59% Qualifier (110 train questions). The rest are redundant
  same-key facts, which act as relevant noise.
- **Prompt length** (Mimir tokens, GRL prompt):
  - KQA Pro median 1,298, p95 1,690, max 2,767; 49 prompts are over 2,048, the RL prompt cap.
  - MetaQA median 1,093, max 1,353.
- **Tests:** `tests/test_graph_bench.py::test_kqapro_gold_validation_rules` (name merge, empty-set rule).

**Impact.** The GRPO smoke data (`data/graphs/bench/verl_smoke`) was exported from the mixed MetaQA files, 3-hop
included. Re-export from the gold-only files before any real SFT or RL run.
