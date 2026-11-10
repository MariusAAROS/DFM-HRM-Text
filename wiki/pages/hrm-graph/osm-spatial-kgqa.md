---
type: Runbook
title: OSM Spatial Multi-Hop QA as KGQA
description: The OSM Paris multi-hop spatial question set (data/osm), evidence that it can be cast as KGQA (entity linking, relation rules recovered from the gold answers, re-answering from the tables), and the train/val/test KGQA build in data/osm/kgqa.
tags: [graphs, kgqa, osm, spatial, dataset]
status: stable
last_updated: 2026-11-10
confidence: high
---
# OSM Spatial Multi-Hop QA as KGQA

Can the OSM multi-hop spatial questions over Paris be cast as knowledge-graph QA, like GraphQA, MetaQA and
KQA Pro? Each question has a set of POI ids as its answer. A typical question is "restaurant indian food at less
than 1000 meters from [cafe … near Le Florida and following Rue de Cléry]". The answer is yes. This page
records the evidence and the train/val/test build.

- **Source data** (`data/osm/`, git-ignored):

  | file | rows | content |
  | --- | --- | --- |
  | `df_osm` | 17,457 | POIs: `poi_id`, name, category, cuisine, seating, Lambert-93 x/y |
  | `df_street` | 7,193 | street polylines (6,462 names) |
  | `df_area` | 1,046 | arrondissements, quartiers, parks (polygons) |
  | `df_multihop` | 1,177,868 | questions |
  | `df_dump_question` | 800 | eval subset, 200 per depth |

- **Analysis notebook:** [`notebooks/osm.ipynb`](../../../notebooks/osm.ipynb) (maps, relation histograms,
  verification, a worked 3-hop example).
- **Code:**
  - [`scripts/osm/osm_kg.py`](../../../scripts/osm/osm_kg.py): WKB decoding, question parser, relation
    rules, solver.
  - [`scripts/osm/build_kgqa.py`](../../../scripts/osm/build_kgqa.py): dataset build.
  - [`scripts/osm/check_kgqa.py`](../../../scripts/osm/check_kgqa.py): checks.
- **Output:** `data/osm/kgqa/{train,val,test}.jsonl` (see its `README.md`). Build steps and noising, current
  version: [Build recipe and noising steps](#build-recipe-and-noising-steps-of-the-current-build-2026-11-09). No model has been trained or
  evaluated on it yet.
  Superseded (2026-11-09): a zero-shot eval of the Mimir benchmark models is queued, see
  [Zero-shot transfer eval](#zero-shot-transfer-eval-of-the-mimir-benchmark-models-2026-11-09).

## Dataset structure (2026-11-06)

**Question structure.**
- `query_texte` nests sub-questions in `[...]`.
- `template[i]` is the number of clauses at level *i* (outermost first, terminated by 0), so depth =
  `len(template) - 1`.
- Depth counts: 43,565 questions at 1 hop, 391,225 at 2, 396,161 at 3 and 345,314 at 4 (after deduplication).
- Hops chain only through `within N`, `dir_of_poi` and `near`. Every other clause names a POI, a street or an
  area.
- `features` / `values` give the attribute filter of the outermost level only. Inner levels' attributes exist
  only in the text.
- `index_*` and `dic_query` refer to the generator's constraint objects. They match no table and are not
  needed.

**Quirks.**
- 1,603 rows repeat a question text.
- `drop_duplicates()` on the whole frame fails, because the array columns are unhashable. Deduplicate on
  `query_texte`.
- Answer sets are large. Depth 1: 72% have exactly one answer and 97% have ≤ 10. Depths 2–4: 38–43% have one
  answer and 75–77% have ≤ 10. The maximum is 2,406.
- Some outdoor-seating values (`street`, `garden`, `summer`, `patio`, `limited`) and a placeholder category
  `pois` never appear in the question text. Those constraints exist only in the metadata.
- Geometries are WKB in Lambert-93, in metres. shapely is not installed; `osm_kg.py` decodes the WKB itself.

## Linking and relation rules (2026-11-06)

**Linking.**
- All 17,443 answer ids are `df_osm.poi_id`s.
- Question anchors are linked by exact name, in a 30k-question sample: 88.8% to a single entity, 11.2% to a
  POI name shared by several POIs (Starbucks, Le Rallye, …), 0.04% not found.
- Street and area anchors always resolve. All segments of one street name are merged.

**Rules.** Fitted on the gold answers of 4,000 one-hop questions. Every gold answer satisfies every clause of
its question, so a hard cut-off in a relation's distribution gives that relation's rule:

| relation | rule | gold answers within it |
| --- | --- | --- |
| `at less than N meters from X` | d < N (N = 100/300/500/1000) | 100% |
| `near X` | d < 1000 m (not k-nearest: tested, rank is unbounded) | 100% |
| `near X towards Y` | d(X) < 1000 m and angle(X→p, X→Y) ≤ 90° | 100% |
| `to the <dir> of X` | d < 1000 m and bearing within 45° of the direction | 99.5% |
| `between X and Y` | < 200 m from segment XY | 100% |
| `at the intersection of S1 and S2` | < 1000 m from the closest vertex pair of S1, S2 | 99.7% |
| `following S` | < 50 m from the street | 100% |
| `at the border of A` | < 200 m from the boundary | 100% |
| `inside A` | point in polygon | 100% |
| `at the <dir> of A` | inside A and on that half of A's bounding box | 99.5% |

Values below 100% come from anchor names shared by several POIs.

## Verification: answering from the tables (2026-11-06)

`osm_kg.solve` answers each question using only the text and the tables:
- It parses the text, then solves the innermost level first and feeds each answer set into the next level.
- Gold intermediate answers are never used.
- The metadata supplies only the outer attribute filter.

On `df_dump_question` (800 questions):

| depth | exact set | recall | Jaccard |
| --- | --- | --- | --- |
| 1 | 82.5% | 0.995 | 0.919 |
| 2 | 75.5% | 0.995 | 0.920 |
| 3 | 79.5% | 0.965 | 0.917 |
| 4 | 83.0% | 0.967 | 0.930 |
| all | 80.1% | 0.981 | 0.922 |

- **Errors are supersets:** a few POIs just past a threshold, plus anchor names shared by several POIs.
- **No compounding with depth,** because every hop's relation has a tolerance of ≥ 100 m.
- **Cause not identified:** the generator's exact projection and tie-breaking are unknown. Its
  `(answers == rules)` agreement is 80%, so the build keeps only questions where the two agree.

## KGQA build (2026-11-06)

**Format.** The same JSONL as `data/graphs/*/hrm-text-grl`:
- `instruction` is the GRL prompt: "#### Question", the fact list, then `Q: <nested query_texte>\nA: `.
- `response` is the answer names, `|`-joined, so `eval_graphqa.py`'s set P/R/F1 applies.
- `condition` is `direct`; `task` is `<d>hop`.
- Metadata fields are added (`answers`, `answer_poi_ids`, `qid`, `prompt_tokens`, …). data_io reads columns by
  name, so it ignores them.

**Facts.** Each prompt is a closed world: every true fact between the described entities, and nothing else.
- **Facts are relations, not coordinates.** Each fact is a relation phrase worded as in the questions:
  `(Combat, at less than 300 meters from, Naam)`, `(La Bicyclette, to the west of, Les Buttes Chaumont)`.
  Answering then needs multi-hop graph traversal and set intersection, not geometry. This matches MetaQA's
  setup.
- **Described entities:**
  - the gold path: every answer and, at each inner hop, its nearest valid supporting POI;
  - per hop, up to 3 near-miss POIs (right attributes, fail exactly one clause);
  - per hop, 1 attribute-miss POI (all clauses hold, wrong attribute);
  - hops 2+: up to 2 **hop distractors**: everything holds except the hop relation, and the POI is linked
    to an *invalid* previous-hop POI, which is described too.
- **Why hop distractors:** without them, a shortcut that checks only the outermost hop and accepts any link
  of the right relation type scored 0.81 on depth-2–4 test questions. With them it scores 0.40–0.56.
  Superseded (2026-11-06): the hop distractors now come from the whole map and may link to any invalid POI.
  The shortcut scores 0.21–0.26 on test; see
  [Hop-distractor fix](#hop-distractor-fix-and-where-the-remaining-shortcuts-come-from-2026-11-06).

**Filters, in order.** 16% of the 174k candidates were kept:
- the outer filter is stated in the text;
- every anchor links to a uniquely named entity;
- 1–10 answers;
- the rules reproduce the gold set exactly;
- inner hops have ≤ 50 POIs;
- answer and path POI names are unique, case-insensitively;
- ≤ 200 facts and ≤ 3,000 prompt tokens.

The main losses: an ambiguous anchor (98.8k), an ambiguous answer name (16.3k), answer count (15.3k), and
rules disagreeing with gold (6.6k).

**Splits.**
- Grouped by the innermost sub-question. Roughly 1,000 innermost sub-questions are each reused by about 1,000
  deeper questions, so without grouping, a test question's inner hops would be train questions.
- The groups of the 800 `df_dump_question` questions are forced into test. Sampling is per (split, depth)
  quota.
- Command: `.venv/bin/python scripts/osm/build_kgqa.py --out data/osm/kgqa` (32 workers, ~4 min). Defaults:
  5,000 / 250 / 500 questions per depth.

**Result** (`check_kgqa.py`: PASS):

| split | rows | facts (mean) | prompt tokens mean / max | answers mean | 1 answer | outer-hop shortcut, depth 2 / 3 / 4 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 20,000 | 52.7 | 972 / 2,986 | 1.63 | 70.2% | 0.47 / 0.38 / 0.44 |
| val | 1,000 | 51.5 | 946 / 2,634 | 1.64 | 69.7% | 0.50 / 0.39 / 0.46 |
| test | 2,000 | 54.2 | 995 / 2,992 | 1.62 | 70.9% | 0.56 / 0.40 / 0.45 |

Superseded (2026-11-06): these are the first build's numbers; the data was rebuilt after the hop-distractor fix
(table in the last section).

- **No overlap:** no innermost sub-question or question is shared between splits.
- **Derivable:** a symbolic solver that reads only the fact lines and the question returns exactly `answers`
  for 23,000 of 23,000 rows.
- **Shortcut baseline:** depth 1 is 1.0 by construction, because the outer hop is the only hop.
- **Test includes eval questions:** 160 of the 2,000 test questions come from `df_dump_question`.

**Fixes found by the checker.**
- **Unstated attributes:** 0.4% of rows were unanswerable, because their outer filter (`pois`, seating =
  `garden` / `street`) is not in the text. Added the filter `attrs_not_in_text`.
- **Case-only name differences:** 9 rows had answer names that differ only in case ("Café NATA" / "Café
  Nata"). `eval_graphqa` lowercases answers, so those rows are ambiguous. Name uniqueness is now
  case-insensitive.

**Open.**
- No training run yet. Evaluation needs `--max_generation` ≥ the longest `|`-joined answer list, and
  `--max_tokens` ≥ 3,000 plus the generation length.
- Shortcut accuracy is still 0.40–0.56 at depths 2–4, because hop distractors cannot be found for every
  question. See the next section: this is a known limitation and is not filtered for now. Superseded
  (2026-11-06): 0.21–0.26 after the hop-distractor fix, and what remains is caused by the dataset (last
  section).

## Known limitation: about half the multi-hop rows don't need their inner hops (2026-11-06)

**Shortcut.** It checks only the outermost hop and accepts any link of the right relation type to the previous
hop, without checking that the linked entity is valid. When its answer equals the gold answer, the question can be
solved without the chain. `needs_inner_hops` = the shortcut is wrong, as computed by
[`check_kgqa.py`](../../../scripts/osm/check_kgqa.py) `solve_from_prompt(rec, shortcut=True)`.

| split | 2 hops | 3 hops | 4 hops |
| --- | --- | --- | --- |
| train | 2,660 / 5,000 (53%) | 3,124 / 5,000 (62%) | 2,822 / 5,000 (56%) |
| val | 124 / 250 (50%) | 153 / 250 (61%) | 135 / 250 (54%) |
| test | 218 / 500 (44%) | 298 / 500 (60%) | 275 / 500 (55%) |

**Why it happens.** When every other final-hop candidate already fails on its own attributes or named anchors,
the answer is the only entity left in the last hop. Example: test qid 639761, where the only other hotel lacks
"no outdoor seating".

Superseded (2026-11-06): these shares and the explanation are from the first build. The cause was split into a
builder part (fixed) and a dataset part (kept); qid 639761 now needs its inner hops. See the next section.

**Example that needs the chain.** Test qid 431362: Ambroise Paré satisfies its own hop, but its "to the west of"
link goes to an invalid ?x2. Both are drawn in [`notebooks/osm.ipynb`](../../../notebooks/osm.ipynb) §12.

**Decision (user, 2026-11-06).** The dataset is **not** filtered on this for now. Every score on
`data/osm/kgqa` should therefore be read with the limitation in mind: on 2–4-hop rows, a model can be right
without following the chain.

**Fixes considered, if needed later:**
- **Filter:** keep only `needs_inner_hops` rows at depths 2–4. The builder would scan about twice as many
  candidates to fill the same quotas.
- **Tag:** add a `needs_inner_hops` field and report test scores on all rows and on that subset.

## Hop-distractor fix and where the remaining shortcuts come from (2026-11-06)

**Two causes of the shortcut.** They are told apart on the **full map**, not the prompt
(`check_kgqa.shortcut_cause`). The question is whether some POI passes the last hop's own conditions
(attributes and named anchors) but fails the hop:
- **Dataset: hop redundant.** No such POI exists. The last hop's own conditions already single out the answers,
  so the "[sub-question]" part filters nothing, and no prompt can make the chain necessary.
  - Example: test qid 532314. Team Brothers is the only cafe near Zara food within 100 m of Café Wam.
  - It is already present in the source: a random sample of the unfiltered `df_multihop` has 11–14% of
    multi-hop questions with a redundant hop. The build's filters raise it to about 20%, because ≤ 10 answers
    favours tightly constrained final hops.
- **Builder: hop distractor left out.** Such a POI exists and can be linked, but it was not in the prompt.
  - In the first build this was 26% of the 2–4-hop test rows (389 / 1,500), against 21% (320) for the dataset
    cause.
  - Cause: a hop distractor was added only if it lay within 1.5 km of the path and could be linked to an
    invalid previous-hop POI **with that hop's attributes**.
  - Example: test qid 118082 left out Mama Shelter, a bar that fails only "within 100 m of ?x1".

**Fix** ([`build_kgqa.py`](../../../scripts/osm/build_kgqa.py)).
- Hop-distractor candidates come from the whole map: every POI that passes the hop's attributes and named
  clauses but fails the hop relation.
- Each is linked to an invalid POI in range of the hop relation. One with the previous hop's attributes is
  preferred; otherwise any category is used. The shortcut accepts any link, so any link is enough to defeat
  it.
- Up to 2 per hop.

**Residual: shared names.** Some hop distractors exist but have a name shared by several POIs
(case-insensitive), or can link only to such POIs. The builder never puts a shared name in a prompt, so they
cannot be added. Example: qid 118082, whose only hop distractor, Mama Shelter Paris La Défense, shares its name
with another POI. Fixable only by disambiguating names (e.g. appending the arrondissement).

**Result after the rebuild** (`check_kgqa.py`: PASS; split, quotas and filters unchanged; 23,000 / 23,000
answerable from the prompt):

| split | facts (mean) | prompt tokens mean / max | shortcut accuracy, depth 2 / 3 / 4 |
| --- | --- | --- | --- |
| train | 58.2 | 1,053 / 2,999 | 0.14 / 0.23 / 0.20 |
| val | 57.1 | 1,028 / 2,801 | 0.09 / 0.24 / 0.23 |
| test | 59.4 | 1,074 / 2,990 | 0.23 / 0.26 / 0.21 |

Why the shortcut still works, test rows with 2–4 hops (1,500):

| depth | needs inner hops | dataset: hop redundant | names shared | builder |
| --- | --- | --- | --- | --- |
| 2 | 383 | 110 | 7 | 0 |
| 3 | 372 | 118 | 10 | 0 |
| 4 | 397 | 94 | 9 | 0 |
| all | 1,152 (76.8%) | 322 (21.5%) | 26 (1.7%) | 0 |

In train, the dataset cause is 12.9% / 21.6% / 19.1% at depths 2 / 3 / 4, and shared names 0.7–1.4%.

**Decision (user, 2026-11-06).**
- Dataset-caused rows are kept; no filter, no tag.
- Scores on 2–4-hop rows include about 20% of questions that can be answered from the last hop alone.
- The notebook explains the causes with maps: [`notebooks/osm.ipynb`](../../../notebooks/osm.ipynb) §12–§14.

## Zero-shot transfer eval of the Mimir benchmark models (2026-11-09)

**Question.** Can the GRPO-tuned Mimir v1.5 (arm E3 of the
[Mimir graph benchmark](/pages/hrm-graph/mimir-graph-benchmark.md)) answer OSM questions zero-shot? OSM is a
downstream task outside the benchmark: no arm trains on it, and its relations (distances, directions, streets,
areas) appear in no training set.

- **Conversion:** [`graph_bench/sources/osm.py`](../../../graph_bench/sources/osm.py) writes
  `data/graphs/bench/osm/{val,test}.jsonl` as bench records.
  - The graph block is kept verbatim as `context`, not split into triples, because POI names contain commas. So
    the benchmark's anonymised-entity variant does not apply to OSM.
  - The rendered prompt equals the source `instruction` on every row (asserted). `answer_format` is `set`.
  - `meta.needs_inner_hops` is the outer-hop shortcut test of `check_kgqa.py`, used only to split scores. The
    source files stay untagged, as decided above.
- **Queue:** [`graph_bench/eval/run_osm.sh`](../../../graph_bench/eval/run_osm.sh) waits for E3's merged model,
  then evaluates, on all 2,000 test rows:
  - Mimir v1.5: E3, E2 (trace SFT) and the base model;
  - Qwen2.5-7B-Instruct and OLMo-3-7B-Instruct, zero-shot and after their own trace SFT (E2).
- **Settings:** those of E3's own eval (`named_t3k`): instruct prompt, thinking off, greedy, up to 3,072
  generated tokens, eager vLLM, Mimir context 4,096.
  - Longest OSM prompt in Mimir tokens: 3,002 (mean 1,091). Every row keeps at least ~1,090 generation tokens.
- **Outputs:**
  - Reports: `mimir_bench/<arm>/<model>/osm_t3k.json`, logged to W&B `Mimir-Graph-RLVR`, group
    `mimir-bench-<arm>`.
  - Breakdown: [`graph_bench/eval/osm_breakdown.py`](../../../graph_bench/eval/osm_breakdown.py) writes
    `mimir_bench/osm_breakdown.json` with set accuracy and set-F1 per depth.
  - 2–4-hop rows are also split into those that need the chain (1,152) and those the shortcut answers (348).
- **Comparisons:**
  - E3 − E2: what RL adds to transfer.
  - E2 − base: what the trace SFT adds.
  - The 7B models show whether Mimir's score is high or low for this task.
- **Status:** queued (`mimir_bench/osm_queue.log`); no results yet. If E3 fails instead of finishing, the queue
  waits indefinitely; then point it at the last E3 checkpoint. Superseded (2026-11-09): first results in
  [Zero-shot results: E3 and E2](#zero-shot-results-e3-and-e2-2026-11-09).

## Zero-shot results: E3 and E2 (2026-11-09)

All 2,000 test rows, greedy, settings above. GLM-5.3 (API, zero-shot, reasoning + answer ≤ 3,072 tokens, effort
low; see [GLM reference](/pages/hrm-graph/mimir-graph-benchmark-results.md#glm-53-at-3072-tokens-results-2026-11-09))
added as a frontier reference. Cells: set accuracy / set-F1.

| Model | All | 1 hop | 2 hops | 3 hops | 4 hops | 2–4 hops, chain (1,152) | 2–4 hops, shortcut (348) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mimir E2 (trace SFT) | 0.046 / 0.210 | 0.096 / 0.465 | 0.058 / 0.253 | 0.018 / 0.076 | 0.014 / 0.048 | 0.024 / 0.114 | 0.049 / 0.165 |
| Mimir E3 (GRPO from E2) | 0.100 / 0.383 | 0.128 / 0.558 | 0.128 / 0.448 | 0.084 / 0.318 | 0.058 / 0.209 | 0.083 / 0.311 | 0.112 / 0.370 |
| GLM-5.3 | **0.736 / 0.792** | 0.928 / 0.959 | 0.748 / 0.824 | 0.680 / 0.747 | 0.588 / 0.638 | 0.642 / 0.716 | 0.770 / 0.805 |

- **The benchmark-tuned Mimir does not transfer to OSM:** 0.100 set accuracy for E3. RL doubles E2's score and
  the gain grows with depth (4 hops 0.014 → 0.058), but both are far from usable.
- **How E3 fails** (read on samples): 13.5% of rows end in a repetition loop with no `Answer:` line; elsewhere it
  writes its KGQA program-style trace and returns an over-inclusive set (e.g. both "Gare du Nord (Métro)" and
  "Gare du Nord (gare routière)" where one was asked). Set-F1 (0.383) is well above accuracy for this reason.
- **The task itself is solvable from the prompt zero-shot:** GLM gets 0.736 with 17% of rows cut off by the cap
  (345 of 2,000), 0.93 at 1 hop and 0.59 at 4 hops. Shortcut rows are easier than chain rows for every model.
- Pending: base Mimir v1.5, Qwen2.5-7B and OLMo-3-7B (E0 and E2) in `run_osm.sh`; GLM `osm_t16k`. Superseded
  (2026-11-10): [all models](#zero-shot-results-all-models-2026-11-10). W&B: E3/E2 in
  groups `mimir-bench-e3` / `mimir-bench-e2`, GLM run `gcfqvsmo`.

## Zero-shot results: all models (2026-11-10)

All 2,000 test rows, same settings; GLM at both budgets. `mimir_bench/osm_breakdown.json` (rows named
`<arm>/<model>/<report>`). Cells: set accuracy / set-F1.

| Model | All | 1 hop | 2 hops | 3 hops | 4 hops | 2–4 chain (1,152) | 2–4 shortcut (348) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mimir v1.5 base (E0) | 0.003 / 0.021 | 0.008 / 0.057 | 0.002 / 0.020 | 0.000 / 0.005 | 0.000 / 0.001 | 0.001 / 0.009 | 0.000 / 0.006 |
| Mimir E2 | 0.046 / 0.210 | 0.096 / 0.465 | 0.058 / 0.253 | 0.018 / 0.076 | 0.014 / 0.048 | 0.024 / 0.114 | 0.049 / 0.165 |
| Mimir E3 | **0.100 / 0.383** | 0.128 / 0.558 | 0.128 / 0.448 | 0.084 / 0.318 | 0.058 / 0.209 | 0.083 / 0.311 | 0.112 / 0.370 |
| Qwen2.5-7B E0 | 0.090 / 0.278 | 0.078 / 0.365 | 0.118 / 0.311 | 0.106 / 0.261 | 0.058 / 0.178 | 0.082 / 0.230 | 0.132 / 0.316 |
| Qwen2.5-7B E2 | 0.029 / 0.210 | 0.046 / 0.396 | 0.030 / 0.231 | 0.016 / 0.120 | 0.024 / 0.094 | 0.012 / 0.144 | 0.060 / 0.165 |
| OLMo-3-7B E0 | 0.063 / 0.340 | 0.102 / 0.521 | 0.042 / 0.313 | 0.058 / 0.270 | 0.050 / 0.254 | 0.021 / 0.232 | 0.147 / 0.436 |
| OLMo-3-7B E2 | 0.046 / 0.220 | 0.076 / 0.447 | 0.052 / 0.233 | 0.032 / 0.129 | 0.024 / 0.072 | 0.024 / 0.121 | 0.075 / 0.225 |
| GLM-5.3, 3,072 tokens | 0.736 / 0.792 | 0.928 / 0.959 | 0.748 / 0.824 | 0.680 / 0.747 | 0.588 / 0.638 | 0.642 / 0.716 | 0.770 / 0.805 |
| GLM-5.3, 16,384 tokens | **0.857 / 0.863** | 0.980 / 0.987 | 0.910 / 0.913 | 0.802 / 0.813 | 0.736 / 0.737 | 0.784 / 0.791 | 0.922 / 0.922 |

- **Every ≤ 7B model fails OSM zero-shot (≤ 0.10 set accuracy).** E3 is the best of them, just above
  Qwen2.5-7B E0 (0.090); base Mimir v1.5 almost never produces the exact set.
- **Trace SFT (E2) hurts the 7B models on OSM** (Qwen2.5 0.090 → 0.029, OLMo-3 0.063 → 0.046) and helps Mimir only
  from near zero (0.003 → 0.046); GRPO then doubles Mimir's score. The benchmark training transfers format, not
  spatial reasoning.
- **GLM at 16k reaches 0.857** (13% of rows still cut off), 0.74 at 4 hops: the dataset is answerable from the
  prompt, and it is a hard zero-shot probe that separates frontier from ≤ 7B models.
- Answer to the question of this section: no, the GRPO-tuned Mimir is not usable zero-shot for this downstream
  geospatial task; it would need OSM training data.

## Build recipe and noising steps of the current build (2026-11-09)

This section summarises how the current `data/osm/kgqa` is built. It describes the build made after the
hop-distractor fix, with the `build_kgqa.py` defaults (`build_config.json`, seed 0). The sections above keep
the history.

**From the source tables to a sample.**

| # | step | what it does |
| --- | --- | --- |
| 1 | dedup | `df_multihop` on `query_texte`: 1,177,868 → 1,176,265 questions |
| 2 | split | md5 of the innermost sub-question: 10% of groups to test, 5% to val, 85% to train. Groups that contain a `df_dump_question` question are forced into test; they are large, so test holds 41% of the pool |
| 3 | sample | random order per (split, depth); test puts dump-eval questions first. Stops at 5,000 / 250 / 500 rows per depth 1–4 |
| 4 | filter | parse, link anchors by exact name, solve with the `osm_kg.py` rules, drop on any check below. 28,237 of 174,195 candidates pass (16%); 23,000 kept |
| 5 | gold path | final answers; then, for each POI at hop *i*, the **nearest** uniquely named POI of hop *i−1* for which the hop relation holds. One supporter per POI, not the whole inner answer set |
| 6 | noise | distractors per hop (next table) |
| 7 | facts | for every described POI of hop *i*: its `category`, plus the attributes hop *i* filters on (cuisine, seating); every named-anchor clause of hop *i* that holds for it; the hop relation to every described POI of hop *i−1* for which it holds. Closed world among the described entities; mean 58 facts |
| 8 | prompt | facts deduplicated, sorted, shuffled with a qid-seeded RNG; `#### Question` header + `(s, p, o)` lines + `Q: <question>\nA: `. Response: answer names `\|`-joined |

Filters of step 4, with the candidates each one removed (`build_stats.csv`):

| filter | dropped |
| --- | --- |
| a named POI anchor is shared by several POIs | 98,799 |
| an answer name is not unique (case-insensitive) | 16,345 |
| not 1–10 answers | 15,341 |
| rules do not reproduce the gold set exactly | 6,564 |
| an answer has no uniquely named supporter at an inner hop | 6,548 |
| an inner level has > 50 POIs | 1,145 |
| outer attribute filter not stated in the text | 854 |
| an anchor links to no entity | 271 |
| > 3,000 prompt tokens / > 200 facts | 78 / 13 |

**Noising steps (distractors), per hop.**
- **Never used as a distractor:** a POI that is valid at that hop, a POI whose name is shared by several POIs
  (case-insensitive), or a POI named like a question anchor.
- **How "fails a clause" is judged:** the hop clause is evaluated against the full previous answer set.

| distractor | hops | max per hop | candidate pool | passes | fails | purpose |
| --- | --- | --- | --- | --- | --- | --- |
| near-miss | all | 3 | ≤ 1.5 km of that hop's gold-path POIs | attributes and all clauses but one | exactly one clause | forces checking every clause |
| 2-clause miss (fill) | all, if the two rows above give < 4 and the hop has ≥ 2 clauses | up to 4 in total | same | attributes | two clauses | tops up rows with few near-misses |
| attribute miss | all | 1 | same | all clauses | an attribute (same category preferred, e.g. wrong cuisine) | forces checking attributes |
| hop distractor | 2+ | 2 | whole map | attributes and every named-anchor clause | the hop relation to every valid previous-hop POI | defeats the outer-hop shortcut |

- **How a hop distractor is linked:** each one gets a true hop-relation link to an **invalid** POI of the
  previous hop. A POI with the previous hop's attributes is preferred, otherwise any category.
- **That invalid POI is described too.** It enters the previous hop's described set, so it gets its own facts.

**Remaining shortcut on the whole dataset** (`check_kgqa.shortcut_cause` on all 23,000 rows; 1-hop rows cannot
have it):

| split | rows | unfixable: dataset | % of 2–4-hop rows | % of split | names shared | needs inner hops |
| --- | --- | --- | --- | --- | --- | --- |
| train | 20,000 | 2,681 | 17.9% | 13.4% | 159 | 12,160 (81.1%) |
| val | 1,000 | 134 | 17.9% | 13.4% | 6 | 610 (81.3%) |
| test | 2,000 | 322 | 21.5% | 16.1% | 26 | 1,152 (76.8%) |
| all | 23,000 | **3,137** | **18.2%** | **13.6%** | 191 (0.8%) | 13,922 (80.7%) |

- **What "unfixable: dataset" counts:** 3,136 hop-redundant rows plus 1 row where the hop removes only isolated
  POIs. No prompt can make the chain necessary for them.
- **By depth (all splits):** 13.5% at 2 hops, 21.8% at 3, 19.3% at 4.
- **`builder` cause:** 0 in every split.
