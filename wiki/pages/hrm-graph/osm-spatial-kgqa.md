---
type: Runbook
title: OSM Spatial Multi-Hop QA as KGQA
description: The OSM Paris multi-hop spatial question set (data/osm), evidence that it can be cast as KGQA (entity linking, relation rules recovered from the gold answers, re-answering from the tables), and the train/val/test KGQA build in data/osm/kgqa.
tags: [graphs, kgqa, osm, spatial, dataset]
status: stable
last_updated: 2026-11-09
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
- **Output:** `data/osm/kgqa/{train,val,test}.jsonl` (see its `README.md`). No model has been trained or
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
  waits indefinitely; then point it at the last E3 checkpoint.
