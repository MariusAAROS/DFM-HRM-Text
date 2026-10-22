---
type: Runbook
title: KG / Graph Pretraining Corpus (Tier 1+2)
description: Sources, download, triple store, holdout, decontamination, data_io formatting and verified sizes of the knowledge-graph and graph-reasoning corpora prepared for Mimir pretraining.
tags: [graphs, knowledge-graphs, pretraining, data, wikidata, yago, data_io, mimir]
status: stable
last_updated: 2026-10-22
confidence: high
---
# KG / Graph Pretraining Corpus

This corpus is the first step toward improving Mimir pretraining on graph data. It contains the
Tier 1 and Tier 2 sources of `corpus_graphes_llm.csv`, which the user supplied on 2026-10-21. They
were fetched on 2026-10-22 and formatted as data_io rows. Mixing, tokenization at scale and training
are not done yet.

- Code: `scripts/kg_pretrain/`.
- Data: `data/kg_pretrain/` (gitignored).
- Everything here was checked against direct tool output. Confidence is high unless a line says
  otherwise.

## Scope decisions

- **Included:** Tier 1 and Tier 2 of the CSV, except DBpedia, which is redundant with Wikidata
  (~20 Md tokens).
- **Skipped:** all of Tier 3, and the KGQA evaluation sets (WebQSP, CWQ, GrailQA, FB15k/WN18RR), so
  they cannot leak into training.
- **GraphPile:** no public dump exists. To get it, contact the authors (HKUST-DSAIL).
- **Language:** English labels only.
- **Labelled KGs** are stored twice:
  - as a structured triple store, with a holdout split fixed before any training view was written;
  - as entity-centric plain documents: `condition=direct`, `instruction=""`, response = document.
- **QA-native sources** keep their question → answer shape: GraphQA generator (`direct`) and
  GraphInstruct (`cot`).
- **Plain rows are meant to be a capped slice of the mix.** The repo trains PrefixLM, and the
  [continuation-data policy](/pages/data-mix-policy/continuation-data.md) caps continuation at
  5–10% (DFM2 used 1X plain to 5X tasks). Most KG tokens should come from phase-2 KG tasks
  (see below).

## Layout

```
data/kg_pretrain/
  raw/<source>/          downloads + SOURCE.json (urls, date, sha256 per file, license)
  kg/<source>/           entities.parquet, relations.parquet, triples-NNNNN.parquet, holdout.json
  canonical/<source>/    {"id","source","text","meta"} gz JSONL (plus "instruction"/"condition" for QA)
  converted/kg-<source>/ {"condition","instruction","response"} gz JSONL  <- data_io input root
  stats/                 per-source *_store.json, kg-*.json (tokens), summary.md, verify_*.json, decontam.json
  work/wikidata/pass1/   61 GB pass-1 intermediate (only needed to re-run Wikidata pass 2)
```

**Triple schema.** Triples are `s, r, o | o_literal, split`.
- Exactly one of `o` (an entity id) and `o_literal` (a rendered value) is set.
- Shards are partitioned by subject, so all facts of a subject sit in one shard.

**data_io.** The tokenizer folder name for each converted file is `kg-<source>__<file>`, so the
sampler prefix rule is `kg-<source>__`.

## Pipeline

`bash scripts/kg_pretrain/run_all.sh` runs every step in dependency order, and `STEPS="..."` picks a
subset. Run it detached with `setsid nohup ... &`.

| Step | Script | Notes |
|---|---|---|
| fetch | `fetch.sh <source\|all>` | One empty dir per source. SOURCE.json records sha256 for every file. |
| small KGs | `convert_{conceptnet,atomic2020,hetionet,primekg}.py` → `holdout.py` → `render_docs.py` | |
| QA | `convert_graphinstruct.py`, `gen_graphqa.py` | |
| Wikidata5M | `convert_wikidata5m.py` | Must run before YAGO and Wikidata: it writes `benchmark_triples.tsv` |
| YAGO | `yago.sh` | |
| Wikidata | `wikidata.sh` (pass 1, pass 2, holdout, render) | See below |
| relabel | `convert_wikidata5m.py --relabel` | Needs `kg/wikidata`: Wikidata5M's first-alias labels are unreliable ("united stated") |
| T-REx, KELM | `convert_trex.py`, `convert_kelm.py` | Need `kg/wikidata` labels |
| checks | `decontam.py`, `verify.py`, `stats.py`, `smoke_tokenize.sh` | |

### Rendering

`render_docs.py` turns one subject's train triples into one document:

```
Copenhagen
capital city of Denmark
- instance of: city; capital
- country: Denmark
```

- **Relation order:** relations given in `--first` come first, then the rest by global frequency.
- **Values:** repeated values are deduplicated.
- **Long documents:** a document over 6,000 chars is split on fact boundaries, and the header
  repeats in every chunk.
- **Options:**
  - `--inverse` also renders incoming edges (Hetionet, "bound by");
  - `--types-line` adds the node types (Hetionet, PrimeKG);
  - `--skip-relations` and `--min-facts` are used for ConceptNet.

### Holdout (`holdout.py`)

- **Hash mode:**
  - 0.1% of triples go to `valid` and 0.1% to `test`.
  - Only entity-object triples are candidates. The hash is seeded on `(min(s,o), r, max(s,o))`, where
    `r` drops any `|direction` suffix, so both directions of an edge always share a split.
  - Both entities must keep degree ≥ 3.
- **Official splits:** used for ATOMIC-2020 (train / dev / test) and Wikidata5M (transductive).
- **`--force-test`:** the 10,291 Wikidata5M valid/test triples are forced to `test` in the Wikidata
  store. In YAGO they are mapped through `owl:sameAs` and `ys:fromProperty` first (5,297 mapped).
- **Training views** read only `split == train`.

### Wikidata (`wikidata.sh`)

- **Input:** `wikidata-20261014-truthy-BETA.nt.bz2` (43.55 GB), which matches the official sha1 in
  `wikidata-20261014-sha1sums.txt`.
- **Pass 1** (36 min; 61 GB of TSV output):
  - lbzip2 decompresses, `fanout.py` hands 8 MB blocks to 48 workers, and each worker runs
    `grep | wikidata_pass1.py` (`wikidata_pass1_filter.sh`);
  - the workers write subject-sharded TSVs: 256 shards by Q number mod 256.
- **Pass 2** (`wikidata_pass2.py`, ~40 min, nearly all of it single-process preparation):
  - **Entities:** items with an English label, minus `wikidata_blocklist.json` classes. The blocklist
    covers scholarly articles and other article types, Wikimedia pages, genes, proteins, chemical
    entities, stars, galaxies and asteroids.
  - **Relations kept:** WikibaseItem, Time, Monolingualtext (`@en`), Quantity without an
    allowed-units constraint, and 7 allowlisted String properties.
  - **Literals:** dates are rendered, with `-01-01` read as year-only because truthy dumps carry no
    precision.
- **Throughput limits found while tuning:**
  - one `grep` caps the stream at ~450 MB/s;
  - `split -n r/N` writes line by line and caps it at ~500 MB/s;
  - lbzip2 on this single-stream bz2 reaches ~540 MB/s of output (~18 MB/s of compressed input)
    and was the final bottleneck. indexed_bzip2 was slower (370 MB/s).
- **Two pyarrow features failed at full scale** with an abort. Both were replaced:
  - `group_by(...).aggregate(list)` over 125M rows aborted with `std::length_error`; it is now a numpy
    sort plus run boundaries;
  - the per-shard dedupe now uses pandas `drop_duplicates`.

### Download notes

- **Wikidata (most of the file came from mirrors):**
  - dumps.wikimedia.org serves ~5 MB/s per connection.
  - After the first 12.85 GB, the rest came in parallel byte ranges from the Wikimedia mirrors
    mirror.accum.se (Umeå ACC) and dumps.wikimedia.your.org, using `fetch_ranges.sh` (chunked,
    multi-mirror, resumable).
  - The mirrors' speed varied: accum.se went from 122 MB/s down to 2 MB/s, and your.org gave
    ~24 MB/s per connection.
  - The whole file is verified by sha1.
- **ATOMIC-2020:** downloaded from its Google Drive link. The AI2 S3 URL returns 404.
- **Wikidata property tables:** `properties.tsv` (datatypes and English labels) and
  `property_units.tsv` (allowed units) are SPARQL exports from query.wikidata.org.

## Sources, licenses and verified sizes

Token counts are estimated from a 10k-row sample of each converted folder. "Gemma" is the
DFM-Mimir v1/v1.5 tokenizer, and the CSV column is the user's estimate.

| Folder | Source (license) | Rows | Gemma tokens | HRM BPE tokens | CSV estimate |
|---|---|---:|---:|---:|---:|
| kg-wikidata | Wikidata truthy 2026-10-14 (CC0) | 39,411,031 | 2,678M | 2,848M | 3,000M |
| kg-yago | YAGO 4.6 facts + beyond-Wikipedia + taxonomy (CC BY 4.0) | 29,274,923 | 1,372M | 1,440M | 2,004M |
| kg-kelm | KELM generated corpus (CC BY-SA 2.0, **to verify**) | 590,153 | 919M | 989M | 900M |
| kg-trex | T-REx (CC BY 4.0) | 4,650,000 | 748M | 842M | 450M |
| kg-wikidata5m | Wikidata5M (CC BY 4.0) | 4,568,596 | 713M | 803M | 750M |
| kg-graphqa_gen | google-research/graphqa generator (Apache 2.0) | 852,627 | 262M | 254M | – |
| kg-graphinstruct | GraphWiz GraphInstruct + RFT-72K (RFT card: Apache 2.0; base card: **none stated**) | 71,658 | 47M | 51M | 87M |
| kg-primekg | PrimeKG (CC0) | 139,530 | 41M | 44M | 49M |
| kg-hetionet | Hetionet v1.0 (CC0; some integrated sources have their own terms) | 48,826 | 21M | 22M | 27M |
| kg-conceptnet | ConceptNet 5.7 (CC BY-SA 4.0, **share-alike**) | 440,469 | 11M | 12M | 340M |
| kg-atomic2020 | ATOMIC-2020 (CC BY 4.0) | 36,987 | 6M | 6M | 20M |
| **total** | | 80,084,800 | **6,819M** | 7,311M | |

Disk usage:
- raw downloads: 63 GB;
- triple stores: 7.5 GB;
- canonical documents: 7.1 GB;
- converted rows: 6.1 GB;
- Wikidata pass-1 intermediate: 61 GB.

### Per-source notes

- **Wikidata:**
  - 93.2M English-labelled items, of which 67.3M are blocked (46.0M scholarly articles, 5.8M
    categories, 3.3M stars, 2.1M galaxies). That leaves 39.7M entities.
  - 1.68B fact lines in, 260.6M triples out (230M entity links, 19M dates, 7.0M counts,
    4.3M English texts, 2.8M strings).
  - Split: 260,109,215 train / 224,298 valid / 233,444 test, with 9,296 Wikidata5M benchmark
    triples forced to `test`.
  - Large classes that remain and could be blocked next: painting (1.08M), edition (0.85M),
    street (0.73M), collection (0.50M), US patent (0.32M).
- **YAGO 4.6:**
  - 126.9M triples, 29.3M entities and 146 relations.
  - Quantities keep their units (`31112 United States dollar`).
  - The description is `rdfs:comment @en`.
  - Some class labels carry YAGO disambiguation (`Female (gender)`).
  - 29.7M beyond-Wikipedia facts were dropped because their subject has no English label.
- **KELM:**
  - 1,319,411 sentences dropped for the generator's `⁇` artifact, and 4,455 because they contain a
    Wikidata5M benchmark triple (matched on labels).
  - Fact values are detokenized (`clean_value`).
  - **The generated sentences sometimes contradict their own facts** (e.g. "home to 2160 people" next
    to "population 2116"), so give KELM a low mix weight.
  - Year-precision dates still read `01 January 1966`.
- **T-REx:**
  - Wikipedia abstract plus `Facts:` lines. Entities use their Wikidata labels, with the surface form
    as fallback.
  - 14.8M facts kept. 9,969 of the 10,291 Wikidata5M benchmark triples appeared in T-REx and were
    removed from the fact lines; the abstracts may still state them.
  - `kg/trex/aligned-*.parquet` and `kg/kelm/aligned.parquet` keep (sentence, triples) pairs for
    phase 2.
- **Wikidata5M:**
  - Relabelled with Wikidata labels: 3.3M of 4.8M labels changed, and 813 of 819 relations got
    property labels.
  - The header is the Wikipedia first paragraph.
- **GraphQA generator:**
  - The generator's task and encoder classes are called directly, without TensorFlow.
  - 7 algorithms × 12 tasks × 9 encoders, using seeds different from the benchmark's
    (9876 / 5432 / 1234).
  - 32% of generated examples are exact duplicates, and a larger pool does not help: complete, star
    and path graphs give only ~15 distinct graphs each.
- **GraphInstruct:** the union is deduplicated on (query, answer): 71,658 rows over 24,311 unique
  queries.
- **ConceptNet:**
  - English↔English edges only, weight ≥ 1.
  - `FormOf` is not rendered, and subjects with fewer than 2 facts are skipped.
  - This explains the 11M tokens against the CSV's 340M, which counted all languages.
- **ATOMIC-2020:** relations use COMET-ATOMIC phrasing.
- **Hetionet:** documents include incoming edges.
- **PrimeKG:** relations are directional (`indication|drug|disease` → "indicated for").

## Redundancy (for the mix design)

Five sources are derived from Wikidata:
- Wikidata;
- YAGO (a cleaned version of Wikidata);
- Wikidata5M (a 2019 subset with Wikipedia text);
- KELM (Wikidata facts verbalized);
- T-REx (Wikipedia text aligned to Wikidata triples).

Recommendation (medium confidence; a design judgment, not measured):
- Use **either YAGO or Wikidata** as the main KG slice. YAGO is cleaner, with typed facts and units.
  Wikidata is about 2× larger and newer.
- Keep Wikidata5M mainly for its benchmark splits.
- Weight KELM low.

Measuring the actual triple overlap is still to do.

## Verification (2026-10-22)

- **Triple stores** (`verify.py`), for all 7 stores:
  - every id resolves;
  - the split counts match `holdout.json`;
  - 0 leaks among up to 10k sampled held-out triples. A held-out object never appears on the subject's
    relation line; the 1–23 hits per store are explained by a train fact with the same label.
- **Benchmark triples:** every Wikidata5M benchmark triple present in Wikidata (9,296) and in YAGO
  (4,487) is `test`.
- **Decontamination** (`decontam.py`) against 13,509 val/test prompts from 32 files under
  `data/graphs`: 439 GraphQA-generator rows removed, 0 GraphInstruct rows.
- **Smoke tokenization** (`smoke_tokenize.sh`, one shard per folder):
  - HRM path (Rust tokenizer + `sample_tokenized.py`): every row tokenized, and the sampler keeps all
    867,423 rows;
  - Mimir path (`tokenize_chat_template.py` with the Gemma template): OK.
  - Max row: 4,747 tokens (GraphInstruct).
  - A plain row becomes an empty user turn plus a model turn in the Gemma format.

## Phase 2 (not built): tasks from the triple store

- **Link prediction:**
  - closed-book (`Copenhagen | country | ?`);
  - in context (the k-hop neighbourhood, minus the target fact).
- **Multi-hop path queries:** 2- and 3-hop, as in MetaQA.
- **Triple classification:** negatives drawn by type.
- **Node and type prediction.**
- **KELM / T-REx:** sentence ↔ triples, from the `aligned` tables.

The held-out splits become their evaluation sets. Each task gets its own `converted/kg-<source>-<task>/`
folder and prefix rule.
