---
type: Runbook
title: Graph-Structure Injection on GraphQA, MetaQA and KQA Pro (graph_inject)
description: Standalone package that fine-tunes HRM-Text-1B, DFM-Mimir v1 and v1.5 on GraphQA, MetaQA and KQA Pro with hard-prompt, soft-prompt, absolute-PE and relative-attention-bias graph injection under one LoRA trainer; design, verification, run matrix and results.
tags: [graphs, graphqa, metaqa, kqapro, lora, soft-prompts, positional-encoding, attention-bias, hrm-text-1b, mimir]
status: stable
last_updated: 2026-10-23
confidence: high
---
# Graph-Structure Injection on GraphQA, MetaQA and KQA Pro

The question is which way of injecting graph structure helps the pretrained models (HRM-Text-1B, DFM-Mimir v1,
DFM-Mimir v1.5) on GraphQA at small scale. The method families come from the user's survey of graph → LLM
techniques: hard prompting, soft prompting, and embedding-level injection.

- Code: [`graph_inject/`](../../../graph_inject/README.md). It is standalone: `pretrain.py`, `models/` and the eval
  scripts are untouched and reused by import.
- Tests: `tests/test_graph_inject.py`.
- Queue: `scripts/graph_inject/run_matrix.sh`.
- W&B: project **`Graph-Positional-Encoding-Exploration`**, groups `graphqa-inject`, `metaqa-inject` and
  `kqapro-inject`. Smoke and bench runs are logged offline (not synced) since 2026-10-23.

## Design (2026-10-22)

**Trainer.**
- `graph_inject/train.py` runs on one GPU, without FSDP or compile.
- It loads the converted HF dirs through `inference_load_checkpoint`.
- Default regime: LoRA (r=16, α=32 on all H/L projections, lr 2e-4), with the base model frozen in bf16. The
  method's own modules train at lr 1e-3.
- Clip 1.0, 8192 tokens/step, H2L3, bp_max_steps 5, 10 epochs.
- Every epoch generates greedily on val; test runs at the best val `tf_em` epoch.
- `fullft-text` repeats the `pretrain.py` full-FT recipe as a check on the trainer.

**Methods.**

| Method | Family | Implementation |
| --- | --- | --- |
| `text` | hard prompting | Less is More: GraphQA text + LoRA |
| `text-incident` | hard prompting | Talk like a Graph incident encoding |
| `text-colors` | hard prompting | Colorful Talks C-OWL |
| `prompt-tuning` | static soft prompt | peft PromptEncoder; capacity control |
| `graphtoken`, `graphtoken-frozen` | graph-level soft tokens | PyG GIN |
| `node-tokens` | node-level soft tokens | LLaGA hop-field template, ported from VITA-Group/LLaGA |
| `abs-pe` | absolute additive PE | φ(c(i)) shared by all tokens of a node or edge; the proposed method |
| `abs-id` | absolute additive PE | random-identity control for `abs-pe` |

The relative attention bias (GTLM / Graphormer) is phase 2, below.

**Data.**
- The graph is parsed from the prompt text, with the parser ported from GRL branch `graph-enc` (a3d3418).
- Prompt and response ids come from `InferenceCheckpoint.tokenize_prompt`. Character offsets come from a
  second fast-tokenizer pass, and the two are asserted equal.
- Soft-token slots go right before the prompt content.
- Packed batches have the same layout as `dataset_new._load_batch`.

## Verification (2026-10-22, `tests/test_graph_inject.py`, 21 passed)

- **Parser.** All 3,850 GraphQA rows parse, and node and edge counts equal `nnodes`/`nedges` in
  `baseline/standard/*.json`. Graphs have 5-19 nodes and up to 171 edges.
- **Token ids.** Prompt and response ids equal the data_io arrays in `data/graphs/tokenized/` for all 3,080 train
  rows: HRM-Text-1B (`graphqa_grl`), and Mimir v1 and v1.5 (`graphqa_mimir`). Every token aligned to node v
  decodes to digits of v or a space.
- **Zero-init identity.** `text` with LoRA (B=0), `abs-pe` and `abs-id` give logits bitwise equal to the
  pretrained LMHead.
- **Gradients.** Every trainable tensor of every method receives a nonzero gradient, except the ones behind a
  zero-init matrix. With `inject_x=False` the injection enters only through the initial `z_H`, and the gradient
  still reaches it under bp_max_steps 5.
- **Train vs inference.** Packed varlen training logits and KV-cache prefill logits have the same argmax on each
  prompt's last token (`text`, `graphtoken`).
- **Generation.** `graph_inject.generate` reproduces `simple_inference_engine.inference_generate` token for token
  on 24 val prompts.

## Speed and memory (RTX PRO 6000, eager)

- HRM-Text-1B, `text` + LoRA: about 0.95 it/s with a 53 GB peak. Generating 64 val samples takes about 20 s.
- Steps per epoch at 8192 tokens: adjacency 113-116, incident 141-143, colors 155. Mean sample length is 292,
  360 and 399 tokens.
- Estimate: about 45 min per 10-epoch run, including val generation every epoch (confidence: medium).

## Caveats

- MaximumFlow capacities are not in the GraphQA text, so that task is unanswerable. It is 34/385 test samples,
  with about 0.18 accuracy for every model. Reports also give `greedy_acc_answerable`, which excludes it.
- Splits are by question: every test graph also appears in train. A graph encoder can memorize graphs, so the
  `abs-id` and `prompt-tuning` controls are needed.
- Prior result: GRL `graph-enc` gated node and WL embeddings under full FT gave no gain (0.901 vs 0.897, control
  0.896). The gate stayed at about 0.002.

## Results, phase 1 (seed 0, 2026-10-22/23)

Ran 2026-10-22 17:12 to 2026-10-23 09:40: 30 runs, about 33 min each. The table gives GraphQA test `greedy_acc`
(385 samples) at the best val `tf_em` epoch. One sample is 0.26 points, and the binomial SE is about 1.9 points.
The mean column also gives the mean without MaximumFlow. Δ is the pooled paired difference against `text` over
the 3 models (samples won / lost, exact McNemar p-value).

| Method | HRM-1B | Mimir v1 | Mimir v1.5 | Mean (w/o MaxFlow) | Pooled vs `text` |
| --- | ---: | ---: | ---: | ---: | --- |
| `text` (LoRA, no graph module) | 0.829 | **0.857** | 0.842 | **0.842** (0.909) | reference |
| `text-colors` | 0.831 | 0.855 | 0.847 | **0.844** (0.906) | +24 / -22, p=0.88 |
| `node-tokens` | 0.821 | 0.849 | **0.855** | **0.842** (0.905) | +23 / -24, p=1.00 |
| `abs-id` (control) | 0.826 | 0.836 | **0.855** | 0.839 (0.901) | +24 / -28, p=0.68 |
| `fullft-text` | **0.847** | 0.839 | 0.829 | 0.838 (0.900) | +29 / -34, p=0.62 |
| `text-incident` | 0.834 | 0.842 | 0.831 | 0.835 (0.896) | +25 / -33, p=0.36 |
| `graphtoken` | 0.813 | 0.849 | 0.844 | 0.835 (0.901) | +20 / -28, p=0.31 |
| `abs-pe` | 0.813 | 0.836 | 0.852 | 0.834 (0.898) | +19 / -29, p=0.19 |
| `prompt-tuning` (control) | 0.810 | 0.831 | 0.842 | 0.828 (0.893) | +19 / -36, p=0.03 |
| `graphtoken-frozen` | 0.709 | 0.787 | 0.805 | 0.767 (0.830) | +16 / -103, p<0.001 |

- **No injection method beats the no-PE `text` baseline.** Every LoRA method is within about 2 points of it, both
  per model and on average. All pooled paired tests are non-significant, except two losses: frozen GraphToken
  (-7.5 points) and prompt tuning (-1.4 points, p=0.03).
- **`abs-pe` vs its identity control `abs-id`:** 23 wins and 29 losses pooled (p=0.49), so structure adds
  nothing over node identity.
- **φ is used; it just does not help.** At the best epoch, the RMS of φ on node tokens is 0.30 (HRM-1B),
  0.25 (Mimir v1) and 0.17 (Mimir v1.5), against a token-embedding RMS of 1.02, 0.99 and 0.43. That is unlike
  GRL `graph-enc`, whose gate stayed at about 0.002. `abs-id` learns a larger φ than `abs-pe` on HRM-1B
  (0.78 vs 0.30).
- **Freezing the LLM costs 4-12 points.** GraphToken needs the LLM adapted, as in the graph pooling tokens paper.
- **The trainer check passes.** `fullft-text` reaches 0.847 / 0.839 / 0.829, against `pretrain.py` clip-1.0 runs
  of 0.824 / 0.843 / 0.818. LoRA `text` is as good as full FT on average (0.842 vs 0.838).
- **The remaining errors are counting tasks, not structure lookup.** Per task:
  - CycleCheck, EdgeExistence, NodeCount and Reachability are at 1.00 for every method, and ShortestPath,
    ConnectedNodes and DisconnectedNodes are at 0.94-1.00.
  - The errors are in TriangleCounting (0.42-0.60), EdgeCount (0.42-0.73) and the unanswerable MaximumFlow
    (0.06-0.26).
  - No method moves TriangleCounting.
  - `text-incident` hurts EdgeCount (0.42-0.50 vs 0.58-0.69): the incident list states each edge twice.
- **Interpretation (confidence: medium).** On GraphQA at this scale, the adjacency text plus LoRA already
  exposes the structure the tasks need. The ceiling is arithmetic over the graph (counting), which none of the
  injected encodings provide.
  - The benchmark is also saturated: the answerable tasks are at 0.90-0.92.
  - Test graphs also appear in train.
  - Only one seed was run.

## Phase 1 on MetaQA and KQA Pro, and phase 2 (design, 2026-10-23)

GraphQA cannot separate the methods (above), so the same matrix runs on the two knowledge-graph datasets, which
are far from saturated. Full-FT `pretrain.py` baselines (test greedy_acc, [graph reasoning
experiments](graph-reasoning-experiments.md)):

| | HRM-1B | Mimir v1 | Mimir v1.5 |
| --- | ---: | ---: | ---: |
| MetaQA | 0.836 | 0.837 | 0.650 (zero-shot 0.761) |
| KQA Pro | 0.632 | 0.698 | 0.600 |

**Data.** GRL's gold-subgraph prompts, `data/graphs/{metaqa,kqapro}/hrm-text-grl/standard`.
- MetaQA: 7,967 / 796 / 1,197 rows; tasks 1hop, 2hop, 3hop. About 72 facts, 78 nodes per prompt.
- KQA Pro: 6,824 / 760 / 1,134 rows; tasks Basic, Qualifier, Comparison, Count, Verify, Logical, Multihop. About 57
  facts and 86 nodes; 38% of facts carry qualifiers; 4% of prompts have no facts.
- Prompts: 944 (MetaQA) and 1,309 (KQA Pro) HRM tokens on average, at most 1,324 and 2,996.
- 3 epochs (as the full-FT configs, which peaked at epoch 1-2); KQA Pro eval batch 8 (KV cache).
- **Validation subset (differs from GraphQA and from the `pretrain.py` baselines).** The best epoch is selected on a
  fixed 400-sample val subset, not the full val split (796 MetaQA, 760 KQA Pro). The subset is the same for every
  run, method, model and epoch, which makes the epoch choice a paired comparison. Drawing a new subset per epoch would
  add about ±2 points of sampling noise per epoch, comparable to the gap between epochs.
  - The draw is seeded (seed 0, not the run seed) and stratified by task, with largest-remainder rounding.
    MetaQA: 133 / 133 / 134 for 1/2/3-hop. KQA Pro: Basic 100, Comparison 86, Qualifier 85, Verify 50, Count 43,
    Logical 27, Multihop 9.
  - Config: `eval.val_subset: 400` in `configs/datasets/{metaqa,kqapro}.yaml`; code:
    `data.stratified_subset`; test: `test_val_subset_fixed_and_stratified`.
  - Why: greedy generation costs about 0.8 s per KG sample, so full val would take about 11 min per epoch. The
    subset saves about 16 h over the matrix.
  - **Test is always the full split** (1,197 MetaQA, 1,134 KQA Pro), so reported numbers are unaffected. Only the
    epoch choice can differ from a full-val selection. The `pretrain.py` full-FT baselines selected on full val.

**Parsing** (`graph_inject/kg_structure.py`).
- A node is a distinct entity or value string; an edge is a fact (s, o); each qualifier `k: v` is an extra edge
  (s, v). Hyper-relational facts therefore hang off the subject (a simplification).
- 1.3% (MetaQA) and 2.1% (KQA Pro) of facts are ambiguous when split on `", "`, because entity names and relations
  contain commas. The split is resolved against the KB vocabulary (`kb.txt`, `kb.json`, longest relation wins).
- All 18,678 rows parse. Each fact re-serializes to its exact line, and the fact count equals `nedges` in the raw
  JSON.
- Entity names on the `Q:` line are node mentions: 99.9% of MetaQA questions and 82% of KQA Pro ones have one (KQA
  Pro often describes the entity instead of naming it).

**Method adaptations.**
- `text-incident`: facts grouped by subject, objects of one relation merged (ours; Talk like a Graph has no KG
  encoder).
- `text-colors`: `Color(<entity>):c` from 1-WL on the undirected entity graph. It adds about 50% to the prompt.
- `node-tokens`: the entity name's mean frozen token embedding replaces the digits of v. Adds 3 soft tokens per
  node (about 230-300 per prompt).
- Every other method is unchanged.

**Phase 2: relative attention bias** (`graph_inject/attention_bias.py`, `methods/relative_bias.py`).
- `rel-bias`: a learned β[layer, head, bucket], zero-init, added to the score of every pair of graph-element
  tokens. Buckets are Levi-graph distances 0..8 (0 = same element; adjacent nodes are at 2), >8, and unreachable.
  Pairs where either token has no element get 0.
- `rel-bias-elem`: buckets same element / other element only. This is the identity control.
- Implementation: every `Attention` is switched to `GraphBiasAttention` by `__class__`; `models/` is not edited.
  - Training uses FlexAttention with the packed document + PrefixLM mask.
  - The prompt prefill adds the bias to the SDPA float mask.
  - Decode steps and other methods run the parent class.
- **The bias gradient is the trap.** Indexing a [12, 12] table inside `score_mod` makes the backward ~17x slower than
  varlen (48 ms vs 2.9 ms per call, atomic adds on a few hundred entries). Gathering it per layer into
  [H, NDOC, EL, EL] first brings that to 7.3 ms (probe: T = 8192, 12 x 128, bf16, RTX PRO 6000).

**Verification** (`tests/test_graph_inject.py`, 65 passed).
- KG parse and counts; token ids equal data_io for HRM and Gemma on 1,000 train rows per dataset; node tokens decode
  to parts of the entity name.
- Gradients reach every trainable tensor for all 10 methods on all 3 datasets, including a KQA Pro batch with an
  empty graph.
- Rel-bias:
  - Levi buckets on a hand-built graph.
  - At β = 0, FlexAttention gives the varlen logits (argmax agreement > 0.98).
  - At random β, packed training logits and KV prefill logits differ no more than at β = 0 (bf16 kernel noise),
    while β moves the logits more than 5x that.
  - Generation at β = 0 equals `text`.

- **LapPE on graphs with ≥ 100 nodes.** PyG switches to ARPACK (`eigsh`) there. ARPACK starts from a random vector
  and failed on some disconnected KQA Pro subgraphs (ARPACK error 3). `features.laplacian_pe` now forces PyG's
  dense `eigh` path at every size, so below 100 nodes (all of GraphQA) the features are unchanged. All 14,791 KG
  train graphs pass.
- Smoke: every method on MetaQA and KQA Pro, plus rel-bias on GraphQA (HRM-1B, 24-48 rows, 1 epoch), all pass.

**Speed and memory** (bench, HRM-1B unless stated, 8192 tokens/step, eager).

| | text | rel-bias | node-tokens | peak memory |
| --- | --- | --- | --- | --- |
| MetaQA | 1.14 s/step | 1.9 s/step | 0.95 s/step | 47-49 GB |
| KQA Pro | 0.97 s/step | 2.5 s/step | | 47-49 GB |

- These timings include warmup and the FlexAttention compile (several padded shapes). Steady-state rel-bias is about
  1.8-2x text on KG, more than the +35% of the single-layer probe, because KG element tables are larger (EL up to 512).
- Mimir v1.5 on KQA Pro: rel-bias peaks at 50 GB, and full FT at 72.5 GB.
- Greedy generation takes about 0.8 s per KG sample.
- Estimate (confidence: medium):
  - about 1.4-1.5 h per phase-1 KG run;
  - about 2.4 h per KG rel-bias run;
  - about 1 h per GraphQA rel-bias run;
  - about 120 h (5 days) for the whole queue.

**Run matrix** (seed 0). Queue started 2026-10-23 14:10; log `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject/queue_kg.log`. `run_matrix.sh` in priority order: text, abs-pe, abs-id, rel-bias, rel-bias-elem,
node-tokens, graphtoken, prompt-tuning, text-incident, text-colors, graphtoken-frozen, fullft-text, over MetaQA,
KQA Pro and GraphQA (GraphQA phase-1 runs are skipped as done). That is 60 phase-1 KG runs plus 18 phase-2 runs.
