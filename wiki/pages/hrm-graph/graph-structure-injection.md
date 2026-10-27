---
type: Runbook
title: Graph-Structure Injection on GraphQA, MetaQA and KQA Pro (graph_inject)
description: Standalone package that fine-tunes HRM-Text-1B, DFM-Mimir v1 and v1.5 on GraphQA, MetaQA and KQA Pro with hard-prompt, soft-prompt, absolute-PE and relative-attention-bias graph injection under one LoRA trainer; design, verification, run matrix and results.
tags: [graphs, graphqa, metaqa, kqapro, lora, soft-prompts, positional-encoding, attention-bias, hrm-text-1b, mimir]
status: stable
last_updated: 2026-10-27
confidence: high
---
# Graph-Structure Injection on GraphQA, MetaQA and KQA Pro

The question is which way of injecting graph structure helps the pretrained models (HRM-Text-1B, DFM-Mimir v1,
DFM-Mimir v1.5) on GraphQA at small scale. The method families come from the user's survey of graph → LLM
techniques: hard prompting, soft prompting, and embedding-level injection.

- What each method is, its paper and its control: [Graph-Injection Methods](/pages/hrm-graph/graph-injection-methods.md).
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
| `text` | hard prompting | GraphQA text + LoRA, direct answers (the LoRA arm of Less is More, not its CoT recipe) |
| `text-incident` | hard prompting | Talk like a Graph incident encoding |
| `text-colors` | hard prompting | Colorful Talks C-OWL, official CL-OWL colour naming (since 2026-10-24) |
| `prompt-tuning` | static soft prompt | peft PromptEncoder + text; capacity control (Less is More's P-Tuning arm) |
| `graphtoken`, `graphtoken-frozen` | graph-level soft tokens | PyG GIN, graph text kept ("Graph Tokens + Text") |
| `node-tokens` | node-level soft tokens | LLaGA hop-field template + text + LoRA, ported from VITA-Group/LLaGA |
| `graphtoken-faithful`, `node-tokens-faithful` | soft tokens | GraphToken / LLaGA in their own setting: frozen LLM, no graph text (added 2026-10-24, below) |
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
- Fidelity review against the source papers (2026-10-24):
  [`tests/reports/graph_inject_paper_review.md`](../../../tests/reports/graph_inject_paper_review.md). Its claims
  were checked against the papers and official code and applied the same day; see
  [Paper-fidelity fixes](#paper-fidelity-fixes-2026-10-24) below, and the full deviation table in
  [`graph_inject/README.md`](../../../graph_inject/README.md#deviations-from-the-papers).
  - Re-checked the same day after the fixes: all 18 pass-1 findings are fixed or documented. Ten minor items
    remain (N1–N10 in the report), for example LLaGA's paper (mean aggregation) and code (D^-1/2 A D^-1/2)
    disagree, and the `-faithful` arms on MetaQA / KQA Pro are close to closed-book. The GPU tests were not
    re-run in that pass.

## Results, phase 1 (seed 0, 2026-10-22/23)

Ran 2026-10-22 17:12 to 2026-10-23 09:40: 30 runs, about 33 min each. The table gives GraphQA test `greedy_acc`
(385 samples) at the best val `tf_em` epoch. One sample is 0.26 points, and the binomial SE is about 1.9 points.
The mean column also gives the mean without MaximumFlow. Δ is the pooled paired difference against `text` over
the 3 models (samples won / lost, exact McNemar p-value).

| Method | HRM-1B | Mimir v1 | Mimir v1.5 | Mean (w/o MaxFlow) | Pooled vs `text` |
| --- | ---: | ---: | ---: | ---: | --- |
| `text` (LoRA, no graph module) | 0.829 | **0.857** | 0.842 | **0.842** (0.909) | reference |
| `text-colors16` (16-name palette, superseded) | 0.831 | 0.855 | 0.847 | **0.844** (0.906) | +24 / -22, p=0.88 |
| `node-tokens` | 0.821 | 0.849 | **0.855** | **0.842** (0.905) | +23 / -24, p=1.00 |
| `abs-id` (control) | 0.826 | 0.836 | **0.855** | 0.839 (0.901) | +24 / -28, p=0.68 |
| `fullft-text` | **0.847** | 0.839 | 0.829 | 0.838 (0.900) | +29 / -34, p=0.62 |
| `text-incident` | 0.834 | 0.842 | 0.831 | 0.835 (0.896) | +25 / -33, p=0.36 |
| `graphtoken` | 0.813 | 0.849 | 0.844 | 0.835 (0.901) | +20 / -28, p=0.31 |
| `abs-pe` | 0.813 | 0.836 | 0.852 | 0.834 (0.898) | +19 / -29, p=0.19 |
| `prompt-tuning` (control) | 0.810 | 0.831 | 0.842 | 0.828 (0.893) | +19 / -36, p=0.03 |
| `graphtoken-frozen` | 0.709 | 0.787 | 0.805 | 0.767 (0.830) | +16 / -103, p<0.001 |

- `text-colors16` used our own 16-name hue palette, not the paper's. Its W&B runs were deleted on 2026-10-24, the
  outputs are archived as `<model>/text-colors16_s0`, and `text-colors` is rerun with the official palette.
- **No injection method beats the no-PE `text` baseline.** Every LoRA method is within about 2 points of it, both
  per model and on average. All pooled paired tests are non-significant, except two losses: frozen GraphToken
  (-7.5 points) and prompt tuning (-1.4 points, p=0.03).
- **`abs-pe` vs its identity control `abs-id`:** 23 wins and 29 losses pooled (p=0.49), so structure adds
  nothing over node identity.
- **φ is used; it just does not help.** At the best epoch, the RMS of φ on node tokens is 0.30 (HRM-1B),
  0.25 (Mimir v1) and 0.17 (Mimir v1.5), against a token-embedding RMS of 1.02, 0.99 and 0.43. That is unlike
  GRL `graph-enc`, whose gate stayed at about 0.002. `abs-id` learns a larger φ than `abs-pe` on HRM-1B
  (0.78 vs 0.30).
- **Freezing the LLM costs 4-12 points** (`graphtoken-frozen`, graph text kept): GraphToken needs the LLM adapted.
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
  Pairs where either token has no element get 0. Since 2026-10-24, so do two tokens of the same span occurrence
  (GTLM's b_SPD(u, u) = 0, below).
- `rel-bias-elem`: buckets same element (other occurrence) / other element only. This is the identity control.
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
- Stopped 2026-10-24 08:45 after 18 KG runs (text, abs-pe, abs-id), for the paper-fidelity fixes (below).
- Restarted the same day, with `graphtoken-faithful` and `node-tokens-faithful` appended. Remaining: 81 runs,
  about 110 h (confidence: medium).
  - 66 KG runs: 12 rel-bias at about 2.4 h, 42 others at about 1.4 h, and 12 faithful at about 0.7 h.
  - 15 GraphQA runs: rel-bias and rel-bias-elem, the `text-colors` reruns, and the two faithful arms.

## Paper-fidelity fixes (2026-10-24)

The review ([`tests/reports/graph_inject_paper_review.md`](../../../tests/reports/graph_inject_paper_review.md)) was
checked against the primary sources. Its claims hold, apart from the `text-incident` neighbour order (Low), which
cannot be checked here because GraphQA's own incident files are not on disk. Two of the checks found more than the
review:
- **GTLM** (`DarioVajda/graph_model` @ f4feefb): `SPDBias.forward` multiplies by `spd > 0`, and `finalize_node_bias`
  zeroes the intra-node diagonal, so the same-node bias is fixed at 0 in the code as well as in the paper.
- **CL-OWL** (`angelozangari/CL-OWL` @ 1f22ea7): `_compute_colors_for_iteration` returns RGB / 255, but
  `rgb_to_color_name` thresholds expect 0-255; as written, every node would be "lime". On 0-255 RGB, amber, yellow
  and cyan are shadowed by earlier branches, so the reachable names are red, crimson, orange, lime, green and teal.
  Those are exactly the colours of the paper's Figure 5 example, which suggests the authors ran the 0-255 version.

**Behaviour changes.**
- **`rel-bias`, `rel-bias-elem`.** Token pairs within one span occurrence (one node mention, one edge tuple, one
  KG fact line or qualifier) get bias 0, as GTLM's b_SPD(u, u) = 0 (Property 2: text inside a node attends as in
  the base LLM). GTLM has one text segment per node; here a node is mentioned several times, so distance 0 between
  *different* occurrences of one element stays a learned bucket. It is what ties a question mention to the node's
  facts. Implementation: `PromptGraph.char_spans` → `EncodedPrompt.occurrences` → `graph_bias_occ`, applied as
  `where(occ[q] == occ[k], 0, bias)` in the FlexAttention `score_mod` and in the prefill mask
  (`GraphBiasContext.token_bias`).
- **`text-colors`.** CL-OWL's own colour naming (ported verbatim, on 0-255 RGB) replaces our 16-name palette, and
  the block carries the paper's four Figure 5 instruction sentences (WL similarity, structural role, `Color(i)=c`,
  "similarity metric among nodes"). Note: the official mapping is not hue-monotone ("lime" covers two hue ranges).
- **`text-colors` on KQA Pro: context cap (a latent bug, found by the smoke run).** The colour block takes KQA Pro
  prompts up to 4,496 HRM tokens (5,233 in train), past the 4,096-position RoPE table of all three models. This
  happens on 0.6% of prompts, and the run crashes at the first one (CUDA gather assert). The old 16-name block was
  as long, so the queued KQA Pro `text-colors` runs would have failed too. The block is now cut to the first nodes
  that fit 3,990 prompt tokens (`text.PROMPT_TOKEN_BUDGET`). This truncates 9 val / 4 test prompts (HRM) and
  5 / 3 (Mimir); the colours still come from the whole graph.
- **New arms `graphtoken-faithful` and `node-tokens-faithful`.** These are GraphToken's and LLaGA's own settings. In
  both, the LLM is frozen, and `graph_text: false` drops the graph description (the question stays).
  - `graphtoken-faithful`: RWSE + LapPE + learned IDX features (16-d, `idx_max` 256; the data has at most 180 nodes).
    Readout [mean ‖ sum ‖ h_a ‖ h_b], where a and b are the first two nodes named in the question (zeros if absent).
    That is GraphToken's node- and edge-level readouts, applied to every task. Unlike the paper, it uses AdamW at
    1e-3 (the paper: Lion, 0.05).
  - `node-tokens-faithful`: 4 hops (LLaGA HO's `use_hop=4`), 5 tokens per node.
  - Both take at most 8 samples per step (`samples_per_step: 8`). Question-only prompts are 10-20x shorter, so
    8192 tokens would mean about 15-50 optimizer steps per epoch. With the cap they get about `text`'s step count
    on MetaQA and KQA Pro, and 3.4x on GraphQA. Bench: 0.7 s/step, 26-31 GB peak (HRM-1B).
  - On MetaQA and KQA Pro these arms are close to closed-book: without the facts text the answer strings are not in
    the prompt. They are meaningful mainly on GraphQA.

**Relabelled, unchanged runs.**
- `text` is the LoRA arm of Less is More, not its CoT recipe.
- `graphtoken` and `graphtoken-frozen` are "Graph Tokens + Text". The "graph pooling tokens" (Grover et al.) label
  was wrong: their k tokens come from k pooling clusters, here one vector is reshaped.
- `graphtoken-frozen` is not GraphToken's original setting, because the graph text is kept.
- `node-tokens` uses 2 hops (the ND template's value; HO uses 4), LoRA and the text.
- The P-Tuning citation is now Liu et al. 2021, "GPT Understands, Too". Its encoder is BiLSTM + MLP; peft's is
  an MLP.
- `abs-id` is a random identity control, not GraphToken's learned IDX.
- `ToEmbeddingScale` is a deviation shared by every soft-token method.

**Impact on runs.**
- The 45 finished runs other than `text-colors` are unaffected, because their code paths did not change. That is
  27 GraphQA phase-1 runs, and KG `text`, `abs-pe` and `abs-id` × 3 models × 2 datasets.
- GraphQA `text-colors` × 3 is superseded (above).
- `metaqa-hrm1b-rel-bias` was running with the learned same-element bias. It was killed after about 30 min, and its
  W&B run deleted. No other rel-bias run had started.
- Tests: `tests/test_graph_inject.py`, 81 passed. New:
  - the KQA Pro colour block fits the context;
  - the `graphtoken-faithful` query readout on an empty graph;
  - occurrences on 3 datasets;
  - the CL-OWL palette;
  - no-graph-text prompts;
  - the rel-bias same-occurrence zero;
  - gradients for both faithful arms.

## Interim: rel-bias on MetaQA, and the bias learning-rate check (2026-10-24)

MetaQA test `greedy_acc` (seed 0; full test split, 1,197 samples; best epoch on the fixed 400-sample val subset):

| | HRM-1B | Mimir v1 | Mimir v1.5 |
| --- | ---: | ---: | ---: |
| `text` | 0.865 | 0.875 | 0.877 |
| `abs-pe` | 0.871 | **0.878** | 0.873 |
| `abs-id` | 0.863 | 0.870 | 0.866 |
| `rel-bias` | **0.871** | 0.870 | **0.881** |

- **Pooled paired comparisons** over the 3 models:

  | Comparison | Won / lost | p |
  | --- | ---: | ---: |
  | `rel-bias` vs `text` | +56 / −49 | 0.56 |
  | `rel-bias` vs `abs-pe` | +60 / −60 | 1.00 |
  | `rel-bias` vs `text`, 1-hop | +11 / −8 | 0.65 |
  | `rel-bias` vs `text`, 2-hop | +17 / −19 | 0.87 |
  | `rel-bias` vs `text`, 3-hop | +28 / −22 | 0.48 |

- **3-hop stays at 0.756–0.776 for every method.**
- **The learned bias barely moved from zero.** At the best epoch, mean |β| over layers and heads is 0.02–0.04 in
  every distance bucket, for all 3 models, and the largest entry is 0.14–0.18 logits.
  - The buckets are flat: distance 2 (adjacent nodes) is no different from "unreachable".
  - With AdamW at the method lr of 1e-3 over about 3,000 steps, a consistent gradient sign would have moved β by
    O(1). So the gradient on β has no consistent direction, and `rel-bias` at this lr is effectively the text model.
- **Reference values from GTLM.** GTLM (`DarioVajda/graph_model` @ f4feefb) trains its biases in a separate
  parameter group:
  - `bias_lr` 5e-3 in its GraphQA defaults (`src/experiments/graphqa/__main__.py`), against LLM lr 3e-5, 20 epochs
    and 32 samples per step;
  - its bias sweeps (`src/experiments/bias_experiments/landmark/README.md`) pick 2e-2 in 8 of 9 cells, and note that
    |b| grows roughly as `bias_lr` × steps.
- **The check: `rel-bias-hilr`** (`configs/methods/rel-bias-hilr.yaml`, `method_lr: 2.0e-2`, 20× ours).
  - Run: MetaQA, HRM-1B, seed 0, everything else as `rel-bias`. Queued ahead of the matrix on 2026-10-24.
    `priority_rel_bias_hilr.sh` in the graph_inject output root waited for the running KQA Pro run, stopped the queue
    between runs, ran the check, then restarted the queue.
  - Outcomes:
    - If β grows and the bucket profile becomes distance-dependent, but accuracy stays at `text`: the bias is used
      but adds nothing over the graph text.
    - If accuracy rises (3-hop above all) or β grows: the remaining `rel-bias` runs should use the higher lr.
    - If β stays near 0 even at 2e-2: the text model has no use for the bias signal.
  - **Result (2026-10-24, 17:05-18:39):**
    - Test `greedy_acc` is 0.871, the same as `rel-bias` at 1e-3 (0.871). Paired: +27 / −28 vs `rel-bias` (p = 1.00),
      and +19 / −12 vs `text` (p = 0.28).
    - Per task (1-hop / 2-hop / 3-hop): 0.962 / 0.887 / 0.761, against `text` 0.950 / 0.892 / 0.751. On 3-hop that is
      +9 / −5 vs `text` (p = 0.42).
    - **β grew about 20×** (mean |β| 0.6, max 3.6; per head, the d = 1..8 range is 2.2 logits at the median), **but has
      no distance structure.**
      - Spearman correlation of β with distance over d = 1..8, per (layer, head): median 0.02, |ρ| > 0.7 for 5% of
        heads. That is the chance rate for 8 random values.
      - Head means are about 0 in every bucket.
      - Isolated heads have spikes. For example, layer 10, head 4: +3.3 at d = 4, −1.6 at d = 5.
    - **Reading:** with a large enough lr the bias is used, but it does not encode graph distance in any systematic
      way, and accuracy does not move. The graph text already carries what the model uses. So the null result of
      `rel-bias` is not an lr artifact. The remaining runs keep 1e-3, which keeps them comparable with the finished
      ones.

**Phase-2 results so far** (test `greedy_acc`, seed 0; HRM-1B / Mimir v1 / Mimir v1.5):

| | GraphQA | MetaQA | KQA Pro |
| --- | --- | --- | --- |
| `text` | 0.829 / 0.857 / 0.842 | 0.865 / 0.875 / 0.877 | 0.739 / 0.742 / 0.748 |
| `abs-pe` | 0.813 / 0.836 / 0.852 | 0.871 / 0.878 / 0.873 | 0.747 / 0.757 / 0.743 |
| `rel-bias` | 0.821 / 0.842 / 0.844 | 0.871 / 0.870 / 0.881 | 0.731 / 0.753 / 0.741 |
| `rel-bias-elem` | pending | 0.864 / 0.870 / 0.882 | running |

Pooled `rel-bias` vs `text` (won / lost, p):

| Dataset | Won / lost | p |
| --- | ---: | ---: |
| GraphQA | +18 / −26 | 0.29 |
| MetaQA | +56 / −49 | 0.56 |
| KQA Pro | +124 / −128 | 0.85 |

- On MetaQA, `rel-bias` vs its identity control `rel-bias-elem` is +59 / −51 (p = 0.50).
- The learned β stayed at |β| ≈ 0.03, flat, on all three datasets (HRM-1B).
- KQA Pro Count is lower with `rel-bias` on all 3 models: 0.557 / 0.626 / 0.588, against 0.634 / 0.634 / 0.672.
  That is about 40 test samples per model; noted, not tested.

## Interim effectiveness of all methods, and the GNN-encoder question (2026-10-26)

All three datasets, seed 0, test `greedy_acc` (HRM-1B / Mimir v1 / Mimir v1.5). Δ is the mean change vs `text` over the
models run. "Won / lost" counts test samples that flip relative to `text`, pooled over the models, with an exact
binomial (McNemar) p. This supersedes the "pending" / "running" cells of the phase-2 table above.

| Method | GraphQA | Δ, won / lost, p | MetaQA | Δ, won / lost, p | KQA Pro | Δ, won / lost, p |
| --- | --- | --- | --- | --- | --- | --- |
| `text` | 0.829 / 0.857 / 0.842 | reference | 0.865 / 0.875 / 0.877 | reference | 0.739 / 0.742 / 0.748 | reference |
| `abs-pe` | 0.813 / 0.836 / 0.852 | −0.9, +19/−29, 0.19 | 0.871 / 0.878 / 0.873 | +0.2, +53/−46, 0.55 | 0.747 / 0.757 / 0.743 | +0.6, +128/−108, 0.22 |
| `abs-id` | 0.826 / 0.836 / 0.855 | −0.3, +24/−28, 0.68 | 0.863 / 0.870 / 0.866 | −0.6, +42/−63, 0.05 | 0.749 / 0.739 / 0.750 | +0.3, +116/−106, 0.55 |
| `rel-bias` | 0.821 / 0.842 / 0.844 | −0.7, +18/−26, 0.29 | 0.871 / 0.870 / 0.881 | +0.2, +56/−49, 0.56 | 0.731 / 0.753 / 0.741 | −0.1, +124/−128, 0.85 |
| `rel-bias-elem` | 0.818 / 0.842 / 0.852 | −0.5, +19/−25, 0.45 | 0.864 / 0.870 / 0.882 | 0.0, +43/−44, 1.00 | 0.749 / 0.742 / 0.752 | +0.5, +115/−99, 0.31 |
| `node-tokens` | 0.821 / 0.849 / 0.855 | −0.1, +23/−24, 1.00 | 0.863 / 0.873 / 0.873 | −0.3, +46/−55, 0.43 | 0.739 / 0.754 / 0.769 | **+1.1, +151/−113, 0.023** |
| `graphtoken` | 0.813 / 0.849 / 0.844 | −0.7, +20/−28, 0.31 | 0.858 / 0.873 / 0.875 | −0.4, +48/−61, 0.25 | 0.732 / 0.754 / 0.753 | +0.4, +123/−111, 0.47 |
| `prompt-tuning` (no graph) | 0.810 / 0.831 / 0.842 | **−1.5, +19/−36, 0.03** | 0.861 / 0.871 / 0.887 | +0.1, +57/−54, 0.85 | 0.735 / running / queued | HRM-1B: +40/−44 |
| `graphtoken-frozen` | 0.709 / 0.787 / 0.805 | **−7.5, +16/−103, <0.001** | queued | | queued | |
| `text-incident` | 0.834 / 0.842 / 0.831 | −0.7, +25/−33, 0.36 | queued | | queued | |
| `fullft-text` | 0.847 / 0.839 / 0.829 | −0.4, +29/−34, 0.62 | queued | | queued | |

- **No structural method beats `text` robustly.** Every method with an unfrozen LLM is within about ±1 pt of `text`. That
  includes the no-graph control `prompt-tuning` on MetaQA (+0.1).
  - **Only two results clearly lose:** `graphtoken-frozen` (−7.5, GraphQA) and `prompt-tuning` (−1.5, GraphQA).
  - **The one nominal gain is `node-tokens` on KQA Pro** (+1.1, p = 0.023).
    - It is driven by Verify (+19/−7, p = 0.029) and Qualifier (+23/−13, p = 0.13).
    - About 25 tests have been run against `text`, so it does not survive a multiple-comparison correction (that would
      need p ≲ 0.002).
    - Against its identity control `abs-id` it is +146/−118 (p = 0.096), and against `abs-pe` +126/−108 (p = 0.27).
    - Next: a second seed on KQA Pro for `text`, `node-tokens` and `abs-id` once the queue ends.
- **The identity controls are not beaten either.**
  - `abs-pe` beats `abs-id` only on MetaQA, and there it is because `abs-id` drops (−0.6), not because `abs-pe` gains.
  - `rel-bias` vs `rel-bias-elem` on MetaQA is +59/−51 (p = 0.50).
- **The hard cases do not move for any method:**
  - MetaQA 3-hop: 0.75–0.78;
  - GraphQA TriangleCounting and EdgeCount, which are the remaining GraphQA errors.
- **Reading (medium confidence):** with the graph written in the prompt, LoRA on the text already gets what these
  encodings could add. The faithful arms (frozen LLM, no graph text) are the open test, because there the structure
  can only arrive through the method.

**Why the GNN architecture is not the lever on the KGs.** The question asked was whether a Graph Transformer (for
example GPS) should replace GraphToken's GIN.
- **With the graph text kept**, the soft tokens are redundant with the text, so a stronger encoder has little room to
  help. Not yet tested: zeroing the soft tokens at eval on the `graphtoken` checkpoints, which would show whether they
  are used at all.
- **In `graphtoken-faithful` on MetaQA and KQA Pro**, the encoder only sees anonymous topology.
  - Nodes carry RWSE + LapPE + IDX features. There are no entity names and no relation types.
  - `normalized_adjacency` / `batched_edge_index` make edges undirected.
  - Example (MetaQA test): "when were the films written by David Morlet released", gold `2009`. The subgraph is a star
    around `Mutants` with 8 facts.
    - The answer node `2009` has the same features and the same position as `French`, `Horror` and the actors, so no
      encoder can single it out.
    - `written by` and `directed by` collapse into one duplicated edge.
    - The LLM never sees the string `2009`.
- **What would address it:** text-derived node and edge features, G-Retriever style. That means the LLM's
  mean-pooled token embeddings of the entity and relation names, fed to a relation-aware encoder (GINE, or
  TransformerConv with `edge_attr`). Per-node tokens (LLaGA style) are probably also needed so the answer string can
  be decoded.
- **A GPS arm** (PyG `GPSConv`: GINE + global attention on the same PEs) is worth trying only on GraphQA, where nodes
  have no attributes. It is worth it only if `graphtoken-faithful` lags `text` mainly on the long-range tasks
  (ShortestPath, Reachability, CycleCheck), since the 3-layer GIN sees only 3 hops.
- **Decision:** wait for the remaining queue (about 50 runs at 2026-10-26 10:20, faithful arms last, estimated to end
  2026-10-28 morning) before adding any arm.

## Per-cell significance and the seed plan (2026-10-27)

**Per-cell paired tests, seed 0.**
- Every method was compared against `text` in each model × dataset cell: 96 comparisons, on per-sample test
  `greedy_acc`, with a paired bootstrap 95% CI and an exact McNemar test.
- The `text-colors`, `text-incident` and `graphtoken-frozen` KG cells and `fullft-text` on GraphQA are included;
  the `-faithful` arms and the KG `fullft-text` runs were still queued.
- **Typical 95% CI half-width of a difference:** about ±1 pt on MetaQA (n=1197), ±1.6 pt on KQA Pro (n=1134) and
  ±2.2 pt on GraphQA (n=385).
- **`graphtoken-frozen` is the only real effect.** It is significant in 8 of 9 cells and survives Bonferroni (p < 0.0005):

  | Dataset | HRM-1B | Mimir v1 | Mimir v1.5 |
  | --- | ---: | ---: | ---: |
  | MetaQA | −5.9 | −5.0 | +0.0 (p = 1) |
  | KQA Pro | −14.5 | −10.2 | −4.6 |
  | GraphQA | −11.9 | −7.0 | −3.6 |

- **The other 85 comparisons are consistent with chance.** 4 reach p < 0.05, about what 85 tests at α = 0.05 give,
  none survives a correction, and their signs disagree:

  | Dataset | Model | Method | Diff vs `text` (pt) | p |
  | --- | --- | --- | ---: | ---: |
  | MetaQA | HRM-1B | `text-colors` | +1.3 | 0.049 |
  | KQA Pro | HRM-1B | `text-colors` | −2.8 | 0.006 |
  | KQA Pro | Mimir v1.5 | `node-tokens` | +2.1 | 0.012 |
  | GraphQA | Mimir v1 | `prompt-tuning` | −2.6 | 0.031 |

- **These CIs cover only test-set sampling, not training noise.** In `graph_inject`, `seed` sets both the new modules'
  init and the data order (`rng([seed, epoch])` in `train.py`). This differs from `pretrain.py` fine-tunes, where
  only a reseeded dataset changes the run.

**Seed plan.** This supersedes the 2026-10-26 "second seed on KQA Pro for `text`, `node-tokens`, `abs-id`".
1. After the current queue: `SEEDS="1 2" METHODS="text"` on all 9 cells (about 15 h, estimated). This gives the
   seed-to-seed sd of the baseline.
2. Compare that sd with the observed gaps of 1–2 pt.
   - sd ≲ 0.5 pt: the gaps may be partly real. Run seeds 1–2 for the 2–3 best candidates (for example
     `node-tokens`, `text-incident`, `abs-pe`), about 15 h each.
   - sd ≳ 1 pt: two single-seed runs of one method already differ by about 1.4 pt, so the gaps are noise. More seeds
     are needed only for a multi-seed null claim.
   - A full method × seed sweep (about 200 h) is not justified.

**Fail: queue relaunched in a new SSH session without W&B credentials.**
- **Symptom:** every run died at `wandb.init` with `UsageError: No API key configured`. `run_matrix.sh` logs `FAILED`
  and moves on, so the queue would have gone through every remaining run in minutes.
- **Fix:** `wandb login --host https://forge.coreweave.com`, then relaunch. Finished runs are skipped. Before a
  relaunch, check that `wandb.Api().viewer` works.
