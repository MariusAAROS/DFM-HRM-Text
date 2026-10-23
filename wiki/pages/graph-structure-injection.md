---
type: Runbook
title: Graph-Structure Injection on GraphQA (graph_inject)
description: Standalone package that fine-tunes HRM-Text-1B, DFM-Mimir v1 and v1.5 on GraphQA with hard-prompt, soft-prompt and embedding-level graph-injection methods under one LoRA trainer; design, verification, run matrix and results.
tags: [graphs, graphqa, lora, soft-prompts, positional-encoding, hrm-text-1b, mimir]
status: stable
last_updated: 2026-10-22
confidence: high
---
# Graph-Structure Injection on GraphQA

The question is which way of injecting graph structure helps the pretrained models (HRM-Text-1B, DFM-Mimir v1,
DFM-Mimir v1.5) on GraphQA at small scale. The method families come from the user's survey of graph → LLM
techniques: hard prompting, soft prompting, and embedding-level injection.

- Code: [`graph_inject/`](../../graph_inject/README.md). It is standalone: `pretrain.py`, `models/` and the eval
  scripts are untouched and reused by import.
- Tests: `tests/test_graph_inject.py`.
- Queue: `scripts/graph_inject/run_matrix.sh`.
- W&B: project **`Graph-Positional-Encoding-Exploration`**, group `graphqa-inject`; smoke and bench runs use
  group `graphqa-inject-bench`.

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

The relative attention bias (GTLM / Graphormer) is phase 2: the SM120 FA2 varlen training kernel takes no bias.

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
