---
type: Runbook
title: Graph-Structure Injection on GraphQA (graph_inject)
description: Standalone package that fine-tunes HRM-Text-1B, DFM-Mimir v1 and v1.5 on GraphQA with hard-prompt, soft-prompt and embedding-level graph-injection methods under one LoRA trainer; design, verification, run matrix and results.
tags: [graphs, graphqa, lora, soft-prompts, positional-encoding, hrm-text-1b, mimir]
status: in-progress
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

## Results

Pending: the run matrix has not been queued yet.
