---
type: Reference
title: Graph-Injection Methods and Their Papers (graph_inject)
description: Explains each graph_inject method run (text, colors, soft tokens, additive PE, attention bias), the paper behind it, what was adapted, and which control each result must be read against.
tags: [graphs, graphqa, metaqa, kqapro, soft-prompts, positional-encoding, attention-bias, papers]
status: stable
last_updated: 2026-10-27
confidence: high
---
# Graph-Injection Methods and Their Papers (graph_inject)

A guide to the methods compared in [Graph-Structure Injection](/pages/hrm-graph/graph-structure-injection.md):
what each run does, which paper it comes from, and how to read it. The per-paper deviation list and the code map
are in [`graph_inject/README.md`](../../../graph_inject/README.md). The fidelity review against the papers and
the official code (2026-10-24) is `tests/reports/graph_inject_paper_review.md`.

## Shared setup

- **Same base for every run.** One of HRM-Text-1B, DFM-Mimir v1 or v1.5 is frozen and trained with LoRA
  (r=16) at the same budget. The exceptions are `fullft-text` and the frozen arms.
- **The graph is in the prompt as text** in every run except the two `-faithful` arms:
  - GraphQA: integer nodes and `(u, v)` edge tuples;
  - MetaQA / KQA Pro: `(subject, relation, object)` fact lines.
- **The difference is the extra injection.** Each method is `text` plus one way of adding structure, so
  "method − `text`" is what the injection adds.
- **Each injection enters the model at one of four levels:**
  1. the prompt words (hard prompting);
  2. extra input embeddings (soft tokens);
  3. a vector added to existing token embeddings (absolute PE);
  4. the attention scores (relative bias).

## Run → paper

| Run | Level | Paper | What it asks |
| --- | --- | --- | --- |
| `text` | words | Less is More (arXiv 2508.10115), its LoRA arm | Reference: plain graph text + LoRA |
| `fullft-text` | words | none (trainer check) | Does full fine-tuning instead of LoRA change things? |
| `text-incident` | words | Talk like a Graph (Fatemi, Halcrow, Perozzi, ICLR 2024) | Does another text serialization help? |
| `text-colors` | words | Colorful Talks (arXiv 2602.10386), C-OWL | Do 1-WL "colour" hints written in text help? |
| `prompt-tuning` | soft tokens | P-Tuning, "GPT Understands, Too" (Liu et al. 2021, arXiv 2103.10385); Less is More's P-Tuning arm | Control: 8 extra trainable tokens, no graph information |
| `graphtoken` | soft tokens | GraphToken, "Let Your Graph Do the Talking" (Perozzi et al. 2024, arXiv 2402.05862), as Less is More's "Graph Tokens + Text" | GNN graph tokens + text + LoRA |
| `graphtoken-frozen` | soft tokens | same | Same, LLM frozen (only the GNN trains) |
| `graphtoken-faithful` | soft tokens | GraphToken in its own setting | Frozen LLM, no graph text |
| `node-tokens` | soft tokens | LLaGA (Chen et al., ICML 2024, arXiv 2402.08170) | Per-node hop tokens + text + LoRA |
| `node-tokens-faithful` | soft tokens | LLaGA in its own setting | Frozen LLM, no graph text, 4 hops |
| `abs-pe` | added to embeddings | none: the proposed method, from GRL's `graph-enc` branch | Add a structural vector to every token of a node or edge |
| `abs-id` | added to embeddings | none (control) | Same, with random vectors: identity only, no structure |
| `rel-bias` | attention | GTLM (arXiv 2605.10247, `DarioVajda/graph_model`); the distance bias comes from Graphormer (Ying et al. 2021) | Learned attention bias by graph distance |
| `rel-bias-hilr` | attention | same | `rel-bias` with a 20× higher bias LR (MetaQA, HRM-1B only) |
| `rel-bias-elem` | attention | none (control) | Only "same element" vs "other element", no distances |

`text-colors16` (GraphQA only) is a superseded `text-colors` run that used our own 16-colour palette instead of the
paper's.

Building blocks shared by several methods:
- **RWSE**: random-walk return probabilities, k steps.
- **LapPE**: Laplacian eigenvectors, sign-flipped at random in training.
- **GIN** (Xu et al. 2019): a message-passing GNN as expressive as 1-WL.
- **1-WL**: Weisfeiler-Lehman colour refinement.

## Hard prompting: changing the words

- **`text`.**
  - Less is More compares text prompting, soft prompts, graph tokens and LoRA for graph tasks. Its main recipe adds
    instructive chain-of-thought and random-word node labels.
  - Only its **LoRA + text, direct-answer** arm is used here, with integer node ids.
- **`fullft-text`.**
  - Not from a paper. It reruns the `pretrain.py` full fine-tuning recipe inside the new trainer.
  - It checks that the trainer reproduces the earlier GraphQA 0.82–0.84, and on the KGs it shows whether LoRA
    costs anything compared with full fine-tuning.
- **`text-incident`.**
  - Talk like a Graph measures how the wording of a graph changes LLM accuracy. Each wording is an "encoder".
  - The incident encoder writes, per node, `Node 0 is connected to nodes 1, 2.` The code is the paper's
    google-research encoder, ported in GRL.
  - The paper has no KG encoder. On MetaQA / KQA Pro, facts are grouped by subject
    (`Santa Sangre: written by X, Y; has tags circus.`), which is our own adaptation.
- **`text-colors`.**
  - Colorful Talks runs 1-WL on the graph and writes each node's WL class as a colour word (`Color(3):teal`), so
    nodes in similar structural positions get similar colours.
  - It uses the paper's prompt wording and the official CL-OWL colour naming, applied to 0–255 RGB. The repo divides
    by 255 before its 0–255 thresholds, which would make every node "lime".
  - Deviations:
    - C-OWL, without CL-OWL's extra WL labels;
    - LoRA, where the paper only prompts;
    - on 0.6% of KQA Pro prompts, the colour block is cut short to fit the context.

## Soft tokens: extra embeddings the model reads

- **`prompt-tuning`.**
  - 8 trainable vectors in front of the prompt, produced by an MLP (P-Tuning v1 used a BiLSTM + MLP; peft uses the
    MLP).
  - **No graph information.** This is the capacity control: a soft-token method that beats `text` but not
    `prompt-tuning` gained from extra trainable tokens, not from the graph.
- **GraphToken arms.**
  - GraphToken runs a GNN over the graph, pools it and projects the result into a few soft tokens. The LLM stays
    frozen, and the graph is never written as text.
  - The encoder here: GIN, 3 layers, on RWSE + LapPE node features → pool → 8 tokens.

  | Arm | Graph text | LLM | Notes |
  | --- | --- | --- | --- |
  | `graphtoken` | kept | LoRA | Less is More's "Graph Tokens + Text", not GraphToken's own setting |
  | `graphtoken-frozen` | kept | frozen | |
  | `graphtoken-faithful` | none (question only) | frozen | Adds learned node-identity (IDX) features and a readout of the queried nodes. AdamW instead of Lion; RWSE+LapPE instead of the paper's small LPE/IDX |

- **LLaGA arms.**
  - LLaGA has no GNN. For each node it stacks the node's features at 0, 1, 2, … hops (features, then averaged over
    neighbours, then over 2-hop neighbours) and projects them with a 2-layer MLP into one token per hop.
  - So the LLM gets a token sequence per node (node-level), where GraphToken gives one summary of the whole graph.
  - Node features: mean LLM embedding of the node name + RWSE + LapPE. LLaGA uses a sentence-encoder embedding.

  | Arm | Hops | Graph text | LLM |
  | --- | --- | --- | --- |
  | `node-tokens` | 2 (comparable with GraphQA phase 1) | kept | LoRA |
  | `node-tokens-faithful` | 4 (LLaGA's HO setting) | none | frozen; only the projector trains, as in LLaGA |

- **The `-faithful` arms on MetaQA / KQA Pro are close to closed-book.**
  - Without the fact text, the answer string is in the prompt in about 0% of cases, and the encoder never sees
    relation labels.
  - Neither paper evaluated KGQA. These arms are meaningful mainly on GraphQA.

## Additive PE: `abs-pe` (proposed) and `abs-id`

- **`abs-pe` adds a vector φ to the embedding of every token that belongs to a graph element.** There are no new
  tokens and attention is unchanged.
  - **Node v:** φ(v) = W·MLP(RWSE + LapPE of v). It goes on every token of v at every mention: the node list, the
    edge tuples, the question.
  - **Edge (u, w):** a vector computed from both endpoints, added to the `(`, `,` and `)` of the tuple.
  - Template words get nothing.
  - W is zero-initialized, so step 0 is exactly `text`.
- **It has no paper reference.** It continues GRL `graph-enc`'s node / WL embeddings behind a scalar gate, which did
  not help: the gate stayed at about 0.002. Differences from that version:
  - zero-init projections instead of the gate;
  - edges get their own vectors;
  - its own, higher learning rate.
- **`abs-id`** has the same architecture, but each node gets a random vector, redrawn for every sample. That still
  says "these tokens are the same node", with no structure.

## Relative attention bias: `rel-bias` and its controls

- **`rel-bias`.**
  - A learned scalar β[layer, head, distance] is added to the attention score of every pair of graph-element tokens.
  - The distance is the shortest path in the **Levi graph**, where each edge is its own node: a node and its edge are
    at distance 1, and two adjacent nodes at 2.
  - Zero-initialized. Two tokens of the same mention get 0, as in GTLM.
  - Two *different* mentions of one node (the question's mention and an edge tuple, say) get a learned distance-0
    bucket. This ties the question to the node's facts.
  - It is GTLM's distance (SPD) bias on top of the text model, the attention-level counterpart of `abs-pe`. GTLM
    itself goes further:
    - it drops the edge-list text;
    - it restarts RoPE in each node's segment;
    - it adds RRWP and magnetic-Laplacian biases.
- **`rel-bias-hilr`.** At method lr 1e-3 the learned β stayed at |β| ≈ 0.03, flat across distances. This arm uses
  2e-2, the value GTLM's sweeps usually pick.
- **`rel-bias-elem`.** Two learned buckets only, "same element, other mention" and "other element". There is no
  distance information.

## Which control to read each result against

| Question | Compare |
| --- | --- |
| Does the injection help at all? | method vs `text` |
| Is a soft-token gain just extra capacity? | `graphtoken` / `node-tokens` vs `prompt-tuning` |
| Is an additive-PE gain structure, or just node identity? | `abs-pe` vs `abs-id` |
| Does graph distance in attention matter? | `rel-bias` vs `rel-bias-elem` |
| Can graph tokens replace the text? | `-faithful` arms vs `text` (GraphQA only) |
| Is LoRA limiting? | `fullft-text` vs `text` |

- **GraphQA graphs can be memorized.** GraphQA is split by question, not by graph, so every test graph is also in
  train, and an encoder can memorize graphs. This is one more reason to check `abs-id` and `prompt-tuning`.
- **MaxFlow can't be answered from the GraphQA text.** Capacities are not printed, which is why the results also
  show accuracy without MaxFlow.
