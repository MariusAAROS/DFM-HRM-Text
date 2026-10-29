---
type: Runbook
title: Graph-Structure Injection on GraphQA, MetaQA and KQA Pro (graph_inject)
description: Standalone package that fine-tunes HRM-Text-1B, DFM-Mimir v1 and v1.5 on GraphQA, MetaQA and KQA Pro with hard-prompt, soft-prompt, absolute-PE and relative-attention-bias graph injection under one LoRA trainer; design, verification, results and fixes.
tags: [graphs, graphqa, metaqa, kqapro, lora, soft-prompts, positional-encoding, attention-bias, hrm-text-1b, mimir]
status: stable
last_updated: 2026-10-29
confidence: high
---
# Graph-Structure Injection on GraphQA, MetaQA and KQA Pro

The question is which way of injecting graph structure helps the pretrained models (HRM-Text-1B, DFM-Mimir v1,
DFM-Mimir v1.5). The method families come from the user's survey of graph → LLM techniques:
- hard prompting;
- soft prompting;
- embedding-level injection (additive PE);
- attention-level injection (relative bias).

- What each method is, its paper and its control: [Graph-Injection Methods](/pages/hrm-graph/graph-injection-methods.md).
- Code: [`graph_inject/`](../../../graph_inject/README.md) (deviations from the papers listed there). It is
  standalone: `pretrain.py`, `models/` and the eval scripts are untouched and reused by import.
- Tests: `tests/test_graph_inject.py`. Queue: `scripts/graph_inject/run_matrix.sh`.
- W&B: project **`Graph-Positional-Encoding-Exploration`**, groups `graphqa-inject`, `metaqa-inject` and
  `kqapro-inject`. Smoke and bench runs are logged offline.
- Outputs: `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject{,_metaqa,_kqapro}/<model>/<method>_s<seed>/`.

## Setup

**Trainer** (`graph_inject/train.py`).
- One GPU, no FSDP, no compile. It loads the converted HF dirs through `inference_load_checkpoint`.
- LoRA r=16, α=32 on all H/L projections, lr 2e-4, base model frozen in bf16. The method's own modules train at lr
  1e-3.
- Clip 1.0, 8192 tokens/step, H2L3, bp_max_steps 5.
- Epochs: GraphQA 10; MetaQA and KQA Pro 3 (their full-FT configs peaked at epoch 1–2).
- Greedy generation on val every epoch; test (always the full split) at the best val `tf_em` epoch.
- `seed` sets the new modules' init **and** the data order (`rng([seed, epoch])`). Unlike `pretrain.py`
  fine-tunes, a new seed is a genuinely different training run.

**Data.** The graph is parsed back out of the prompt text; the GraphQA parser was ported from GRL `graph-enc`.

| Dataset | Rows (train / val / test) | Graph | Prompt (HRM tokens) |
| --- | --- | --- | --- |
| GraphQA | 3,080 / 385 / 385 (11 tasks) | 5–19 integer nodes, ≤ 171 `(u, v)` edges | ~290 mean (with answer) |
| MetaQA | 7,967 / 796 / 1,197 (1/2/3-hop) | GRL gold subgraph, ~72 facts, 9 relations | 944 mean, 1,324 max |
| KQA Pro | 6,824 / 760 / 1,134 (7 tasks) | ~57 facts, 767 relations, 38% with qualifiers, 4% empty | 1,309 mean, 2,996 max |

- **KG parsing** (`kg_structure.py`):
  - A node is a distinct entity or value string, and an edge is a fact (s, o).
  - Each qualifier `k: v` is an extra edge (s, v), so hyper-relational facts hang off the subject.
  - Facts are split where the middle field is a known relation (`kb.txt` / `kb.json`, longest match wins). This is
    needed because 1.3% (MetaQA) and 2.1% (KQA Pro) of facts contain `", "` inside names.
  - All 18,678 rows parse and re-serialize to their own lines.
- **KG validation subset.** The epoch is chosen on a fixed, task-stratified 400-sample val subset (seed 0, the same
  for every run). This makes the epoch choice a paired comparison and saves about 16 h of generation over the
  matrix. Config: `eval.val_subset`.
- **Method adaptations for the KGs:**
  - `text-incident` groups facts by subject.
  - `text-colors` runs 1-WL on the undirected entity graph, which adds about 50% to the prompt.
  - `node-tokens` embeds the entity name (mean frozen token embedding) instead of v's digits.

**Implementation notes.**
- **Soft tokens** are slots placed right before the prompt content, inside the bidirectional PrefixLM prefix.
- **Relative bias.** Every `Attention` is switched to `GraphBiasAttention` by `__class__`. Training uses
  FlexAttention with the packed document + PrefixLM mask; the prompt prefill adds the bias to the SDPA float mask.
- **The bias gradient is the trap.** Indexing the small [H, buckets] table inside `score_mod` makes the backward
  ~17× slower (48 ms vs 2.9 ms per call, from atomic adds). Gathering it per layer into [H, NDOC, EL, EL] first
  brings that to 7.3 ms.
- **LapPE on graphs with ≥ 100 nodes.** PyG switches to ARPACK there, which failed on some disconnected KQA Pro
  subgraphs (ARPACK error 3). `features.laplacian_pe` forces the dense `eigh` path at every size, which leaves
  GraphQA unchanged.

## Verification (`tests/test_graph_inject.py`, 81 passed at 2026-10-24)

- **Parsers:** GraphQA node and edge counts equal `nnodes` / `nedges`. KG fact counts equal the raw JSON.
- **Token ids** equal the data_io arrays: all GraphQA train rows, and 1,000 KG rows per dataset, for both the HRM and
  the Gemma tokenizer. Element-aligned tokens decode to the node's name.
- **Zero-init identity:** `text` (LoRA B=0), `abs-pe` and `abs-id` give logits bitwise equal to the pretrained model.
  At β = 0, `rel-bias` generation equals `text`.
- **Gradients:** a nonzero gradient reaches every trainable tensor of every method on all 3 datasets, including an
  empty KQA Pro graph. This holds with `inject_x=False`, where the injection enters only through the initial `z_H`.
- **Train vs inference:** packed training logits match KV-cache prefill logits, and that includes `rel-bias` at
  random β. `graph_inject.generate` reproduces `inference_generate` token for token.

## Cost (RTX PRO 6000, eager, measured from the queue logs)

| | GraphQA | MetaQA | KQA Pro | Peak memory |
| --- | --- | --- | --- | --- |
| `text` and most LoRA arms | ~33 min | ~52 min | ~63 min | 47–53 GB |
| `rel-bias`, `rel-bias-elem` | ~1.3 h | ~1.6 h | ~2.7 h | ~50 GB |
| `text-colors` | ~40 min | ~1.5 h | ~1.9 h | |
| `fullft-text` | ~30 min | ~55 min | ~65 min | 72.5 GB (Mimir v1.5) |
| `-faithful` arms | ~43 min | ~32 min | ~30 min | 26–31 GB |

## Caveats

- **MaximumFlow** capacities are not in the GraphQA text, so the task is unanswerable: 34 of 385 test samples, at
  about 0.18 for every model. Reports also give `greedy_acc_answerable`.
- **GraphQA splits are by question:** every test graph also appears in train. Graph encoders can memorize graphs,
  hence the `abs-id` and `prompt-tuning` controls.
- **Paired test error.** The typical 95% CI half-width of a method − `text` difference is about ±2.2 pt on GraphQA,
  ±1 pt on MetaQA and ±1.6 pt on KQA Pro. Training-seed noise comes on top of that (see Next).
- **Prior result:** in GRL `graph-enc`, gated node and WL embeddings under full FT gave no gain, and the gate stayed
  at about 0.002.

## Results (seed 0, final 2026-10-28)

Test `greedy_acc` (HRM-1B / Mimir v1 / Mimir v1.5). Δ is the mean difference from `text` over the 3 models, in
points.

| Method | GraphQA | Δ | MetaQA | Δ | KQA Pro | Δ |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `text` (reference) | 0.829 / 0.857 / 0.842 | | 0.865 / 0.875 / 0.877 | | 0.739 / 0.742 / 0.748 | |
| `text-incident` | 0.834 / 0.842 / 0.831 | −0.7 | 0.871 / 0.876 / 0.886 | +0.5 | 0.748 / 0.750 / 0.762 | +1.0 |
| `text-colors` | 0.810 / 0.860 / 0.839 | −0.6 | 0.877 / 0.881 / 0.871 | +0.4 | 0.711 / 0.753 / 0.753 | −0.4 |
| `prompt-tuning` (no graph) | 0.810 / 0.831 / 0.842 | −1.5 | 0.861 / 0.871 / 0.887 | +0.1 | 0.735 / 0.733 / 0.750 | −0.4 |
| `graphtoken` | 0.813 / 0.849 / 0.844 | −0.7 | 0.858 / 0.873 / 0.875 | −0.4 | 0.732 / 0.754 / 0.753 | +0.3 |
| `node-tokens` | 0.821 / 0.849 / 0.855 | −0.1 | 0.863 / 0.873 / 0.873 | −0.3 | 0.739 / 0.754 / 0.769 | +1.1 |
| `abs-pe` | 0.813 / 0.836 / 0.852 | −0.9 | 0.871 / 0.878 / 0.873 | +0.2 | 0.747 / 0.757 / 0.743 | +0.6 |
| `abs-id` (control) | 0.826 / 0.836 / 0.855 | −0.4 | 0.863 / 0.870 / 0.866 | −0.6 | 0.749 / 0.739 / 0.750 | +0.3 |
| `rel-bias` | 0.821 / 0.842 / 0.844 | −0.7 | 0.871 / 0.870 / 0.881 | +0.2 | 0.731 / 0.753 / 0.741 | −0.1 |
| `rel-bias-elem` (control) | 0.818 / 0.842 / 0.852 | −0.5 | 0.864 / 0.870 / 0.882 | 0.0 | 0.749 / 0.742 / 0.752 | +0.5 |
| `fullft-text` | 0.847 / 0.839 / 0.829 | −0.4 | 0.803 / 0.812 / 0.693 | **−10.3** | 0.683 / 0.706 / 0.585 | **−8.5** |
| `graphtoken-frozen` | 0.709 / 0.787 / 0.805 | **−7.6** | 0.805 / 0.825 / 0.877 | **−3.7** | 0.594 / 0.639 / 0.702 | **−9.8** |
| `graphtoken-faithful` | 0.514 / 0.504 / 0.532 | **−32.6** | 0.116 / 0.110 / 0.129 | **−75.4** | 0.319 / 0.304 / 0.310 | **−43.2** |
| `node-tokens-faithful` | 0.473 / 0.538 / 0.538 | **−32.6** | 0.271 / 0.265 / 0.292 | **−59.6** | 0.317 / 0.333 / 0.360 | **−40.6** |

Other runs:
- `rel-bias-hilr` (MetaQA, HRM-1B only): 0.871.
- `text-colors16` (GraphQA, our own 16-name palette, superseded): 0.831 / 0.855 / 0.847.

**Significance.** Each method was compared with `text` per cell (2026-10-27): exact McNemar plus a paired bootstrap,
96 comparisons, before the KG `fullft-text` and faithful runs existed.
- `graphtoken-frozen` is significant in 8 of 9 cells and survives Bonferroni.
- Of the other 85 comparisons, 4 reach p < 0.05, which is the chance rate. None survives a correction, and their
  signs disagree: `text-colors` HRM-1B is +1.3 on MetaQA but −2.8 on KQA Pro.
- The nominal `node-tokens` gain on KQA Pro (+1.1 pooled, p = 0.023) does not survive either. Against its identity
  control `abs-id` it is +146 / −118 (p = 0.096).

## Findings

- **No structural injection beats `text` + LoRA.** Every LoRA arm is within about ±1.5 pt of `text` on average,
  including the no-graph control `prompt-tuning`. The identity controls are not beaten either:
  - `abs-pe` vs `abs-id`: GraphQA pooled +23 / −29;
  - `rel-bias` vs `rel-bias-elem`: MetaQA pooled +59 / −51 (p = 0.50).

  Reading (medium confidence): with the graph written in the prompt, LoRA on the text already gets what these
  encodings could add.
- **The injected signals are used, but carry nothing new.**
  - **`abs-pe`:** φ is not ignored. Its RMS on node tokens is 0.17–0.30, against a token-embedding RMS of 0.43–1.02,
    unlike the GRL gate. `abs-id` learns an even larger φ on HRM-1B (0.78).
  - **`rel-bias` at method lr 1e-3:** β stays at |β| ≈ 0.03, flat across distances, on all 3 datasets.
  - **`rel-bias-hilr` at 2e-2** (20×, the value GTLM's sweeps pick): β grows about 20× (max 3.6) but has no distance
    structure. The median per-head Spearman correlation with distance is 0.02, which is chance level, and accuracy is
    unchanged (0.871). So the null result is not a learning-rate artifact.
- **The hard cases don't move for any method.**
  - MetaQA 3-hop stays at 0.75–0.78.
  - On GraphQA the remaining errors are counting tasks: TriangleCounting 0.42–0.60 and EdgeCount 0.42–0.73.
    `text-incident` lowers EdgeCount because it states each edge twice.
  - CycleCheck, EdgeExistence, NodeCount and Reachability are at 1.00 for every LoRA arm.
  - KQA Pro Count is lower with `rel-bias` on all 3 models (about 40 samples each; not tested).
- **Freezing the LLM costs 4–15 pt** (`graphtoken-frozen`, graph text kept). The exception is Mimir v1.5 on MetaQA,
  which ties. The losses are on node-list and counting tasks.
- **Without the graph text, graph tokens don't replace it** (`-faithful` arms).
  - **GraphQA:** about 30 pt below `text`. CycleCheck and Reachability stay at 0.94–1.00, and EdgeExistence (0.68–0.91)
    and ShortestPath (0.65–0.82) partly hold. Node lists and counts collapse:
    - ConnectedNodes 0.32–0.44;
    - DisconnectedNodes 0.05–0.12;
    - EdgeCount 0.15–0.35;
    - NodeDegree 0.28–0.42.

    The tokens carry coarse topology, not node-level detail.
  - **MetaQA / KQA Pro:** near closed-book (0.11–0.36), because the answer strings never reach the model. LLaGA's
    tokens, which carry the entity-name embeddings, beat GraphToken's on MetaQA (0.27–0.29 vs 0.11–0.13).
- **Full fine-tuning overfits on the KGs** (−4 to −18 pt vs LoRA; tied on GraphQA, which confirms the trainer).
  - Train loss goes to about 0. Val peaks at epoch 1–2, then falls; for example, Mimir v1.5 on KQA Pro goes 0.515 →
    0.463 → 0.445.
  - The `pretrain.py` KG fine-tunes in [Graph Reasoning Experiments](/pages/hrm-graph/graph-reasoning-experiments.md)
    land at the same level (MetaQA 0.836 / 0.837 / 0.650, KQA Pro 0.632 / 0.698 / 0.600).
  - So that recipe (`cfg_{metaqa,kqapro}_*`, lr 5e-5) understates these models by about 5–15 pt. LoRA r=16 at lr
    2e-4 is the stronger baseline. Mimir v1.5 is hurt most.
- **A stronger GNN is not the lever on the KGs.**
  - With the graph text kept, the soft tokens are redundant with it.
  - In the faithful arms the encoder sees only anonymous, undirected topology: RWSE + LapPE + IDX, with no entity
    names and no relation types. For example, the answer node `2009` looks identical to `French`, `Horror` and the
    actors in its star subgraph.
  - What would help is text-derived node and edge features fed to a relation-aware encoder (GINE /
    TransformerConv with `edge_attr`), plus per-node tokens, G-Retriever style.
  - A GPS arm is only worth trying on GraphQA.

## Fixes and fails

**Paper-fidelity fixes (2026-10-24).** These come from the review in
[`tests/reports/graph_inject_paper_review.md`](../../../tests/reports/graph_inject_paper_review.md), checked against the
papers and official code. Ten minor items (N1–N10) remain and are documented in the README.
- **`rel-bias`:** token pairs within one span occurrence now get bias 0, as GTLM's b_SPD(u, u) = 0. This is confirmed
  in `DarioVajda/graph_model` @ f4feefb, where `SPDBias` multiplies by `spd > 0`. Different occurrences of one element
  keep a learned distance-0 bucket, which ties question mentions to the node's facts.
- **`text-colors`:** CL-OWL's own colour naming and the paper's instructions now replace our 16-name palette.
  - The official repo divides RGB by 255 before its 0–255 thresholds, which would make every node "lime".
  - On 0–255 RGB, the reachable names are exactly the paper's example colours. The GraphQA `text-colors16` runs are
    superseded.
- **New arms:** `graphtoken-faithful` and `node-tokens-faithful`, at most 8 samples per step. Question-only prompts
  would otherwise give only 15–50 steps per epoch.
- **Relabelled, code unchanged:**
  - `text` is Less is More's LoRA arm, not its CoT recipe.
  - `graphtoken` and `graphtoken-frozen` are "Graph Tokens + Text". The Grover et al. label was wrong.
  - `abs-id` is a random identity control, not GraphToken's IDX.
- **Runs:** the 45 finished runs on unchanged code paths were kept. The `rel-bias` run in flight was killed and
  deleted from W&B.

**Fail: KQA Pro `text-colors` overflowed the context (2026-10-24, found by the smoke run).**
- **Symptom:** a CUDA gather assert.
- **Cause:** the colour block pushes 0.6% of prompts past the models' 4,096-position RoPE table (up to 5,233 tokens).
- **Fix:** the block is cut to the first nodes that fit 3,990 prompt tokens (`text.PROMPT_TOKEN_BUDGET`). That
  affects 4 test prompts for HRM and 3 for Mimir; the colours are still computed on the whole graph.

**Fail: queue relaunched in a new SSH session without W&B credentials (2026-10-27).**
- **Symptom:** every run died at `wandb.init` with `UsageError: No API key configured`. `run_matrix.sh` logs `FAILED`
  and moves on, so it would have gone through the whole queue in minutes.
- **Fix:** `wandb login --host https://forge.coreweave.com`, check that `wandb.Api().viewer` works, then relaunch.
  Finished runs are skipped.

## Seed variance of `text` (2026-10-29)

`text` was rerun with seeds 1 and 2 on all 9 cells (2026-10-28 08:55 to 23:46, no failures). Test `greedy_acc` in
points:

| Dataset | HRM-1B (s0 / s1 / s2) | Mimir v1 | Mimir v1.5 | Pooled seed sd |
| --- | --- | --- | --- | ---: |
| GraphQA | 82.9 / 81.6 / 83.4 | 85.7 / 82.9 / 83.9 | 84.2 / 85.5 / 85.7 | 1.1 |
| MetaQA | 86.5 / 87.5 / 86.5 | 87.5 / 86.7 / 87.2 | 87.7 / 87.8 / 88.7 | 0.5 |
| KQA Pro | 73.9 / 74.9 / 73.5 | 74.2 / 75.0 / 75.3 | 74.8 / 76.4 / 74.8 | 0.75 |

- **This is training noise only.** The test set is fixed, so the spread comes from the init and the data order.
  Changing only the seed moves one run by up to 2.9 pt (GraphQA, Mimir v1). The best epoch also moves: on GraphQA it
  was 8–9 for seed 0 and 4 for seed 1.
- **Every seed-0 method is now compared with the 3-seed `text` mean.**
  - Δ is averaged over the 3 models.
  - z = Δ / (sd·√(4/3)/√3) uses the dataset's pooled seed sd: one method run against a 3-seed mean, over 3
    models.
  - Leaving out the arms that clearly lose (frozen, faithful, KG `fullft-text`), the largest effects are:

  | Method | GraphQA Δ (z) | MetaQA Δ (z) | KQA Pro Δ (z) |
  | --- | ---: | ---: | ---: |
  | `text-incident` | −0.4 (−0.5) | +0.4 (+1.3) | +0.6 (+1.1) |
  | `node-tokens` | +0.2 (+0.3) | −0.4 (−1.1) | +0.6 (+1.3) |
  | `abs-pe` | −0.6 (−0.8) | +0.1 (+0.2) | +0.1 (+0.2) |
  | `prompt-tuning` (no graph) | −1.2 (−1.6) | 0.0 (−0.1) | −0.8 (−1.6) |
  | `abs-id` (control) | −0.1 (−0.1) | −0.7 (−2.1) | −0.2 (−0.4) |

  - No LoRA arm reaches |z| > 2 except `abs-id` on MetaQA (−2.1), which is one of 33 tests.
  - The `node-tokens` KQA Pro gain shrinks from +1.1 against seed 0 to +0.6 against the 3-seed mean.
- **Conclusion.** The seed sd is ≳ 1 pt on GraphQA and about 0.5–0.75 on the KGs, and no LoRA method moves more than
  ±1.2 pt on average. **Structural injection does not measurably improve on `text` + LoRA** for these models and
  datasets. The negative effects (frozen LLM, no graph text, KG full fine-tuning) are 4–75 pt, far outside the
  noise.
- **The one consistent pattern is `text-incident` on the KGs.** It is above the 3-seed `text` mean in all 6 KG cells
  (+0.1 to +0.9; a sign test gives p = 0.03), but below it on GraphQA. It is the only candidate a seeds-1–2 check
  could confirm. That check is 12 runs, about 10 h (estimated from the seed-0 run times). Not run.
