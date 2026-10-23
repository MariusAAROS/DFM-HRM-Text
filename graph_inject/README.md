# graph_inject: graph-structure injection for pretrained HRM-Text models

This package fine-tunes HRM-Text-1B, DFM-Mimir v1 and DFM-Mimir v1.5 on GraphQA, MetaQA and KQA Pro. Each run uses
one way of injecting the graph structure, and every method is trained and evaluated by the same trainer, so the
results compare like for like.

The package is standalone. It does not modify `pretrain.py`, `models/` or the eval scripts. It reuses them by
import:
- the model loader and decoding from `simple_inference_engine`;
- the PrefixLM batch helpers;
- the metrics from `scripts/graphs/eval_graphqa.py`.

## Running

```bash
# one run (dataset defaults to graphqa)
PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m graph_inject.train dataset=metaqa model=mimir1 method=graphtoken [seed=1 epochs=5 ...]

# the whole matrix, detached, behind the GPU lock (skips finished runs)
setsid nohup flock /work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock bash scripts/graph_inject/run_matrix.sh \
  >> /work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject/queue.log 2>&1 < /dev/null &

# results table
.venv/bin/python -m graph_inject.summarize --dataset metaqa --per_task

# tests (CPU parity + GPU method checks on HRM-Text-1B); single-threaded BLAS, or LapPE crawls
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_graph_inject.py
```

- **Configs.** `configs/base.yaml` is overridden by `configs/datasets/<dataset>.yaml` (data, epochs, out_root,
  W&B group), then `configs/models/<model>.yaml`, then `configs/methods/<method>.yaml` (its `train:` block), then
  the CLI. `data.limit=N` keeps the first N rows of every split (smoke runs).
- **Outputs.** `<out_root>/<model>/<method>_s<seed>/` holds `config.yaml`, `eval_val_ep<N>.json`,
  `eval_test.json` and `best_trainable.pt`. The out_root is `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject`
  for GraphQA, and `graph_inject_metaqa` / `graph_inject_kqapro` next to it.
- **W&B.** Project `Graph-Positional-Encoding-Exploration`, group `<dataset>-inject` (always online). Run names
  are `<model>-<method>-s<seed>`, prefixed with the dataset outside GraphQA.

## Datasets

| Dataset | Graph in the prompt | Parser | Epochs |
| --- | --- | --- | --- |
| GraphQA | integer nodes, `(u, v)` edge tuples | `structure.parse_graphqa` | 10 |
| MetaQA | GRL gold subgraph, about 72 `(s, p, o)` facts, 9 relations | `kg_structure.parse_kg_facts` | 3 |
| KQA Pro | GRL gold subgraph, about 57 facts, 767 relations / attributes, qualifiers `[k: v; ...]` | same | 3 |

On the knowledge graphs, a node is a distinct entity or value string, an edge is a fact (s, o), and each qualifier
`k: v` is an extra edge (s, v). Entity names and relations may contain `", "`, so facts are split where the middle
field is a known relation (`kb.txt` / `kb.json`). Every row parses and re-serializes to its own line. 4% of KQA Pro
prompts have no facts at all; every method handles the empty graph. Entity names on the `Q:` line are node mentions
(99.9% of MetaQA questions, 82% of KQA Pro ones, which often describe the entity instead of naming it).

## Recipe

| Setting | Value |
| --- | --- |
| Regime | LoRA r=16, α=32 on every H/L projection (`gqkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`) |
| Base weights | frozen, bf16 |
| Optimizer | AdamW, LoRA lr 2e-4; the method's own modules lr 1e-3, no weight decay |
| Schedule | 20 warmup steps, then constant LR; clip 1.0 |
| Batching | 8192 tokens per step (packed PrefixLM) |
| Model | H2L3, bp_max_steps 5 |
| Length | 10 epochs |
| Evaluation | greedy generation on val every epoch; test (always the full split) at the best val epoch by `tf_em`. MetaQA / KQA Pro: val is a fixed, task-stratified 400-sample subset (`eval.val_subset`, seed 0), the same for every run and epoch |

`seed` sets the data order and the initialization of the new modules.

## Methods

| Config | Family | What it is | Source of the code |
| --- | --- | --- | --- |
| `text` | hard prompting | GraphQA adjacency text + LoRA ("Less is More", arXiv 2508.10115). **Reference.** | `lora.py` (own: peft cannot wrap `LinearInit`) |
| `fullft-text` | hard prompting | Same text, full fine-tune, the `cfg_graphqa_hrm1b` recipe. Trainer check against `pretrain.py` (0.82-0.84 test). | |
| `text-incident` | hard prompting | Talk like a Graph incident encoding ("Node 0 is connected to nodes 1, 2.") | google-research encoders as ported in GRL `graph_text_encoders.py` |
| `text-colors` | hard prompting | Colorful Talks C-OWL: ordered 1-WL classes as colour words in a `<<COLORS>>` block | no public code: paper template; WL from GRL `graph-enc` |
| `prompt-tuning` | static soft prompt | 8 P-Tuning tokens (MLP reparametrization). Capacity control: no graph information. | `peft.tuners.p_tuning.PromptEncoder` |
| `graphtoken` | graph-level soft tokens | GIN(RWSE+LapPE) → [mean ‖ sum] pool → 8 soft tokens, + LoRA (= graph pooling tokens) | PyG `GIN`; no official PyTorch GraphToken code found |
| `graphtoken-frozen` | graph-level soft tokens | Same, with the LLM frozen (the original GraphToken setting) | |
| `node-tokens` | node-level soft tokens | LLaGA hop-field template: per node, [name emb ‖ RWSE ‖ LapPE] at hops 0..2 → 2-layer MLP → 3 tokens | ported from `VITA-Group/LLaGA` @ 288240f |
| `abs-pe` | absolute (additive) | φ(c(i)) added to every token of a node or edge (zero-init). Node: MLP(RWSE+LapPE). Edge: MLP of its endpoints. **The proposed method.** | parser and alignment ported from GRL `graph-enc` (a3d3418) |
| `abs-id` | absolute (additive) | Same architecture, with random per-node vectors redrawn every sample. **Identity control.** | |
| `rel-bias` | relative (attention bias) | Learned β[layer, head, bucket] added to the attention score of every pair of graph-element tokens; bucket = Levi-graph distance 0..8, >8, unreachable (GTLM / Graphormer spatial encoding). Zero-init. | no public GTLM code; `attention_bias.py` (FlexAttention) |
| `rel-bias-elem` | relative (attention bias) | Same, with buckets same element / other element only. **Identity control.** | |

On the knowledge graphs, `text-incident` groups the facts by subject (`Santa Sangre: written by Claudio Argento,
Roberto Leoni; has tags circus.`): Talk like a Graph has no KG encoder, so this one is ours. `text-colors` lists
`Color(<entity name>):c` from 1-WL on the undirected entity graph. `node-tokens` embeds the entity name (mean of the
frozen token embeddings of its tokens) instead of the digits of v.

### How the methods plug in

- **Soft tokens.** Soft-token methods insert placeholder slots before the prompt content (after
  `<|im_start|><cond>` / `<bos><|turn>user\n`) and overwrite their embeddings. The prompt is the bidirectional
  PrefixLM prefix, so the slots see the whole graph text and the text sees the slots.
- **Absolute φ.** φ follows the graph element, not the token. Every token of node 12 (`␣`, `1`, `2`) gets φ(12)
  at every mention: the node list, the edge tuples and the question. The `(`, `,` and `)` of the tuple
  `(3, 12)` get φ(edge). Template words get nothing. In `Micky Mike --is friend--> Jacky Jack`, the two names
  would share φ(node₁) and φ(node₂), and `is friend` would share φ(edge).
- **Relative bias.** `install_relative_bias` switches every `models.layers.Attention` to `GraphBiasAttention`
  by `__class__`; `models/` is not edited. While the method's context is set, training uses FlexAttention with the
  document + PrefixLM mask of the packed batch (the FA2 varlen kernel takes no bias), and the prompt prefill adds the
  same bias to the SDPA float mask. Decode steps and every other method run the parent class. The bias is gathered
  per layer into `[H, NDOC, EL, EL]` once per forward: indexing the small `[H, buckets]` table inside the kernel
  makes its backward ~17x slower (atomic adds). Pairs where either token has no element (template, response) get 0.
- **Where the injection enters the model.** HRM-Text-1B and Mimir have `inject_x=False`, so the token embedding
  enters only as the initial `z_H`. A test checks that gradient still reaches every method's modules.

### Caveats

- MaximumFlow is unanswerable from the GraphQA text, because capacities are not printed. It is 34/385 test
  samples, and every model scores about 0.18 on it. `greedy_acc_answerable` excludes it.
- Splits are by question, not by graph: every test graph also appears in train. A graph encoder can memorize
  graphs, so compare against `abs-id` and `prompt-tuning`, and look at the per-task results.
- Under full fine-tuning in GRL `graph-enc`, node and WL embeddings behind a scalar gate gave no gain (the gate
  stayed at ~0.002). `abs-pe` differs: it uses zero-init projections, a separate higher LR, LoRA and edge
  vectors.

## Set aside for now

| Method | Why |
| --- | --- |
| Prefix tuning, GraphLLM | Per-layer KV prefixes across the H/L recursion and the varlen kernel. |
| K-BERT | Needs a visibility mask in attention. |
| GraphGPT, NT-LLM | Need their own alignment or anchor pre-training stages. |
| G-Retriever, GNP | Retrieval over a knowledge graph. GRL's MetaQA / KQA Pro prompts already carry a gold subgraph. |
| Weaver, NAG-LoRA | Formulas not verified against the papers. |

## Files

- `structure.py`: `PromptGraph` and the GraphQA parser. It turns a prompt into a graph and gives the character →
  graph element map.
- `kg_structure.py`: the MetaQA / KQA Pro fact parser into the same `PromptGraph`.
- `attention_bias.py`: Levi-graph buckets, `GraphBiasAttention` and its FlexAttention / SDPA paths.
- `features.py`: RWSE, LapPE and the LLaGA hop features.
- `tokenize.py`: ids identical to data_io, the token → element map and the soft slots.
- `data.py`: encoding and packed batches.
- `lora.py`: the LoRA layer.
- `model.py`: `GraphInjectedLM`.
- `methods/`: one class per method family, sharing the `base.GraphMethod` interface.
- `generate.py`: KV-cache greedy decoding and metrics.
- `train.py`: the trainer.
- `summarize.py`: the results table.
