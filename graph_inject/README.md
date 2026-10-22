# graph_inject: graph-structure injection for pretrained HRM-Text models on GraphQA

This package fine-tunes HRM-Text-1B, DFM-Mimir v1 and DFM-Mimir v1.5 on GraphQA. Each run uses one way of
injecting the graph structure, and every method is trained and evaluated by the same trainer, so the results
compare like for like.

The package is standalone. It does not modify `pretrain.py`, `models/` or the eval scripts. It reuses them by
import:
- the model loader and decoding from `simple_inference_engine`;
- the PrefixLM batch helpers;
- the metrics from `scripts/graphs/eval_graphqa.py`.

## Running

```bash
# one run
PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m graph_inject.train model=mimir1 method=graphtoken [seed=1 epochs=5 ...]

# the whole matrix, detached, behind the GPU lock (skips finished runs)
setsid nohup flock /work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock bash scripts/graph_inject/run_matrix.sh \
  >> /work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject/queue.log 2>&1 < /dev/null &

# results table
.venv/bin/python -m graph_inject.summarize --per_task

# tests (CPU parity + GPU method checks on HRM-Text-1B)
.venv/bin/python -m pytest -q tests/test_graph_inject.py
```

- **Configs.** `configs/base.yaml` is overridden by `configs/models/<model>.yaml`, then
  `configs/methods/<method>.yaml` (its `train:` block), then the CLI.
- **Outputs.** `/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject/<model>/<method>_s<seed>/` holds
  `config.yaml`, `eval_val_ep<N>.json`, `eval_test.json` and `best_trainable.pt`.
- **W&B.** Project `Graph-Positional-Encoding-Exploration`, group `graphqa-inject` (always online).

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
| Evaluation | greedy generation on val every epoch; test at the best val epoch by `tf_em` |

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

### How the methods plug in

- **Soft tokens.** Soft-token methods insert placeholder slots before the prompt content (after
  `<|im_start|><cond>` / `<bos><|turn>user\n`) and overwrite their embeddings. The prompt is the bidirectional
  PrefixLM prefix, so the slots see the whole graph text and the text sees the slots.
- **Absolute φ.** φ follows the graph element, not the token. Every token of node 12 (`␣`, `1`, `2`) gets φ(12)
  at every mention: the node list, the edge tuples and the question. The `(`, `,` and `)` of the tuple
  `(3, 12)` get φ(edge). Template words get nothing. In `Micky Mike --is friend--> Jacky Jack`, the two names
  would share φ(node₁) and φ(node₂), and `is friend` would share φ(edge).
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
| Relative attention bias (GTLM, Graphormer bias, GLM) | Phase 2. The SM120 training kernel (FA2 varlen) takes no bias, so this needs a FlexAttention training path and an SDPA bias at inference. |
| Prefix tuning, GraphLLM | Per-layer KV prefixes across the H/L recursion and the varlen kernel. |
| K-BERT | Needs a visibility mask in attention. |
| GraphGPT, NT-LLM | Need their own alignment or anchor pre-training stages. |
| G-Retriever, GNP | Retrieval over a knowledge graph. Revisit for MetaQA / KQA Pro. |
| Weaver, NAG-LoRA | Formulas not verified against the papers. |

## Files

- `structure.py`: the GraphQA parser. It turns a prompt into a graph and gives the character → graph element map.
- `features.py`: RWSE, LapPE and the LLaGA hop features.
- `tokenize.py`: ids identical to data_io, the token → element map and the soft slots.
- `data.py`: encoding and packed batches.
- `lora.py`: the LoRA layer.
- `model.py`: `GraphInjectedLM`.
- `methods/`: one class per method family, sharing the `base.GraphMethod` interface.
- `generate.py`: KV-cache greedy decoding and metrics.
- `train.py`: the trainer.
- `summarize.py`: the results table.
