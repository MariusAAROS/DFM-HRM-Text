---
type: Runbook
title: Mimir v1.5 Graph Benchmark - Results and Run Log
description: "Results and run log of the Mimir v1.5 graph benchmark from 2026-11-03: E0 zero-shot matrix (7 models), E1/E2 LoRA SFT for Mimir v1.5, Qwen2.5-7B and OLMo-3-7B, the generation-cap artefact on trace-trained models, pass@16 headroom for GRPO, and the failures and protocol changes made on the way (memory guard, eval split keys, GRPO response length)."
tags: [graphs, mimir, benchmark, sft, lora, grpo, kgqa, clrs, clutrr, qwen, olmo]
status: draft
last_updated: 2026-11-05
confidence: medium
---
# Mimir v1.5 Graph Benchmark - Results and Run Log

Results of the benchmark designed in [Mimir v1.5 Graph-Reasoning Benchmark](/pages/hrm-graph/mimir-graph-benchmark.md)
(plan, hypotheses, datasets, arms). That page holds the design and the launch; this one holds the numbers and what
changed while running.

- Reports: `/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench/<arm>/<model>/{named,anon}[_t3k].json` with
  `.samples.jsonl` next to each. Per-split numbers: `graph_bench.eval.log_wandb.metrics` on the samples file.
- Queues: `graph_bench/eval/run_e0.sh`, `graph_bench/run_sft.sh` (Mimir E1/E2), `graph_bench/run_sft_baselines.sh`
  (baseline E1/E2, Mimir E2 re-eval, GRPO resume check), `graph_bench/rl/run_rl_arms.sh` (GRPO).
- Baseline SFT trainer: `graph_bench/sft_hf.py`.
- W&B: project `Mimir-Graph-RLVR`; eval runs in `mimir-bench-{e0,e1,e2,mergecheck}`, baseline SFT training in
  `mimir-bench-sft-baselines`, Mimir SFT training in `mimir-bench-sft`.

Conditions for every table: greedy, one seed, up to 1,000 random rows per test file, metric `acc` (normalised answer
match on the `Answer:` line). "Named" = the 12 test files; "anon" = MetaQA / KQA Pro / CWQ test with entity names
replaced (rows whose question loses its entity are dropped, which removes all of WebQSP). Macro = mean over the 7
datasets, each pooling its files. Differences under ~0.02 are within noise.

## E0 zero-shot matrix (2026-11-03)

Instruct models: instruct prompt, thinking off, 1,024 tokens. HRM-Text-1B: base style, 1 shot. Gemma-4-E4B failed to
load (below), so 7 models.

| Model | Macro named | Macro anon | Share with an `Answer:` line |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | **0.405** | 0.438 | 0.94 |
| Qwen3.5-4B | 0.392 | **0.464** | 0.62 |
| OLMo-3-7B-Instruct | 0.377 | 0.433 | 0.91 |
| SmolLM3-3B | 0.289 | 0.339 | 0.86 |
| DFM-Mimir v1 | 0.233 | 0.257 | 0.40 |
| **DFM-Mimir v1.5** | 0.198 | 0.280 | 0.38 |
| HRM-Text-1B (base, 1-shot) | 0.184 | 0.138 | n/a (answers directly) |

- **Mimir v1.5 is last among the instruct models zero-shot**, mainly from format: with thinking off it still
  reasons at length, often loops, and ends with an `Answer:` line in 38% of rows (KQA Pro 33%, WebQSP 26%, CLRS 2%).
- **CLRS is ~0 for every model** (≤ 0.006): an untrained task with matrix answers.
- **CLUTRR is near 0 for Mimir (0.010)** partly from scoring: it answers in sentences ("Lynn is Wesley's aunt"),
  which never match the one-word gold.
- Retrieved subgraphs cost every model 0.1-0.3 against gold subgraphs (Mimir v1.5: MetaQA 0.584 → 0.390, KQA Pro
  0.259 → 0.131).

## E1 / E2 SFT: Mimir v1.5 and baselines (2026-11-04/05)

Same rows (17,917; E2 17,916, see fails), same recipe: LoRA r16 / alpha 32 on every block projection, lr 2e-4,
20 warmup steps then constant, 8,192 tokens/step, 3 epochs, last epoch merged. Mimir via `graph_inject`; baselines via
`graph_bench/sft_hf.py`. E1 evaluated at 1,024 tokens (as E0); E2 at 3,072 (see the cap artefact below).

| Split | Mimir E0 | **Mimir E1** | Mimir E2 | Qwen2.5 E0 | Qwen2.5 E1 | Qwen2.5 E2 | OLMo-3 E0 | OLMo-3 E1 | OLMo-3 E2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GraphQA test | 0.423 | 0.858 | **0.960** | 0.543 | 0.676 | 0.923 | 0.522 | 0.698 | 0.932 |
| GraphQA large | 0.354 | 0.732 | **0.800** | 0.431 | 0.597 | 0.747 | 0.433 | 0.621 | 0.785 |
| MetaQA gold | 0.584 | **0.971** | 0.939 | 0.689 | 0.878 | 0.939 | 0.662 | 0.860 | 0.865 |
| MetaQA retrieved | 0.390 | **0.732** | 0.645 | 0.606 | 0.730 | 0.704 | 0.620 | 0.709 | 0.655 |
| KQA Pro gold | 0.259 | 0.924 | **0.936** | 0.671 | 0.913 | 0.904 | 0.556 | 0.886 | 0.911 |
| KQA Pro retrieved | 0.131 | **0.498** | 0.412 | 0.329 | 0.495 | 0.413 | 0.280 | 0.453 | 0.417 |
| WebQSP | 0.157 | **0.781** | 0.712 | 0.565 | 0.755 | 0.701 | 0.479 | 0.754 | 0.703 |
| CWQ (held out) | 0.141 | 0.675 | 0.529 | 0.468 | **0.694** | 0.504 | 0.530 | 0.640 | 0.393 |
| CLUTRR (held out) | 0.010 | 0.192 | 0.118 | 0.188 | **0.252** | 0.013 | 0.096 | 0.223 | 0.026 |
| CLRS test | 0.003 | **0.725** | 0.641 | 0.006 | 0.527 | 0.458 | 0.003 | 0.608 | 0.623 |
| CLRS OOD (held out) | 0.002 | **0.212** | 0.060 | 0.001 | 0.060 | 0.023 | 0.000 | 0.023 | 0.056 |
| **Macro named** | 0.198 | **0.642** | 0.583 | 0.405 | 0.589 | 0.541 | 0.377 | 0.577 | 0.538 |
| **Macro anon** | 0.280 | **0.814** | 0.750 | 0.438 | 0.732 | 0.723 | 0.433 | 0.758 | 0.685 |

- **After the same answer-only SFT, Mimir v1.5 (the smallest) is the best of the three** (macro 0.642 vs 0.589 /
  0.577), with the largest leads on CLRS and GraphQA, and it keeps the lead with entities anonymised (0.814), so it
  is not entity memorisation. The baselines are ahead only on the held-out CWQ and CLUTRR (Qwen2.5).
- **Training on traces (E2) helps GraphQA in every model** (Mimir 0.858 → 0.960, Qwen2.5 0.676 → 0.923, OLMo-3
  0.698 → 0.932, large graphs included) **and lowers macro accuracy in every model** (−0.04 to −0.06): retrieved
  subgraphs, WebQSP, CWQ and CLUTRR all drop. For KGQA the hop-by-hop path trace adds length without accuracy.
- **E2 breaks CLUTRR transfer for the baselines** (Qwen2.5 0.252 → 0.013, OLMo-3 0.223 → 0.026). Qwen2.5 E2 (checked)
  writes an `Answer:` line 98% of the time but applies the KGQA "find the entity" pattern and answers a person's name
  ("Answer: Melody.") where the question asks for a relation ("niece"); OLMo-3's outputs were not inspected. Mimir E2
  keeps answering relations (0.118).
- **Gaps left for RL in every model:** retrieved vs gold subgraph (KQA Pro ~0.45 vs ~0.92), CLRS OOD (≤ 0.212),
  CLUTRR (≤ 0.252).
- Training cost on the RTX PRO 6000: Mimir E1 1,636 s/epoch, E2 ~2,250-2,530 s/epoch; Qwen2.5-7B 4,700 tokens/s
  (E1 ~3,050 s/epoch, 36 GB peak); OLMo-3-7B E1 ~3,900 s/epoch; **Qwen3.5-4B ~1,020 tokens/s (4.0 h/epoch,
  52 GB)**: its linear-attention layers run on the PyTorch fallback, as `flash-linear-attention` and `causal-conv1d`
  are not installed.
- E2 runs more optimizer steps than E1 at the same epochs (Mimir 8,010 vs 5,859): steps hold 8,192 tokens and the
  traces make an epoch 1.34× longer (19.3M vs 14.4M tokens with the Qwen tokenizer).

## The generation cap understated E2 (2026-11-04)

E2 was first evaluated at 1,024 tokens, like E0/E1. Its traces are often longer: in the training data, 20% of CLRS and
22% of GraphQA traces exceed 1,024 Mimir tokens (p90 ~1,500-1,700, max 3,239). Cut-off traces end without an
`Answer:` line and score 0 (CLRS OOD: 28% of rows had one). Re-evaluated at 3,072 (`named_t3k`, `anon_t3k`;
`eval_vllm` clamps each prompt to the 4,096 context):

| Mimir E2 | @1,024 | @3,072 |
| --- | --- | --- |
| GraphQA test / fresh / large | 0.910 / 0.931 / 0.749 | 0.960 / 0.968 / 0.800 |
| CLRS test / OOD | 0.519 / 0.036 | 0.641 / 0.060 |
| KGQA, CWQ, CLUTRR | | unchanged (±0.005) |

- **Rule: evaluate trace-trained models with room for the longest training trace.** The 1,024-token E2 reports are
  kept for comparison only.
- The same cap applied to GRPO responses, where a cut-off trace gets reward 0 and RL would push towards shorter
  traces. GRPO now uses 2,048 response + 2,048 prompt tokens (Mimir's 4,096 context). This covers 97.5% of CLRS and
  94% of GraphQA traces and drops no RL rows: the long-trace tasks have short prompts (CLRS p90 595, GraphQA 803),
  the long-prompt KGQA tasks have short traces (p90 ≤ 350).

## GRPO headroom from E2: pass@16 (2026-11-04)

T = 1, 16 samples, 200 rows per file.

| Val split (2,048 tokens) | pass@1 | pass@4 | pass@16 |
| --- | --- | --- | --- |
| GraphQA | 0.972 | 0.990 | 0.995 |
| MetaQA | 0.863 | 0.950 | 0.965 |
| KQA Pro | 0.842 | 0.931 | 0.945 |
| CLRS | 0.648 | 0.793 | 0.885 |
| WebQSP | 0.578 | 0.694 | 0.735 |

- 409 of 1,000 val prompts have mixed outcomes, i.e. a GRPO signal under `filter_groups`; 95 are never solved.
  GraphQA is saturated; the headroom is in CLRS, WebQSP and KQA Pro.
- Held-out (1,024 tokens, so CLRS OOD and GraphQA large are understated): CLUTRR pass@1 0.081 → pass@16 0.270
  (room for E4), CWQ 0.461 → 0.610, CLRS OOD 0.045 → 0.105.

## Fails and fixes (2026-11-03..05)

- **Gemma-4-E4B does not load in vLLM 0.27.1** (transformers `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a
  per-layer attribute`). Not in E0; needs a vLLM/transformers change.
- **E2 training stopped before step 1**: `train rows [10018] exceed 4096 tokens at the coarsest detail` (a CLRS
  Kruskal row, 4,086-character matrix trace). Dropped from `train_cot.jsonl` (original kept as
  `train_cot.jsonl.orig17917`); E1 kept its answer-only version, so the arms differ by that one row.
- **The memory guard killed a healthy eval** (E2 held-out pass@16 at 73%, exit 137): `mem_guard.sh` compared the
  container's `memory.current` with 110 GB, and 104 GB of it was page cache from model loading (the eval used
  6.6 GB). It now counts only unreclaimable memory, `memory.current − file + shmem` (anon, kernel, `/dev/shm`). A
  queue that waits for an output file stalls when that step fails: the GPU sat idle 1 h 15 min.
- **W&B eval runs showed no panels and merged splits.** The logger wrote summary values only, keyed by dataset, so
  GraphQA test/large/fresh and gold/retrieved were one number (the retrieved files reuse the gold ids). `eval_vllm`
  now records each row's `source` file (`file/<dataset>/<stem>` in the summary); `log_wandb` (v2) logs
  `split/<dataset>/<file>/{acc,f1,answer_rate,pass@k}` as history plus a `metrics` table, and upgraded the old runs
  in place (split rebuilt from row order, checked against hand-computed numbers).

## Protocol changes (2026-11-04)

- **Baselines get E1/E2** (user decision): Qwen2.5-7B, OLMo-3-7B, Qwen3.5-4B, before GRPO. `sft_hf.py` trains on the
  exact eval prompt ids (checked against `eval_vllm`; prompt + response + the template's stop token, loss on the
  target only), rows one at a time with gradient accumulation. Its merge adds `B @ A` to the base checkpoint's own
  tensors (suffix-matched keys), so Qwen3.5's vision-language checkpoint stays loadable by vLLM; it matches peft's
  `merge_and_unload` within bf16 rounding.
- **GRPO:** response 2,048 tokens, micro-batches 2 (from 4, so fewer tokens per pass than the smoke setting),
  checkpoints every 25 steps (from 50), all kept (~7 GB each).
- **GRPO resume checked**: a copy of a smoke checkpoint resumed at `global_step_1` and trained step 2 (model,
  optimizer, RNG, LR scheduler and data position reloaded).

## Still running / next (2026-11-05)

- Qwen3.5-4B E1/E2: at ~1,020 tokens/s about 12 h per E1 and 16 h per E2.
- GRPO E3 → E4 (+ SFT control) → E3b after the baselines. Estimate at the longer responses: ~9-12 min/step.
- Not covered: eval of the GRPO checkpoints, retention suite (H4), Gemma-4, Ouro, IFM, Tier 2 sets.
