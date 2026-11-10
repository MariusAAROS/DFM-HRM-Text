---
type: Runbook
title: Mimir v1.5 Graph Benchmark - Results and Run Log
description: "Results and run log of the Mimir v1.5 graph benchmark from 2026-11-03: E0 zero-shot matrix (7 models), E1/E2 LoRA SFT for Mimir v1.5, Qwen2.5-7B and OLMo-3-7B, the generation-cap artefact on trace-trained models, pass@16 headroom for GRPO, E3 GRPO test results, the GLM-5.3 zero-shot frontier reference, and the failures and protocol changes made on the way (memory guard, eval split keys, GRPO response length)."
tags: [graphs, mimir, benchmark, sft, lora, grpo, kgqa, clrs, clutrr, qwen, olmo]
status: draft
last_updated: 2026-11-10
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
- Superseded (2026-11-09): E3 finished and was evaluated, see [E3](#e3-grpo-from-e2-results-2026-11-09).

## GLM-5.3 as a zero-shot frontier reference: setup (2026-11-09)

**Why.** GLM-5.3 gives a frontier-model ceiling for every test set of the E0–E4 tables. It's far larger than the
≤ 7B models, so it's a reference row, not an H1 competitor. It's only reachable through an API, so it's evaluated
zero-shot only.

- **Code:**
  - [`graph_bench/eval/eval_api.py`](../../../graph_bench/eval/eval_api.py) is `eval_vllm.py`'s counterpart for
    OpenAI-compatible APIs. It uses the same `load` subsets (seed 0, ≤ 1,000 rows per file), the same instruct
    prompt and the same `score_record`, and writes the same report / `.samples.jsonl` format. So `log_wandb.py`
    and `osm_breakdown.py` read it unchanged.
  - Queue: [`graph_bench/eval/run_glm.sh`](../../../graph_bench/eval/run_glm.sh). CPU only, no GPU lock.
- **Endpoint:** `GLM_SERVER` / `GLM_API_KEY` in the repo's `.env` (SDU, `ai.cloud.sdu.dk/v1`, model
  `zai-org/GLM-5.3`). The key is never written to reports or logs.
- **Robustness:** runs are resumable (`<out>.partial.jsonl`; failed rows are retried on restart). Rows that still
  fail after the retries score 0 and are counted as `api_errors`.
- **Outputs:**
  - `mimir_bench/e0/glm53/{named,anon,osm}_{t3k,t16k}.json`;
  - W&B `Mimir-Graph-RLVR`, group `mimir-bench-e0`, runs `e0/glm53/<cond>`.

**What the SDU server accepts** (checked 2026-11-09 on single rows):

| Request parameter | Effect |
| --- | --- |
| `chat_template_kwargs.enable_thinking=false`, `thinking.type=disabled`, `/nothink` in the prompt | Ignored: hidden `reasoning` is always returned |
| `max_tokens` | Ignored (a cap of 64 returned 411 tokens) |
| `max_completion_tokens` | Enforced; it covers hidden reasoning and the visible answer together |
| `reasoning_effort` | Only `low` is accepted (`none`, `minimal` and `medium` are rejected). On one CLUTRR row it cut 21,842 tokens to 3,322 |

So GLM can't run "thinking off" like the other chat models. Unlimited, the hidden reasoning reached 22k tokens on
one CLUTRR row; the visible answers stayed short.

**Settings (user, 2026-11-09).** Two runs on the same rows:

| Condition | Token cap (reasoning + answer) | Effort | Meaning |
| --- | --- | --- | --- |
| `_t3k` | 3,072 | low | The other models' budget; GLM splits it between hidden reasoning and the answer |
| `_t16k` | 16,384 | server default | GLM's ceiling |

- **Sets:** the 12 named E0 test files (9,554 rows), the 4 anonymised KGQA files and OSM test (2,000 rows; see
  [OSM KGQA](/pages/hrm-graph/osm-spatial-kgqa.md)). No pass@k.
- **Scoring:** on the visible `content` only. The reasoning is kept in the samples file. A row cut off before its
  `Answer:` line scores 0, as cut-off answers do for every other model.

**Smoke run (15 rows: 5 each from GraphQA, CLUTRR, OSM; `_t3k`):**
- 9 of 15 right, and these are exactly the 9 rows that finished.
- The other 6 hit the cap inside the hidden reasoning (11–13k characters each) with no answer: 4 of 5 CLUTRR,
  1 GraphQA, 1 four-hop OSM row.
- Without the cap (first attempt), 4 of these 6 reached an answer, 3 of them right; the other 2 hadn't finished
  when it was stopped.
- On the full run, 8 of the first 107 named rows were cut off. Throughput was about 2.5 rows/s with 16
  concurrent requests.

**Reading the results:**
- `_t3k` compares with the `_t3k` reports (E2, E3, OSM), not with E0's `named.json`, which the other models ran at
  1,024 tokens.
- The gap between `_t3k` and `_t16k` shows how much of GLM's score the shared budget costs. Report the cut-off
  rate (`finish_length`) next to every GLM score.
- GLM has probably seen WebQSP, CWQ, MetaQA and KQA Pro. The gap between its anonymised and named scores measures
  that reliance. GraphQA-fresh and OSM are the cleanest probes.

**Status:** running (`mimir_bench/e0/glm53/queue.log`); results to be added here. Superseded (2026-11-09): `_t3k` results in
[GLM-5.3 at 3,072 tokens](#glm-53-at-3072-tokens-results-2026-11-09).

## E3 (GRPO from E2): results (2026-11-09)

- **Run:** 200 steps from Mimir E2, W&B run `ypei9ypw`. Crashed at step 186 (2026-11-07) when `/work` vanished
  during a session shutdown; resumed from the step-175 checkpoint into the same W&B run and finished 2026-11-09.
- **Model:** `dfm-hrm-graphs/hf/mimir15-e3_grpo_e2-step200` (merged LoRA; the non-weight files are symlinks, copy
  with `cp -rL`). VeRL checkpoints every 25 steps in `mimir_bench/e3_grpo_e2/ckpt/` (~7 GB each).
- **In-training val** (~60 questions per dataset, greedy): mean 0.842 (step 0) → 0.866 (100) → 0.899 (175) →
  0.906 (200). At step 200: GraphQA, MetaQA, KQA Pro 0.983 each, WebQSP 0.780, CLRS 0.800.

Test, `named_t3k` / `anon_t3k` (3,072 tokens, same rows and settings as E2's `_t3k` reports):

| Split | Mimir E1 | Mimir E2 | **Mimir E3** | E3 − E2 |
| --- | --- | --- | --- | --- |
| GraphQA test / fresh | 0.858 / - | 0.960 / 0.968 | **0.972 / 0.975** | +0.012 / +0.007 |
| GraphQA large | 0.732 | 0.800 | **0.811** | +0.011 |
| MetaQA gold | **0.971** | 0.939 | 0.956 | +0.017 |
| MetaQA retrieved | **0.732** | 0.645 | 0.678 | +0.033 |
| KQA Pro gold | 0.924 | 0.936 | **0.950** | +0.014 |
| KQA Pro retrieved | **0.498** | 0.412 | 0.455 | +0.043 |
| WebQSP | **0.781** | 0.712 | 0.761 | +0.049 |
| CWQ (held out) | 0.675 | 0.529 | 0.652 | +0.123 |
| CLUTRR (held out) | **0.192** | 0.118 | 0.154 | +0.036 |
| CLRS test | 0.725 | 0.641 | **0.778** | +0.137 |
| CLRS OOD (held out) | **0.212** | 0.060 | 0.154 | +0.094 |
| **Macro named** | **0.642** | 0.583 | 0.639 | +0.056 |
| **Macro anon** | **0.814** | 0.750 | 0.806 | +0.056 |

E1 is at 1,024 tokens (answer-only, its outputs are short). Anon by dataset, E2 → E3: CWQ 0.450 → 0.558,
KQA Pro 0.925 → 0.932, MetaQA 0.875 → 0.927.

- **GRPO improves on E2 on every split**, most on CLRS (+0.14), the held-out CWQ (+0.12, +0.11 anonymised) and
  CLRS OOD (+0.09): RL helps more on the splits E2 got worse on than on the ones it already solved.
- **E3 only gets back to E1's macro (0.639 vs 0.642; 0.806 vs 0.814 anon), it does not pass it.** E3 is best on
  GraphQA, KQA Pro gold and CLRS test; answer-only E1 is still best on the retrieved subgraphs, WebQSP, CWQ, CLUTRR
  and CLRS OOD. GRPO on top of trace SFT mostly undoes the cost of the traces. Whether GRPO from E1 or from the
  base (E3b) does better is open.
- OSM zero-shot transfer: E3 doubles E2's set accuracy (0.100 vs 0.046), see
  [OSM KGQA](/pages/hrm-graph/osm-spatial-kgqa.md#zero-shot-results-e3-and-e2-2026-11-09).
- Pending: E3 val pass@16 (headroom left after RL), E4, E3b. Superseded (2026-11-10): pass@16 in
  [E3 pass@16](#e3-pass16-and-glm-at-16k-tokens-2026-11-10); E4 and E3b failed at start and were relaunched,
  see [fails](#fails-and-fixes-2026-11-10).

## GLM-5.3 at 3,072 tokens: results (2026-11-09)

`named_t3k` / `anon_t3k` (effort low, reasoning + answer ≤ 3,072 tokens), 0 API errors. W&B runs: `named_t3k`
`28vkixpo` (the first upload `73wbsy52` is tagged `superseded`, see the id bug below), `anon_t3k` `w6t7jj7l`.
Compared with the zero-shot E0 models (1,024 tokens) and E3:

| Split | Mimir E0 | Qwen2.5-7B E0 | OLMo-3-7B E0 | Qwen3.5-4B E0 | **GLM-5.3** | GLM cut off | GLM acc on uncut rows | Mimir E3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GraphQA test | 0.423 | 0.543 | 0.522 | 0.444 | **0.688** | 17% | 0.829 | 0.972 |
| GraphQA fresh | 0.417 | 0.471 | 0.508 | 0.434 | **0.673** | 16% | 0.799 | 0.975 |
| GraphQA large | 0.354 | 0.431 | 0.433 | 0.471 | **0.616** | 18% | 0.755 | 0.811 |
| MetaQA gold | 0.584 | 0.689 | 0.662 | 0.743 | **0.846** | 6% | 0.904 | 0.956 |
| MetaQA retrieved | 0.390 | 0.606 | 0.620 | 0.514 | **0.631** | 10% | 0.698 | 0.678 |
| KQA Pro gold | 0.259 | 0.671 | 0.556 | 0.682 | **0.769** | 7% | 0.826 | 0.950 |
| KQA Pro retrieved | 0.131 | 0.329 | 0.280 | 0.296 | **0.334** | 34% | 0.503 | 0.455 |
| WebQSP | 0.157 | 0.565 | 0.479 | 0.571 | **0.663** | 13% | 0.764 | 0.761 |
| CWQ | 0.141 | 0.468 | **0.530** | 0.414 | 0.396 | 44% | 0.712 | 0.652 |
| CLUTRR | 0.010 | 0.188 | 0.096 | 0.189 | **0.209** | 77% | 0.897 | 0.154 |
| CLRS test | 0.003 | 0.006 | 0.003 | 0.003 | **0.016** | 91% | 0.172 | 0.778 |
| CLRS OOD | 0.002 | 0.001 | 0.000 | 0.000 | 0.000 | 93% | 0.000 | 0.154 |
| **Macro named** | 0.198 | 0.405 | 0.377 | - | **0.459** | 40% | | 0.639 |

Anon: CWQ 0.204, KQA Pro 0.632, MetaQA 0.792 (macro 0.543; 40% cut off).

- **GLM is the best zero-shot model on 10 of 12 splits** and the trained Mimir E3 beats it on 11 of 12 (all but CLUTRR).
- **The 3,072 cap costs GLM most of CLUTRR, CWQ and CLRS:** hidden reasoning fills the budget before the
  `Answer:` line. On the rows that finish, GLM beats E3 on CLUTRR (0.897) and CWQ (0.712). `_t16k` will measure this.
- **Answer conventions it was never trained on cost the rest.** GraphQA CycleCheck wants "Yes, there is a cycle";
  GLM answers "Yes" (93 rows of GraphQA fresh, +0.09 if accepted). CLRS wants the CLRS array conventions (pred `π`
  arrays, 0/1 masks, matrix layouts); GLM writes NIL / True-False / other layouts.
- **Anonymised CWQ halves GLM's score (0.396 → 0.204)**, more than the trained models lose (E3 0.652 → 0.558):
  consistent with GLM drawing on world knowledge for CWQ, but the cut-off rate also differs, so not proven.

## Fails and fixes (2026-11-09)

- **`eval_api` scored 292 KQA Pro rows against the wrong question.** `kqapro/test` and `kqapro/test_retrieved` reuse
  ids (`kqapro/test/<n>`; 292 shared within the seed-0 named subsets, no other file pair). `eval_api` kept responses
  in a dict keyed by id, so each shared id got one response scored for both rows. Symptom: GLM KQA Pro gold 0.380,
  below every 7B baseline, with answers about entities absent from the prompt. Fix: rows keyed by
  `(source file, id)`; the report was rebuilt from `.partial.jsonl` (every row's own response was stored, no API
  calls): KQA Pro gold 0.380 → 0.769, all other splits unchanged. `eval_vllm` scores by position and was never
  affected. GLM's `named_t16k` started before the fix; `mimir_bench/e0/glm53/fix_named_t16k.sh` rebuilds and
  re-logs it once it is done. **Rule: never key per-row results by `id` alone across bench files.**
- **The queue logs said `exit 0` for failed runs.** `echo "=== $(date ...) exit $?"` reads `$?` after the `date`
  command substitution, i.e. always 0. Fixed (`local rc=$?` first) in `run_glm.sh`, `run_osm.sh`, `run_e0.sh`.
  Effect: GLM `osm_t3k` segfaulted 2 s after start (native crash in the Python process, cause unknown, no core
  dump in the container), was logged as a success and skipped; the identical command ran fine when relaunched by
  hand. A missing report `.json` is the reliable failure signal; a relaunch reruns only those.

## E3 pass@16 and GLM at 16k tokens (2026-11-10)

**E3 val pass@16** (T = 1, 16 samples, 200 rows per file, 2,048 tokens; same rows as E2's):

| Val split | pass@1 E2 → E3 | pass@4 E2 → E3 | pass@16 E2 → E3 |
| --- | --- | --- | --- |
| GraphQA | 0.972 → 0.986 | 0.990 → 0.994 | 0.995 → 0.995 |
| MetaQA | 0.863 → 0.945 | 0.950 → 0.965 | 0.965 → 0.970 |
| KQA Pro | 0.842 → 0.921 | 0.931 → 0.952 | 0.945 → 0.970 |
| CLRS | 0.648 → 0.789 | 0.793 → 0.853 | 0.885 → 0.895 |
| WebQSP | 0.578 → 0.713 | 0.694 → 0.777 | 0.735 → 0.810 |

- **GRPO moved pass@1 up towards E2's pass@16 and pass@16 barely moved** (CLRS +0.14 pass@1, +0.01 pass@16). It
  mostly made reliable what E2 could already sample, the usual RLVR pattern; WebQSP (+0.075 pass@16) is the only
  split where the reachable set grew noticeably.
- E2 CLUTRR test pass@16 at 2,048 tokens (reference for E4): pass@1 0.081, pass@4 0.176, pass@16 0.280.

**GLM-5.3 at 16,384 tokens** (`_t16k`, server-default effort, 0 API errors; W&B `named_t16k` run `z5mf2fdv`, the
first upload `nr6q13mw` is tagged `superseded` (id bug); 58M completion tokens for the named set):

| Split | `_t3k` | `_t16k` | cut off at 16k | Mimir E3 |
| --- | --- | --- | --- | --- |
| GraphQA test / fresh / large | 0.688 / 0.673 / 0.616 | 0.784 / 0.756 / 0.681 | 9% / 9% / 15% | 0.972 / 0.975 / 0.811 |
| MetaQA gold / retrieved | 0.846 / 0.631 | 0.882 / 0.645 | 3% / 13% | 0.956 / 0.678 |
| KQA Pro gold / retrieved | 0.769 / 0.334 | 0.799 / 0.360 | 2% / 33% | 0.950 / 0.455 |
| WebQSP | 0.663 | 0.652 | 20% | 0.761 |
| CWQ | 0.396 | 0.531 | 34% | 0.652 |
| CLUTRR | 0.209 | **0.523** | 44% | 0.154 |
| CLRS test / OOD | 0.016 / 0.000 | 0.286 / 0.173 | 43% / 66% | 0.778 / 0.154 |
| **Macro named** | 0.459 | 0.572 | 27% | 0.639 |
| **Macro anon** (CWQ / KQA Pro / MetaQA) | 0.543 | 0.607 (0.271 / 0.700 / 0.851) | | 0.806 |
| OSM test (set acc / F1) | 0.736 / 0.792 | **0.857 / 0.863** | 13% | 0.100 / 0.383 |

- **More budget helps GLM most where its reasoning is long:** CLUTRR +0.31, CLRS +0.27 / +0.17, CWQ +0.14, OSM
  +0.12. Even at 16k, 27% of named rows are still cut off inside the hidden reasoning.
- **E3 still leads on macro (0.639 vs 0.572)**: GLM wins CLUTRR (0.523 vs 0.154), CLRS OOD (0.173 vs 0.154) and
  OSM; E3 wins the trained in-distribution splits and CWQ.

## Fails and fixes (2026-11-10)

- **E4 GRPO and E3b crashed at start** (2026-11-09 23:32 / 23:47, ~20 s each): `ValueError: [actor_rollout_ref.ref]
  Please set at least one of 'actor_rollout_ref.ref.log_prob_micro_batch_size' or '..._per_gpu'`. They are the first
  arms with `REF_KL=True` (default for every arm but E3, decided 2026-11-06), which adds the reference-policy worker;
  `run_grpo_graph.sh` never set its micro-batch. Fix: `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=
  ${LOGPROB_MICRO_BATCH_PER_GPU}`; checked offline with VeRL's `validate_config` on E4's exact overrides. Queue
  relaunched (`mimir_bench/run_rl_arms.running9.sh`).
- **E4 SFT control crashed at start:** `graph_inject.train` parsed `model=mimir` and `model.hf_dir=...` in one
  OmegaConf dotlist, so the overrides turned `model` into a dict and the config lookup failed
  (`configs/models/{'name': ...}.yaml`). Fix: `load_config` takes `dataset=`/`model=`/`method=` out first, then
  merges the rest; `model.<key>=` overrides now work.
- **Qwen3.5-4B baseline SFT: training finished, merge failed** (`layers.0.mlp.down_proj.weight: 2 matching base
  tensors`): the checkpoint has a multi-token-prediction head (`mtp.layers.*`) with the same suffixes as the LM
  layers. Fix: `sft_hf.merge` skips `mtp.*` (as it skips `visual`). The answer arm was merged by hand (248 pairs).
  The cot arm was training when this was found and will hit the same error; `sft_baselines/qwen35_followup.sh` waits
  for the queue, merges it, and runs the four Qwen3.5-4B evals. **Do not relaunch `run_sft_baselines.sh` for an arm
  that trained but did not merge: it retrains any arm without `merged_ep3` and overwrites `train.log`;** use
  `sft_hf --merge-only 3`.
- The 2026-11-09 watcher that re-logged GLM `named_t16k` could not tag the old run: `.wandb` markers hold two lines
  (run path, version), read with `head -1`, not `awk '{print $1}'`.
