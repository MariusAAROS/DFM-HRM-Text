---
type: Runbook
title: Mimir v1.5 Graph-Reasoning Benchmark (Zero-Shot, SFT, RLVR)
description: "Plan (approved 2026-11-02) and implementation log for testing whether DFM-Mimir-v1.5 is the strongest graph / KG reasoner, zero-shot and after SFT and RLVR: claim challenges, hypotheses, models and baselines, datasets, splits, arms and what each is compared with, the VeRL + vLLM RL stack, and the 2026-11-03 fixes, vLLM parity gate and GRPO smoke result."
tags: [graphs, mimir, rlvr, grpo, vllm, verl, kgqa, benchmark]
status: draft
last_updated: 2026-11-05
confidence: medium
---
# Mimir v1.5 Graph-Reasoning Benchmark

The question is whether DFM-Mimir-v1.5 is the strongest model for reasoning over graphs and knowledge graphs, in
three settings: zero-shot, after SFT, and after SFT followed by RLVR. Both in-distribution and held-out tasks
count. Zero-shot transfer to held-out tasks is the main criterion, because it separates a research-only model from
a general-purpose one. This page records the plan approved on 2026-11-02, the reasons for each choice, and the
implementation log that follows it.

- Code: `graph_bench/` (dataset converters in `sources/`, scoring, `eval/eval_vllm.py`, RL in `rl/`). Bench data:
  `data/graphs/bench/<dataset>/`. MetaQA / KQA Pro gold-only files: [KGQA gold-only rebuild](/pages/hrm-graph/kgqa-gold-rebuild.md).
- W&B: everything (E0 eval, E1/E2 SFT, E3/E4 GRPO) in project `Mimir-Graph-RLVR`, groups `mimir-bench-<arm>`
  (since 2026-11-03; only the first smoke run, `iyp11un6`, is in `HRM-Graph`).
- Run outputs: `/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench/` (`e0/`, `e1/`, `e2/`, `sft_*`, `e3_*`, `e4_*`,
  queue logs); merged models in `/work/dfm/marius-ortega/dfm-hrm-graphs/hf/`.
- Results and run log (E0, E1/E2 with baselines, pass@16, fixes): [Results](/pages/hrm-graph/mimir-graph-benchmark-results.md).
- Prior results it builds on: [Graph Reasoning Experiments](/pages/hrm-graph/graph-reasoning-experiments.md),
  [Graph-Structure Injection](/pages/hrm-graph/graph-structure-injection.md),
  [GraphQA Chain of Thought](/pages/hrm-graph/graphqa-cot.md).
- RL stack: `/work/dfm/marius-ortega/jacob-hrm-text-rlvr/verl_hrm_text/` (VeRL + vLLM). The fallback is
  `/work/dfm/marius-ortega/hrm-text-rlvr/train_grpo.py` (TRL + HF generate).
- KGQA source loaders (URLs and schemas only): `data/graphs/KGQA/*/`.

## Status (updated 2026-11-03)

Superseded (2026-11-05): E0 done (7 models), E1/E2 done for Mimir v1.5, Qwen2.5-7B and OLMo-3-7B, Qwen3.5-4B
training, GRPO queued after it. Results and run log: [Results](/pages/hrm-graph/mimir-graph-benchmark-results.md).

| Item | State |
| --- | --- |
| Tier 1 data | Converted: GraphQA (+ `test_fresh`, `test_large`), MetaQA and KQA Pro (gold-only), WebQSP, CWQ, CLRS-Text, CLUTRR. Not yet: MetaQA 3-hop is in test only; fresh Wikidata probes not built |
| Tier 2 data | Raw files downloaded (NLGraph, GraphInstruct, GraphArena, LC-QuAD 2.0, CronQuestions, PathQuestions, Mintaka); no converters. GrailQA not downloaded |
| vLLM parity gate | Passed after two fixes (tokenizer flag, double `<bos>`); bf16 identical-generation bar waived |
| RL pipeline (E3 gate) | Passed: 15-step GRPO smoke run, rollout-vs-actor Pearson ≥ 0.999 after the token_type_ids fix |
| E0 zero-shot matrix | Running since 2026-11-03 16:10 (8 cached models, see [Launch](#launch-of-e0e4-2026-11-03)) |
| E1 / E2 SFT | Queued (data built; trains when the GPU lock frees) |
| E3 / E4 / E3b RL | Queued; start when E2's merged model exists |
| Not covered yet | Ouro (needs its own vLLM venv), IFM and Qwen3-4B / HRM-Text-1B-v3.2 (not downloaded), Tier 2 sets, retention suite (H4), eval of the RL checkpoints |

## Design (2026-11-02)

### Challenges to the claim, from our own results

- **Direct answers do not favour v1.5.**
  - With full fine-tuning, Mimir v1 beats v1.5 on GraphQA (0.843 vs 0.818, 3 seeds), MetaQA (0.837 vs 0.650) and
    KQA Pro (0.698 vs 0.600).
  - Full fine-tuning leaves v1.5 *below its own zero-shot* on MetaQA (0.650 vs 0.761). The cause is unexplained:
    bp 8 recovers 6 points at most.
  - With LoRA `text`, the three models are within ±1.5 pt on all three datasets.
- **Chain of thought favours v1.5.** On GraphQA CoT (3 seeds, `text` + LoRA), v1.5 scores 96.0 (sd 0.6), v1 92.9
  (sd 1.5) and HRM-1B 89.4 (sd 3.1). v1.5 also leads zero-shot MetaQA, but only just (0.761 vs v1 0.758).
- **So the claim must be scoped and fixed before the new runs.** Picking the protocol where v1.5 already wins is a
  forking-paths problem.
- **Breadth is not proof.** "Best" needs four things:
  - a primary metric and aggregation fixed in advance;
  - paired tests with a multiple-comparison correction;
  - 3 seeds where the gaps are small;
  - the same tuning budget per model. Our lr 5e-5 recipe was tuned on HRM, so each external model gets a small
    LR sweep.
- **Compute is a confound.** Mimir has 1.79B parameters, but each token runs 8 block applications of 16 layers
  (2 H + 6 L). That is roughly 5–6B dense-equivalent FLOPs per token (estimate, not measured). The fair
  comparison is therefore against 4–7B dense models. Report params, FLOPs per token and pretraining tokens next to
  every score.
- **Closed-book KGQA measures memorised knowledge.** Qwen-class models have probably seen WebQSP, CWQ and MetaQA.
  So:
  - every KGQA set puts its subgraph in the prompt;
  - each in-context set gets an **entity-anonymised variant** (entity labels replaced by random IDs). The gap to
    the named version measures how much the model relies on parametric knowledge;
  - freshly generated synthetic sets serve as contamination-free probes.
- **General purpose also needs retention.** A general-purpose model must keep its general skills after SFT and RL.
  Retention suite: MMLU, GSM8K, ARC-C, IFEval and Danish (EuroEval). SFT mixes in ~10–20% general instruction data
  as replay.
- **RL on a held-out set makes it in-distribution.** "Let RL discover traces on the held-out set" is therefore its
  own arm (E4, answer-only adaptation), not evidence for zero-shot transfer.
- **RLVR sharpens; it does not create.** Yue et al. 2025 found that RLVR raises pass@1 but rarely pass@k at large
  k. So measure the SFT model's pass@16/64 before RL, and train only on prompts with a pass rate strictly between
  0 and 1.

### Hypotheses (fixed before running)

| | Claim | Test |
| --- | --- | --- |
| H1 | Zero-shot: v1.5 ≥ every model of ≤ 7B dense-equivalent compute on the held-out tasks | macro-average `greedy_acc` (set-F1 for multi-answer KGQA); paired bootstrap 95% CI |
| H2 | After the same LoRA SFT, v1.5 leads on ID test and within-task OOD splits | same; 3 seeds for v1.5, v1 and Qwen3-4B |
| H3 | SFT + RLVR beats SFT alone on the *held-out* tasks, not only on ID tasks | held-out macro-average, SFT vs SFT+RL |
| H4 | After SFT + RL, the retention suite drops by ≤ 2 pt | retention suite before and after |

Secondary analyses: results per FLOP, per parameter, and the gap between anonymised and named variants.
Significance: McNemar and paired bootstrap per dataset, with Holm correction. Epoch selection uses the full val
set or the last epoch, never a saturated subset (see the [CoT seed variance](/pages/hrm-graph/graphqa-cot.md)).

### Models

Base and chat models need different zero-shot protocols:
- Base models get 3-shot prompts and are compared with each other.
- Chat models use their own chat template.
- After SFT, every model is comparable.

| Group | Core | Extended |
| --- | --- | --- |
| HRM family | Mimir v1.5, Mimir v1 (chat); `sapientinc/HRM-Text-1B-v3.2` (base) | HRM-Text-1B (base) |
| IFM looped, P1 (base, ~1B dense, all trained on the same 336B tokens) | `IFM/LoopedLM-P1-dense-{d28,split-hl,ouro,huginn}-336b` | `dense-shared-hl`, `dense-middle`, `dense-d112-500b` (4B) |
| Ouro | `ByteDance/Ouro-2.6B-Thinking`, `Ouro-2.6B` (base) | `Ouro-1.4B(-Thinking)` |
| Dense SLM | Qwen3-4B, one ~7B (Qwen2.5-7B-Instruct or OLMo-2-7B-Instruct) | Gemma-3-4B / Gemma-4 small, SmolLM3-3B, Phi-4-mini |

- **The IFM P1 1B variants form a controlled architecture ablation.** They share data and token count, so after
  the same SFT, the differences between them come from the looping scheme alone. `split-hl` is the HRM-like
  variant. Still to check per variant: context length (prompts reach ~3k tokens), loading code, and whether `d28`
  is the non-looped control.
- **Environments differ.** Ouro needs `transformers<4.56`, and Mimir in vLLM/VeRL runs on transformers 5.20.

### Datasets

Rules for inclusion:
- the answer is automatically checkable;
- the graph fits in the prompt (≤ ~4k tokens);
- preferably, a gold program exists, so traces can be produced by executing it, under the CoT rule "each step is a
  copy or a lookup".

The HF `datasets` scripts in `data/graphs/KGQA/` no longer run on recent `datasets` releases. Use them for URLs
and schemas only.

**Tier 1**

| Dataset | Graph in prompt | Traces | Role |
| --- | --- | --- | --- |
| GraphQA (have) | synthetic | `graph_inject/cot.py` | train |
| MetaQA (have) | KB subgraph | hop-path expansion | train 1–2 hop; 3-hop is within-task OOD |
| KQA Pro (have) | subgraph | KoPL program executed step by step | train |
| WebQSP | RoG subgraphs (`rmanluo/RoG-webqsp`) | SPARQL → relation paths | train |
| CWQ | RoG subgraphs | SPARQL | near-OOD, held out |
| CLRS-Text (graph algorithms) | graph | native execution traces | train small n; larger n is within-task OOD |
| CLUTRR | kinship graph | proof chain | far-OOD, held out |
| Fresh synthetic (GraphQA generator with new seeds; Wikidata held-out link prediction and 2/3-hop paths) | synthetic | generator | contamination-free probe |

**Tier 2, cleaned to the reasoning subsets.** Each filter uses the dataset's own annotation, so it can be
reproduced.

| Dataset | Keep | Drop | Cleaning | Role / feasibility |
| --- | --- | --- | --- | --- |
| GrailQA | `num_edge ≥ 2` or `function ∈ {count, comparative, superlative}` | 1-edge questions without a function | Test answers are hidden: dev (6.8k) becomes test, val comes out of train. Keep `level`. Drop questions whose S-expression is empty on the extracted subgraph. | train i.i.d.; compositional and zero-shot levels are OOD. **Medium–low**: needs a Freebase dump, so it goes last. |
| LC-QuAD 2.0 | multi-fact, two-intention, count, rank, boolean with ≥ 2 facts (by template) | single-fact templates; string operations (lexical); **all qualifier templates** (the truthy Wikidata store has no qualifiers) | Empty or "n/a" `question` → fall back to `paraphrased_question`, else drop. Re-run the SPARQL on the 2026-10 store and drop empty answers (2019 answers are stale). Dedupe. | train or near-OOD. Medium. |
| CronQuestions | `before_after`, `first_last`, `time_join` | `simple_entity`, `simple_time` | Subgraph = the released temporal facts of the annotated entities, capped to fit 4k tokens. Subsample. | far-OOD (temporal), held out. High. |
| NLGraph | all 8 tasks | – | Tag the 4 tasks GraphQA covers. Topological sort, matching and Hamilton path have several valid answers, so they need a **checker**, not EM; the RL reward too. | official test as held-out eval; training data regenerated. High. |
| GraphWiz GraphInstruct | all 9 tasks | its RFT traces (filtered on the final answer only) | Regenerate traces by program. Dedupe against NLGraph and GraphQA test. Use the checker. | train. High. |
| GraphArena | 4 polynomial + 6 NP-hard tasks, small instances | instances over 4k tokens | Score with GraphArena's own feasible / optimal / hallucinated scorer | far-OOD, held out. High. |

Optional additions:
- 2WikiMultiHopQA: its gold evidence triples serve as the graph and as traces.
- Reasoning Gym graph generators: procedural, with verifiers, so they suit RL.

**Tier 3 verdict.** Mostly no use here.

| Dataset | Verdict | Reason |
| --- | --- | --- |
| SimpleQuestions-Wikidata, SimpleDBpediaQA | drop | 1-hop lookups |
| LC-QuAD 1.0 | drop | superseded by LC-QuAD 2.0; would need DBpedia |
| RuBQ 1/2 | drop | ~2.9k, mostly 1-hop, machine-translated English |
| QALD-9/10 | drop | a few hundred test questions (SE ~2.5 pt) |
| EventQA | drop | ~1k questions on a third KG (EventKG) |
| ComplexQuestions, GraphQuestions | drop | GrailQA supersedes them |
| TempQuestions | drop | TimeQuestions supersedes it |
| QALM | drop (contents not checked) | looks like a multilingual re-release of existing questions |
| **PathQuestions** | **keep, eval-only** | 2/3-hop questions over a small *released* Freebase subset: MetaQA's skill on another KB, at almost no cost |
| **Mintaka** | **keep, eval-only, closed-book** | every question carries a complexity label (count, comparative, superlative, ordinal, multi-hop, ...); a general-purpose probe, not an in-context graph test |
| **TimeQuestions** | blocked | real temporal reasoning, but most answers sit in Wikidata qualifiers. Usable only if EXAQT's released per-question evidence works. |

### Splits

1. **Train (SFT and RL):**
   - GraphQA, MetaQA 1–2 hop, KQA Pro, WebQSP;
   - CLRS-Text at small n;
   - generated NLGraph/GraphInstruct-style tasks;
   - the LC-QuAD 2.0 reasoning subset;
   - GrailQA i.i.d. if it gets built.
2. **Within-task OOD, never trained:**
   - MetaQA 3-hop;
   - CLRS-Text at larger n;
   - larger GraphQA graphs;
   - GrailQA's compositional and zero-shot levels.
3. **Held-out tasks, never trained, chosen by skill family:**
   - near: CWQ, PathQuestions, NLGraph test;
   - far: CronQuestions complex, CLUTRR, GraphArena;
   - closed-book probe: Mintaka.

Within-task OOD splits separate "new composition" from "new domain". A held-out set is chosen by skill family,
not at random.

### Training protocol and arms

The recipe follows DeepSeek-R1, with the DAPO and Dr. GRPO fixes:
1. cold-start SFT on verified traces, plus general replay;
2. GRPO on the same task pool, with:
   - EM / set-F1 / checker rewards, plus a small format reward;
     Superseded (2026-11-05): the implemented reward has no format term, see
     [GRPO reward as implemented](#grpo-reward-as-implemented-2026-11-05);
   - prompts filtered to SFT pass rates strictly between 0 and 1;
   - clip-higher, dynamic sampling and token-level loss;
   - no KL term;
3. optionally, rejection sampling followed by re-SFT.

LoRA is the main protocol for both SFT and RL, because full fine-tuning degrades v1.5 for reasons not yet
understood.

| Arm | What | Models |
| --- | --- | --- |
| E0 | Zero-shot matrix: greedy + pass@k, named and anonymised | all |
| E1 | LoRA SFT on the train mixture, answer-only | core |
| E2 | LoRA SFT on the train mixture, with traces (Mimir: in the `<|channel>thought` channel, opener included in the targets) | core |
| E3 | GRPO from E2 | Mimir v1.5, Qwen3-4B (Ouro-2.6B if time allows) |
| E3b | GRPO from the base model (R1-Zero style), as an ablation | Mimir v1.5 |
| E4 | Answer-only RL adaptation on one held-out task (e.g. CronQuestions) from E2, against SFT on answers | Mimir v1.5 |

Gates:
- after E0: if v1.5 is not in the top 2 on the held-out macro-average, reframe the claim (per FLOP, or within
  the family) before spending SFT and RL compute;
- before RL: SFT pass@16 must be above 0 on most train prompts.

**What each arm is compared with (clarified 2026-11-03).**

| Comparison | Question it answers | Hypothesis |
| --- | --- | --- |
| E0 across models | Zero-shot ranking on held-out tasks | H1 |
| E2 − E1 (same data and budget, traces vs answers only) | Does training on reasoning traces help? Our earlier results favour v1.5 only with CoT, so the gap per model is part of the claim | H2 |
| E3 − E2, on held-out and within-task OOD sets | Does RL add zero-shot transfer beyond SFT? This is the main zero-shot-after-RL result | H3 |
| E3 − E3b | Does RL need the trace SFT first? | ablation |
| E4 − SFT on the same task's answers, both from E2 | For a new task with answers but no traces, is RL a better adaptation than SFT on answers? E4 trains on the task, so it is adaptation, never zero-shot evidence | – |
| Retention suite before / after SFT + RL | MMLU, GSM8K, ARC-C, IFEval, Danish (EuroEval): non-graph skills only. OOD graph data belongs to H1–H3 | H4 |

For H3, also report the SFT model's pass@k on the held-out sets: when pass@64 is already high, RL mostly selects
among answers the model could already produce.

### Feasibility (one RTX PRO 6000, estimates)

| Item | Estimate |
| --- | --- |
| Tier 1 curation | ~1 week |
| Tier 2 curation (GrailQA's Freebase extraction is the hard part) | 1–2 weeks |
| E0 with the native HRM engine (CoT decoding at 0.14 s/step, batch 8 max) | ~60–90 h for the HRM family. Through vLLM, hours, once parity holds. |
| E1/E2 SFT (~200–300M-token mixture) | ~1 day per model per arm |
| E3 GRPO on Mimir with VeRL + vLLM | ~15–25 h per run with 2–3k-token graph prompts (scaled from GSM8K, below) |

Total: ~5–6 weeks of single-GPU time for the core set, plus ~1 week to validate the RL stack.

Measured since (2026-11-03): Tier 1 conversion took about a day, not a week. Mimir GRPO with eager vLLM, 16 prompts ×
8 rollouts and up to 1,024 response tokens takes about 6.2 min per step.

## RL stack: VeRL + vLLM in jacob-hrm-text-rlvr (checked 2026-11-02)

The plan first assumed that `hrm_text` had no vLLM backend, as `hrm-text-rlvr/README.md` says. **That is out of
date.**

- **vLLM 0.27.1 implements `HrmTextForCausalLM` natively** (`vllm/model_executor/models/hrm_text.py`).
  - Each of the 8 recurrence steps gets its own KV-cache slot.
  - Prefill uses `PrefillPrefixLMAttention`, which sets `causal=False` on every prefill. During single-token
    decoding that is a no-op.
  - This non-causal prefill turns off chunked prefill and prefix caching.
  - It is installed in `/home/ucloud/miniforge3` (python 3.13, torch 2.13, transformers 5.20.2, trl 1.13,
    verl 0.9.0). That is a different environment from this repo's `.venv`.
- **`jacob-hrm-text-rlvr/verl_hrm_text/hrm_verl_ext.py`** is loaded through
  `actor_rollout_ref.model.external_lib` and patches VeRL to:
  - disable the trl value-head monkey-patch;
  - provide a pure-PyTorch replacement for flash_attn's padding utilities;
  - append the thought prefill `<|channel>thought\n`;
  - inject PrefixLM `token_type_ids` into the FSDP scoring and training passes (prompt = 1; response and prefill
    = 0);
  - fuse q/k/v/gate into `gqkv_proj` and gate/up into `gate_up_proj` for the FSDP → vLLM weight sync;
  - scale the LR per module (present but disabled).

  The repo also has an HF SFT script with PrefixLM (`train_sft_toolmind.py`) and a llama.cpp rollout. The llama.cpp
  rollout re-exports a GGUF and restarts its server at every step, so it is a fallback only.
- **Measured** (`grpo_run_v13.log`: GSM8K, internal Mimir checkpoint, 32 prompts/step, max response 1,024,
  vLLM rollout with `enforce_eager=True` and `gpu_memory_utilization=0.4`):
  - 102 s/step on average, of which 55 s is generation, over 233 steps (≈ 6.6 h);
  - val accuracy 64.7% → 88.9% (README).
  - The equivalent through HF `generate` was estimated at ~50 h per run.
- **`run_grpo_hrm_text.sh` defaults to `ROLLOUT_NAME=hf`.** Set `vllm` explicitly, as the ToolMind scripts do.

Gaps to close before E3 (CPU work first, GPU checks once the GPU is free):

1. **Prefill mismatch.** vLLM treats the 3 prefill tokens as bidirectional prompt, while the FSDP patch scores
   them as causal. Rollout and training distributions then differ silently.
   - For E3: E2's SFT targets include the opener, so RL needs no prefill.
   - For E3b: score the prefill tokens as prompt (`token_type_ids = 1`).
2. **Parity.** vLLM's greedy outputs must match `scripts/graphs/eval_graphqa.py` on GraphQA, MetaQA and KQA Pro
   test. During RL, run with `calculate_log_probs=True` and watch the rollout-vs-actor logprob gap.
3. **LoRA.** vLLM's `HrmTextForCausalLM` does not implement `SupportsLoRA`. Use VeRL's
   `actor_rollout_ref.model.lora.merge=True` (merge, then sync full weights), and check that the fusion patch sees
   the merged names.
4. **KV budget.** The cache costs 768 KB per token in bf16, and the rollouts of a GRPO group cannot share their
   prompt (no prefix caching).
   - Cap RL prompts at ~2k tokens.
   - Try fp8 KV, more GPU memory, and CUDA graphs: the model has `@support_torch_compile`, but CUDA graphs are
     untested.
5. **Baselines.**
   - Qwen3: standard VeRL + vLLM.
   - Ouro: vLLM lists `OuroForCausalLM` as previously supported (0.26.0), so it needs an older vLLM in its own
     venv.
   - IFM: probably no vLLM support, so SFT and HF eval only.

Gate for E3: parity holds, the prefill fix is verified, and a 50-step GraphQA-CoT GRPO smoke run with LoRA merge
shows rising reward and a stable logprob gap.

**Gap status (2026-11-03)**, details in the implementation log below:
1. Prefill: `GRAPH_RL_PREFILL` in `graph_bench/rl/verl_ext.py` picks one convention (`none`: no prefill, for E3 / E4;
   `prompt`: prefill scored as prompt, for E3b).
2. Parity: holds, after two fixes (tokenizer flag, double `<bos>`).
3. LoRA merge sync: works. The rollout-vs-actor gap stays at about 0.003 across 15 steps of LoRA updates.
   `GRAPH_RL_FUSE_SYNC=0` sends separate weight names, for Bifrost.
4. KV budget: RL prompts are capped at 2,048 tokens by the export; fp8 KV and CUDA graphs are still untested.
5. Baselines' environments: not started.
- The E3 gate was met with a 15-step run on GraphQA + MetaQA + KQA Pro (not 50 GraphQA-CoT steps). The rollout-vs-actor
  gap was stable; the reward gain was mostly answer format.

## Implementation log (2026-11-03)

### Out-of-memory crash of 2026-11-02 16:50

- **Symptom.** The container restarted at 16:50:59. It killed every job in it, including the GraphQA-v3 Mimir training
  at its epoch-8 val step. The user saw two crashes in a row.
- **Cause (most likely; the kernel counters were reset).** `graph_bench/sources/rog.py --dataset cwq`, started at 16:47
  next to `clrs.py` and the training.
  - It ran `pq.read_table(...).to_pylist()` on CWQ train: 8.1 GB of Arrow, about 30 GB of Python objects.
  - It then forked `Pool(32)` off that heap. Each child copies inherited pages as Python's refcounts and GC write to
    them.
  - The container's cgroup limit is **128 GB** (the host has 1 TB; there is no swap). WebQSP, 9× smaller, had passed.
- **Fix.**
  - **`rog.py`:** forks its workers before reading anything and streams rows (`iter_batches`, ordered `imap`).
    CWQ now converts in 3 min at a **4.0 GB peak**.
  - **`clrs.py`:** streams too (1.4 GB peak).
  - **`scripts/graphs/mem_guard.sh <cap_GB> -- <cmd>`:** kills the command's process group above the cap, or when the
    container passes 110 GB, and logs the peak.
  - **Rule:** converters run one at a time, under the guard.
- **Also fixed: `rog.py` was not deterministic.** Its output changed from run to run, because set iteration order
  depends on the per-process string hash seed. With sorted iteration, runs under different `PYTHONHASHSEED`s are
  byte-identical, and the WebQSP files were rebuilt (same counts).

### RL code fixes (`graph_bench/rl`, `graph_bench/scoring.py`)

- **Reward was 0 for correct answers.** `hrm_verl_ext` makes VeRL decode with `skip_special_tokens=False`, so a
  completion reaches the reward as `...Answer: Paris<turn|>`, which scored 0. `scoring.extract_answer` now cuts at
  end-of-turn tokens and turns `<channel|>` into a line break; tests cover both cases.
- **Ray caps.** `run_grpo_graph.sh` passes `ray_kwargs.ray_init.num_cpus=16` and `object_store_memory=8e9`. Ray would
  otherwise size its `/dev/shm` store from system memory and see 188 CPUs.
- **`GRAPH_RL_FUSE_SYNC`** (default 1): weight sync with fused `gqkv_proj` / `gate_up_proj`. Set it to 0 for the
  Bifrost plugin (`/work/dfm/marius-ortega/Bifrost_vllm`), which only accepts the separate projections.

### vLLM parity gate

- **Backend decision:** native vLLM 0.27.1 `hrm_text`, with Bifrost kept as the reference. Bifrost serves the same
  prefix-LM contract (prompt bidirectional, no chunked prefill, no prefix caching), but only on vLLM 0.32 and only with
  the Triton backend.
- **Setup.** Mimir v1.5 zero-shot, greedy, 96 tokens, on 200-row seeded subsets of GraphQA / MetaQA / KQA Pro test
  (`data/graphs/bench/parity/`). Native `eval_graphqa.py` against `graph_bench/eval/eval_vllm.py`.
- **Bug 1: tokenizer.** The DFM-Mimir checkpoints (v1, v1.5, HRM-Mimir-v1) set `"fix_mistral_regex": true`.
  transformers then prepends a regex pre-tokenizer that `tokenizer.json` does not have (`" Question"` becomes
  `" "` + `"Question"`).
  - Effect: 0/600 prompts tokenized as in data_io / the native engine. Everything that loads Mimir through
    transformers is affected: vLLM, VeRL, probably GRL.
  - Fix: `graph_bench/eval/hf_view.py` makes a symlinked model dir with the flag set to false (600/600), at
    `/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5-hfview`.
  - `eval_vllm.py` now refuses to run when its tokenizer disagrees with `tokenizer.json`.
- **Bug 2: double `<bos>`.** vLLM tokenizes text prompts with `add_special_tokens=True`, so the chat template's `<bos>`
  got a second one in front. `eval_vllm.py` now sends token ids.
- **Result after both fixes.**
  - vLLM in **fp32** gives the HF fp32 argmax as first token on every prompt, except one tie at 0.004 logits. The
    five other apparent misses were artefacts of re-tokenizing the decoded text. So the vLLM implementation is exact.
  - In bf16:

| 200-row subset | Identical generations | acc native | acc vLLM |
| --- | --- | --- | --- |
| GraphQA | 0.885 | 0.455 | 0.445 |
| MetaQA | 0.995 | 0.785 | 0.790 |
| KQA Pro | 0.890 | 0.460 | 0.460 |

- **Reading of the bf16 numbers.**
  - The two engines' bf16 kernels round differently, which flips near-tied argmaxes (zero-shot count answers, for
    example).
  - The 95% identical-generation bar set in the plan is too strict for that, and was waived (2026-11-03).
  - Rule: compare scores within one engine only.
### GRPO smoke run

**Run.** W&B `HRM-Graph/iyp11un6` (`smoke_grpo_graph`), driver `/work/dfm/marius-ortega/dfm-hrm-graphs/grpo_smoke/run_smoke.sh`.
- Model: zero-shot Mimir v1.5 + LoRA r32 (`DFM-Mimir-v1.5-hfview`), 15 steps.
- Batch: 16 prompts × 8 rollouts, responses up to 1,024 tokens, eager vLLM.
- Data: `data/graphs/bench/verl_smoke`, exported from the gold-only files: 2,086 train / 208 val prompts over
  GraphQA, MetaQA and KQA Pro, at most 2,048 prompt tokens.

**Bug that made the first attempt wrong: token_type_ids misaligned.**
- **Symptom.** Step 1: `training/rollout_probs_diff_mean` 0.155, rollout-vs-actor Pearson 0.77, `rollout_corr/k3_kl`
  696. A correct setup gives about 0.01 / > 0.99 / about 0.
- **Cause.** `jacob-hrm-text-rlvr/verl_hrm_text/hrm_verl_ext.py` injects the PrefixLM `token_type_ids` into the FSDP
  passes, but verl 0.9 hands it a nested `response_mask` holding only each row's response. The patch pads that mask
  on the right to the full row.
  - So the causal marks land on the first prompt tokens, and the response stays bidirectional.
  - Seen with a debug hook: a row of 1,129 real tokens with a 128-token response got `0x128 1x1001` instead of
    `1x1001 0x128`.
- **Fix.** `graph_bench/rl/verl_ext.py` installs its own `prepare_model_inputs`, which marks each row's last
  len(response) valid tokens causal. `GRAPH_RL_DEBUG_TTI=1` prints the layout of the first batches.
- **Result after the fix.** Prob diff 0.0028, Pearson 0.99985, k3_kl 0.0003.
- **Likely affects Jacob's GSM8K GRPO runs.** Same verl 0.9 install, same patch, but `grpo_run_v13.log` has no
  rollout-vs-actor metric, so this is unverified.

**Launch fixes**, all in `run_grpo_graph.sh`:
- **Ray startup:** the raylet timed out waiting for its runtime-env agent. Fix: `RAY_agent_register_timeout_ms`,
  `RAY_raylet_start_wait_time_s`, dashboard off.
- **W&B:** miniforge's wandb 0.30 defaults to api.wandb.ai, while `~/.netrc` only has the CoreWeave key. Fix:
  `WANDB_BASE_URL=https://api.forge.coreweave.com`.
- **Stale checkpoint:** VeRL auto-resumes from `default_local_dir` (a 1-step debug run "finished" instantly on an old
  checkpoint). Smoke and debug runs pass `trainer.resume_mode=disable`.

**Result.** Val subsets of 68–70 questions per dataset (SE ~5–6 pt).

| Val (greedy) | acc step 0 → 15 | has "Answer:" line 0 → 15 | acc among answers with the line, 0 → 15 |
| --- | --- | --- | --- |
| KQA Pro | 0.29 → 0.69 | 0.37 → 0.94 | 0.80 → 0.73 |
| GraphQA | 0.43 → 0.53 | 0.70 → 0.87 | 0.61 → 0.61 |
| MetaQA | 0.70 → 0.87 | 0.73 → 0.89 | 0.96 → 0.98 |

- **Pipeline passes the gate.**
  - Rollout-vs-actor prob diff 0.003–0.0045 and Pearson ≥ 0.999 on every step.
  - About 6.2 min per step (generation about half).
  - No OOM kill; about 45–49 GB anonymous host memory, of which the 10 agent-loop workers take about 2 GB each.
  - The only tracebacks are DataLoader workers killed at shutdown.
- **The validation gain is answer format, not reasoning.** Accuracy among answers that have an "Answer:" line did not
  improve. Train reward (`critic/score/mean`) is noisy: 0.52 at step 1, 0.57–0.59 at steps 14–15. The real E3 runs
  start from an SFT model that already emits the format.
- **To watch in longer runs.**
  - Entropy rose from 2.4 to 3.0.
  - Responses got longer (253 → 337 tokens), and the share cut at 1,024 tokens rose from 7% to 12–16%.

**Next:**
1. SFT data from the gold-only files.
2. E1 / E2 LoRA SFT.
3. E3 GRPO from E2, in W&B `Mimir-Graph-RLVR`.

## Launch of E0–E4 (2026-11-03)

**Decisions (user, 2026-11-03).**
- E0 uses cached models only: Mimir v1.5, Mimir v1, Qwen3.5-4B (stands in for Qwen3-4B), Qwen2.5-7B-Instruct,
  Olmo-3-7B-Instruct, Gemma-4-E4B-it, SmolLM3-3B, HRM-Text-1B (base). Phi-4-mini's cache is empty, so it is out.
- E4's held-out task is CLUTRR (already converted, short prompts), not CronQuestions.
- SFT replay is dfm-dyna-instruct's `apertus-sft-mixture` config, read from the local copy
  `/work/dfm/Mimir/MTCv1/data/dfm7_special_sources/dfm_dyna_instruct_apertus_sft_mixture_shards` (no download).
- Every run of the benchmark logs to W&B `Mimir-Graph-RLVR`.

**Protocol choices.**
- **One prompt for everything.** E0, SFT and GRPO all use the instruct message (`export_verl.user_message`: reason,
  then an `Answer:` line), thinking off. SFT prompts tokenize exactly as the vLLM eval prompts (200/200 checked).
- **Eager vLLM for every eval.** CUDA graphs are 25% faster, but only 42/128 long generations were identical to
  eager (accuracy 20 vs 21 of 64). VeRL rollouts and the parity gate use eager, so scores stay in one engine setting.
- **Mimir's context is 4,096 tokens.** `eval_vllm.py` now shortens the generation budget of a prompt that does not
  leave room for `--max-tokens`; with fewer than 16 tokens left, the row is not generated and scores 0 (it stays in
  the denominator).
- **E1 and E2 train on the same questions.** Rows without a trace are dropped from both (WebQSP: 2,004 of 2,826), so
  only the target differs: `Answer: X.` vs trace + `Answer: X.`
- **Epoch selection:** the last epoch (3), no val-based selection.

**Data** (`data/graphs/bench/sft/`, `verl/`, `verl_clutrr/`):

| Set | Rows |
| --- | --- |
| SFT mixture (E1 `train_answer`, E2 `train_cot`) | 17,917: GraphQA 3,648, MetaQA 2,588, KQA Pro 3,377, WebQSP 2,004, CLRS 3,700 (cap), replay 2,600 (14.5%) |
| GRPO train / val (E3, E3b) | 16,055 / 298, prompts ≤ 2,048 tokens (84 dropped) |
| GRPO CLUTRR train / val (E4) | 12,064 / 300 |
| E4 SFT control | CLUTRR train, answer line only |

**Queues** (each takes the GPU lock per run and skips finished runs):

| Driver | What | Estimate |
| --- | --- | --- |
| `graph_bench/eval/run_e0.sh` | 8 models × (12 named test files + 4 anonymised KGQA files), up to 1,000 rows per file, greedy, 1,024 tokens | Mimir about 1.5 h for the named files (1.85 prompts/s) |
| `graph_bench/run_sft.sh` | merge check, then E1 and E2: graph_inject LoRA r16, 3 epochs, 8,192 tokens/step, last epoch merged into an HF dir (`graph_bench/merge_lora_hf.py`), eval like E0, E2 pass@16 on val and held-out subsets | about 1.5 h of training per arm (from earlier Mimir LoRA runs: 1.2 steps/s) |
| `graph_bench/rl/run_rl_arms.sh` | after E2's merge: E3 (GRPO from E2, 200 steps), E4 (GRPO on CLUTRR from E2, 100 steps; then LoRA SFT on CLUTRR answers from E2), E3b (GRPO from base, 200 steps) | about 21 h per 200 GRPO steps (6.2 min/step in the smoke run) |

- **GRPO settings:** 16 prompts × 8 rollouts, LoRA r32, val every 25 steps, checkpoints every 50.
  Superseded (2026-11-04): responses 2,048 tokens (from 1,024), micro-batches 2, checkpoints every 25, see
  [Results: generation cap](/pages/hrm-graph/mimir-graph-benchmark-results.md#the-generation-cap-understated-e2-2026-11-04). DAPO
  `filter_groups` drops all-0 / all-1 groups online, which applies the plan's "pass rate strictly between 0 and 1"
  rule without a separate pass@k filtering pass.
- **Merge check:** a GraphQA LoRA with a known native test score (0.842) is merged and run through vLLM on the
  200-row parity subset before the SFT runs.
- **Still to do:** convert the VeRL checkpoints (LoRA) to HF dirs and evaluate them like E0; the retention suite;
  the remaining baselines.


## GRPO reward as implemented (2026-11-05)

`graph_bench/rl/reward.py` (used by E3, E3b and E4) scores the final answer only:
- no `Answer:` line (after stripping special tokens) → 0;
- otherwise the eval metric of `graph_bench/scoring.py`: `acc` (1/0) by default; with `GRAPH_REWARD=f1`, set-answer
  records get their set-F1 as partial credit. `run_grpo_graph.sh` defaults to `acc`.
- **No format reward**, despite the design above: the `Answer:`-line rule is a gate, not a bonus.
- **GRPO never sees a gold trace.** The VeRL ground truth (`graph_bench/export_verl.py`) carries only `answers`,
  `answer_format` and `checker`. Traces reach RL only through the starting point: E3 and E4 start from E2, which
  imitated the gold traces, and RL then reinforces whatever reasoning reaches the right answer. E3b starts from the
  base model, so E3 − E3b measures what that trace initialisation buys.

**To do: a format reward.** Not implemented. Candidate: a small bonus (≪ 1, e.g. 0.1) for a well-formed completion
(thought channel closed, exactly one `Answer:` line, answer parses in the record's `answer_format`). The likely use
is E3b: if the base model rarely writes an `Answer:` line, most groups are all-0, DAPO `filter_groups` drops them and
little is learned. Keep `filter_groups.metric=acc`, so the bonus shapes the gradient without letting formatted wrong
answers count as solved. Check E3b's `has_answer` rate in its first steps before deciding.
