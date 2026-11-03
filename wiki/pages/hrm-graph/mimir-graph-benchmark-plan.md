---
type: Experiment Plan
title: Is Mimir v1.5 Best at Graph Reasoning? (Zero-Shot, SFT, RLVR)
description: "Approved plan for testing whether DFM-Mimir-v1.5 is the strongest graph / KG reasoner, zero-shot and after SFT and RLVR: claim challenges, hypotheses, models, dataset tiers with cleaning rules, splits, arms, feasibility, and the VeRL + vLLM RL stack found in jacob-hrm-text-rlvr."
tags: [graphs, mimir, rlvr, grpo, vllm, verl, kgqa, benchmark, plan]
status: draft
last_updated: 2026-11-02
confidence: medium
---
# Is Mimir v1.5 Best at Graph Reasoning?

The question is whether DFM-Mimir-v1.5 is the strongest model for reasoning over graphs and knowledge graphs, in
three settings: zero-shot, after SFT, and after SFT followed by RLVR. Both in-distribution and held-out tasks
count. Zero-shot transfer to held-out tasks is the main criterion, because it separates a research-only model from
a general-purpose one. This page records the plan approved on 2026-11-02 and the reasons for each choice. No
experiment has run yet.

- Plan source: the approved plan of 2026-11-02 (Claude Code session).
- Prior results it builds on: [Graph Reasoning Experiments](/pages/hrm-graph/graph-reasoning-experiments.md),
  [Graph-Structure Injection](/pages/hrm-graph/graph-structure-injection.md),
  [GraphQA Chain of Thought](/pages/hrm-graph/graphqa-cot.md).
- RL stack: `/work/dfm/marius-ortega/jacob-hrm-text-rlvr/verl_hrm_text/` (VeRL + vLLM). The fallback is
  `/work/dfm/marius-ortega/hrm-text-rlvr/train_grpo.py` (TRL + HF generate).
- KGQA source loaders (URLs and schemas only): `data/graphs/KGQA/*/`.

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

### Feasibility (one RTX PRO 6000, estimates)

| Item | Estimate |
| --- | --- |
| Tier 1 curation | ~1 week |
| Tier 2 curation (GrailQA's Freebase extraction is the hard part) | 1–2 weeks |
| E0 with the native HRM engine (CoT decoding at 0.14 s/step, batch 8 max) | ~60–90 h for the HRM family. Through vLLM, hours, once parity holds. |
| E1/E2 SFT (~200–300M-token mixture) | ~1 day per model per arm |
| E3 GRPO on Mimir with VeRL + vLLM | ~15–25 h per run with 2–3k-token graph prompts (scaled from GSM8K, below) |

Total: ~5–6 weeks of single-GPU time for the core set, plus ~1 week to validate the RL stack.

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

## Status (2026-11-02)

- The GPU is busy, so only CPU work is approved: dataset download and conversion, trace generators, reward
  functions and their unit tests, parquet builders, and eval and launch scripts written but not run.
- E0–E4, the parity gate and all SFT runs wait for the GPU.
