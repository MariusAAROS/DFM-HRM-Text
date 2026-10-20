---
type: Runbook
title: Graph Reasoning Experiments (ported from HRM-Text-for-Graphs)
description: Datasets, tokenization parity, SM120 attention backend, configs, verified smoke runs and the HRM-Text-1B fine-tune for the GraphQA/KGQA recursion-depth study.
tags: [graphs, graphqa, kgqa, recursion, sm120, tokenization, attention, fine-tuning, huggingface]
status: stable
last_updated: 2026-10-20
confidence: high
---
# Graph Reasoning Experiments

Recursion-depth study of HRM on graph reasoning (GraphQA main; MetaQA, KQA Pro, pointer chasing, GSM8k,
MATH, MMLU), ported on 2026-10-19 from `HRM-Text-for-Graphs` (fork of sapientinc HRM-Text @ `da566c9`,
FlexAttention stack) onto this training stack. Only data, configs, experiment drivers and the model
knobs the experiments need were ported; the old `pretrain.py`, FlexAttention and RecursionProbe
instrumentation were not. Models are size B, trained from scratch.

## Tokenization parity (verified)

The old `scripts/prepare_sft_data.py` and `data_io` (`--template-mode hrm`) produce the same
sequences: `<|im_start|> <cond> instruction <|im_end|>` | `response <|box_end|>`, same condition map.
The HRM-Text-1B `tokenizer.json` and `data_io/trained_tokenizers/bpe/tokenizer.json` have identical
vocab, merges and 31 added tokens (only `trim_offsets`/decoder flags differ; IDs are unaffected).
Confidence: high.

- `scripts/graphs/check_tokenization_parity.py` on GraphQA standard train: 3,080/3,080 samples
  token-identical to the old prepared dir; `InferenceCheckpoint.tokenize_prompt` reproduces the
  training instruction tokens 3,080/3,080. Token totals also match exactly for `graphqa_scaled`
  (345,441,884/epoch) and `pointer` (5,095,582/epoch). Confidence: high.
- Differences without effect on these datasets: `data_io` encodes literal special-token strings as
  text (no JSONL contains any), supports multi-conditions, truncates/drops overlong rows instead of
  raising, uses a different epoch-shuffle RNG stream (same samples, different order), and adds
  `tokenizer_info.template_mode`. Both drop empty responses.
- `data_io` records `--tokenizer-path` verbatim and inference reloads it: always pass an absolute path.
- The Rust tokenizer's directory walk does not follow symlinks: stage inputs as hard links or copies.
- rustup on this machine has `stable` and `1.95.0` but no default toolchain: set `RUSTUP_TOOLCHAIN=stable`.

## SM120 attention (RTX PRO 6000 Blackwell)

`accelerator_type=sm120` (CC 12.x) routes PrefixLM attention to
`models/flash_attention_prefixlm_torch_varlen.py`, which uses PyTorch's built-in FA2 varlen kernels
(`torch.nn.attention.varlen.varlen_attn`, torch 2.11.0+cu130 ships `sm_120`). Routing is the ROCm
backend's gather scheme: a bidirectional prefix pass, then a causal pass of response queries over
prefix + response keys with bottom-right alignment (`window_size=(-1, 0)`). Confidence: high.

- FA4 CuTe does not work for SM120 varlen here: `flash-attn-4` 4.0.0b14 @ `4178915` with
  cutlass-dsl 4.5.2 (the `hrm` conda env stack) aborts with `Operation creation failed`, and 4.0.0b19
  with cutlass-dsl 4.6.2 fails with a CuTe layout-congruence MLIR error. FA3 is SM90-only. Confidence: high.
- `tests/test_flash_attention_prefixlm_sm120.py`: forward and dq/dk/dv match the dense SDPA reference
  within 2x the bf16 dense error on mixed, long-prefix and GraphQA-like packs, causal and PrefixLM,
  with and without precomputed routing; padding rows exactly zero.
- KV-cache inference uses SDPA and was checked against the training forward on a step-250 checkpoint:
  410/410 teacher-forced answer positions agree on argmax, max |logit diff| 0.125 (bf16).

## Training stack facts that matter for these runs

- bf16 compute on CUDA exists only under `torch.distributed` (FSDP mixed-precision policy). A plain
  `python pretrain.py` runs fp32 and the FA2 kernel rejects it ("FlashAttention only support fp16 and
  bf16"). Launch with `torchrun --standalone --nproc_per_node=1`. Confidence: high.
- `lr_auto` (default true) divides H/L learning rates by their backward call counts; the graph configs
  set `lr_auto: false` to keep the old single-LR recipe. With full/explicit BPTT budgets the auto rates
  now use the resolved budget (`models/module_learning_rates.py`). Confidence: high.
- `pretrain.py` writes `all_config.yaml` / `train_metadata.yaml` (needed by eval) only with a W&B run.
- Logged `bp_steps` = block applications with gradient (H_bp + L_bp). For fixed budgets
  (`full_backprop` or explicit H/L_bp_steps) it reports the resolved budget, e.g. 21 = 3×(6+1) for
  H3L6 full BP. Before this fix (smoke run `k1qnn8oi`) it showed the unused warmup value 5.

## Model knobs ported

- HRM/TRM: `full_backprop`, `H_bp_steps`/`L_bp_steps` (-1 = all; L counts across all H cycles),
  `inject_x`; shared `models/common.py:resolve_bp_steps`. Defaults keep the legacy warmup path
  bit-identical (`tests/test_hrm_bp_budget.py`).
- `norm_type=peri` (Peri-LN: sublayer inputs and outputs normalised, final norm kept) for the
  recursion-hypothesis runs (`tests/test_transformer_peri_norm.py`). No stability diagnostics for peri.

## Verified smoke runs (2026-10-19, one RTX PRO 6000, torchrun, W&B online)

| Run | Steps | Speed | Peak memory | Result |
| --- | ---: | ---: | ---: | --- |
| GraphQA H2L3 truncated | 250 | 5.3 it/s | n/a | loss 0.71, token acc 0.75, sharded `fsdp2_step_250` written |
| GraphQA H3L6 full BP | 150 | ~1.27 it/s | 37.3 GB | loss decreasing, no NaN, whole step compiled |
| GraphQA H1L1 full BP (queue `BENCH=1`, 150 s) | ~1,000 | 7.3 it/s | 12.2 GB | old FlexAttention notes: 12.4 steps/s; small cells may be slower here, not profiled (confidence: medium) |

Eval smoke (`scripts/graphs/eval_graphqa.py`, step 250, val): runs end to end, 10.6% accuracy
(early checkpoint mostly answers "Yes."). W&B: entity `m-ortega-p-le-l-onard-de-vinci` on
`forge.coreweave.com`, projects `HRM-GraphQA-Smoke` and `HRM-GraphQA-Smoke-eval`.

## W&B layout (from 2026-10-19)

Every run logs to project `HRM-Graph` (entity `m-ortega-p-le-l-onard-de-vinci`). `wandb_group` is the experiment
(graphqa, graphqa-fullbp, recursion-hyp, graphqa-hrm1b, ...), `job_type` is train or eval, bench runs use
`<experiment>-bench`. Earlier projects (HRM-GraphQA-*, *-eval, *-smoke) hold the old-stack and smoke results and are
not used for comparisons.

## Fine-tuning pretrained HRM-Text-1B (verified 2026-10-19)

`conversion/convert_from_hf.py` maps an HF HRM-Text checkpoint (hub layout = `convert_to_hf.py`'s vllm_packed) onto the
native model; `init_weights_path` loads it for fine-tuning. HRM-Text-1B is size XL with half_layers (2 x 16 layers,
hidden 1536, 12 heads, MLP 4096, lecun init so embedding scale sqrt(1536)), H2L3, `L_bp_cycles=[0,3]` = `bp_max_steps 5`.
Confidence: high.

- Parity: tiny random model, fp32 CPU, native cache path vs transformers 5.20.1 `HrmTextForCausalLM` with
  `token_type_ids`: max logit diff 1e-6 (`tests/test_convert_from_hf.py`, 1e-5 tolerance). In float64 the gap stays
  ~1e-7 because HF RMSNorm always computes in fp32. Real 1B in bf16 on 32 GraphQA val prompts: 202/203 answer argmaxes
  agree, max logit diff 0.25 (logit scale 19).
- The native `zL_init` buffer is allocated in `fwd_bwd_dtype` (bf16). The hub checkpoint stores it in bf16, so nothing is lost.
- Recipe `config/cfg_graphqa_hrm1b.yaml` follows Graph-Representation-Learning-for-LLM (GRL) `configs/hrm_text.yaml`:
  - same splits (row-identical to GRL `data/graphqa/baseline/standard`);
  - same prompt (`#### Question\n` + question, condition `direct`);
  - lr 5e-5 constant, betas 0.9/0.999, wd 0.01, no EMA.
  - Differs from GRL in: AdamATan2 on fp32 master weights instead of AdamW on bf16 weights, 8192-token packed batches
    (~28 samples instead of 32), 20 warmup steps, a supervised `<|box_end|>`, and 10 epochs without early stopping.
- GRL's `val/exact_match` is teacher-forced: argmax == label at each answer token. `eval_graphqa.py` reports it as
  `tf_exact_match` (greedy output starts with the gold answer). Once fine-tuned, it equalled greedy normalized exact
  match at every epoch.
  - The two are equivalent under greedy decoding (by induction over answer positions), up to bf16 KV-cache vs
    full-forward argmax flips, **only if `--max_generation` covers the whole answer**. Teacher forcing has no length
    limit; generation does.
  - Decision (2026-10-20): compare models on greedy-generation EM (`eval/accuracy`). Teacher-forced EM does not measure
    whether the model can generate the answer. GRL only reports teacher-forced EM, so GRL comparisons use that metric
    as given.
- One RTX PRO 6000: 1.3 it/s, 76 GB peak, 1,100 steps in about 18 minutes. Evaluating 385 samples takes about 75 s at
  `--max_tokens 1536 --batch_size 32`.
  - Breakdown (2026-10-20, 64 samples at batch 8, sharing the GPU with another eval): imports 11 s, load 3.5 s,
    prefill 137 ms per prompt, decode 111 ms per step.
  - Prefill is mostly compute: 128 layer applications per token (8 H/L x 16), ~500k prompt tokens. Decode is
    overhead-bound: eager, one GPU sync per step.
  - Kept as is (normal). Possible speedups: one process for all epochs, CUDA-graph decode, batched prefill.

Results, seed 0 (`scripts/graphs/run_graphqa_hrm1b.sh`; W&B group `graphqa-hrm1b`).
**Superseded 2026-10-20:** these evals used `--max_generation 32`, which cut off 9 test and 10 val answers. See
[Generation-length bug and GRL test comparison](#generation-length-bug-and-grl-test-comparison-2026-10-20).

| Checkpoint | val tf_EM | test tf_EM | test gen EM |
| --- | ---: | ---: | ---: |
| zero-shot (imported, step 0) | 0.000 | 0.000 | 0.384 |
| best val epoch (7 of 10) | 0.829 | 0.795 | 0.795 |
| `gradient_clip_norm=1.0`, best val epoch (10 of 10) | 0.829 | 0.797 | 0.797 |
| GRL `hrm-text-id` (2 runs), best val epoch | 0.849 / 0.852 | not logged | |
| GRL `graphqa-hrm-text-clip1-id`, best val epoch | 0.865 | not logged | |

- Per-epoch val tf_EM: 0.769, 0.761, 0.816, 0.790, 0.787, 0.803, 0.829, 0.821, 0.797, 0.818.
- Clipped at 1.0 (GRL `-clip1` recipe): per-epoch val 0.709, 0.735, 0.777, 0.808, 0.808, 0.821, 0.784, 0.821, 0.816,
  0.829. It starts slower, ends at the same best val, and gives the same test score (0.797 vs 0.795, one sample apart).
  Here clipping does not reproduce the +1.3 points GRL saw (0.865 vs 0.852). Only its test report was logged to W&B.
- Test per task at epoch 7 matches GRL's per-task profile:
  - MaximumFlow 0.18, TriangleCounting 0.53, EdgeCount 0.62;
  - CycleCheck, EdgeExistence, NodeCount and Reachability at 1.0.
- This run is about 2-3 points below GRL's best val. One val sample is 0.26 points, and the binomial SE at 0.83 on
  385 samples is ~1.9 points. One seed per side, so the gap is not established. Confidence: medium.
  (Superseded 2026-10-20: most of the gap was the 32-token generation cap, see below.)
- Zero-shot, the model answers in a different format ("Yes", "10"), so tf_EM is 0. Normalized generation EM reaches
  0.38-0.39 only because `normalize` strips trailing periods.

### Generation-length bug and GRL test comparison (2026-10-20)

Seeds 0-2, unclipped and `gradient_clip_norm=1.0` (`scripts/graphs/run_graphqa_hrm1b_seeds.sh`; seed s trains on
`graphqa_grl_s<s>_e20`, so only the data order changes). Test at the best val epoch, 385 samples, as first evaluated
with `--max_generation 32` (W&B group `graphqa-hrm1b`, 2026-10-19):

| Arm | s0 | s1 | s2 | mean |
| --- | ---: | ---: | ---: | ---: |
| unclipped | 0.795 (ep7) | 0.800 (ep8) | 0.792 (ep10) | 0.796 |
| clip 1.0 | 0.797 (ep10) | 0.803 (ep7) | 0.803 (ep8) | 0.801 |

- GRL's test score for H2L3 is in W&B project `HRM-Pretrained-Recursion`, run `b-graphqa-H2L3` (`o2n3mu05`):
  **0.834** (321/385), teacher-forced EM. It evaluates GRL's `graphqa-hrm-text-clip1-id` (Meta-ICL `8dfnfqn3`)
  `last.ckpt`, which is epoch 8 (best val 0.865) or epoch 9 (val 0.852). The timestamps don't say which.
  GRL's per-sample predictions: `../Graph-Representation-Learning-for-LLM/predictions/depth-b-graphqa/H2L3/`. They are
  row-aligned with our test JSONL.
- Root cause of most of the gap: `eval_graphqa.py --max_generation 32`. GraphQA gold answers reach 54 tokens on test,
  57 on val and 62 on train. 9 test answers and 10 val answers exceed 32 tokens: long ConnectedNodes and
  DisconnectedNodes node lists. In all 6 runs, our output on those 9 test items stops mid-list on a correct prefix.
  GRL gets all 9 right. That is why ConnectedNodes was exactly 28/34 in every run. Confidence: high.
- Upper bound if those answers had finished correctly: clip 0.821 / 0.826 / 0.826 (mean 0.824), unclipped 0.818 /
  0.823 / 0.816 (mean 0.819), about 1 point under GRL's 0.834. Confidence: medium (bound, not a measurement).
- Clipping: +0.5 points mean, inside noise (seed sd ~0.4 points, binomial SE ~2 points). It fired on 37-43% of steps.
- Fix: `--max_generation` default and every driver set to 96 (longest prompt 1,324 + 96 < `--max_tokens 1536`).
  The 32-token reports moved to `<ckpt>/gen32/`. The from-scratch queues (`lib_local_queue.sh`,
  `run_graphqa_scaled.sh`) had the same 32 cap, so their existing GraphQA results undercount long list answers.
- Re-evaluation at 96 tokens (partial, 2026-10-20 07:50): seed-0 val gains 8-10 samples at every epoch (+2.1 to +2.6
  points). Epochs 1-6: 0.795, 0.787, 0.842, 0.810, 0.813, 0.829. ConnectedNodes val goes to 0.95-1.0. Zero-shot
  gains 2 samples (test 0.390). Test and the other seeds: pending. Confidence: high.

### DFM-Mimir-v1.5 fine-tune (queued 2026-10-20)

`danish-foundation-models/DFM-Mimir-v1.5` (revision `521b40b`) is the same network as HRM-Text-1B (XL, 2 x 16 layers,
hidden 1536, H2L3, 1.787B params). It uses the 262,144-token Gemma tokenizer and the Gemma chat format, and was
pre-trained with `L_bp_cycles=[3, 3]` (bp 8). Confidence: high.

- Data: `CHAT_TEMPLATE=data_io/chat_templates/gemma4_native_chat.jinja TOKENIZER=<HF_DIR>/tokenizer.json
  prepare_dataset.sh` runs `scripts/tokenize_chat_template.py` (`template_mode: jinja_chat_template`):
  `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n` | `{answer}<turn|>\n`. Condition `direct` adds no system turn.
  Sets: `graphqa_mimir_e20`, `graphqa_mimir_s{1,2}_e20` and `graphqa_mimir_val_e32`, with 922,395 train
  tokens/epoch and nothing dropped.
- The Rust tokenizer's `--template-mode gemma4-chat` is not usable for eval. It writes `template_mode: "Gemma4Chat"`
  with HRM `boq`/`eoq`/`eoa`, which `simple_inference_engine.py` does not recognize, so it would build HRM-format
  prompts and find no stop token.
- Parity:
  - All 3,080 train prompts are token-identical to `apply_chat_template` with the engine's tokenizer settings
    (`bos_token=<bos>`; without it the leading `<bos>` is missing).
  - Imported weights vs transformers `HrmTextForCausalLM`, fp32 CPU, one GraphQA sample: max logit diff 1.6e-4
    (scale 26), every answer-token argmax agrees.
- Recipe: `config/cfg_graphqa_mimir.yaml` inherits `cfg_graphqa_hrm1b` unchanged (8192 tokens/step, bp 5, lr 5e-5).
  One-epoch bench (W&B `graphqa-mimir-bench`): 87 GB reserved of 96, ~1.2 it/s, validation and checkpoint save OK.
- Driver: `scripts/graphs/run_graphqa_mimir_seeds.sh` (seeds 0-2 x {unclipped, clip 1.0}, through
  `run_graphqa_hrm1b.sh` with `CONFIG`/`GROUP`/`TAG`/`HF_DIR`/`ROOT` overrides). Outputs go to
  `/work/dfm/marius-ortega/dfm-hrm-graphs/graphqa_mimir`, W&B group `graphqa-mimir`. Results: pending.

## Fixes made 2026-10-19

- `run_graphqa_fullbp_local.sh` / `run_recursion_hyp_local.sh` defaulted their eval dirs to `data/graphqa/...` and
  `data/pointer*/...`, which do not exist. They now point to `data/graphs/...`. Before this fix, every queue eval would
  have failed.
- In-training validation needs one sampled val epoch per validation call: each restart of the val iterator reads the
  next `epoch_N/` dir. With a 1-epoch val set, the first fine-tune crashed at step 110 (`epoch_1/inst_start.npy`
  missing). `validation_batches` must exceed the val set's batch count, otherwise alternate calls score only a remainder.

## Layout

- Raw/JSONL data: `data/graphs/` (gitignored); sampled training sets: `data/graphs/sampled/<name>_e<epochs>`.
- Configs: `config/cfg_graphqa.yaml` (inherits `cfg_pretrain`) and children; `config/data/graphs.yaml`.
- Tooling: `scripts/graphs/` (setup, prepare, parity, eval, queues, summaries, `run_graphqa_hrm1b.sh`).
- HF imports: `/work/dfm/marius-ortega/dfm-hrm-graphs/hf/<model>`; fine-tune outputs: `/work/dfm/marius-ortega/dfm-hrm-graphs/graphqa_hrm1b`.
- Queue outputs: `/work/dfm/marius-ortega/dfm-hrm-graphs/<experiment>` (old-stack outputs stay in
  `/work/dfm/marius-ortega/<experiment>` and are not mixed in).
