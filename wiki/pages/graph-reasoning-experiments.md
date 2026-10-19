---
type: Runbook
title: Graph Reasoning Experiments (ported from HRM-Text-for-Graphs)
description: Datasets, tokenization parity, SM120 attention backend, configs and verified smoke runs for the GraphQA/KGQA recursion-depth study.
tags: [graphs, graphqa, kgqa, recursion, sm120, tokenization, attention]
status: stable
last_updated: 2026-10-19
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

## Layout

- Raw/JSONL data: `data/graphs/` (gitignored); sampled training sets: `data/graphs/sampled/<name>_e<epochs>`.
- Configs: `config/cfg_graphqa.yaml` (inherits `cfg_pretrain`) and children; `config/data/graphs.yaml`.
- Tooling: `scripts/graphs/` (setup, prepare, parity, eval, queues, summaries).
- Queue outputs: `/work/dfm/marius-ortega/dfm-hrm-graphs/<experiment>` (old-stack outputs stay in
  `/work/dfm/marius-ortega/<experiment>` and are not mixed in).
