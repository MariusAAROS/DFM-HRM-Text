# CLAUDE.md

## Response style

- Give factual, straight-to-the-point answers. No filler, no padding, no restating the question.
- State results and numbers directly; flag uncertainty only when it matters.

## What this repo is

Fork of the DFM HRM-Text training stack (hierarchical recurrent LM, H/L modules; FSDP2, PrefixLM packing,
FA3/FA4 attention). It also hosts the **graph-reasoning recursion-depth study** ported from
`../HRM-Text-for-Graphs` (the old FlexAttention fork): GraphQA (main), MetaQA (+ gold/retrieved
variants), KQA Pro, pointer chasing, GSM8k, MATH, MMLU. These models are trained **from scratch** at size B.
Only data, configs, experiment drivers and the model knobs they need were ported. The old `pretrain.py`,
FlexAttention and RecursionProbe instrumentation were not.
It also fine-tunes **pretrained Hugging Face HRM-Text checkpoints** (e.g. `sapientinc/HRM-Text-1B`), to compare with
`../Graph-Representation-Learning-for-LLM` (GRL), which fine-tuned the same model through transformers/Lightning.
Results from the old stack are not used for comparisons any more.

- Read [`AGENTS.md`](AGENTS.md) and the OKF wiki ([`wiki/index.md`](wiki/index.md), rules in
  [`wiki/schema.md`](wiki/schema.md)) for DFM-wide knowledge, and record durable findings there.
  AGENTS.md's operational notes (`hrm` conda env, eval scheduler, `/work/dfm/HRM-Text`, B200) describe
  the team's production checkout, not this one.
- Graph-experiment details and verified numbers: [`wiki/pages/graph-reasoning-experiments.md`](wiki/pages/graph-reasoning-experiments.md).

## Environment

- Use the repo-local venv **`.venv`** (Python 3.13, torch 2.11.0+cu130), built by `bash scripts/graphs/setup_venv.sh`.
  No FlashAttention package is installed (see below). The vLLM/euroeval extras are not installed; base
  `lm-eval` and `math-verify` are, because `evaluation/benchmarks.py` imports them.
- Machine: one **RTX PRO 6000 Blackwell (SM120, CC 12.0, 96 GB)**, no SLURM.
- Always set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for training, and put `.venv/bin` on `PATH` (torchrun).
- **Run every training and eval detached from the SSH/Claude session**, so quitting or a dropped session does not
  kill it: `setsid nohup bash <script> > <log> 2>&1 < /dev/null &`. Never launch long jobs as children of the
  interactive shell (including Claude Code background tasks).
- `data_io/` is a submodule (`git submodule update --init data_io`). Its Rust tokenizer must be built and run
  from `data_io/tokenizer` with `RUSTUP_TOOLCHAIN=stable`: rustup has toolchains but no default here.

## Attention backends (`accelerator_type`)

| Value | GPU | PrefixLM backend |
| --- | --- | --- |
| `sm90` | H100 | FA3 (`models/flash_attention_prefixlm_v2.py`) |
| `sm100` | B200 | FA4 CuTe (`models/flash_attention_prefixlm_fa4.py`) |
| `sm120` | RTX PRO 6000 / RTX 50xx | PyTorch built-in FA2 varlen (`models/flash_attention_prefixlm_torch_varlen.py`) |

- Graph configs default to `sm120`. FA4's SM120 varlen kernels fail to compile with every available build
  (4.0.0b14 @ 4178915 and 4.0.0b19), so `sm120` uses `torch.nn.attention.varlen.varlen_attn`.
  It uses the ROCm backend's prefix/causal gather routing.
- KV-cache inference (`simple_inference_engine.py`) uses SDPA on every GPU.

## Data pipeline (data_io, verified identical to the old prep)

- Raw/JSONL: `data/graphs/<dataset>/...` (gitignored, copied from the old repo). Sampled training sets go in
  `data/graphs/sampled/<name>_e<epochs>/`.
- `bash scripts/graphs/prepare_dataset.sh <name> <train.jsonl> <epochs>` runs the Rust tokenizer, then
  `sample_tokenized.py`, then a no-drop check. `bash scripts/graphs/prepare_all.sh` rebuilds every dataset.
- **`<epochs>` must equal `epochs` in the training config**: one `epoch_N/` dir is consumed per epoch.
- Template: `<|im_start|> <cond> instruction <|im_end|>` | `response <|box_end|>`, using the HRM-Text-1B BPE
  tokenizer. Its vocab, merges and special tokens are identical to `data_io/trained_tokenizers/bpe`.
- Pass an **absolute** tokenizer path: data_io stores it verbatim, and inference reloads the tokenizer from it.
- `scripts/graphs/check_tokenization_parity.py` compares against an old prepared dir and against
  inference-time prompt tokenization.

## Configs

- `config/cfg_graphqa.yaml` is the recipe: size B, 4096 tokens/step, lr 3e-4, warmup 200, cosine to 0.1, EMA
  0.999, seed 0. It inherits `cfg_pretrain`, so any training key can be overridden. Children change only
  data/epochs/W&B group: `cfg_graphqa_scaled`, `cfg_metaqa`, `cfg_kqapro`, `cfg_gsm8k`, `cfg_math`, `cfg_mmlu`,
  `cfg_pointer`, `cfg_pointer2`. Pass `data.path=data/graphs/sampled/...`.
- `config/cfg_graphqa_hrm1b.yaml`: full fine-tune of HRM-Text-1B (size XL = 2 x 16 layers, hidden 1536) on GraphQA
  with GRL's prompt and hyperparameters (lr 5e-5 constant, betas 0.9/0.999, wd 0.01, no EMA, bp_max_steps 5).
- **W&B: every run goes to project `HRM-Graph`**. `wandb_group` names the experiment (graphqa, graphqa-fullbp,
  recursion-hyp, graphqa-hrm1b, ...), `job_type` is train or eval, bench runs use group `<experiment>-bench`.
  `wandb_tags` is also available. The old per-experiment projects (HRM-GraphQA-*, *-eval) are history.
- `lr_auto: false` keeps the old single-LR recipe. The new stack's default (`true`) divides the H/L LRs by
  their backward call counts.
- Old → new keys: `resume_from` → `resume_checkpoint_path`; `compile_scope=step|block` →
  `compile_train_batch=true|false`; `arch.grad_checkpointing` → `activation_checkpointing`;
  `instrumentation.*` → not ported.
- Model knobs:
  - `arch.full_backprop=True`;
  - `arch.H_bp_steps`/`arch.L_bp_steps` (HRM: `+arch.` prefix; -1 = all; L counts across all H cycles);
  - `arch.inject_x=True`;
  - `arch.norm_type=peri` (Peri-LN).

## Pretrained Hugging Face checkpoints

- `conversion/convert_from_hf.py` is the inverse of `convert_to_hf.py` (vLLM-packed layout = the hub layout). It composes
  a training config, refuses to write if that config's architecture differs from the HF `config.json`, and writes a
  native checkpoint dir: `unsharded_step_0.pt`, `all_config.yaml`, `train_metadata.yaml`.
- Fine-tune: `init_weights_path=<dir>/unsharded_step_0.pt` loads the weights only (fresh optimizer, step 0). It is
  ignored when resuming. Zero-shot eval: `eval_graphqa.py --ckpt_path <dir> --ckpt_tag step_0`.
- Parity: `tests/test_convert_from_hf.py` (tiny random model, fp32) matches transformers' `HrmTextForCausalLM` to
  1e-5. The real 1B in bf16 agrees on 202/203 teacher-forced answer argmaxes (max logit diff 0.25 on a scale of 19).
- The HRM-Text-1B tokenizer equals `data_io/trained_tokenizers/bpe`, so the graph datasets need no re-tokenization.
  GRL's prompt adds `#### Question\n` before the question: `data/graphs/graphqa/hrm-text-grl` (with `task` labels),
  sampled as `graphqa_grl_e20` (train) and `graphqa_grl_val_e32` (val).
- `eval_graphqa.py` metrics (renamed 2026-10-20), all from one greedy generation:
  - `greedy_acc` (was `accuracy`): normalized full-answer match, nothing generated after it. **Compare models on this.**
  - `greedy_em` (was `tf_exact_match`): raw `pred.startswith(gold)`. Whitespace sensitive, and it is NOT teacher-forced.
  - `tf_em`: GRL's teacher-forced `val/exact_match` (argmax == label on every answer token, stop token excluded),
    read off the generation as "generated token ids start with the gold training token ids". Selects the best epoch.
  - Old reports and W&B runs carry `accuracy`/`tf_exact_match`; readers fall back to them.
- GRL's H2L3 test score is in W&B project `HRM-Pretrained-Recursion`, run `b-graphqa-H2L3` (0.834, clip1 GRL model).
- `--max_generation` must cover the longest answer (GraphQA: 62 tokens). It is 96 everywhere; the old 32 cut off long
  ConnectedNodes/DisconnectedNodes lists (9 test samples, ~2.3 points).
- 1B fine-tune on one RTX PRO 6000: ~1.3 it/s at 8192 tokens/step, 76 GB peak. Eval: `--max_tokens 1536
  --batch_size 32` (GraphQA prompts reach ~1,330 tokens; the KV cache of 8 H/L applications x 16 layers is large).
- Driver: `bash scripts/graphs/run_graphqa_hrm1b.sh` (import, train, val per epoch, test at the best val epoch,
  zero-shot baseline, W&B group `graphqa-hrm1b`).
- DFM-Mimir-v1.5: same network, Gemma tokenizer (262k vocab) and chat format. Data via
  `CHAT_TEMPLATE=data_io/chat_templates/gemma4_native_chat.jinja TOKENIZER=... prepare_dataset.sh` (the Python
  `tokenize_chat_template.py`; the Rust `gemma4-chat` mode's metadata is not understood by inference). Config
  `cfg_graphqa_mimir`, driver `scripts/graphs/run_graphqa_mimir_seeds.sh`, W&B group `graphqa-mimir`, 87 GB peak.
  DFM-Mimir v1 (`danish-foundation-models/DFM-Mimir`): same tokenizer and data, `cfg_graphqa_mimir1`,
  `MIMIR=v1 bash scripts/graphs/run_graphqa_mimir_seeds.sh`, W&B group `graphqa-mimir1`.
- Queues share the GPU via `flock /work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock`. Never edit a bash
  driver while a queue is running it: bash reads scripts incrementally, so the running instance can execute garbage.

## Commands

```bash
# Train (single GPU: always torchrun -- bf16 compute exists only under torch.distributed)
PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=1 pretrain.py --config-name cfg_graphqa \
  data.path=data/graphs/sampled/graphqa_e20 arch.H_cycles=2 arch.L_cycles=3 \
  run_name=graphqa_H2_L3 checkpoint_path=/work/dfm/marius-ortega/dfm-hrm-graphs/ckpts/graphqa_H2_L3

# Evaluate (greedy_acc, greedy_em, tf_em, set P/R/F1/hits@1; reads H/L from all_config.yaml, --L_cycles
# overrides). Logs to W&B HRM-Graph in the checkpoint's wandb_group; --no_wandb to skip.
.venv/bin/python scripts/graphs/eval_graphqa.py --ckpt_path <ckpt_dir> \
  --data data/graphs/graphqa/hrm-text/standard/test.jsonl --use_ema \
  --out <ckpt_dir>/eval_test.json

# Import + fine-tune + evaluate HRM-Text-1B on GraphQA
bash scripts/graphs/run_graphqa_hrm1b.sh

# Experiment queues (one worker per visible GPU; BENCH=1 for speed/memory)
bash scripts/graphs/run_graphqa_fullbp_local.sh
bash scripts/graphs/run_recursion_hyp_local.sh
bash scripts/graphs/run_graphqa_scaled.sh

# Tests for the ported pieces (the SM120 parity tests need the GPU)
.venv/bin/python -m pytest -q tests/test_flash_attention_prefixlm_sm120.py tests/test_hrm_bp_budget.py \
  tests/test_transformer_peri_norm.py tests/test_flash_attention_prefixlm_bounds.py tests/test_convert_from_hf.py
```

## Gotchas

- Plain `python pretrain.py` runs fp32 on CUDA (no FSDP mixed precision without a process group) and fails in
  the FA2 kernel. Use `torchrun --nproc_per_node=1`.
- Log every run to W&B: `all_config.yaml` and `train_metadata.yaml`, which eval needs, are only written with a
  W&B run. Logged in as `m-ortega` (entity `m-ortega-p-le-l-onard-de-vinci`, host `forge.coreweave.com`).
  Use `WANDB_MODE=offline` only without network, and never `disabled`.
- Queue roots default to `/work/dfm/marius-ortega/dfm-hrm-graphs/<experiment>`. Old FlexAttention results stay
  in `/work/dfm/marius-ortega/<experiment>`. Pointing a queue there would skip runs as "already trained".
- Run-time estimates in the queue tables come from the old FlexAttention stack; re-bench with `BENCH=1`.
  Measured here: H2L3 truncated 5.3 it/s; H3L6 full BP ~1.27 it/s at 37 GB peak; H1L1 full BP 7.3 it/s
  (the old FlexAttention notes say 12.4, so small cells may be slower here; not yet profiled).
- Truncated BPTT budget shrinks with H (H_bp = min(H, bp-1), L_bp = bp - H_bp), which confounds depth with
  gradient signal. The full-BP arm de-confounds it. The logged `bp_steps` is the number of block applications
  with gradient (H_bp + L_bp): H×(L+1) under full BP, e.g. 21 for H3L6.
- The `data_io` tokenizer does not follow symlinks; `prepare_dataset.sh` hard-links or copies its input.
- Seeds: `seed` in the training config only seeds the random weight init. The data order (shuffle + packing per epoch)
  is fixed when the dataset is sampled (`SEED=<s> bash scripts/graphs/prepare_dataset.sh ...` →
  `<name>_s<s>_e<E>`), and the dataloader adds no randomness. A fine-tune with `init_weights_path` is therefore
  identical across `seed` values (up to GPU kernel noise): replicate it by training on a reseeded dataset.
- In-training validation (`data.validation_path`): every validation call restarts the val iterator, which moves to the
  next `epoch_N/` dir. Sample the val set with at least one epoch per validation call, and set `validation_batches`
  above its batch count, or alternate calls score a leftover fragment.
- Pre-existing test failures, unrelated to graphs: `tests/test_gradient_skip.py` (2 failures; the config stub
  lacks `experiment_update_probe_interval`), tests that need `psutil` (handoff, `test_training_safety.py`), and
  `tests/test_hf_export_rope.py` (imports `eval_scheduler.runtime`, absent from this checkout).
