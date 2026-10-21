#!/bin/bash
# Does the bp mismatch explain DFM-Mimir-v1.5 degrading under fine-tuning (MetaQA test 0.650 < zero-shot 0.761)?
# v1.5 was pre-trained with bp_max_steps 8; the MetaQA fine-tune (metaqa_mimir_s0) used bp 5. bp 8 does not fit at
# 8192 tokens per microbatch, so it runs with gradient_accumulation_steps=2 (same 8192 tokens per optimizer step).
#   1. Control, bp 5 + accum 2, 500 steps (W&B group metaqa-mimir-accum-check): must track metaqa_mimir_s0's train/val
#      loss (same data order), which clears accumulation as a confounder.
#   2. Full bp 8 run through run_hf_finetune.sh (3 epochs, val per epoch, test at the best val epoch; W&B group
#      metaqa-mimir-bp8). The zero-shot reports are copied from metaqa_mimir (same model), so they are not recomputed.
#
# Detached from the shell / SSH session, behind any other job holding the GPU lock:
#   setsid nohup flock /work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock bash scripts/graphs/run_metaqa_mimir_bp8.sh \
#     >> /work/dfm/marius-ortega/dfm-hrm-graphs/metaqa_mimir_bp8/queue.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
H=/work/dfm/marius-ortega/dfm-hrm-graphs
ROOT=$H/metaqa_mimir_bp8
export PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$ROOT/logs" "$ROOT/ckpts/zeroshot"

RUN=metaqa_mimir_accum2_check
if [[ -f "$ROOT/logs/train_$RUN.log" ]] && grep -q "View run $RUN" "$ROOT/logs/train_$RUN.log"; then
  echo "=== [skip] $RUN done"
else
  echo "=== $(date '+%F %T') $RUN"
  torchrun --standalone --nnodes=1 --nproc_per_node=1 pretrain.py --config-name cfg_metaqa_mimir \
    init_weights_path=$H/hf/DFM-Mimir-v1.5/unsharded_step_0.pt gradient_accumulation_steps=2 max_steps=500 \
    wandb_group=metaqa-mimir-accum-check run_name=$RUN checkpoint_path=$ROOT/ckpts/$RUN \
    > "$ROOT/logs/train_$RUN.log" 2>&1 || echo "FAILED: $RUN"
  rm -f "$ROOT"/ckpts/$RUN/*.pt; rm -rf "$ROOT"/ckpts/$RUN/fsdp2_*  # control only: keep no weights
fi

cp -n "$H"/metaqa_mimir/ckpts/zeroshot/eval_grl_{val,test}_ep0.json "$ROOT/ckpts/zeroshot/"
echo "=== $(date '+%F %T') metaqa_mimir_bp8_s0"
ROOT=$ROOT HF_MODEL=danish-foundation-models/DFM-Mimir-v1.5 HF_DIR=$H/hf/DFM-Mimir-v1.5 CONFIG=cfg_metaqa_mimir_bp8 \
  GROUP=metaqa-mimir-bp8 TAG=mimir-finetune RUN=metaqa_mimir_bp8_s0 OVERRIDES="seed=0 gradient_accumulation_steps=2" \
  LOG_SPLITS=test EVAL_DIR=data/graphs/metaqa/hrm-text-grl/standard \
  bash scripts/graphs/run_hf_finetune.sh || echo "FAILED: metaqa_mimir_bp8_s0"
echo "=== $(date '+%F %T') done"
