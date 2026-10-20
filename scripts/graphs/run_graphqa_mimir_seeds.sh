#!/bin/bash
# DFM-Mimir-v1.5 GraphQA fine-tune (config/cfg_graphqa_mimir.yaml), the counterpart of run_graphqa_hrm1b_seeds.sh:
# seeds 0-2, unclipped and gradient_clip_norm=1.0, each through run_graphqa_hrm1b.sh (train, val per epoch, test at
# the best val epoch, zero-shot baseline). Only the best-epoch test report goes to W&B (group graphqa-mimir).
# Finished runs and existing eval reports are skipped, so a rerun resumes at the first unfinished run.
#
# Data (Mimir's Gemma chat format; seed s changes only the data order):
#   export CHAT_TEMPLATE=$PWD/data_io/chat_templates/gemma4_native_chat.jinja \
#          TOKENIZER=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5/tokenizer.json
#   SEED=<s> bash scripts/graphs/prepare_dataset.sh graphqa_mimir data/graphs/graphqa/hrm-text-grl/standard/train.jsonl 20
#   bash scripts/graphs/prepare_dataset.sh graphqa_mimir_val data/graphs/graphqa/hrm-text-grl/standard/val.jsonl 32
#
# ACCUM=<n> sets gradient_accumulation_steps (same 8192 tokens per step, n microbatches).
#
# Detached from the shell / SSH session, behind any other job holding the GPU lock:
#   setsid nohup flock /work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock bash scripts/graphs/run_graphqa_mimir_seeds.sh \
#     >> /work/dfm/marius-ortega/dfm-hrm-graphs/graphqa_mimir/logs/seeds_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
export ROOT=/work/dfm/marius-ortega/dfm-hrm-graphs/graphqa_mimir
export HF_MODEL=danish-foundation-models/DFM-Mimir-v1.5
export HF_DIR=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5
export CONFIG=cfg_graphqa_mimir GROUP=graphqa-mimir TAG=mimir-finetune LOG_SPLITS=test
ACCUM=${ACCUM:-1}
mkdir -p "$ROOT/logs"
for S in ${SEEDS:-0 1 2}; do
  for C in "" clip1; do
    RUN=graphqa_mimir${C:+_$C}_s$S
    DATA=data/graphs/sampled/graphqa_mimir$( [[ $S == 0 ]] || echo "_s$S" )_e20
    OV="data.path=$DATA seed=$S gradient_accumulation_steps=$ACCUM${C:+ gradient_clip_norm=1.0}"
    echo "=== $(date '+%F %T') $RUN ($OV)"
    RUN=$RUN OVERRIDES="$OV" EVAL_DIR=data/graphs/graphqa/hrm-text-grl/standard \
      bash scripts/graphs/run_graphqa_hrm1b.sh || echo "FAILED: $RUN"
  done
done
echo "=== $(date '+%F %T') done"
