#!/bin/bash
# Fine-tune HRM-Text-1B, DFM-Mimir v1 and DFM-Mimir v1.5 on MetaQA and KQA Pro (GRL prompt), one seed, unclipped,
# 3 epochs (config/cfg_<dataset>_<model>.yaml), each through run_hf_finetune.sh: train, val per epoch, test at the
# best val epoch (tf_em), zero-shot baseline. Only test reports go to W&B (groups <dataset>-<model>).
# Finished runs and existing eval reports are skipped, so a rerun resumes at the first unfinished step.
#
# Data (GRL prompt "#### Question\n" + question; HRM template for HRM-Text-1B, Gemma chat template for both Mimirs):
#   format_graphqa_to_hrm.py --input-dir data/graphs/<d>/baseline[-grl] --output-dir data/graphs/<d>/hrm-text-grl \
#     --instruction-prefix $'#### Question\n' --keep-task
#   prepare_dataset.sh <d>_grl .../train.jsonl 3 ; prepare_dataset.sh <d>_grl_val .../val.jsonl 8
#   CHAT_TEMPLATE=... TOKENIZER=<Mimir tokenizer.json> prepare_dataset.sh <d>_mimir[_val] ...
#
# Detached from the shell / SSH session, behind any other job holding the GPU lock:
#   setsid nohup flock /work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock bash scripts/graphs/run_kgqa_finetunes.sh \
#     >> /work/dfm/marius-ortega/dfm-hrm-graphs/kgqa_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
H=/work/dfm/marius-ortega/dfm-hrm-graphs
for D in ${DATASETS:-metaqa kqapro}; do
  if [[ $D == kqapro ]]; then EVAL_MAX_TOKENS=3200 EVAL_BATCH=16; else EVAL_MAX_TOKENS=1536 EVAL_BATCH=32; fi
  for M in ${MODELS:-hrm1b mimir1 mimir}; do
    case $M in
      hrm1b)  HF_MODEL=sapientinc/HRM-Text-1B;                 HF_DIR=$H/hf/HRM-Text-1B ;;
      mimir1) HF_MODEL=danish-foundation-models/DFM-Mimir;      HF_DIR=$H/hf/DFM-Mimir ;;
      mimir)  HF_MODEL=danish-foundation-models/DFM-Mimir-v1.5; HF_DIR=$H/hf/DFM-Mimir-v1.5 ;;
    esac
    RUN=${D}_${M}_s0
    echo "=== $(date '+%F %T') $RUN"
    ROOT=$H/${D}_$M HF_MODEL=$HF_MODEL HF_DIR=$HF_DIR CONFIG=cfg_${D}_$M GROUP=$D-$M TAG=$M-finetune \
      RUN=$RUN OVERRIDES="seed=0" LOG_SPLITS=test EVAL_DIR=data/graphs/$D/hrm-text-grl/standard \
      EVAL_MAX_TOKENS=$EVAL_MAX_TOKENS EVAL_BATCH=$EVAL_BATCH \
      bash scripts/graphs/run_hf_finetune.sh || echo "FAILED: $RUN"
  done
done
echo "=== $(date '+%F %T') done"
