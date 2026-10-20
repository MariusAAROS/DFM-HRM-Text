#!/bin/bash
# Seed replicates of the HRM-Text-1B GraphQA fine-tune (unclipped and gradient_clip_norm=1.0), one after another.
# Seed S trains on data/graphs/sampled/graphqa_grl_s<S>_e20 (SEED=S bash scripts/graphs/prepare_dataset.sh ...):
# the data order is the only thing a seed changes in a fine-tune. Only the best-epoch test report goes to W&B.
# Finished runs and existing eval reports are skipped, so a rerun resumes at the first unfinished run.
#
# Detached from the shell / SSH session:
#   setsid nohup bash scripts/graphs/run_graphqa_hrm1b_seeds.sh \
#     > /work/dfm/marius-ortega/dfm-hrm-graphs/graphqa_hrm1b/logs/seeds_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
for S in ${SEEDS:-1 2}; do
  for C in "" clip1; do
    RUN=graphqa_hrm1b${C:+_$C}_s$S
    OV="data.path=data/graphs/sampled/graphqa_grl_s${S}_e20 seed=$S${C:+ gradient_clip_norm=1.0}"
    echo "=== $(date '+%F %T') $RUN ($OV)"
    RUN=$RUN OVERRIDES="$OV" LOG_SPLITS=test bash scripts/graphs/run_graphqa_hrm1b.sh || echo "FAILED: $RUN"
  done
done
echo "=== $(date '+%F %T') done"
