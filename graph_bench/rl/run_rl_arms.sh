#!/usr/bin/env bash
# RL arms of the Mimir v1.5 graph benchmark (wiki: mimir-graph-benchmark), W&B project Mimir-Graph-RLVR:
#   E3b  GRPO from the base model (R1-Zero style), train mixture (data/graphs/bench/verl)
#   E3   GRPO from E2 (merged SFT-with-traces model), same data and budget
#   E4   answer-only GRPO on CLUTRR from E2 (data/graphs/bench/verl_clutrr), and its control: LoRA SFT on CLUTRR
#        answers from E2 (graph_inject, 1 epoch)
# GRPO: 16 prompts x 8 rollouts, responses <= 1,024 tokens, LoRA r32 merged at sync, DAPO filter_groups (drops
# all-0 / all-1 groups: the "pass rate strictly between 0 and 1" rule, applied online), val every 25 steps.
# Order: E3, E4 (GRPO, then the SFT control), E3b; all start after E2's merged dir exists (graph_bench/run_sft.sh), so
# RL never takes the GPU ahead of the SFT on the critical path. GPU lock per run; a run with a done marker is skipped.
# Do not edit while it runs.
#   setsid nohup bash graph_bench/rl/run_rl_arms.sh > $ROOT/rl_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
BASE=$HF/DFM-Mimir-v1.5-hfview
E2=$HF/mimir15-sft-cot-ep3
B=$PWD/data/graphs/bench
STEPS=${STEPS:-200}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_BASE_URL=https://api.forge.coreweave.com

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }

grpo() {  # name model train val steps
  local name=$1 dir=$ROOT/$1
  [ -f $dir/DONE ] && { echo "=== skip $name"; return; }
  mkdir -p $dir
  lock grpo $name
  ( export MODEL_PATH=$2 TRAIN_FILE=$3 TEST_FILE=$4 TOTAL_TRAINING_STEPS=$5 EXPERIMENT_NAME=$name
    export GRAPH_RL_PREFILL=none TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 PPO_MINI_BATCH=8 SAVE_FREQ=50 TEST_FREQ=25
    export EXTRA_VERL_ARGS="trainer.default_local_dir=$dir/ckpt trainer.resume_mode=auto"
    bash scripts/graphs/mem_guard.sh 90 -- bash graph_bench/rl/run_grpo_graph.sh > $dir/train.log 2>&1 < /dev/null )
  local status=$?
  [ $status = 0 ] && touch $dir/DONE
  unlock $status
}

wait_for() { until [ -f "$1" ]; do sleep 300; done; }

echo "=== $(date '+%F %T') waiting for $E2"
wait_for $E2/MERGED_FROM  
grpo e3_grpo_e2 $E2 $B/verl/train.parquet $B/verl/val.parquet $STEPS
grpo e4_grpo_clutrr $E2 $B/verl_clutrr/train.parquet $B/verl_clutrr/val.parquet 100

# E4 control: LoRA SFT on CLUTRR answers from E2 (native import of the merged model)
NATIVE=$HF/mimir15-sft-cot-ep3-native
if [ ! -f $NATIVE/unsharded_step_0.pt ]; then
  PATH="$PWD/.venv/bin:$PATH" bash scripts/graphs/mem_guard.sh 32 -- python -m conversion.convert_from_hf \
    --hf_model $E2 --out_dir $NATIVE --config-name cfg_graphqa_mimir > $ROOT/e4_native_import.log 2>&1 < /dev/null
fi
run=$ROOT/sft_clutrr/mimir15-e2/text_s0
if [ ! -f $run/trainable_ep1.pt ]; then
  mkdir -p $ROOT/sft_clutrr
  lock sft e4_sft_clutrr
  PATH="$PWD/.venv/bin:$PATH" bash scripts/graphs/mem_guard.sh 48 -- python -m graph_inject.train \
    dataset=bench_clutrr_answer model=mimir method=text model.name=mimir15-e2 model.hf_dir=$NATIVE \
    > $ROOT/sft_clutrr/train.log 2>&1 < /dev/null
  unlock $?
fi
grpo e3b_grpo_base $BASE $B/verl/train.parquet $B/verl/val.parquet $STEPS
echo "=== $(date '+%F %T') RL queue done"
