#!/usr/bin/env bash
# RL arms of the Mimir v1.5 graph benchmark (wiki: mimir-graph-benchmark), W&B project Mimir-Graph-RLVR:
#   E3b  GRPO from the base model (R1-Zero style), train mixture (data/graphs/bench/verl)
#   E3   GRPO from E2 (merged SFT-with-traces model), same data and budget
#   E4   answer-only GRPO on CLUTRR from E2 (data/graphs/bench/verl_clutrr), and its control: LoRA SFT on CLUTRR
#        answers from E2 (graph_inject, 1 epoch)
# GRPO: 16 prompts x 8 rollouts, responses <= 2,048 tokens, LoRA r32 merged at sync, DAPO filter_groups (drops
# all-0 / all-1 groups: the "pass rate strictly between 0 and 1" rule, applied online), val every 25 steps.
# Each arm stores its rollouts (<arm>/rollouts/<step>.jsonl: the trained groups) and val generations
# (<arm>/val_generations/), and checkpoints every 25 steps. A relaunch resumes from the last checkpoint into the same
# W&B run (id read from train.log, which is appended to) and skips the resume step's val (already logged).
# After each arm: merge its last checkpoint into an HF dir (graph_bench/rl/merge_verl_lora.py) and evaluate it like
# E2's _t3k reports (named + anon at 3,072 tokens, val pass@16 at 2,048; E4 and its control: CLUTRR test pass@16)
# into $ROOT/<e3|e4|e4sft|e3b>/mimir15/, which log_wandb uploads.
# Order: E3, E4 (GRPO, then the SFT control), E3b; all start after E2's merged dir exists (graph_bench/run_sft.sh).
# GPU lock per run; a run with a done marker (or an eval with its report) is skipped.
# WAIT_ALSO="file ...": also wait for these files before the first GPU step. Writes $ROOT/RL_QUEUE_DONE at the end.
# Do not edit while it runs (run a copy).
#   setsid nohup bash graph_bench/rl/run_rl_arms.sh > $ROOT/rl_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
BASE=$HF/DFM-Mimir-v1.5-hfview
E2=$HF/mimir15-sft-cot-ep3
B=$PWD/data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
STEPS=${STEPS:-200}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1

NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
VAL="$B/graphqa/val.jsonl $B/metaqa/val.jsonl $B/kqapro/val.jsonl $B/webqsp/val.jsonl $B/clrs/val.jsonl"
GREEDY="--limit 1000 --n 1 --temperature 0 --pass-k 1"
PASS16="--limit 200 --n 16 --temperature 1.0 --pass-k 1 4 16"

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }

grpo() {  # name model train val steps
  local name=$1 dir=$ROOT/$1
  [ -f $dir/DONE ] && { echo "=== skip $name"; return; }
  mkdir -p $dir
  local extra="trainer.default_local_dir=$dir/ckpt trainer.resume_mode=auto" run_id=""
  if [ -f $dir/ckpt/latest_checkpointed_iteration.txt ]; then
    extra="$extra trainer.val_before_train=False"  # resuming: that step's val is already logged
    run_id=$(grep -oE "wandb.ai/[^ ]*/runs/[a-z0-9]+" $dir/train.log 2>/dev/null | tail -1 | sed 's|.*/runs/||')
    echo "=== $name: resuming from step $(cat $dir/ckpt/latest_checkpointed_iteration.txt), W&B run ${run_id:-new}"
  fi
  lock grpo $name
  ( export MODEL_PATH=$2 TRAIN_FILE=$3 TEST_FILE=$4 TOTAL_TRAINING_STEPS=$5 EXPERIMENT_NAME=$name
    export GRAPH_RL_PREFILL=none TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 PPO_MINI_BATCH=8 SAVE_FREQ=${SAVE_FREQ:-25} TEST_FREQ=25
    export MAX_RESPONSE_LENGTH=2048 PPO_MICRO_BATCH_PER_GPU=2 LOGPROB_MICRO_BATCH_PER_GPU=2  # traces reach 3.2k; prompt 2,048 + 2,048 = context
    export ROLLOUT_DATA_DIR=$dir/rollouts VAL_DATA_DIR=$dir/val_generations
    [ -n "$run_id" ] && export WANDB_RUN_ID=$run_id WANDB_RESUME=allow
    export EXTRA_VERL_ARGS="$extra"
    bash scripts/graphs/mem_guard.sh 90 -- bash graph_bench/rl/run_grpo_graph.sh >> $dir/train.log 2>&1 < /dev/null )
  local status=$?
  [ $status = 0 ] && touch $dir/DONE
  unlock $status
}

evaluate() {  # out model tokens extra files...  (Mimir: instruct, thinking off, 4,096 context, as run_sft.sh)
  local out=$1 model=$2 tokens=$3 extra=$4; shift 4
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style instruct \
    --enable-thinking false --max-tokens $tokens --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
    $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  unlock $?
}

evaluate_model() {  # out_dir model pass16_name pass16_files...  (the E2 _t3k / _t2k set)
  local d=$1 model=$2 p16=$3; shift 3
  evaluate $d/named_t3k.json $model 3072 "$GREEDY" $NAMED
  evaluate $d/anon_t3k.json $model 3072 "$GREEDY --anonymize" $ANON
  evaluate $d/${p16}_pass16_t2k.json $model 2048 "$PASS16" "$@"
}

grpo_eval() {  # name eval_arm base_model pass16_name pass16_files...: merge the last checkpoint of a finished arm, then evaluate
  local name=$1 arm=$2 base=$3 dir=$ROOT/$1; shift 3
  [ -f $dir/DONE ] || { echo "=== $name not done: no eval"; return; }
  local step; step=$(cat $dir/ckpt/latest_checkpointed_iteration.txt)
  local merged=$HF/mimir15-$name-step$step
  if [ ! -f $merged/MERGED_FROM ]; then
    echo "=== $(date '+%F %T') merge $name step $step"
    OMP_NUM_THREADS=8 bash scripts/graphs/mem_guard.sh 30 -- $PY -m graph_bench.rl.merge_verl_lora \
      --ckpt $dir/ckpt/global_step_$step/actor --base $base --out $merged > $dir/merge.log 2>&1 < /dev/null
  fi
  [ -f $merged/MERGED_FROM ] || { echo "=== $name: merge failed (see $dir/merge.log)"; return; }
  evaluate_model $ROOT/$arm/mimir15 $merged "$@"
}

echo "=== $(date '+%F %T') waiting for $E2 ${WAIT_ALSO:-}"
wait_for() { until [ -f "$1" ]; do sleep 300; done; }
wait_for $E2/MERGED_FROM
for f in ${WAIT_ALSO:-}; do wait_for $f; done
grpo e3_grpo_e2 $E2 $B/verl/train.parquet $B/verl/val.parquet $STEPS
grpo_eval e3_grpo_e2 e3 $E2 val $VAL
evaluate $ROOT/e2/mimir15/clutrr_pass16_t2k.json $E2 2048 "$PASS16" $B/clutrr/test.jsonl  # E2 reference for E4
grpo e4_grpo_clutrr $E2 $B/verl_clutrr/train.parquet $B/verl_clutrr/val.parquet 100
grpo_eval e4_grpo_clutrr e4 $E2 clutrr $B/clutrr/test.jsonl

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
CTRL=$HF/mimir15-e4-sft-clutrr-ep1
if [ -f $run/trainable_ep1.pt ] && [ ! -f $CTRL/MERGED_FROM ]; then
  PATH="$PWD/.venv/bin:$PATH" bash scripts/graphs/mem_guard.sh 24 -- python -m graph_bench.merge_lora_hf \
    --base $E2 --lora $run/trainable_ep1.pt --config $run/config.yaml --out $CTRL > $ROOT/sft_clutrr/merge.log 2>&1 < /dev/null
fi
[ -f $CTRL/MERGED_FROM ] && evaluate_model $ROOT/e4sft/mimir15 $CTRL clutrr $B/clutrr/test.jsonl

grpo e3b_grpo_base $BASE $B/verl/train.parquet $B/verl/val.parquet $STEPS
grpo_eval e3b_grpo_base e3b $BASE val $VAL
touch $ROOT/RL_QUEUE_DONE  # also on failed arms: the Qwen3.5-4B baseline SFT waits for it
echo "=== $(date '+%F %T') RL queue done"
