#!/usr/bin/env bash
# E1 / E2 for the baselines (wiki: mimir-graph-benchmark): the Mimir SFT recipe (graph_bench/sft_hf.py: same rows,
# LoRA r16, 3 epochs, 8,192 tokens/step) on Qwen2.5-7B-Instruct, OLMo-3-7B-Instruct and Qwen3.5-4B, then the E0 eval
# (named + anonymised files, same flags as each model's E0 run) of the merged last epoch into e1/<model>, e2/<model>.
# Qwen2.5-7B / OLMo-3-7B ran before GRPO; Qwen3.5-4B (~1k tok/s on the PyTorch linear-attention fallback, ~28 h)
# runs after it (WAIT_ALSO=$ROOT/RL_QUEUE_DONE, decided 2026-11-05). First step: a GRPO resume check
# (a copy of the smoke debug checkpoint, resumed for one real step), then Mimir E2 re-evaluated at the trace-length
# cap. Starts after the Mimir SFT queue's last eval.
# GPU lock per step; finished steps are skipped. Do not edit while it runs.
#   setsid nohup bash graph_bench/run_sft_baselines.sh > $ROOT/sft_baselines/queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
OUT=$ROOT/sft_baselines
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
PY=/home/ucloud/miniforge3/bin/python
B=data/graphs/bench
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODELS=${MODELS:-"qwen25_7b|Qwen/Qwen2.5-7B-Instruct olmo3_7b|allenai/Olmo-3-7B-Instruct qwen35_4b|Qwen/Qwen3.5-4B"}
NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
mkdir -p $OUT

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }
wait_for() { until [ -f "$1" ]; do sleep 300; done; }

evaluate() {  # out model tokens maxlen extra files...  (instruct, thinking off, greedy: the E0 flags)
  local out=$1 model=$2 tokens=$3 maxlen=$4 extra=$5; shift 5
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style instruct \
    --enable-thinking false --max-tokens $tokens --max-model-len $maxlen --gpu-memory-utilization 0.85 --enforce-eager \
    --limit 1000 $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  unlock $?
}
GREEDY="--n 1 --temperature 0 --pass-k 1"

echo "=== $(date '+%F %T') waiting for the Mimir SFT queue (E2 pass@16)"
wait_for $ROOT/e2/mimir15/heldout_pass16.json

# 0. GRPO resume check: one real step after resuming global_step_1 of the smoke debug run (a copy)
RC=$OUT/grpo_resume_check
if [ ! -f $RC/DONE ]; then
  rm -rf $RC; mkdir -p $RC/ckpt
  cp -r /work/dfm/marius-ortega/dfm-hrm-graphs/grpo_smoke/ckpt_debug/global_step_1 $RC/ckpt/
  echo 1 > $RC/ckpt/latest_checkpointed_iteration.txt
  lock grpo_resume_check
  ( export MODEL_PATH=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5-hfview
    export TRAIN_FILE=$PWD/$B/verl_smoke/train.parquet TEST_FILE=$PWD/$B/verl_smoke/val.parquet
    export TOTAL_TRAINING_STEPS=2 EXPERIMENT_NAME=grpo_resume_check GRAPH_RL_PREFILL=none
    export TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH=4 MAX_RESPONSE_LENGTH=128 SAVE_FREQ=1000 TEST_FREQ=1000 FILTER_GROUPS=False
    export EXTRA_VERL_ARGS="trainer.val_before_train=False trainer.logger=[console] trainer.default_local_dir=$RC/ckpt trainer.resume_mode=auto"
    bash scripts/graphs/mem_guard.sh 90 -- bash graph_bench/rl/run_grpo_graph.sh > $RC/train.log 2>&1 < /dev/null )
  status=$?
  # passes if it resumed at step 1 and then trained step 2
  grep -q 'Resuming from .*global_step_1' $RC/train.log && grep -q 'step:2 ' $RC/train.log && [ $status = 0 ] \
    && touch $RC/DONE && echo "=== resume check passed" || echo "=== resume check FAILED (see $RC/train.log)"
  unlock $status
fi

# 1. Mimir E2 at the trace-length cap (E2 traces reach 3.2k tokens; the 1,024 cap cut 20% of CLRS / GraphQA traces):
#    named + anon at 3,072 (eval_vllm clamps to the 4,096 context), val pass@16 at 2,048 = the GRPO response length
E2M=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/mimir15-sft-cot-ep3
VAL="$B/graphqa/val.jsonl $B/metaqa/val.jsonl $B/kqapro/val.jsonl $B/webqsp/val.jsonl $B/clrs/val.jsonl"
evaluate $ROOT/e2/mimir15/named_t3k.json $E2M 3072 4096 "$GREEDY" $NAMED
evaluate $ROOT/e2/mimir15/anon_t3k.json $E2M 3072 4096 "$GREEDY --anonymize" $ANON
evaluate $ROOT/e2/mimir15/val_pass16_t2k.json $E2M 2048 4096 "--limit 200 --n 16 --temperature 1.0 --pass-k 1 4 16" $VAL

# WAIT_ALSO="file ...": wait for these before the baselines (Qwen3.5-4B runs after GRPO: $ROOT/RL_QUEUE_DONE)
for f in ${WAIT_ALSO:-}; do echo "=== $(date '+%F %T') waiting for $f"; wait_for $f; done
# 2. baselines: E1 (answer) evaluated at 1,024 tokens like E0, E2 (traces) at 3,072
for spec in $MODELS; do
  name=${spec%%|*} repo=${spec#*|}
  for arm in answer cot; do
    dir=$OUT/$name/$arm
    if [ ! -f $dir/merged_ep3/MERGED_FROM ]; then
      mkdir -p $dir
      lock sft $name $arm
      bash scripts/graphs/mem_guard.sh 64 -- $PY -m graph_bench.sft_hf --model $repo \
        --train $B/sft/train_$arm.jsonl --out $dir --merge --run-name sft_$arm/$name > $dir/train.log 2>&1 < /dev/null
      unlock $?
    fi
    [ -f $dir/merged_ep3/MERGED_FROM ] || { echo "=== $name $arm: no merged model, next"; continue; }
    e=$([ $arm = answer ] && echo e1 || echo e2)
    tokens=$([ $arm = answer ] && echo 1024 || echo 3072)
    evaluate $ROOT/$e/$name/named.json $dir/merged_ep3 $tokens 8192 "$GREEDY" $NAMED
    evaluate $ROOT/$e/$name/anon.json $dir/merged_ep3 $tokens 8192 "$GREEDY --anonymize" $ANON
  done
done
touch $OUT/DONE
echo "=== $(date '+%F %T') baseline SFT queue done"
