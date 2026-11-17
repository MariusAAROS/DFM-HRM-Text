#!/usr/bin/env bash
# E3e1: GRPO from E1 (answer-only SFT) with E3's data and settings (wiki: mimir-graph-trace-sft-analysis), W&B project
# Mimir-Graph-RLVR. Decided 2026-11-16: E1's val pass@16 - pass@1 is 0.101 macro (E2: 0.124), above the 0.03 rule, so
# GRPO from E1 has room. With E3c (GRPO from the repaired trace model) it gives the headline comparison: RL on traces
# against RL on direct answers, same data, same RL budget.
#   1. wait until the relaunch queue (graph_bench/run_sft_v2.sh) has exited
#   2. GRPO 200 steps from hf/mimir15-sft-answer-ep3 on verl/train.parquet (acc reward, group filter on acc, no
#      reference-KL pass, 16 x 8 rollouts, 2,048-token responses), checkpoints every 25 steps
#   3. merge, then the same evals as E3c into $ROOT/e3e1/mimir15/: named_t3k, anon_t3k, val_pass16_t2k, osm_t3k,
#      holdback_t3k (3,072 tokens: RL may lengthen the answers)
#   4. rewrite the breakdowns with E3e1 added ($ROOT/trace_gap_v2.txt, trace_gap_v2_holdback.txt)
# GPU lock per step; a finished step is skipped; GRPO resumes from its last checkpoint into the same W&B run.
# Do not edit while it runs (run a copy):
#   cp graph_bench/rl/run_e3_from_e1.sh $ROOT/run_e3_from_e1.running.sh
#   setsid nohup bash $ROOT/run_e3_from_e1.running.sh > $ROOT/e3e1_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
E1=$HF/mimir15-sft-answer-ep3
B=$PWD/data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
STEPS=${STEPS:-200}
WAIT_PATTERN=${WAIT_PATTERN:-run_sft_v2.running}
NAME=e3_grpo_e1
# $PY's wandb (0.30.0) defaults to api.wandb.ai, where the login fails (CLAUDE.md)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 WANDB_BASE_URL=https://api.forge.coreweave.com

NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
VAL="$B/graphqa/val.jsonl $B/metaqa/val.jsonl $B/kqapro/val.jsonl $B/webqsp/val.jsonl $B/clrs/val.jsonl"
HOLD="$B/cwq/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl $B/graphqa/test_fresh.jsonl $B/graphqa/test_large.jsonl"
GREEDY="--n 1 --temperature 0 --pass-k 1"
HOLDBACK="--limit 1000 --complement --n 1 --temperature 0 --pass-k 1"
PASS16="--limit 200 --n 16 --temperature 1.0 --pass-k 1 4 16"

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }

evaluate() {  # out model tokens extra files...  (as run_sft_v2.sh), then W&B
  local out=$1 model=$2 tokens=$3 extra=$4; shift 4
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style instruct \
    --enable-thinking false --max-tokens $tokens --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
    $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  local rc=$?
  unlock $rc
  [ -f "$out" ] && .venv/bin/python -m graph_bench.eval.log_wandb $out >> $ROOT/e3e1_wandb.log 2>&1 < /dev/null
}

grpo() {  # name model train val steps: E3's settings, as run_sft_v2.sh
  local name=$1 dir=$ROOT/$1
  [ -f $dir/DONE ] && { echo "=== skip $name"; return 0; }
  mkdir -p $dir
  local extra="trainer.default_local_dir=$dir/ckpt trainer.resume_mode=auto" run_id=""
  if [ -f $dir/ckpt/latest_checkpointed_iteration.txt ]; then
    extra="$extra trainer.val_before_train=False"  # resuming: that step's val is already logged
    run_id=$(grep -oE "/runs/[a-z0-9]+" $dir/train.log 2>/dev/null | tail -1 | sed 's|.*/runs/||')
    echo "=== $name: resuming from step $(cat $dir/ckpt/latest_checkpointed_iteration.txt), W&B run ${run_id:-new}"
  fi
  lock grpo $name
  ( export MODEL_PATH=$2 TRAIN_FILE=$3 TEST_FILE=$4 TOTAL_TRAINING_STEPS=$5 EXPERIMENT_NAME=$name
    export GRAPH_REWARD=acc FILTER_METRIC=acc REF_KL=False  # as E3 (run_grpo_graph.sh turns REF_KL on for other names)
    export GRAPH_RL_PREFILL=none TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 PPO_MINI_BATCH=8 SAVE_FREQ=25 TEST_FREQ=25
    export MAX_RESPONSE_LENGTH=2048 PPO_MICRO_BATCH_PER_GPU=2 LOGPROB_MICRO_BATCH_PER_GPU=2
    export ROLLOUT_DATA_DIR=$dir/rollouts VAL_DATA_DIR=$dir/val_generations
    [ -n "$run_id" ] && export WANDB_RUN_ID=$run_id WANDB_RESUME=allow
    export EXTRA_VERL_ARGS="$extra"
    bash scripts/graphs/mem_guard.sh ${GRPO_RSS_CAP_GB:-105} -- bash graph_bench/rl/run_grpo_graph.sh >> $dir/train.log 2>&1 < /dev/null )
  local status=$?
  [ $status = 0 ] && touch $dir/DONE
  unlock $status
  return $status
}

echo "=== $(date '+%F %T') waiting until no process matches '$WAIT_PATTERN'"
while pgrep -f "$WAIT_PATTERN" > /dev/null; do sleep 300; done
[ -f $E1/MERGED_FROM ] || { echo "=== $E1 missing: stop"; exit 1; }

grpo $NAME $E1 $B/verl/train.parquet $B/verl/val.parquet $STEPS
if [ -f $ROOT/$NAME/DONE ]; then
  step=$(cat $ROOT/$NAME/ckpt/latest_checkpointed_iteration.txt)
  merged=$HF/mimir15-$NAME-step$step
  if [ ! -f $merged/MERGED_FROM ]; then
    echo "=== $(date '+%F %T') merge $NAME step $step"
    OMP_NUM_THREADS=8 bash scripts/graphs/mem_guard.sh 30 -- $PY -m graph_bench.rl.merge_verl_lora \
      --ckpt $ROOT/$NAME/ckpt/global_step_$step/actor --base $E1 --out $merged > $ROOT/$NAME/merge.log 2>&1 < /dev/null
  fi
  if [ -f $merged/MERGED_FROM ]; then
    d=$ROOT/e3e1/mimir15
    evaluate $d/named_t3k.json $merged 3072 "--limit 1000 $GREEDY" $NAMED
    evaluate $d/anon_t3k.json $merged 3072 "--limit 1000 $GREEDY --anonymize" $ANON
    evaluate $d/val_pass16_t2k.json $merged 2048 "$PASS16" $VAL
    evaluate $d/osm_t3k.json $merged 3072 "$GREEDY" $B/osm/test.jsonl
    evaluate $d/holdback_t3k.json $merged 3072 "$HOLDBACK" $HOLD
  else
    echo "=== $NAME: merge failed (see $ROOT/$NAME/merge.log)"
  fi
else
  echo "=== $NAME not done: no eval (relaunch this queue to resume it)"
fi

# breakdowns with E3e1 added (E1 first: the reference)
breakdown() {  # out answer_cond trace_cond
  local files="$ROOT/e1/mimir15/$2.samples.jsonl" names=E1 a f
  [ -f $files ] || { echo "=== no E1 report for $1"; return; }
  for a in e1p:$2 e2:$3 e3:$3 e2c:$3 e2p:$3 e3c:$3 e3e1:$3; do
    f=$ROOT/${a%%:*}/mimir15/${a##*:}.samples.jsonl
    [ -f $f ] && { files="$files $f"; names="$names,${a%%:*}"; }
  done
  .venv/bin/python -m graph_bench.eval.trace_gap_breakdown --names $names $files > $1 2>&1
}
breakdown $ROOT/trace_gap_v2.txt named named_t3k
breakdown $ROOT/trace_gap_v2_holdback.txt holdback holdback_t3k
echo "=== $(date '+%F %T') E3e1 queue done"
