#!/usr/bin/env bash
# Relaunch of the trace-SFT and GRPO arms of the Mimir v1.5 graph benchmark (wiki: mimir-graph-trace-sft-analysis),
# W&B project Mimir-Graph-RLVR. Data: bash graph_bench/build_sft_v2.sh (it must have printed OK).
#   0. smoke: 1 epoch on 200 rows of train_cot_p (offline W&B, own root), merge, 50-row eval; the queue stops if it fails
#   1. E1 val pass@16 (existing answer-only model): bounds what GRPO from E1 could gain
#   2. SFT from base Mimir v1.5, E1 / E2's recipe (graph_inject, LoRA r16, 3 epochs, last epoch merged), then vLLM eval:
#        E1p  answer_p  E1 + 523 WebQSP questions without a gold path (answers in the prompt)   named, anon @1,024
#        E2c  cot_c     E2 with merged repeated CLRS states                                     named, anon @3,072, val pass@16
#        E2p  cot_p     E2c + the same 523 questions, answer line only                          as E2c
#   3. E3c's start model: E2p if its macro val pass@1 is >= MARGIN above E2c's, else E2c -> $ROOT/e3c_base.txt
#   4. E3c: GRPO from it, 200 steps, E3's data and settings (acc reward, group filter on acc, no reference-KL pass),
#      merge, eval (named, anon, val pass@16, OSM)
#   5. holdback eval of the existing E1, E2, E3 models
#   6. trace_gap_breakdown over E1, E2, E3 and the new arms -> $ROOT/trace_gap_v2.txt, trace_gap_v2_holdback.txt
# Holdback (every model, reports holdback[_t3k].json): the test rows that the --limit 1000 --seed 0 subset of every
# earlier report leaves out (eval_vllm --complement: CWQ 2,531, WebQSP 628, CLRS test 200, GraphQA fresh / large 100
# each). No model was scored on them before, and the fixes of this relaunch were chosen without them.
# GPU lock per step; a finished step (its output exists) is skipped, GRPO resumes from its last checkpoint into the same
# W&B run. A training step without its checkpoint stops the queue. Starts once no process matches WAIT_PATTERN (E5's
# queue). Do not edit while it runs (run a copy).
#   cp graph_bench/run_sft_v2.sh $ROOT/run_sft_v2.running.sh
#   setsid nohup bash $ROOT/run_sft_v2.running.sh > $ROOT/sft_v2_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
BASE=$HF/DFM-Mimir-v1.5-hfview
E1=$HF/mimir15-sft-answer-ep3
B=$PWD/data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
VENV="$PWD/.venv/bin:$PATH"   # graph_inject / merge only: the vLLM and VeRL steps run in $PY's environment
STEPS=${STEPS:-200}
MARGIN=${MARGIN:-0.01}
WAIT_PATTERN=${WAIT_PATTERN:-run_e5.running}
SMOKE=/work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject_bench/sft_v2_smoke
# $PY's wandb only has credentials for this host (api.wandb.ai answers 401 there, checked 2026-11-16)
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

evaluate() {  # out model tokens extra files...  (Mimir: instruct, thinking off, 4,096 context), then W&B unless NO_WANDB
  local out=$1 model=$2 tokens=$3 extra=$4; shift 4
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style instruct \
    --enable-thinking false --max-tokens $tokens --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
    $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  unlock $?
  [ -f "$out" ] && [ -z "${NO_WANDB:-}" ] && .venv/bin/python -m graph_bench.eval.log_wandb $out >> $ROOT/sft_v2_wandb.log 2>&1 < /dev/null
}

sft() {  # root dataset_config epochs merged_dir [extra graph_inject overrides]: train + merge the last epoch
  local root=$1 cfg=$2 ep=$3 merged=$4 run=$1/mimir/text_s0
  if [ ! -f $run/trainable_ep$ep.pt ]; then
    mkdir -p $root
    lock train $cfg
    PATH="$VENV" bash scripts/graphs/mem_guard.sh 48 -- python -m graph_inject.train dataset=$cfg model=mimir \
      method=text ${5:-} > $root/train.log 2>&1 < /dev/null
    unlock $?
  fi
  [ -f $run/trainable_ep$ep.pt ] || { echo "=== $cfg: no trainable_ep$ep.pt (see $root/train.log)"; return 1; }
  [ -f $merged/MERGED_FROM ] || PATH="$VENV" bash scripts/graphs/mem_guard.sh 24 -- python -m graph_bench.merge_lora_hf \
    --base $BASE --lora $run/trainable_ep$ep.pt --config $run/config.yaml --out $merged > $root/merge.log 2>&1 < /dev/null
  [ -f $merged/MERGED_FROM ] || { echo "=== $cfg: merge failed (see $root/merge.log)"; return 1; }
}

grpo() {  # name model train val steps: E3's settings (run_rl_arms.sh), with E5's memory cap
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
    # summed-RSS cap: 90 GB killed E5 at a checkpoint save while the container used 59 GB (2026-11-15)
    bash scripts/graphs/mem_guard.sh ${GRPO_RSS_CAP_GB:-105} -- bash graph_bench/rl/run_grpo_graph.sh >> $dir/train.log 2>&1 < /dev/null )
  local status=$?
  [ $status = 0 ] && touch $dir/DONE
  unlock $status
  return $status
}

echo "=== $(date '+%F %T') waiting until no process matches '$WAIT_PATTERN'"
while pgrep -f "$WAIT_PATTERN" > /dev/null; do sleep 300; done

# --- 0. smoke: the new configs, merge and eval paths end to end, before any long step ---
if [ ! -f $SMOKE/DONE ]; then
  mkdir -p $SMOKE
  head -200 $B/sft/train_cot_p.jsonl > $SMOKE/train.jsonl
  ( export WANDB_MODE=offline
    sft $SMOKE/run bench_sft_cot_p 1 $SMOKE/merged \
      "data.train=$SMOKE/train.jsonl epochs=1 out_root=$SMOKE/run wandb.mode=offline" ) \
    || { echo "=== smoke failed: queue stopped"; exit 1; }
  NO_WANDB=1 evaluate $SMOKE/eval.json $SMOKE/merged 3072 "--limit 25 $GREEDY" $B/webqsp/test.jsonl $B/clrs/test.jsonl
  [ -f $SMOKE/eval.json ] || { echo "=== smoke eval failed (see $SMOKE/eval.log): queue stopped"; exit 1; }
  rm -rf $SMOKE/merged $SMOKE/run && touch $SMOKE/DONE
  echo "=== $(date '+%F %T') smoke passed"
fi

# --- 1. E1 val pass@16 ---
evaluate $ROOT/e1/mimir15/val_pass16.json $E1 1024 "$PASS16" $VAL

# --- 2. SFT arms ---
for arm in e1p:answer_p e2c:cot_c e2p:cot_p; do
  e=${arm%%:*} name=${arm##*:}
  merged=$HF/mimir15-sft-${name//_/-}-ep3
  sft $ROOT/sft_$name bench_sft_$name 3 $merged || { echo "=== $e: queue stopped"; exit 1; }
  if [ $e = e1p ]; then  # as E1
    evaluate $ROOT/$e/mimir15/named.json $merged 1024 "--limit 1000 $GREEDY" $NAMED
    evaluate $ROOT/$e/mimir15/anon.json $merged 1024 "--limit 1000 $GREEDY --anonymize" $ANON
    evaluate $ROOT/$e/mimir15/holdback.json $merged 1024 "$HOLDBACK" $HOLD
  else                   # as E2's _t3k reports and E3
    evaluate $ROOT/$e/mimir15/named_t3k.json $merged 3072 "--limit 1000 $GREEDY" $NAMED
    evaluate $ROOT/$e/mimir15/anon_t3k.json $merged 3072 "--limit 1000 $GREEDY --anonymize" $ANON
    evaluate $ROOT/$e/mimir15/val_pass16_t2k.json $merged 2048 "$PASS16" $VAL
    evaluate $ROOT/$e/mimir15/holdback_t3k.json $merged 3072 "$HOLDBACK" $HOLD
  fi
done

# --- 3. E3c's start model (rule fixed 2026-11-16, before any result) ---
if [ ! -f $ROOT/e3c_base.txt ]; then
  $PY - <<PYEOF || { echo "=== no start model for E3c: queue stopped"; exit 1; }
import json
DATASETS = ("graphqa", "metaqa", "kqapro", "webqsp", "clrs")
score = {}
for arm in ("e2c", "e2p"):
    s = json.load(open("$ROOT/%s/mimir15/val_pass16_t2k.json" % arm))["summary"]
    score[arm] = sum(s[d]["pass@1"] for d in DATASETS) / len(DATASETS)
pick = "e2p" if score["e2p"] - score["e2c"] >= $MARGIN else "e2c"
open("$ROOT/e3c_base.txt", "w").write(pick + "\n")
print(f"=== E3c starts from {pick}: macro val pass@1 E2c {score['e2c']:.4f}, E2p {score['e2p']:.4f} (margin $MARGIN)")
PYEOF
fi
base=$(cat $ROOT/e3c_base.txt)
START=$HF/mimir15-sft-cot-${base#e2}-ep3
[ -f $START/MERGED_FROM ] || { echo "=== $START missing: queue stopped"; exit 1; }

# --- 4. E3c: GRPO, merge, eval ---
name=e3c_grpo_$base
grpo $name $START $B/verl/train.parquet $B/verl/val.parquet $STEPS
if [ -f $ROOT/$name/DONE ]; then
  step=$(cat $ROOT/$name/ckpt/latest_checkpointed_iteration.txt)
  merged=$HF/mimir15-$name-step$step
  if [ ! -f $merged/MERGED_FROM ]; then
    echo "=== $(date '+%F %T') merge $name step $step"
    OMP_NUM_THREADS=8 bash scripts/graphs/mem_guard.sh 30 -- $PY -m graph_bench.rl.merge_verl_lora \
      --ckpt $ROOT/$name/ckpt/global_step_$step/actor --base $START --out $merged > $ROOT/$name/merge.log 2>&1 < /dev/null
  fi
  if [ -f $merged/MERGED_FROM ]; then
    evaluate $ROOT/e3c/mimir15/named_t3k.json $merged 3072 "--limit 1000 $GREEDY" $NAMED
    evaluate $ROOT/e3c/mimir15/anon_t3k.json $merged 3072 "--limit 1000 $GREEDY --anonymize" $ANON
    evaluate $ROOT/e3c/mimir15/val_pass16_t2k.json $merged 2048 "$PASS16" $VAL
    evaluate $ROOT/e3c/mimir15/osm_t3k.json $merged 3072 "$GREEDY" $B/osm/test.jsonl
    evaluate $ROOT/e3c/mimir15/holdback_t3k.json $merged 3072 "$HOLDBACK" $HOLD
  else
    echo "=== $name: merge failed (see $ROOT/$name/merge.log)"
  fi
else
  echo "=== $name not done: no eval (relaunch this queue to resume it)"
fi

# --- 5. holdback rows for the existing models ---
evaluate $ROOT/e1/mimir15/holdback.json $E1 1024 "$HOLDBACK" $HOLD
evaluate $ROOT/e2/mimir15/holdback_t3k.json $HF/mimir15-sft-cot-ep3 3072 "$HOLDBACK" $HOLD
evaluate $ROOT/e3/mimir15/holdback_t3k.json $HF/mimir15-e3_grpo_e2-step200 3072 "$HOLDBACK" $HOLD

# --- 6. breakdowns: the usual rows, then the holdback rows (E1 first: the reference) ---
breakdown() {  # out answer_cond trace_cond
  local files="$ROOT/e1/mimir15/$2.samples.jsonl" names=E1 a f
  [ -f $files ] || { echo "=== no E1 report for $1"; return; }
  for a in e1p:$2 e2:$3 e3:$3 e2c:$3 e2p:$3 e3c:$3; do
    f=$ROOT/${a%%:*}/mimir15/${a##*:}.samples.jsonl
    [ -f $f ] && { files="$files $f"; names="$names,${a%%:*}"; }
  done
  .venv/bin/python -m graph_bench.eval.trace_gap_breakdown --names $names $files > $1 2>&1
}
breakdown $ROOT/trace_gap_v2.txt named named_t3k
breakdown $ROOT/trace_gap_v2_holdback.txt holdback holdback_t3k
echo "=== $(date '+%F %T') SFT v2 queue done"
