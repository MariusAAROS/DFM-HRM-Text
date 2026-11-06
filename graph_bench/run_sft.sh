#!/usr/bin/env bash
# E1 / E2 LoRA SFT of Mimir v1.5 on the graph_bench train mixture (wiki: mimir-graph-benchmark), then vLLM eval.
#   0. merge check: a GraphQA LoRA of known native score (0.842 test), merged and run through vLLM
#   1. train E1 (answer line) and E2 (traces + answer line): graph_inject, LoRA r16, 3 epochs, train only
#   2. merge the last epoch into an HF dir (graph_bench/merge_lora_hf.py)
#   3. evaluate each merged model on the E0 files (named + anonymised, same flags as run_e0.sh)
#   4. E2 pass@16 (T=1) on 200-row subsets: val (RL headroom) and the held-out / OOD files (H3)
# GPU lock per step; a finished step (its output exists) is skipped. Do not edit while it runs.
#   setsid nohup bash graph_bench/run_sft.sh > $ROOT/sft_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
PY=/home/ucloud/miniforge3/bin/python
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
BASE=$HF/DFM-Mimir-v1.5-hfview
B=data/graphs/bench
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$PWD/.venv/bin:$PATH"

NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
VAL="$B/graphqa/val.jsonl $B/metaqa/val.jsonl $B/kqapro/val.jsonl $B/webqsp/val.jsonl $B/clrs/val.jsonl"
HELDOUT="$B/cwq/test.jsonl $B/clutrr/test.jsonl $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl $B/metaqa/test.jsonl"

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }

evaluate() {  # out model style tokens extra files...
  local out=$1 model=$2 style=$3 tokens=$4 extra=$5; shift 5
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style $style \
    --enable-thinking false --max-tokens $tokens --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
    $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  unlock $?
}

# 0. merge check
evaluate $ROOT/mergecheck/mimir15/graphqa_sft.json $HF/mergecheck-mimir15-graphqa-text sft 96 "--n 1 --temperature 0 --pass-k 1" \
  data/graphs/bench/parity/graphqa.jsonl

for arm in answer cot; do
  run=$ROOT/sft_$arm/mimir/text_s0
  # 1. train
  if [ ! -f $run/trainable_ep3.pt ]; then
    mkdir -p $ROOT/sft_$arm
    lock train sft_$arm
    bash scripts/graphs/mem_guard.sh 48 -- python -m graph_inject.train dataset=bench_sft_$arm model=mimir method=text \
      > $ROOT/sft_$arm/train.log 2>&1 < /dev/null
    unlock $?
  fi
  [ -f $run/trainable_ep3.pt ] || { echo "=== sft_$arm: no trainable_ep3.pt, stop"; exit 1; }
  # 2. merge
  merged=$HF/mimir15-sft-$arm-ep3
  [ -f $merged/MERGED_FROM ] || bash scripts/graphs/mem_guard.sh 24 -- python -m graph_bench.merge_lora_hf \
    --base $BASE --lora $run/trainable_ep3.pt --config $run/config.yaml --out $merged
  # 3. eval, as E0
  e=$([ $arm = answer ] && echo e1 || echo e2)
  evaluate $ROOT/$e/mimir15/named.json $merged instruct 1024 "--limit 1000 --n 1 --temperature 0 --pass-k 1" $NAMED
  evaluate $ROOT/$e/mimir15/anon.json $merged instruct 1024 "--limit 1000 --n 1 --temperature 0 --pass-k 1 --anonymize" $ANON
done

# 4. E2 pass@16
merged=$HF/mimir15-sft-cot-ep3
evaluate $ROOT/e2/mimir15/val_pass16.json $merged instruct 1024 "--limit 200 --n 16 --temperature 1.0 --pass-k 1 4 16" $VAL
evaluate $ROOT/e2/mimir15/heldout_pass16.json $merged instruct 1024 "--limit 200 --n 16 --temperature 1.0 --pass-k 1 4 16" $HELDOUT
echo "=== $(date '+%F %T') SFT queue done"
