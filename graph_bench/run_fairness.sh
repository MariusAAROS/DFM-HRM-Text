#!/usr/bin/env bash
# Fairness runs for the Mimir-vs-baselines claim (wiki: mimir-graph-benchmark-results; plan approved 2026-11-23). The
# benchmark plan (wiki: mimir-graph-benchmark) asked for a small LR sweep per external model and 3 seeds; the baselines
# got neither. This queue, for the best E1 baseline (Qwen3.5-4B) and Mimir E1:
#   1. Qwen3.5-4B E1 (lr 2e-4, existing) on val: greedy, 200 rows per val file, 1,024 tokens
#   2. Mimir E1, seeds 1 and 2 (graph_inject seed = data order + LoRA init; E1's recipe otherwise), merged, then
#      named + anon test as E1 (1,024 tokens, 4,096 context) -> e1/mimir15_s<seed>/
#   3. Qwen3.5-4B E1 at lr 5e-5, 1e-4, 4e-4 (sft_hf, E1's recipe otherwise), merged, each on val as step 1
#   4. the LR, chosen on val only (rule fixed 2026-11-23, before any sweep result): highest macro val accuracy, a tie
#      goes to 2e-4 -> sft_baselines/qwen35_4b/chosen_lr.txt; if not 2e-4, its seed-0 model on test -> e1/qwen35_4b_lr<lr>/
#   5. Qwen3.5-4B E1 at the chosen LR, seeds 1 and 2, on test -> e1/qwen35_4b[_lr<lr>]_s<seed>/
# Every test eval: named + anonymised files, seed-0 1,000-row subsets, greedy, as the existing E1 reports.
# GPU lock per step; a finished step (its output exists) is skipped. Do not edit while it runs (run a copy):
#   cp graph_bench/run_fairness.sh $ROOT/run_fairness.running.sh
#   setsid nohup bash $ROOT/run_fairness.running.sh > $ROOT/fairness_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
OUT=$ROOT/sft_baselines
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
BASE=$HF/DFM-Mimir-v1.5-hfview
B=$PWD/data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
VENV="$PWD/.venv/bin:$PATH"
SEEDS=${SEEDS:-"1 2"}
SWEEP=${SWEEP:-"5e-5 1e-4 4e-4"}
DEFAULT_LR=2e-4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 WANDB_BASE_URL=https://api.forge.coreweave.com
NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
VAL="$B/graphqa/val.jsonl $B/metaqa/val.jsonl $B/kqapro/val.jsonl $B/webqsp/val.jsonl $B/clrs/val.jsonl"
GREEDY="--n 1 --temperature 0 --pass-k 1"

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }

evaluate() {  # out model maxlen extra files...  (E1 flags: instruct, thinking off, 1,024 answer tokens), then W&B
  local out=$1 model=$2 maxlen=$3 extra=$4; shift 4
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style instruct \
    --enable-thinking false --max-tokens 1024 --max-model-len $maxlen --gpu-memory-utilization 0.85 --enforce-eager \
    $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  unlock $?
  [ -f "$out" ] && .venv/bin/python -m graph_bench.eval.log_wandb $out >> $ROOT/fairness_wandb.log 2>&1 < /dev/null
}
test_evals() {  # dir model maxlen
  evaluate $1/named.json $2 $3 "--limit 1000 $GREEDY" $NAMED
  evaluate $1/anon.json $2 $3 "--limit 1000 $GREEDY --anonymize" $ANON
}

qwen_sft() {  # dir lr seed: train + merge the last epoch
  local dir=$1 lr=$2 seed=$3
  if [ ! -f $dir/merged_ep3/MERGED_FROM ]; then
    mkdir -p $dir
    lock sft qwen35_4b answer lr $lr seed $seed
    bash scripts/graphs/mem_guard.sh 64 -- $PY -m graph_bench.sft_hf --model Qwen/Qwen3.5-4B \
      --train $B/sft/train_answer.jsonl --out $dir --lr $lr --seed $seed --merge \
      --run-name sft_answer/qwen35_4b_lr${lr}_s$seed > $dir/train.log 2>&1 < /dev/null
    unlock $?
  fi
  [ -f $dir/merged_ep3/MERGED_FROM ] || { echo "=== $dir: no merged model (see $dir/train.log)"; return 1; }
}

mimir_sft() {  # seed: E1's graph_inject recipe with another seed, merged
  local seed=$1 run=$ROOT/sft_answer/mimir/text_s$1 merged=$HF/mimir15-sft-answer-s$1-ep3
  if [ ! -f $run/trainable_ep3.pt ]; then
    lock train mimir E1 seed $seed
    PATH="$VENV" bash scripts/graphs/mem_guard.sh 48 -- python -m graph_inject.train dataset=bench_sft_answer \
      model=mimir method=text seed=$seed > $ROOT/sft_answer/train_s$seed.log 2>&1 < /dev/null
    unlock $?
  fi
  [ -f $run/trainable_ep3.pt ] || { echo "=== mimir seed $seed: no trainable_ep3.pt"; return 1; }
  [ -f $merged/MERGED_FROM ] || PATH="$VENV" bash scripts/graphs/mem_guard.sh 24 -- python -m graph_bench.merge_lora_hf \
    --base $BASE --lora $run/trainable_ep3.pt --config $run/config.yaml --out $merged > $ROOT/sft_answer/merge_s$seed.log 2>&1 < /dev/null
  [ -f $merged/MERGED_FROM ] || { echo "=== mimir seed $seed: merge failed"; return 1; }
}

lr_dir() { [ $1 = $DEFAULT_LR ] && echo $OUT/qwen35_4b/answer || echo $OUT/qwen35_4b/answer_lr$1; }
lr_tag() { [ $1 = $DEFAULT_LR ] && echo qwen35_4b || echo qwen35_4b_lr$1; }

# 1. val of the existing Qwen3.5-4B E1
evaluate $ROOT/e1/qwen35_4b/val.json $OUT/qwen35_4b/answer/merged_ep3 8192 "--limit 200 $GREEDY" $VAL

# 2. Mimir E1 seeds
for s in $SEEDS; do
  mimir_sft $s && test_evals $ROOT/e1/mimir15_s$s $HF/mimir15-sft-answer-s$s-ep3 4096
done

# 3. LR sweep
for lr in $SWEEP; do
  qwen_sft $(lr_dir $lr) $lr 0 && \
    evaluate $ROOT/e1/$(lr_tag $lr)/val.json $(lr_dir $lr)/merged_ep3 8192 "--limit 200 $GREEDY" $VAL
done

# 4. the LR, on val only
CHOSEN=$OUT/qwen35_4b/chosen_lr.txt
if [ ! -f $CHOSEN ]; then
  .venv/bin/python - "$ROOT" "$DEFAULT_LR" $SWEEP > $CHOSEN.tmp <<'PYEOF' || { echo "=== LR choice failed: queue stopped"; exit 1; }
import json, sys
root, default, sweep = sys.argv[1], sys.argv[2], sys.argv[3:]
score = {}
for lr in [default, *sweep]:
    tag = "qwen35_4b" if lr == default else f"qwen35_4b_lr{lr}"
    score[lr] = json.load(open(f"{root}/e1/{tag}/val.json"))["summary"]["macro_acc"]
best = max(score.values())
pick = default if score[default] == best else max(score, key=score.get)
print(pick)
print("val macro: " + ", ".join(f"{lr} {v:.4f}" for lr, v in score.items()), file=sys.stderr)
PYEOF
  mv $CHOSEN.tmp $CHOSEN
fi
lr=$(cat $CHOSEN)
echo "=== chosen lr $lr"
[ $lr = $DEFAULT_LR ] || test_evals $ROOT/e1/$(lr_tag $lr) $(lr_dir $lr)/merged_ep3 8192

# 5. Qwen3.5-4B seeds at the chosen LR
for s in $SEEDS; do
  qwen_sft $(lr_dir $lr)_s$s $lr $s && test_evals $ROOT/e1/$(lr_tag $lr)_s$s $(lr_dir $lr)_s$s/merged_ep3 8192
done
echo "=== $(date '+%F %T') fairness queue done"
