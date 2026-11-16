#!/usr/bin/env bash
# E5 of the Mimir v1.5 graph benchmark (wiki: mimir-graph-e5-multidataset-rl), W&B project Mimir-Graph-RLVR:
# from E3 (GRPO after trace SFT), add datasets no arm trained Mimir on (OSM, CLUTRR, CWQ) and measure their test sets,
# the old test sets (forgetting) and the two sets still held out (CLRS OOD, GraphQA large).
#   control  LoRA SFT from E3 on the same questions (answer line for the new sets, traces for the old), 1 epoch
#   GRPO     200 steps, E3's settings, but reward = set-F1 on set answers and group filtering on the reward
# Data: bash graph_bench/rl/build_e5_data.sh. Step 0 (E3 pass@16 on the new val sets, $ROOT/e5_headroom.sh) must have
# written $ROOT/e3/mimir15/e5new_pass16_t2k.json: a new dataset whose share of val groups with a non-constant reward
# is under MIN_MIXED is left out of the GRPO parquet (it stays in the control).
# Order: 2-step GRPO smoke (offline W&B; the queue stops if it fails), control + its evals, GRPO + its evals.
# GPU lock per run; a run with a done marker (or an eval with its report) is skipped. Do not edit while it runs.
#   setsid nohup bash graph_bench/rl/run_e5.sh > $ROOT/e5_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
E3=$HF/mimir15-e3_grpo_e2-step200
B=$PWD/data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
STEPS=${STEPS:-200}
MIN_MIXED=${MIN_MIXED:-0.2}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 WANDB_BASE_URL=https://api.forge.coreweave.com

NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
VAL="$B/graphqa/val.jsonl $B/metaqa/val.jsonl $B/kqapro/val.jsonl $B/webqsp/val.jsonl $B/clrs/val.jsonl
     $B/osm/val.jsonl $B/clutrr/val.jsonl $B/cwq/val.jsonl"
GREEDY="--n 1 --temperature 0 --pass-k 1"
PASS16="--limit 200 --n 16 --temperature 1.0 --pass-k 1 4 16"

lock() { exec 9>$LOCK; flock 9; echo "=== $(date '+%F %T') $* (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"; }
unlock() { echo "=== $(date '+%F %T') exit $1"; exec 9>&-; }

grpo() {  # name model train val steps [extra VeRL args]: E3's settings + F1 reward, filtering on the reward
  local name=$1 dir=$ROOT/$1
  [ -f $dir/DONE ] && { echo "=== skip $name"; return 0; }
  mkdir -p $dir
  local extra="trainer.default_local_dir=$dir/ckpt trainer.resume_mode=auto ${6:-}" run_id=""
  if [ -f $dir/ckpt/latest_checkpointed_iteration.txt ]; then
    extra="$extra trainer.val_before_train=False"  # resuming: that step's val is already logged
    run_id=$(grep -oE "/runs/[a-z0-9]+" $dir/train.log 2>/dev/null | tail -1 | sed 's|.*/runs/||')
    echo "=== $name: resuming from step $(cat $dir/ckpt/latest_checkpointed_iteration.txt), W&B run ${run_id:-new}"
  fi
  lock grpo $name
  ( export MODEL_PATH=$2 TRAIN_FILE=$3 TEST_FILE=$4 TOTAL_TRAINING_STEPS=$5 EXPERIMENT_NAME=$name
    export GRAPH_REWARD=f1 FILTER_METRIC=score
    export GRAPH_RL_PREFILL=none TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 PPO_MINI_BATCH=8 SAVE_FREQ=${SAVE_FREQ:-25} TEST_FREQ=${TEST_FREQ:-25}
    export MAX_RESPONSE_LENGTH=2048 PPO_MICRO_BATCH_PER_GPU=2 LOGPROB_MICRO_BATCH_PER_GPU=2
    export ROLLOUT_DATA_DIR=$dir/rollouts VAL_DATA_DIR=$dir/val_generations
    [ -n "$run_id" ] && export WANDB_RUN_ID=$run_id WANDB_RESUME=allow
    export EXTRA_VERL_ARGS="$extra"
    # Cap on the summed RSS of the group. It counts shared pages once per process (Ray object store, shared libs):
    # 91 GB at the step-125 checkpoint save of E5 while the container used 59 GB, and the 90 GB cap killed the run
    # (2026-11-15). The container cap (110 GB unreclaimable) stays the real protection against the 128 GB limit.
    bash scripts/graphs/mem_guard.sh ${GRPO_RSS_CAP_GB:-105} -- bash graph_bench/rl/run_grpo_graph.sh >> $dir/train.log 2>&1 < /dev/null )
  local status=$?
  [ $status = 0 ] && touch $dir/DONE
  unlock $status
  return $status
}

evaluate() {  # out model tokens extra files...  (Mimir: instruct, thinking off, 4,096 context), then W&B
  local out=$1 model=$2 tokens=$3 extra=$4; shift 4
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  mkdir -p $(dirname $out)
  lock eval $out
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $model --style instruct \
    --enable-thinking false --max-tokens $tokens --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
    $extra --out $out "$@" > ${out%.json}.log 2>&1 < /dev/null
  unlock $?
  [ -f "$out" ] && .venv/bin/python -m graph_bench.eval.log_wandb $out >> $ROOT/e5_wandb.log 2>&1 < /dev/null
}

evaluate_model() {  # out_dir model: E3's report set + OSM test (all 2,000 rows) + pass@16 on the 8 val sets
  local d=$1 model=$2
  evaluate $d/named_t3k.json $model 3072 "--limit 1000 $GREEDY" $NAMED
  evaluate $d/osm_t3k.json $model 3072 "$GREEDY" $B/osm/test.jsonl
  evaluate $d/anon_t3k.json $model 3072 "--limit 1000 $GREEDY --anonymize" $ANON
  evaluate $d/val_pass16_t2k.json $model 2048 "$PASS16" $VAL
}

# --- GRPO parquet: drop the new datasets without enough mixed-reward groups (step 0) ---
HEAD=$ROOT/e3/mimir15/e5new_pass16_t2k.samples.jsonl
TRAIN=$B/verl_e5/train_grpo.parquet
echo "=== $(date '+%F %T') waiting for step 0 ($HEAD; $ROOT/e5_headroom.sh)"
until [ -f ${HEAD%.samples.jsonl}.json ]; do sleep 120; done
if [ ! -f $TRAIN ]; then
  $PY - <<PYEOF || exit 1
import json, collections
import pyarrow.parquet as pq
g = collections.defaultdict(list)
for line in open("$HEAD"):
    r = json.loads(line)
    g[r["dataset"]].append(len(set(r["f1"][:8])) > 1)  # GRAPH_REWARD=f1: set-F1 on set answers, = acc elsewhere
keep = {d for d, v in g.items() if sum(v) / len(v) >= $MIN_MIXED}
for d, v in sorted(g.items()):
    print(f"=== headroom {d}: {sum(v) / len(v):.3f} of {len(v)} val groups have a non-constant reward (8 samples) "
          f"-> {'in' if d in keep else 'OUT of'} the GRPO mixture")
t = pq.read_table("$B/verl_e5/train.parquet")
mask = [s not in g or s in keep for s in t.column("data_source").to_pylist()]
t = t.filter(mask)
pq.write_table(t, "$TRAIN")
print(f"=== $TRAIN: {t.num_rows} rows")
PYEOF
fi

# --- smoke: 2 GRPO steps, offline W&B, own root; stop the queue on failure ---
SMOKE=$ROOT/e5_smoke
if [ ! -f $SMOKE/e5_grpo_smoke/DONE ]; then
  ( export ROOT=$SMOKE WANDB_MODE=offline SAVE_FREQ=1000 TEST_FREQ=1000
    grpo e5_grpo_smoke $E3 $TRAIN $B/verl_e5/val.parquet 2 "trainer.val_before_train=False" ) \
    || { echo "=== smoke failed (see $SMOKE/e5_grpo_smoke/train.log): queue stopped"; exit 1; }
  $PY - <<PYEOF
import json, glob, collections
c = collections.Counter()
for p in glob.glob("$SMOKE/e5_grpo_smoke/rollouts/*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        c["rows"] += 1; c["fractional reward"] += 0 < r["score"] < 1
print("=== smoke rollouts:", dict(c))
PYEOF
fi

# --- control: LoRA SFT from E3 (native import of the merged model) ---
NATIVE=$HF/mimir15-e3-native
if [ ! -f $NATIVE/unsharded_step_0.pt ]; then
  PATH="$PWD/.venv/bin:$PATH" bash scripts/graphs/mem_guard.sh 32 -- python -m conversion.convert_from_hf \
    --hf_model $E3 --out_dir $NATIVE --config-name cfg_graphqa_mimir > $ROOT/e5_native_import.log 2>&1 < /dev/null
fi
run=$ROOT/sft_e5/mimir15-e3/text_s0
if [ ! -f $run/trainable_ep1.pt ]; then
  mkdir -p $ROOT/sft_e5
  lock sft e5_sft
  PATH="$PWD/.venv/bin:$PATH" bash scripts/graphs/mem_guard.sh 48 -- python -m graph_inject.train \
    dataset=bench_e5_answer model=mimir method=text model.name=mimir15-e3 model.hf_dir=$NATIVE \
    > $ROOT/sft_e5/train.log 2>&1 < /dev/null
  unlock $?
fi
CTRL=$HF/mimir15-e5-sft-ep1
if [ -f $run/trainable_ep1.pt ] && [ ! -f $CTRL/MERGED_FROM ]; then
  PATH="$PWD/.venv/bin:$PATH" bash scripts/graphs/mem_guard.sh 24 -- python -m graph_bench.merge_lora_hf \
    --base $E3 --lora $run/trainable_ep1.pt --config $run/config.yaml --out $CTRL > $ROOT/sft_e5/merge.log 2>&1 < /dev/null
fi
[ -f $CTRL/MERGED_FROM ] && evaluate_model $ROOT/e5sft/mimir15 $CTRL || echo "=== control: no merged model"

# --- GRPO ---
name=e5_grpo_multi
grpo $name $E3 $TRAIN $B/verl_e5/val.parquet $STEPS
if [ -f $ROOT/$name/DONE ]; then
  step=$(cat $ROOT/$name/ckpt/latest_checkpointed_iteration.txt)
  merged=$HF/mimir15-$name-step$step
  if [ ! -f $merged/MERGED_FROM ]; then
    echo "=== $(date '+%F %T') merge $name step $step"
    OMP_NUM_THREADS=8 bash scripts/graphs/mem_guard.sh 30 -- $PY -m graph_bench.rl.merge_verl_lora \
      --ckpt $ROOT/$name/ckpt/global_step_$step/actor --base $E3 --out $merged > $ROOT/$name/merge.log 2>&1 < /dev/null
  fi
  [ -f $merged/MERGED_FROM ] && evaluate_model $ROOT/e5/mimir15 $merged || echo "=== $name: merge failed"
fi

.venv/bin/python -m graph_bench.eval.osm_breakdown $ROOT/e*/*/osm_t3k.samples.jsonl --json $ROOT/osm_breakdown.json
echo "=== $(date '+%F %T') E5 queue done"
