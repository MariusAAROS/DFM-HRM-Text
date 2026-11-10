#!/usr/bin/env bash
# Zero-shot transfer to a downstream task (wiki: mimir-graph-benchmark): the OSM Paris spatial KGQA test set
# (data/graphs/bench/osm/test.jsonl, 2,000 rows, 500 per depth; graph_bench/sources/osm.py), never trained by any arm.
# Question: is the E3 (GRPO) Mimir usable zero-shot on geospatial reasoning, and does RL help over E2 / the base?
# Same settings as the E2/E3 named_t3k reports: instruct, thinking off, greedy, up to 3,072 generated tokens
# (shortened when the prompt leaves less room in Mimir's 4,096 context), eager vLLM, all 2,000 rows.
# Reports: $ROOT/<arm>/<model>/osm_t3k.json, logged to W&B Mimir-Graph-RLVR (group mimir-bench-<arm>).
# Order: waits for E3's merged model (run_rl_arms.sh), then E3, E2, base Mimir v1.5, then Qwen2.5-7B / OLMo-3-7B
# zero-shot (E0) and after their trace SFT (E2). GPU lock per run; a run whose report exists is skipped.
# Do not edit while it runs (run a copy).
#   WANDB_BASE_URL=https://api.forge.coreweave.com setsid nohup bash graph_bench/eval/run_osm.sh \
#     > $ROOT/osm_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
LOCK=/work/dfm/marius-ortega/dfm-hrm-graphs/locks/gpu0.lock
HF=/work/dfm/marius-ortega/dfm-hrm-graphs/hf
PY=/home/ucloud/miniforge3/bin/python
DATA=data/graphs/bench/osm/test.jsonl
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

evaluate() {  # arm model_name model max_model_len
  local out=$ROOT/$1/$2/osm_t3k.json
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  [ -e "$3" ] || [[ "$3" != /* ]] || { echo "=== missing model $3: skip $out"; return; }
  mkdir -p $(dirname $out)
  exec 9>$LOCK; flock 9
  echo "=== $(date '+%F %T') eval $out (container $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB)"
  bash scripts/graphs/mem_guard.sh 48 -- $PY -m graph_bench.eval.eval_vllm --model $3 --style instruct \
    --enable-thinking false --max-tokens 3072 --max-model-len $4 --gpu-memory-utilization 0.85 --enforce-eager \
    --n 1 --temperature 0 --pass-k 1 --out $out $DATA > ${out%.json}.log 2>&1 < /dev/null
  local rc=$?; echo "=== $(date '+%F %T') exit $rc"
  exec 9>&-
  .venv/bin/python -m graph_bench.eval.log_wandb $out >> $ROOT/osm_wandb.log 2>&1 < /dev/null
}

echo "=== $(date '+%F %T') waiting for E3 to finish ($ROOT/e3_grpo_e2/DONE) and its merged model"
until [ -f $ROOT/e3_grpo_e2/DONE ]; do sleep 300; done
E3=$HF/mimir15-e3_grpo_e2-step$(cat $ROOT/e3_grpo_e2/ckpt/latest_checkpointed_iteration.txt)
until [ -f $E3/MERGED_FROM ]; do sleep 120; done

evaluate e3 mimir15 $E3 4096
evaluate e2 mimir15 $HF/mimir15-sft-cot-ep3 4096
evaluate e0 mimir15 $HF/DFM-Mimir-v1.5-hfview 4096
evaluate e0 qwen25_7b Qwen/Qwen2.5-7B-Instruct 8192
evaluate e0 olmo3_7b allenai/Olmo-3-7B-Instruct 8192
evaluate e2 qwen25_7b $ROOT/sft_baselines/qwen25_7b/cot/merged_ep3 8192
evaluate e2 olmo3_7b $ROOT/sft_baselines/olmo3_7b/cot/merged_ep3 8192

.venv/bin/python -m graph_bench.eval.osm_breakdown $ROOT/e*/*/osm_t3k.samples.jsonl --json $ROOT/osm_breakdown.json
echo "=== $(date '+%F %T') OSM evals done"
