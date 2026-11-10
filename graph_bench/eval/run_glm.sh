#!/usr/bin/env bash
# GLM-5.3 reference rows for the E0-E4 tables (wiki: mimir-graph-benchmark), through the SDU OpenAI-compatible API
# (graph_bench/eval/eval_api.py; GLM_SERVER / GLM_API_KEY in the repo's .env). CPU only, no GPU lock.
# Same test files and seed-0 subsets as E0 (12 named files, <= 1,000 rows each; 4 anonymised KGQA files) plus OSM test.
# GLM's thinking cannot be turned off on this server, so the budget covers hidden reasoning + answer:
#   _t3k   3,072 tokens, reasoning_effort low: the other models' budget (decided 2026-11-09)
#   _t16k  16,384 tokens, default effort: GLM's ceiling, to show how much the 3,072 cap cuts
# Reports: $ROOT/e0/glm53/<cond>.json, logged to W&B Mimir-Graph-RLVR (group mimir-bench-e0). Each run is resumable
# (.partial.jsonl); a finished report is skipped. Do not edit while it runs (run a copy).
#   WANDB_BASE_URL=https://api.forge.coreweave.com setsid nohup bash graph_bench/eval/run_glm.sh \
#     > $ROOT/e0/glm53/queue.log 2>&1 < /dev/null &
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
OUT=$ROOT/e0/glm53
PY=/home/ucloud/miniforge3/bin/python
B=data/graphs/bench
CONC=${CONC:-16}
NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"
mkdir -p $OUT

run() {  # cond tokens effort extra files...
  local cond=$1 tokens=$2 effort=$3 extra=$4; shift 4
  local out=$OUT/$cond.json
  [ -f "$out" ] && { echo "=== skip $cond"; return; }
  echo "=== $(date '+%F %T') $cond"
  local eff=""; [ -n "$effort" ] && eff="--reasoning-effort $effort"
  bash scripts/graphs/mem_guard.sh 16 -- $PY -m graph_bench.eval.eval_api --model zai-org/GLM-5.3 \
    --max-tokens $tokens $eff --concurrency $CONC $extra --out $out "$@" >> $OUT/$cond.log 2>&1 < /dev/null
  local rc=$?; echo "=== $(date '+%F %T') $cond exit $rc"
  [ -f "$out" ] && .venv/bin/python -m graph_bench.eval.log_wandb $out >> $OUT/wandb.log 2>&1 < /dev/null
}

run named_t3k 3072 low "--limit 1000" $NAMED
run anon_t3k 3072 low "--limit 1000 --anonymize" $ANON
run osm_t3k 3072 low "" $B/osm/test.jsonl
run named_t16k 16384 "" "--limit 1000" $NAMED
run anon_t16k 16384 "" "--limit 1000 --anonymize" $ANON
run osm_t16k 16384 "" "" $B/osm/test.jsonl
echo "=== $(date '+%F %T') GLM done"
