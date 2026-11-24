#!/usr/bin/env bash
# GLM-5.3 at a higher token cap (wiki: mimir-graph-benchmark-results, GLM-5.3 cut-offs). max_completion_tokens covers
# hidden reasoning + answer; a row whose reasoning hits the cap returns no answer and scores 0. At 16k, 2,534 of 9,554
# named rows were cut off that way; at 32k, 1,226 still were (12.9%), so the cap is raised step by step.
#   MODE=pilot   does the cap change GLM's behaviour? 20 rows per dataset that finished at 16k, run again at 16k (serving
#                noise) and at 32k, then glm_cap_pilot compare -> cap_pilot/verdict.json (rule fixed before the run).
#                Verdict 2026-11-23: invisible.
#   MODE=cut     the cap is invisible: reuse the rows that finished in run FROM, call only its cut-off rows at CAP
#                -> named_t<CAP/1024>k, anon_t<CAP/1024>k
#   MODE=full    the cap is visible: every row at CAP
# Same prompt, default effort, same files and seed-0 subsets as run_glm.sh. Resumable; a finished report is skipped.
# CPU only, no GPU lock. Do not edit while it runs (run a copy):
#   cp graph_bench/eval/run_glm_cap.sh $ROOT/e0/glm53/run_glm_cap.running.sh
#   MODE=cut CAP=65536 FROM=t32k setsid nohup bash $ROOT/e0/glm53/run_glm_cap.running.sh > $ROOT/e0/glm53/run_64k_cut.log 2>&1 < /dev/null &
# History: 32k = MODE=cut CAP=32768 FROM=t16k (2026-11-23/24, logs run_32k_*.log).
set -uo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
ROOT=${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench}
OUT=$ROOT/e0/glm53
PILOT=$OUT/cap_pilot
PY=/home/ucloud/miniforge3/bin/python
B=data/graphs/bench
CONC=${CONC:-16}
MODE=${MODE:?set MODE=pilot, cut or full}
CAP=${CAP:-32768}
FROM=${FROM:-t16k}
TAG=t$((CAP / 1024))k
TIMEOUT=${TIMEOUT:-3600}  # per request, s: at 32k a request took ~340 s on average and some came close to 600
export WANDB_BASE_URL=https://api.forge.coreweave.com  # $PY's wandb 0.30.0 defaults to api.wandb.ai (401)
NAMED="$B/graphqa/test.jsonl $B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/clrs/test.jsonl
       $B/graphqa/test_large.jsonl $B/clrs/test_ood.jsonl
       $B/cwq/test.jsonl $B/clutrr/test.jsonl
       $B/graphqa/test_fresh.jsonl $B/metaqa/test_retrieved.jsonl $B/kqapro/test_retrieved.jsonl"
ANON="$B/metaqa/test.jsonl $B/kqapro/test.jsonl $B/webqsp/test.jsonl $B/cwq/test.jsonl"

api() {  # out tokens extra files...
  local out=$1 tokens=$2 extra=$3; shift 3
  [ -f "$out" ] && { echo "=== skip $out"; return; }
  echo "=== $(date '+%F %T') $out"
  bash scripts/graphs/mem_guard.sh 16 -- $PY -m graph_bench.eval.eval_api --model zai-org/GLM-5.3 \
    --max-tokens $tokens --concurrency $CONC --timeout $TIMEOUT $extra --out $out "$@" >> ${out%.json}.log 2>&1 < /dev/null
  echo "=== $(date '+%F %T') $out exit $?"
}

case $MODE in
  pilot)
    [ -d $PILOT/data ] || .venv/bin/python -m graph_bench.eval.glm_cap_pilot prepare --per-dataset 20 --out $PILOT
    files=$(find $PILOT/data -name "*.jsonl" | sort)
    api $PILOT/pilot_t16k.json 16384 "" $files
    api $PILOT/pilot_t32k.json 32768 "" $files
    .venv/bin/python -m graph_bench.eval.glm_cap_pilot compare --out $PILOT
    ;;
  cut|full)
    for cond in named anon; do
      if [ $MODE = cut ] && [ ! -f $OUT/${cond}_$TAG.partial.jsonl ]; then
        .venv/bin/python -m graph_bench.eval.glm_cap_pilot seed --run ${cond}_$FROM --to ${cond}_$TAG || exit 1
      fi
    done
    api $OUT/named_$TAG.json $CAP "--limit 1000" $NAMED
    api $OUT/anon_$TAG.json $CAP "--limit 1000 --anonymize" $ANON
    for cond in named anon; do
      [ -f $OUT/${cond}_$TAG.json ] && .venv/bin/python -m graph_bench.eval.log_wandb $OUT/${cond}_$TAG.json >> $OUT/wandb.log 2>&1 < /dev/null
    done
    ;;
esac
echo "=== $(date '+%F %T') GLM $TAG ($MODE) done"
