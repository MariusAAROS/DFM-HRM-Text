#!/bin/bash
# Wikidata truthy dump -> triple store kg/wikidata -> holdout -> entity documents.
#
#   pass 1: lbzip2 | fanout.py into $WORKERS (grep prefilter | wikidata_pass1.py) workers
#   pass 2: wikidata_pass2.py (entities, relations, filtered triples per subject shard)
#   holdout.py --force-test (Wikidata5M valid+test), render_docs.py
#
# Run detached: setsid nohup bash scripts/kg_pretrain/wikidata.sh > data/kg_pretrain/logs/wikidata.log 2>&1 < /dev/null &
# STAGES="1 2 3 4" picks stages (1 pass1, 2 pass2, 3 holdout, 4 render).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
TOOLS=/work/dfm/marius-ortega/dfm-hrm-graphs/kg_tools/bin
DUMP="${DUMP:-$(ls data/kg_pretrain/raw/wikidata/wikidata-*-truthy-BETA.nt.bz2)}"
WORK="${WORK:-data/kg_pretrain/work/wikidata}"
WORKERS="${WORKERS:-48}"
SHARDS="${SHARDS:-256}"
STAGES="${STAGES:-1 2 3 4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

stage() { [[ " $STAGES " == *" $1 "* ]]; }
ts() { date '+%F %T'; }

if stage 1; then
  [[ -f data/kg_pretrain/raw/wikidata/SOURCE.json ]] || { echo "dump not fully fetched yet" >&2; exit 1; }
  echo "$(ts) pass 1: $DUMP -> $WORK/pass1"
  rm -rf "$WORK/pass1"
  mkdir -p "$WORK/pass1"
  # grep runs inside each worker (wikidata_pass1_filter.sh): a single grep caps the stream at ~450 MB/s, and
  # fanout.py hands out 8 MB blocks (split -n r/N writes line by line and caps it at ~500 MB/s).
  "$TOOLS/lbzip2" -dc -n 96 "$DUMP" \
    | "$PY" -I scripts/kg_pretrain/fanout.py "$WORKERS" -- \
        bash scripts/kg_pretrain/wikidata_pass1_filter.sh "$WORK/pass1" "{worker}" "$SHARDS"
  echo "$(ts) pass 1 done: $(du -sh "$WORK/pass1" | cut -f1)"
fi

if stage 2; then
  echo "$(ts) pass 2"
  "$PY" -I scripts/kg_pretrain/wikidata_pass2.py --pass1 "$WORK/pass1" --shards "$SHARDS" --workers 64
fi

if stage 3; then
  echo "$(ts) holdout"
  "$PY" -I scripts/kg_pretrain/holdout.py wikidata --workers 64 \
    --force-test data/kg_pretrain/kg/wikidata5m/benchmark_triples.tsv
fi

if stage 4; then
  echo "$(ts) render"
  "$PY" -I scripts/kg_pretrain/render_docs.py wikidata --first P31,P279 --workers 64
fi
echo "$(ts) wikidata done: stages $STAGES"
