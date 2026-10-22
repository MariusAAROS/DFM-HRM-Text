#!/bin/bash
# Full KG pretraining corpus pipeline, in dependency order (see wiki/pages/kg-pretraining-corpus.md).
#
#   STEPS="fetch small wikidata5m yago wikidata relabel trex kelm graphqa decontam verify stats smoke" (default: all)
#   setsid nohup bash scripts/kg_pretrain/run_all.sh > data/kg_pretrain/logs/run_all.log 2>&1 < /dev/null &
#
# The Wikidata5M store must exist before YAGO / Wikidata (benchmark triples for the holdout), and Wikidata
# must exist before relabel / trex / kelm (English labels).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -I"
S=scripts/kg_pretrain
STEPS="${STEPS:-fetch small wikidata5m yago wikidata relabel trex kelm graphqa decontam verify stats smoke}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
step() { [[ " $STEPS " == *" $1 "* ]] && echo "$(date '+%F %T') == $1"; }

if step fetch; then bash $S/fetch.sh all; fi
if step small; then
  $PY $S/convert_conceptnet.py && $PY $S/holdout.py conceptnet
  $PY $S/render_docs.py conceptnet --skip-relations /r/FormOf --min-facts 2
  $PY $S/convert_atomic2020.py && $PY $S/holdout.py atomic2020 --keep-existing && $PY $S/render_docs.py atomic2020
  $PY $S/convert_hetionet.py && $PY $S/holdout.py hetionet && $PY $S/render_docs.py hetionet --inverse --types-line
  $PY $S/convert_primekg.py && $PY $S/holdout.py primekg && $PY $S/render_docs.py primekg --types-line
  $PY $S/convert_graphinstruct.py
fi
if step wikidata5m; then $PY $S/convert_wikidata5m.py && $PY $S/holdout.py wikidata5m --keep-existing; fi
if step yago; then bash $S/yago.sh; fi
if step wikidata; then bash $S/wikidata.sh; fi
if step relabel; then
  $PY $S/convert_wikidata5m.py --relabel && $PY $S/render_docs.py wikidata5m --first P31,P279
fi
if step trex; then $PY $S/convert_trex.py; fi
if step kelm; then $PY $S/convert_kelm.py; fi
if step graphqa; then $PY $S/gen_graphqa.py --n 1250000 --graphs-per-algorithm 40000 --workers 48; fi
if step decontam; then $PY $S/decontam.py; fi
if step verify; then
  for s in conceptnet atomic2020 hetionet primekg; do $PY $S/verify.py $s > /dev/null; done
  $PY $S/verify.py wikidata5m > /dev/null
  $PY $S/verify.py yago --benchmark data/kg_pretrain/kg/yago/benchmark_triples.tsv > /dev/null
  $PY $S/verify.py wikidata --benchmark data/kg_pretrain/kg/wikidata5m/benchmark_triples.tsv > /dev/null
fi
if step stats; then $PY $S/stats.py; fi
if step smoke; then bash $S/smoke_tokenize.sh; fi
echo "$(date '+%F %T') run_all done: $STEPS"
