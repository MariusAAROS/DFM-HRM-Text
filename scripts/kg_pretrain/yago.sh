#!/bin/bash
# YAGO 4.6 -> triple store -> holdout (Wikidata5M benchmark forced to test) -> entity documents.
# Run detached: setsid nohup bash scripts/kg_pretrain/yago.sh > data/kg_pretrain/logs/yago.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
date '+%F %T convert'
$PY -I scripts/kg_pretrain/convert_yago.py
date '+%F %T holdout'
$PY -I scripts/kg_pretrain/holdout.py yago --workers 64 --force-test data/kg_pretrain/kg/yago/benchmark_triples.tsv
date '+%F %T render'
$PY -I scripts/kg_pretrain/render_docs.py yago --first rdf:type,rdfs:subClassOf --workers 64
date '+%F %T yago done'
