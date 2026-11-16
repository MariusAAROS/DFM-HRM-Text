#!/usr/bin/env bash
# Data of E5 (wiki: mimir-graph-e5-multidataset-rl): GRPO from E3 on datasets it never trained on, plus replay.
#   verl_e5/train.parquet  OSM 6,000 + CLUTRR 3,000 + CWQ 3,000 (new) + 800 each of GraphQA, CLRS, MetaQA, KQA Pro,
#                          WebQSP (old), minus prompts over 2,048 Mimir tokens (GRPO context = 2,048 + 2,048)
#   verl_e5/val.parquet    E3's val.parquet (298 rows) + 60 each of OSM, CLUTRR, CWQ val
#   sft/e5_answer.jsonl    the SFT control's rows = the questions of train.parquet: answer line for the new datasets,
#                          trace + answer for the old ones (old records without a trace are dropped)
# Usage: bash graph_bench/rl/build_e5_data.sh
set -euo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
B=data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
TOK=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5-hfview
NEW="$B/osm/train.jsonl $B/clutrr/train.jsonl $B/cwq/train.jsonl"
OLD="$B/graphqa/train.jsonl $B/clrs/train.jsonl $B/metaqa/train.jsonl $B/kqapro/train.jsonl $B/webqsp/train.jsonl"
export HF_HUB_OFFLINE=1

$PY -m graph_bench.export_verl --out $B/verl_e5/train.parquet --seed 0 --tokenizer $TOK --max-prompt-tokens 2048 \
  $B/osm/train.jsonl@6000 $B/clutrr/train.jsonl@3000 $B/cwq/train.jsonl@3000 $(for f in $OLD; do echo $f@800; done)
$PY -m graph_bench.export_verl --out $B/verl_e5/val_new.parquet --seed 0 --tokenizer $TOK --max-prompt-tokens 2048 \
  $B/osm/val.jsonl@60 $B/clutrr/val.jsonl@60 $B/cwq/val.jsonl@60
$PY - <<PYEOF
import pyarrow as pa, pyarrow.parquet as pq
t = pa.concat_tables([pq.read_table("$B/verl/val.parquet"), pq.read_table("$B/verl_e5/val_new.parquet")])
pq.write_table(t, "$B/verl_e5/val.parquet")
print("$B/verl_e5/val.parquet:", t.num_rows, "rows")
PYEOF
rm $B/verl_e5/val_new.parquet

$PY -m graph_bench.export_hrm --target cot --prompt instruct --ids-from $B/verl_e5/train.parquet \
  --out $B/sft/e5_old_cot.jsonl $OLD
$PY -m graph_bench.export_hrm --target answer --prompt instruct --ids-from $B/verl_e5/train.parquet \
  --extra $B/sft/e5_old_cot.jsonl --out $B/sft/e5_answer.jsonl $NEW
