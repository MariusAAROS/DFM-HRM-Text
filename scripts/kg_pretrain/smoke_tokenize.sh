#!/bin/bash
# Tokenize the first shard of every converted/kg-* folder on both paths, into a scratch dir:
#   - HRM path: data_io Rust tokenizer (--template-mode hrm), then sample_tokenized.py with a "kg-" prefix rule
#     and a check that no row was dropped;
#   - Mimir path: scripts/tokenize_chat_template.py with the Gemma chat template and tokenizer.
# Usage: bash scripts/kg_pretrain/smoke_tokenize.sh [scratch_dir]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
OUT="${1:-$REPO/data/kg_pretrain/work/smoke}"
HRM_TOK="$REPO/data_io/trained_tokenizers/bpe/tokenizer.json"
GEMMA_TOK=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5/tokenizer.json
TEMPLATE="$REPO/data_io/chat_templates/gemma4_native_chat.jinja"

rm -rf "$OUT"
mkdir -p "$OUT/in"
for d in data/kg_pretrain/converted/kg-*/; do
  name=$(basename "$d")
  first=$(ls "$d" | grep '\.jsonl\.gz$' | head -1)
  [[ -n "$first" ]] || continue
  mkdir -p "$OUT/in/$name"
  ln "$d/$first" "$OUT/in/$name/$first" 2>/dev/null || cp "$d/$first" "$OUT/in/$name/$first"
done

echo "== HRM path"
(cd data_io/tokenizer && RUSTUP_TOOLCHAIN=stable cargo run --release --quiet --bin tokenizer -- \
  "$OUT/in" -o "$OUT/tok_hrm" --tokenizer-path "$HRM_TOK" --template-mode hrm)
printf -- '- prefix: "kg-"\n  long_context: drop\n' > "$OUT/prefix.yaml"
"$PY" data_io/sample_tokenized.py tokenized_path="$OUT/tok_hrm" output_path="$OUT/sampled_hrm" \
  prefix_config_path="$OUT/prefix.yaml" epochs=1 context_size=8193 > "$OUT/sample_hrm.log" 2>&1 \
  || { tail -20 "$OUT/sample_hrm.log"; exit 1; }
# Every input row must be tokenized, and every non-empty row must survive sampling.
"$PY" - "$OUT" <<'EOF'
import gzip, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
bad, expected_total = 0, 0
for d in sorted(p for p in (out / "tok_hrm").iterdir() if p.is_dir()):
    folder, shard = d.name.split("__", 1)
    n_in = sum(1 for _ in gzip.open(out / "in" / folder / shard, "rt"))
    resp_len = np.load(d / "resp_len.npy")
    lens = np.load(d / "inst_len.npy") + resp_len
    expected_total += int((resp_len >= 2).sum())
    print(f"{d.name}: input {n_in}, tokenized {len(resp_len)}, max {int(lens.max())} tokens, "
          f"mean {lens.mean():.0f}")
    bad += len(resp_len) != n_in
got = len(np.load(out / "sampled_hrm" / "epoch_0" / "inst_len.npy"))
print(f"sampled {got} rows, expected {expected_total}")
sys.exit(1 if bad or got != expected_total else 0)
EOF

echo "== Mimir (Gemma chat template) path"
"$PY" scripts/tokenize_chat_template.py "$OUT/in" -o "$OUT/tok_gemma" --tokenizer-path "$GEMMA_TOK" \
  --chat-template "$TEMPLATE" --force --workers 8 > "$OUT/tok_gemma.log" 2>&1 || { tail -20 "$OUT/tok_gemma.log"; exit 1; }
ls "$OUT/tok_gemma"
echo "smoke tokenize OK"
