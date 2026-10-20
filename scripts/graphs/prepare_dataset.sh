#!/usr/bin/env bash
# Tokenize one graph-experiment train JSONL with the data_io pipeline (Rust tokenizer + stratified
# sampler) into the V1Dataset layout read by dataset_new.py.
#
#   bash scripts/graphs/prepare_dataset.sh <name> <train.jsonl> <epochs> [context_size]
#   e.g. bash scripts/graphs/prepare_dataset.sh graphqa data/graphs/graphqa/hrm-text/standard/train.jsonl 20
#
# Output: data/graphs/sampled/<name>_e<epochs>/   (train with data.path=<that dir>)
# SEED=<s> (default 0) reseeds the per-epoch shuffle (sample order and packing) -> <name>_s<s>_e<epochs>/.
# This is the only source of run-to-run variation in a fine-tune: pretrain.py's `seed` only sets the
# random init, which init_weights_path overwrites, and the training dataloader adds no randomness.
# <epochs> MUST equal `epochs` in the training config: one epoch_N/ index dir is consumed per epoch.
#
# Template (data_io --template-mode hrm, identical to the old scripts/prepare_sft_data.py):
#   <|im_start|> <condition token(s)> instruction <|im_end|> | response <|box_end|>
# The tokenizer path is passed as an absolute path because data_io records it verbatim in
# tokenizer_info.json, and simple_inference_engine reloads the tokenizer from there.
#
# CHAT_TEMPLATE=<jinja> switches to scripts/tokenize_chat_template.py (template_mode jinja_chat_template), the
# format of the Gemma-tokenizer DFM models such as DFM-Mimir-v1.5. Condition "direct" adds no system turn, so
# the prompt is one user turn, as InferenceCheckpoint.tokenize_prompt builds it at eval time. Pass TOKENIZER too.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$PWD
NAME=${1:?name}
TRAIN_JSONL=$(realpath "${2:?train.jsonl}")
EPOCHS=${3:?epochs}
CONTEXT_SIZE=${4:-4097}  # max sample length + 1 (AR shift); dataset_new.py subtracts 1
SEED=${SEED:-0}
PY=${PY:-$REPO/.venv/bin/python}
TOKENIZER=${TOKENIZER:-$REPO/data_io/trained_tokenizers/bpe/tokenizer.json}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
ROOT=${ROOT:-$REPO/data/graphs}
# rustup on this machine has toolchains (stable, 1.95.0) but no default; the crate needs edition 2024.
export RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable}

IN_DIR=$ROOT/tok_in/$NAME
TOK_DIR=$ROOT/tokenized/$NAME
OUT_DIR=$ROOT/sampled/${NAME}$( [[ $SEED == 0 ]] || echo "_s$SEED" )_e${EPOCHS}

[[ -f data_io/tokenizer/Cargo.toml ]] || { echo "data_io submodule missing: git submodule update --init data_io" >&2; exit 1; }

# 1. Stage: the tokenizer scans directories (one output subdir per input file) and does not follow
#    symlinks, so hard-link (or copy) the JSONL in.
rm -rf "$IN_DIR"; mkdir -p "$IN_DIR"
ln "$TRAIN_JSONL" "$IN_DIR/train.jsonl" 2>/dev/null || cp -p "$TRAIN_JSONL" "$IN_DIR/train.jsonl"

# 2. Tokenize. Must run from data_io/tokenizer (Cargo.toml). One output tree per dataset:
#    the tokenizer prunes subdirs of the output dir that have no matching input.
mkdir -p "$TOK_DIR"
if [[ -n $CHAT_TEMPLATE ]]; then
  "$PY" scripts/tokenize_chat_template.py "$IN_DIR" -o "$TOK_DIR" --tokenizer-path "$TOKENIZER" \
      --chat-template "$CHAT_TEMPLATE" --force
else
  (cd data_io/tokenizer && cargo run --release --quiet --bin tokenizer -- \
      "$IN_DIR" -o "$TOK_DIR" --tokenizer-path "$TOKENIZER" --template-mode hrm)
fi

# 3. Sample all rows into per-epoch shuffles.
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
"$PY" data_io/sample_tokenized.py \
    tokenized_path="$TOK_DIR" output_path="$OUT_DIR" \
    prefix_config_path="$REPO/scripts/graphs/prefix_config_graphs.yaml" \
    epochs="$EPOCHS" context_size="$CONTEXT_SIZE" seed="$SEED" > "$OUT_DIR/show_analytics.md"
cp "$TOKENIZER" "$OUT_DIR/tokenizer.json"  # self-contained copy, like the old prepared dirs

# 4. No silent drops: every input row (minus empty responses) must be in every epoch.
"$PY" - "$TOK_DIR" "$OUT_DIR" "$EPOCHS" <<'PY'
import sys, json
from pathlib import Path
import numpy as np
tok, out, epochs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
resp_len = np.load(tok / "train.jsonl" / "resp_len.npy")
expected = int((resp_len >= 2).sum())
empty = int((resp_len < 2).sum())
for e in range(epochs):
    got = len(np.load(out / f"epoch_{e}" / "inst_len.npy"))
    if got != expected:
        sys.exit(f"epoch_{e}: {got} rows, expected {expected} (overlong rows dropped? raise context_size)")
meta = json.load(open(out / "metadata.json"))
lens = np.load(tok / "train.jsonl" / "inst_len.npy") + resp_len
print(f"{out}: {expected} rows x {epochs} epochs, {meta['total_length']:,} tokens/epoch, "
      f"max sample {int(lens.max())} tokens, {empty} empty responses dropped")
PY
