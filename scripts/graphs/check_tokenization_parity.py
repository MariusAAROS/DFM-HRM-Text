"""Check that data_io tokenization matches the old prepare_sft_data.py output, and inference prompts.

1. Training parity: the multiset of per-sample (instruction tokens, response tokens) in epoch 0 of a
   data_io sampled dir must equal that of an old HRM-Text-for-Graphs prepared dir. Epoch shuffles use
   different RNG streams, so sample order is not compared.
2. Inference parity: simple_inference_engine.InferenceCheckpoint.tokenize_prompt(condition, prompt)
   must reproduce the training instruction tokens of the same JSONL row (generation is conditioned on
   exactly what the model was trained on).

Usage:
    .venv/bin/python scripts/graphs/check_tokenization_parity.py \
        --new data/graphs/sampled/graphqa_e20 \
        --old /work/dfm/marius-ortega/graphqa_fullbp/std_prepared \
        --tokenized data/graphs/tokenized/graphqa \
        --jsonl data/graphs/graphqa/hrm-text/standard/train.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def epoch_samples(root: Path, epoch: int = 0) -> Counter:
    tokens = np.load(root / "tokens.npy", mmap_mode="r")
    idx = {k: np.load(root / f"epoch_{epoch}" / f"{k}.npy") for k in ("inst_start", "inst_len", "resp_start", "resp_len")}
    samples = Counter()
    for i_s, i_l, r_s, r_l in zip(idx["inst_start"], idx["inst_len"], idx["resp_start"], idx["resp_len"]):
        inst = tokens[i_s:i_s + i_l].astype(np.int32).tobytes()
        resp = tokens[r_s:r_s + r_l].astype(np.int32).tobytes()
        samples[(inst, resp)] += 1
    return samples


def check_training(new: Path, old: Path) -> bool:
    a, b = epoch_samples(new), epoch_samples(old)
    n_new, n_old = sum(a.values()), sum(b.values())
    only_new, only_old = sum((a - b).values()), sum((b - a).values())
    print(f"[train] new={n_new} old={n_old} samples; only-in-new={only_new} only-in-old={only_old}")
    meta_new = json.load(open(new / "metadata.json"))["tokenizer_info"]
    meta_old = json.load(open(old / "metadata.json"))["tokenizer_info"]
    for key in ("boq", "eoq", "eoa", "condition_mapping", "vocab_size"):
        same = meta_new.get(key) == meta_old.get(key)
        print(f"[train] tokenizer_info.{key}: {'same' if same else f'DIFF {meta_new.get(key)!r} vs {meta_old.get(key)!r}'}")
        if not same:
            return False
    return only_new == 0 and only_old == 0 and n_new == n_old


def check_inference(sampled: Path, tokenized: Path, jsonl: Path, limit: int) -> bool:
    from transformers import PreTrainedTokenizerFast

    from simple_inference_engine import InferenceCheckpoint

    tokenizer_info = json.load(open(sampled / "metadata.json"))["tokenizer_info"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_info["tokenizer_path"])
    ckpt = SimpleNamespace(tokenizer=tokenizer, tokenizer_info=tokenizer_info)

    tokens = np.load(tokenized / "train.jsonl" / "tokens.npy", mmap_mode="r")
    inst_start = np.load(tokenized / "train.jsonl" / "inst_start.npy")
    inst_len = np.load(tokenized / "train.jsonl" / "inst_len.npy")
    mismatches = checked = 0
    with open(jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row_id, row in enumerate(rows[:limit]):
        prompt = InferenceCheckpoint.tokenize_prompt(ckpt, row.get("condition", "direct"), row["instruction"])  # type: ignore[arg-type]
        train = tokens[inst_start[row_id]:inst_start[row_id] + inst_len[row_id]]
        checked += 1
        if not np.array_equal(np.asarray(prompt, dtype=np.int64), train.astype(np.int64)):
            mismatches += 1
    print(f"[infer] tokenize_prompt == training instruction tokens: {checked - mismatches}/{checked}")
    return mismatches == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, type=Path, help="data_io sampled dir (prepare_dataset.sh output).")
    ap.add_argument("--old", type=Path, help="Old prepare_sft_data.py output dir.")
    ap.add_argument("--tokenized", type=Path, help="data_io tokenized dir (pre-sampling, JSONL row order).")
    ap.add_argument("--jsonl", type=Path, help="Train JSONL the dirs were built from.")
    ap.add_argument("--limit", type=int, default=1000, help="Rows checked for inference parity.")
    args = ap.parse_args()

    ok = True
    if args.old is not None:
        ok &= check_training(args.new, args.old)
    if args.tokenized is not None and args.jsonl is not None:
        ok &= check_inference(args.new, args.tokenized, args.jsonl, args.limit)
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
