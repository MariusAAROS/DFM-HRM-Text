"""Bench records -> VeRL parquet for GRPO (schema of jacob-hrm-text-rlvr/verl_hrm_text/prepare_data.py).

Columns: data_source (the dataset), prompt (one user message: the GRL prompt plus an instruction to end with an
"Answer:" line), ability ("graph"), reward_model {style: rule, ground_truth: JSON with answers / answer_format /
checker}, extra_info {split, index, id, task}. graph_bench/rl/reward.py decodes ground_truth and calls
scoring.score, so the reward is the same metric as evaluation.

--max-prompt-tokens drops records whose rendered prompt is longer (the KV cache costs 768 KB/token and GRPO groups
cannot share a cached prompt; see the wiki). Token counts use the tokenizer given by --tokenizer.

Run in an env with pyarrow (the repo .venv or /home/ucloud/miniforge3):
    .venv/bin/python -m graph_bench.export_verl --out data/graphs/bench/verl/train.parquet \\
        --tokenizer danish-foundation-models/DFM-Mimir-v1.5 --max-prompt-tokens 2048 data/graphs/bench/*/train.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graph_bench.schema import GRL_PREFIX, prompt_text, read_records

ANSWER_INSTRUCTION = {
    "single": "Answer: <answer>",
    "set": "Answer: <answer 1> | <answer 2> | ...",
    "ordered": "Answer: <item 1>, <item 2>, ...",
    "checker": "Answer: <answer>",
}


def user_message(rec) -> str:
    return (GRL_PREFIX + prompt_text(rec).removesuffix("A: ")
            + "\nReason step by step from the facts, then finish with a final line in exactly this form, with no "
            + f"other text after it:\n{ANSWER_INSTRUCTION[rec.answer_format]}")


def ground_truth(rec) -> str:
    return json.dumps({"answers": rec.answers, "answer_format": rec.answer_format, "checker": rec.checker},
                      ensure_ascii=False)


def build_rows(inputs: list[str], cap: int | None, seed: int, count_tokens=None, max_tokens: int | None = None):
    rng = random.Random(seed)
    rows, stats = [], Counter()
    for path in inputs:
        recs = read_records(path)
        if cap is not None and len(recs) > cap:
            recs = rng.sample(recs, cap)
        for rec in recs:
            content = user_message(rec)
            if max_tokens is not None and count_tokens(content) > max_tokens:
                stats[f"{rec.dataset}_too_long"] += 1
                continue
            rows.append({"data_source": rec.dataset, "prompt": [{"role": "user", "content": content}],
                         "ability": "graph", "reward_model": {"style": "rule", "ground_truth": ground_truth(rec)},
                         "extra_info": {"split": rec.split, "index": len(rows), "id": rec.id, "task": rec.task}})
            stats[rec.dataset] += 1
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["extra_info"]["index"] = i
    return rows, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=None, help="Max records per input file.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for --max-prompt-tokens.")
    ap.add_argument("--max-prompt-tokens", type=int, default=None)
    args = ap.parse_args()
    count = None
    if args.max_prompt_tokens is not None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        def count(text: str) -> int:
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True)
            return len(tok(rendered, add_special_tokens=False)["input_ids"])
    rows, stats = build_rows(args.inputs, args.cap, args.seed, count, args.max_prompt_tokens)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{args.out}: {len(rows)} rows {dict(stats)}")


if __name__ == "__main__":
    main()
