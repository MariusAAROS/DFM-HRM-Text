"""GraphQA v2 with a chain-of-thought target for every task (graph_inject.cot), for dataset graphqa_v2_cot.

    .venv/bin/python scripts/graph_inject/build_graphqa_cot.py [--src ...] [--dst ...]

Same rows, order and prompts as v2; `response` becomes the trace ending in "Answer: <gold>", and two fields are added:
`cot_detail` (the trace detail, see graph_inject.cot.DETAILS) and `cot_gold_replaced` (the 3 ConnectedNodes rows whose
gold order matches no rule). Every row must fit the models' 4,096 RoPE positions (prompt + response incl. the stop
token <= MAX_TOKENS) with every model's tokenizer; TriangleCounting and MaximumFlow rows that don't fit at full detail
take the next coarser detail. v2 is not modified; dedup_report.json is copied.
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from graph_inject.cot import DETAILS, cot_response
from graph_inject.tokenize import PromptTokenizer, tokenizer_checkpoint

SPLITS = ("train", "val", "test")
MAX_TOKENS = 4090  # 4,096 positions, a few spare
MODELS = ("hrm1b", "mimir1", "mimir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/graphs/graphqa/hrm-text-grl-v2/standard")
    ap.add_argument("--dst", default="data/graphs/graphqa/hrm-text-grl-v2-cot/standard")
    args = ap.parse_args()
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise ValueError("--dst must differ from --src")
    tokenizers = [PromptTokenizer(tokenizer_checkpoint(OmegaConf.load(f"graph_inject/configs/models/{m}.yaml").hf_dir))
                  for m in MODELS]

    def length(row, response) -> int:
        return max(len(t.encode_prompt(row["condition"], row["instruction"], None).ids)
                   + len(t.encode_response(row["condition"], row["instruction"], response)) for t in tokenizers)

    os.makedirs(args.dst, exist_ok=True)
    for k in SPLITS:
        rows = [json.loads(line) for line in open(os.path.join(args.src, f"{k}.jsonl"))]
        out, details, replaced, lengths = [], Counter(), [], {}
        for i, r in enumerate(rows):
            for detail in DETAILS.get(r["task"], ("full",)):
                response, gold_replaced = cot_response(r["instruction"], r["task"], r["response"], detail)
                n = length(r, response)
                if n <= MAX_TOKENS:
                    break
            else:
                raise ValueError(f"{k} row {i} ({r['task']}): {n} tokens > {MAX_TOKENS} at the coarsest detail")
            out.append({**r, "response": response, "cot_detail": detail, "cot_gold_replaced": gold_replaced})
            details[(r["task"], detail)] += detail != "full"
            lengths.setdefault(r["task"], []).append(n)
            if gold_replaced:
                replaced.append(i)
        with open(os.path.join(args.dst, f"{k}.jsonl"), "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
        print(f"{k}: {len(out)} rows; gold replaced (ConnectedNodes order) at rows {replaced}; coarser details: "
              f"{ {f'{t}/{d}': c for (t, d), c in details.items() if c} }")
        for t, ns in sorted(lengths.items()):
            print(f"  {t:18s} prompt+response tokens: mean {np.mean(ns):6.0f}  max {max(ns)}")
    shutil.copy(os.path.join(args.src, "dedup_report.json"), os.path.join(args.dst, "dedup_report.json"))


if __name__ == "__main__":
    main()
