"""Bench records -> data_io JSONL ({instruction, response, condition, task, dataset}) for prepare_dataset.sh.

The instruction is GRL's prompt ("#### Question\\n" + graph + "Q: ...\\nA: "), as in data/graphs/*/hrm-text-grl, so the
HRM BPE and Gemma chat paths (scripts/tokenize_chat_template.py) both take the output unchanged.

  --target direct   "<answer>."
  --target cot      trace lines + "Answer: <answer>." (records without a trace are dropped and counted)
  --anonymize       entity-anonymised variant (anonymize.py); records whose question loses its entity are dropped

Traces go in the plain response, not in Mimir's thought channel: the Gemma template's strip_thinking removes every
<|channel>...<channel|> span from model turns, so a channel-wrapped trace would vanish from the SFT target.

Several inputs can be mixed into one file (the SFT train mixture), each capped with --cap and shuffled with --seed:
    .venv/bin/python -m graph_bench.export_hrm --target cot --cap 20000 --seed 0 \\
        --out data/graphs/bench/mix/train_cot.jsonl data/graphs/bench/{metaqa,kqapro,webqsp}/train.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from graph_bench.anonymize import anonymize
from graph_bench.schema import GRL_PREFIX, cot_response, direct_response, prompt_text, read_records


def to_row(rec, target: str) -> dict | None:
    if target == "cot" and not rec.trace:
        return None
    response = cot_response(rec) if target == "cot" else direct_response(rec)
    return {"instruction": GRL_PREFIX + prompt_text(rec), "response": response, "condition": "direct",
            "task": rec.task, "dataset": rec.dataset}


def export(inputs: list[str], out: str, target: str, anonymized: bool = False, cap: int | None = None,
           seed: int = 0) -> Counter:
    rng = random.Random(seed)
    stats, rows = Counter(), []
    for path in inputs:
        recs = read_records(path)
        if cap is not None and len(recs) > cap:
            recs = rng.sample(recs, cap)
        for rec in recs:
            if anonymized:
                rec, grounded = anonymize(rec)
                if not grounded:
                    stats["dropped_ungrounded"] += 1
                    continue
            row = to_row(rec, target)
            if row is None:
                stats["dropped_no_trace"] += 1
                continue
            rows.append(row)
            stats[rec.dataset] += 1
    if len(inputs) > 1:
        rng.shuffle(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Bench JSONL files (graph_bench.schema.Record lines).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["direct", "cot"], default="direct")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--cap", type=int, default=None, help="Max records per input file (random subset).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stats = export(args.inputs, args.out, args.target, args.anonymize, args.cap, args.seed)
    print(f"{args.out}: {dict(stats)}")


if __name__ == "__main__":
    main()
