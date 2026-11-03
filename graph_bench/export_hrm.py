"""Bench records -> data_io JSONL ({instruction, response, condition, task, dataset}) for prepare_dataset.sh.

The instruction is GRL's prompt ("#### Question\\n" + graph + "Q: ...\\nA: "), as in data/graphs/*/hrm-text-grl, so the
HRM BPE and Gemma chat paths (scripts/tokenize_chat_template.py) both take the output unchanged.

  --target direct   "<answer>."
  --target cot      trace lines + "Answer: <answer>." (records without a trace are dropped and counted)
  --anonymize       entity-anonymised variant (anonymize.py); records whose question loses its entity are dropped

Traces go in the plain response, not in Mimir's thought channel: the Gemma template's strip_thinking removes every
<|channel>...<channel|> span from model turns, so a channel-wrapped trace would vanish from the SFT target.

  --target answer  "Answer: <answer>." (E1: the answer line alone, after the instruct prompt)
  --prompt instruct  the RL / zero-shot user message (export_verl.user_message: reason, then an "Answer:" line), so SFT,
                     E0 eval (eval_vllm --style instruct) and GRPO share one prompt
  --traced-only      drop records without a trace for every target (E1 and E2 then train on the same questions)
  --extra FILE       data_io rows added as they are (general-instruction replay, export_replay.py), shuffled in

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
from graph_bench.export_verl import user_message
from graph_bench.schema import GRL_PREFIX, cot_response, direct_response, prompt_text, read_records


def to_row(rec, target: str, prompt: str = "grl", traced_only: bool = False) -> dict | None:
    if (target == "cot" or traced_only) and not rec.trace:
        return None
    response = {"cot": cot_response, "direct": direct_response,
                "answer": lambda r: f"Answer: {direct_response(r)}"}[target](rec)
    instruction = user_message(rec) if prompt == "instruct" else GRL_PREFIX + prompt_text(rec)
    return {"instruction": instruction, "response": response, "condition": "direct", "task": rec.task,
            "dataset": rec.dataset}


def export(inputs: list[str], out: str, target: str, anonymized: bool = False, cap: int | None = None,
           seed: int = 0, prompt: str = "grl", traced_only: bool = False, extra: list[str] = ()) -> Counter:
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
            row = to_row(rec, target, prompt, traced_only)
            if row is None:
                stats["dropped_no_trace"] += 1
                continue
            rows.append(row)
            stats[rec.dataset] += 1
    for path in extra:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    stats[rows[-1].get("dataset", "extra")] += 1
    if len(inputs) + len(extra) > 1:
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
    ap.add_argument("--target", choices=["direct", "cot", "answer"], default="direct")
    ap.add_argument("--prompt", choices=["grl", "instruct"], default="grl")
    ap.add_argument("--traced-only", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="data_io JSONL rows to add as they are (replay).")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--cap", type=int, default=None, help="Max records per input file (random subset).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stats = export(args.inputs, args.out, args.target, args.anonymize, args.cap, args.seed, args.prompt,
                   args.traced_only, args.extra)
    print(f"{args.out}: {dict(stats)}")


if __name__ == "__main__":
    main()
