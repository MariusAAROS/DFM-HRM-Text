"""General-instruction replay rows for the graph SFT mixture (wiki: mimir-graph-benchmark, E1 / E2).

Source: dfm-dyna-instruct's apertus-sft-mixture config, local copy in /work/dfm/Mimir/MTCv1 (parquet, `messages` =
[{role, content}, ...]). Keeps single-turn conversations (one user message, one assistant reply, no system or tool
turn) under --max-chars, without chat-template control strings, and draws --n of them uniformly from --shards random
shards. Output: data_io rows {instruction, response, condition, task, dataset="replay"} for export_hrm.py --extra.

Streams each shard in small batches (the container has 128 GB; see the wiki's OOM entry):
    .venv/bin/python -m graph_bench.export_replay --n 2600 --out data/graphs/bench/sft/replay.jsonl
"""
import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

SRC = "/work/dfm/Mimir/MTCv1/data/dfm7_special_sources/dfm_dyna_instruct_apertus_sft_mixture_shards"
CONTROL = ("<|", "|>", "<turn", "turn|>", "<start_of_turn>", "<end_of_turn>", "<think>", "</think>")


def single_turn(messages, max_chars: int):
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    user, reply = messages[0]["content"].strip(), messages[1]["content"].strip()
    if not user or not reply or len(user) + len(reply) > max_chars:
        return None
    if any(c in user or c in reply for c in CONTROL):
        return None
    return user, reply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--max-chars", type=int, default=6000, help="user + reply characters (~1.5k tokens)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    files = sorted(glob.glob(f"{SRC}/part-*.parquet"))
    pool, seen = [], 0
    for f in sorted(rng.sample(files, args.shards)):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                seen += 1
                turn = single_turn(messages, args.max_chars)
                if turn is not None:
                    pool.append(turn)
    rows = rng.sample(pool, min(args.n, len(pool)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for user, reply in rows:
            f.write(json.dumps({"instruction": user, "response": reply, "condition": "direct", "task": "replay",
                                "dataset": "replay"}, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(rows)} rows from {len(pool)} single-turn of {seen} conversations in {args.shards} shards")


if __name__ == "__main__":
    main()
