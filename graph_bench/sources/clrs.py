"""CLRS-Text (smcleish/CLRS-Text-train/-test), graph algorithms only -> bench records with the native traces.

Tier 1 (wiki: mimir-graph-benchmark): train on small graphs, test both on the train sizes (test.jsonl) and on
larger ones (test_ood.jsonl, within-task OOD size). The 12 graph algorithms: bfs, dfs, dijkstra, bellman_ford,
mst_prim, mst_kruskal, topological_sort, articulation_points, bridges, strongly_connected_components,
dag_shortest_paths, floyd_warshall.

A CLRS-Text question is "<algo>:\\n<inputs>\\ntrace | <outputs>:" and its answer "<state>, <state>, ... | <final>"; each
state holds one value per output name. The record keeps the inputs as `context`, asks for the final output, and the
trace is one line per algorithm state (CLRS's own execution trace, so every step is the algorithm's next state).
The final output is deterministic (CLRS tie-breaking), so it is compared exactly ("ordered" would split the arrays).
Rows whose prompt or answer exceeds MAX_PROMPT_CHARS / MAX_ANSWER_CHARS are skipped (token budget, ~2 chars/token).

    .venv/bin/python -m graph_bench.sources.clrs
"""
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from graph_bench.schema import Record, write_records

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/graphs/bench/raw/clrs"
OUT = REPO / "data/graphs/bench/clrs"
ALGOS = ("bfs", "dfs", "dijkstra", "bellman_ford", "mst_prim", "mst_kruskal", "topological_sort",
         "articulation_points", "bridges", "strongly_connected_components", "dag_shortest_paths", "floyd_warshall")
TRAIN_MAX_N = 12
OOD_N = (13, 20)
PER_ALGO = {"train": 1500, "val": 100, "test": 100, "test_ood": 100}
MAX_PROMPT_CHARS = 4000
MAX_ANSWER_CHARS = 4000
PREAMBLE_NOTE = ("A graph algorithm from CLRS. A is the adjacency matrix (A[i][j] is the weight of edge i -> j, 0 for "
                 "no edge), s the source node; trace values are the algorithm's state arrays, indexed by node.")


def top_level_items(text: str) -> list[str]:
    """Split "a, [b c], [[d], [e]], 3" at commas outside brackets."""
    items, depth, cur = [], 0, []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        items.append("".join(cur).strip())
    return items


def n_nodes(inputs: str) -> int:
    """Rows of the adjacency matrix A (the train shards have no length column)."""
    start = inputs.index("A: [") + 3
    depth = 0
    for j, ch in enumerate(inputs[start:], start):
        depth += ch == "["
        depth -= ch == "]"
        if depth == 0:
            return len(top_level_items(inputs[start + 1:j]))
    raise ValueError("unterminated A")


def parse(row: dict) -> dict | None:
    q, a = row["question"], row["answer"].strip()
    if " | " not in a or "A: [" not in q or "\ntrace | " not in q:
        return None  # malformed rows (counted by the caller)
    head, rest = q.split(":\n", 1)
    inputs, out_line = rest.rsplit("\ntrace | ", 1)
    outputs = [o.strip() for o in out_line.rstrip(":\n ").split(",")]
    states_text, final = a.rsplit(" | ", 1)
    items = top_level_items(states_text)
    k = len(outputs)
    if len(items) % k:
        return None
    states = [", ".join(items[i:i + k]) for i in range(0, len(items), k)]
    return {"algo": head.strip(), "inputs": inputs.strip(), "outputs": outputs, "states": states,
            "final": final.strip(), "n": n_nodes(inputs)}


def to_record(p: dict, split: str, i: int) -> Record:
    out = ", ".join(p["outputs"])
    trace = [f"State {j + 1}: {s}" for j, s in enumerate(p["states"])]
    return Record(id=f"clrs/{split}/{p['algo']}/{i}", dataset="clrs", split=split, task=p["algo"],
                  question=f"Run {p['algo']} on this input. What is its final {out}?", answers=[p["final"]],
                  context=f"{PREAMBLE_NOTE}\n{p['algo']}:\n{p['inputs']}", trace=trace, preamble="",
                  meta={"n": p["n"], "outputs": p["outputs"], "supported": True})


def fits(p: dict, row: dict) -> bool:
    return len(row["question"]) <= MAX_PROMPT_CHARS and len(row["answer"]) <= MAX_ANSWER_CHARS


def _rows(path):
    """Stream rows of the graph algorithms (never the whole table: see rog.py, the 2026-11-02 out-of-memory crash)."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["question", "answer", "algo_name"]):
        batch = batch.filter(pc.is_in(batch.column("algo_name"), value_set=pa.array(ALGOS)))
        yield from batch.to_pylist()


def main():
    rng = random.Random(0)
    stats = Counter()
    # train / val: from the train shards, n <= TRAIN_MAX_N, disjoint samples per algorithm
    pools = defaultdict(list)
    for f in sorted(glob.glob(str(RAW / "train/data/*.parquet"))):
        for row in _rows(f):
            if row["algo_name"] in ALGOS:
                p = parse(row)
                stats["malformed"] += p is None
                if p and p["n"] <= TRAIN_MAX_N and fits(p, row):
                    pools[row["algo_name"]].append(p)
    train, val = [], []
    for algo in ALGOS:
        pool = pools[algo]
        rng.shuffle(pool)
        seen, uniq = set(), []
        for p in pool:  # drop duplicate inputs (small graphs repeat)
            if p["inputs"] not in seen:
                seen.add(p["inputs"])
                uniq.append(p)
        val += uniq[:PER_ALGO["val"]]
        train += uniq[PER_ALGO["val"]:PER_ALGO["val"] + PER_ALGO["train"]]
        stats[f"pool_{algo}"] = len(uniq)
    train_inputs = {p["inputs"] for p in train} | {p["inputs"] for p in val}
    # test (train sizes) / test_ood (larger): from test_1
    tests = {"test": defaultdict(list), "test_ood": defaultdict(list)}
    for row in _rows(RAW / "test/data/test_1-00000-of-00001.parquet"):
        if row["algo_name"] not in ALGOS:
            continue
        p = parse(row)
        if not p or not fits(p, row) or p["inputs"] in train_inputs:
            continue
        if p["n"] <= TRAIN_MAX_N:
            tests["test"][p["algo"]].append(p)
        elif OOD_N[0] <= p["n"] <= OOD_N[1]:
            tests["test_ood"][p["algo"]].append(p)
    out = {"train": train, "val": val}
    for split, by_algo in tests.items():
        out[split] = []
        for algo in ALGOS:
            rows = by_algo[algo]
            rng.shuffle(rows)
            out[split] += rows[:PER_ALGO[split]]
    for split, ps in out.items():
        n = write_records([to_record(p, split, i) for i, p in enumerate(ps)], OUT / f"{split}.jsonl")
        stats[split] = n
    print(dict(stats))


if __name__ == "__main__":
    main()
