"""GraphQA -> bench records: the v3-CoT splits, a fresh-seed probe and a larger-graph OOD split.

Tier 1 (wiki: mimir-graph-benchmark-plan):
  train / val / test   data/graphs/graphqa/hrm-text-grl-v3-cot (graphqa-v3.md), so test is the same 324 questions as
                       every GraphQA CoT run. The rendered prompt is checked byte-identical to the row's instruction
                       (minus GRL's "#### Question\\n", which export_hrm.py adds back); the trace is the row's CoT.
  test_fresh           contamination-free probe: GraphQA's own generator (all 7 algorithms, adjacency encoder) with
                       seeds no other build used, same graph sizes (5-19 nodes) as test.
  test_large           within-task OOD size: the same generator with 20-34 nodes (ER sparsity <= 0.3, so prompts fit
                       ~4k tokens); rows over MAX_PROMPT_CHARS are dropped.
New rows follow build_graphqa_v3: MaximumFlow capacities written into the prompt (build_graphqa_v2.with_capacities),
gold from graph_task, trace from graph_inject.cot (which must reproduce the gold). A new row is dropped when its graph
is isomorphic to any v3 graph of the same task (graphs without self-loops and isolated nodes) or to an earlier new
row of the same task.

    .venv/bin/python -m graph_bench.sources.graphqa
"""
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import json
import numpy as np

from graph_bench.schema import GRL_PREFIX, Record, prompt_text, write_records

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/graphs/graphqa/hrm-text-grl-v3-cot/standard"
OUT = REPO / "data/graphs/bench/graphqa"
sys.path.insert(0, str(REPO / "scripts/graph_inject"))
sys.path.insert(0, str(REPO / "data/kg_pretrain/raw/graphqa_gen/google-research"))
from build_graphqa_v2 import with_capacities  # noqa: E402
from build_graphqa_v3 import IsoIndex, structure  # noqa: E402
from graph_inject.cot import cot_response  # noqa: E402
from graph_inject.structure import parse_graphqa  # noqa: E402
from graphqa import graph_generator_utils, graph_task  # noqa: E402

TASKS = ("EdgeExistence", "NodeDegree", "NodeCount", "EdgeCount", "ConnectedNodes", "DisconnectedNodes", "CycleCheck",
         "Reachability", "ShortestPath", "MaximumFlow", "TriangleCounting")
ALGORITHMS = ("er", "ba", "sbm", "sfn", "complete", "star", "path")
PER_TASK = 100
MAX_PROMPT_CHARS = 9000
SEEDS = {"test_fresh": 20261103, "test_large": 20261104}  # GraphQA 9876/5432/1234, kg_pretrain 20261022+, v3 20261102+
SIZES = {"test_fresh": None, "test_large": {"xl": np.arange(20, 35)}}


def from_row(row: dict, split: str, i: int) -> Record:
    ins = row["instruction"].removeprefix(GRL_PREFIX)
    context, q = ins.rsplit("Q: ", 1)
    *trace, last = row["response"].split("\n")
    if not last.startswith("Answer: "):
        raise ValueError(f"{split}/{i}: response does not end in an Answer line")
    rec = Record(id=f"graphqa/{split}/{i}", dataset="graphqa", split=split, task=row["task"],
                 question=q.removesuffix("\nA: "), answers=[last.removeprefix("Answer: ").removesuffix(".")],
                 context=context, preamble="", trace=trace or None,
                 meta={"supported": bool(trace), **{k: row[k] for k in ("cot_detail", "gen_algorithm") if k in row}})
    if prompt_text(rec) != ins:
        raise ValueError(f"{split}/{i}: rendered prompt differs from the instruction")
    return rec


def generate(split: str, seen: dict, stats: Counter) -> list[Record]:
    seed = SEEDS[split]
    if SIZES[split] is not None:
        graph_generator_utils._NUMBER_OF_NODES_RANGE = SIZES[split]
    pool = []
    for ai, alg in enumerate(ALGORITHMS):
        kw = {"er_max_sparsity": 0.3} if split == "test_large" else {}
        gs = graph_generator_utils.generate_graphs(number_of_graphs=400, algorithm=alg, directed=False,
                                                   random_seed=seed + ai, **kw)
        pool += [(alg, g) for g in gs]
    recs = []
    for ti, task in enumerate(TASKS):
        order = np.random.default_rng([seed, ti]).permutation(len(pool))
        n = 0
        for j in order:
            if n >= PER_TASK:
                break
            alg, graph = pool[j]
            random.seed(seed * 100 + ti * 10_000 + int(j))  # graph_task samples question nodes with `random`
            ex = getattr(graph_task, task)().prepare_examples_dict([graph.copy()], [alg], "adjacency")[0]
            row = {"instruction": GRL_PREFIX + ex["question"], "response": ex["answer"], "task": task}
            if task == "MaximumFlow":
                row = with_capacities(row, np.random.default_rng([seed, ti, int(j)]))
            if len(row["instruction"]) > MAX_PROMPT_CHARS:
                stats[f"{split}_too_long"] += 1
                continue
            g = structure(parse_graphqa(row["instruction"]).to_networkx())
            if g in seen[task]:
                stats[f"{split}_isomorphic"] += 1
                continue
            try:
                row["response"], _ = cot_response(row["instruction"], task, row["response"].strip(), "full")
            except ValueError:
                stats[f"{split}_trace_mismatch"] += 1
                continue
            seen[task].add(g)
            recs.append(from_row(row, split, len(recs)))
            recs[-1].meta["gen_algorithm"] = alg
            n += 1
        stats[f"{split}_{task}"] = n
    return recs


def main():
    stats = Counter()
    seen = defaultdict(IsoIndex)
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in open(SRC / f"{split}.jsonl")]
        recs = [from_row(r, split, i) for i, r in enumerate(rows)]
        for r in rows:
            seen[r["task"]].add(structure(parse_graphqa(r["instruction"]).to_networkx()))
        stats[split] = write_records(recs, OUT / f"{split}.jsonl")
    for split in SEEDS:
        stats[split] = write_records(generate(split, seen, stats), OUT / f"{split}.jsonl")
    print(dict(stats))


if __name__ == "__main__":
    main()
