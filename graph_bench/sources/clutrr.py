"""CLUTRR (gen_train234_test2to10) -> bench records over the kinship graph, with the proof as trace.

Tier 1, far-OOD held-out task (wiki: mimir-graph-benchmark): kinship composition, a skill no train set covers;
train chains have 2-4 edges, test 2-10, so test also measures length generalisation (task = "k<n>" edges).

The prompt is the graph form, not the story: one fact per story edge, "(A, son, B)" meaning B is the son of A
(CLUTRR's edge convention), shuffled so the chain order is not given away. The question asks what the second query
entity is to the first. The trace is CLUTRR's own proof_state, bottom-up: each step composes two known relations
into one (a kinship rule, the only knowledge not in the prompt), until the queried pair.

    .venv/bin/python -m graph_bench.sources.clutrr
"""
import ast
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from graph_bench.schema import Record, write_records

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/graphs/bench/raw/clutrr/gen_train234_test2to10"
OUT = REPO / "data/graphs/bench/clutrr"
PREAMBLE = ("In a family graph, (A, r, B) means that B is the r of A. For example, (Ann, son, Tom) means that Tom is "
            "Ann's son.")
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def proof_steps(proof_state: str, facts: set) -> list[str] | None:
    """Bottom-up order of CLUTRR's composition steps (every premise a fact or an earlier conclusion)."""
    steps = []
    for d in ast.literal_eval(proof_state):
        for concl, premises in d.items():
            steps.append((concl, premises))
    known, ordered = set(facts), []
    while steps:
        ready = [s for s in steps if all(tuple(p) in known for p in s[1])]
        if not ready:
            return None
        for s in ready:
            steps.remove(s)
            known.add(tuple(s[0]))
            ordered.append(s)
    return [f"({a}, {r1}, {b}) and ({b2}, {r2}, {c}) give ({x}, {r}, {y})"
            for (x, r, y), ((a, r1, b), (b2, r2, c)) in ordered]


def main():
    stats = Counter()
    for src, split in SPLITS.items():
        df = pd.read_csv(RAW / f"{src}.csv")
        recs = []
        for _, row in df.iterrows():
            names = [g.split(":")[0] for g in row.genders.split(",")]
            edges = ast.literal_eval(row.story_edges)
            types = ast.literal_eval(row.edge_types)
            facts = [[names[a], t, names[b]] for (a, b), t in zip(edges, types)]
            qa, qb = ast.literal_eval(row.query)
            steps = proof_steps(row.proof_state, {tuple(f) for f in facts})
            final_ok = bool(steps) and steps[-1].endswith(f"give ({qa}, {row.target_text}, {qb})")
            stats[f"{split}_{'trace' if final_ok else 'no_trace'}"] += 1
            random.Random(row.id).shuffle(facts)
            recs.append(Record(id=f"clutrr/{split}/{row.id}", dataset="clutrr", split=split, task=f"k{len(types)}",
                               question=f"What is {qb} to {qa}?", answers=[row.target_text], facts=facts,
                               preamble=PREAMBLE, trace=steps if final_ok else None, entities=sorted(set(names)),
                               meta={"f_comb": row.f_comb, "supported": final_ok}))
        write_records(recs, OUT / f"{split}.jsonl")
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
