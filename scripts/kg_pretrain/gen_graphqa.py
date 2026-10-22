"""Generate GraphQA ("Talk like a Graph") zero-shot examples for pretraining -> kg-graphqa_gen (native Q -> A).

Uses the task and encoder classes of google-research/graphqa (raw/graphqa_gen) directly, without TensorFlow.
The graph pools come from graph_generator_utils.generate_graphs with seeds distinct from the benchmark's
(train 9876, valid 5432, test 1234). Every (task, text encoder) cell gets ceil(N / cells) examples, each
from a graph sampled from the pool. Exact duplicates of the GraphQA val/test instructions in data/graphs are
removed by decontam.py.

    python -I scripts/kg_pretrain/gen_graphqa.py --n 1000000
"""
from __future__ import annotations

import argparse
import collections
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATS, DocWriter, write_json  # noqa: E402

sys.path.insert(0, str(RAW / "graphqa_gen" / "google-research"))
from graphqa import graph_generator_utils, graph_task  # noqa: E402

ALGORITHMS = ["er", "ba", "sbm", "sfn", "complete", "star", "path"]
ENCODERS = ["adjacency", "incident", "coauthorship", "friendship", "south_park", "got", "social_network",
            "politician", "expert"]
TASKS = {
    "edge_existence": graph_task.EdgeExistence, "node_degree": graph_task.NodeDegree,
    "node_count": graph_task.NodeCount, "edge_count": graph_task.EdgeCount,
    "connected_nodes": graph_task.ConnectedNodes, "cycle_check": graph_task.CycleCheck,
    "disconnected_nodes": graph_task.DisconnectedNodes, "reachability": graph_task.Reachability,
    "shortest_path": graph_task.ShortestPath, "maximum_flow": graph_task.MaximumFlow,
    "triangle_counting": graph_task.TriangleCounting, "node_classification": graph_task.NodeClassification,
}
G: dict = {}


def sbm_graph(rng: random.Random, random_state):
    """Same as graph_task_generator.generate_random_sbm_graph (node classification needs block labels)."""
    small, large = rng.uniform(0, 0.05), rng.uniform(0.6, 0.8)
    n = rng.choice(list(range(5, 20)))
    return nx.stochastic_block_model([n // 2, n // 2], [[large, small], [small, large]], seed=random_state)


def _cell(args):
    ci, task_name, encoder, n = args
    seed = G["seed"] * 1000 + ci
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    pool, algos = (G["sbm"], ["sbm"] * len(G["sbm"])) if task_name == "node_classification" else (G["graphs"], G["algos"])
    idx = [random.randrange(len(pool)) for _ in range(n)]
    graphs = [pool[i].copy() for i in idx]  # MaximumFlow adds edge weights in place
    ex = TASKS[task_name]().prepare_examples_dict(graphs, [algos[i] for i in idx], encoder)
    rows = []
    for k, v in ex.items():
        rows.append((v["question"], v["answer"], {"task": task_name, "encoder": encoder, "algorithm": v["algorithm"],
                                                  "nnodes": v["nnodes"], "nedges": v["nedges"]}))
    return ci, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000, help="total examples")
    ap.add_argument("--graphs-per-algorithm", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    graphs, algos = [], []
    for i, alg in enumerate(ALGORITHMS):
        gs = graph_generator_utils.generate_graphs(number_of_graphs=a.graphs_per_algorithm, algorithm=alg,
                                                   directed=False, random_seed=a.seed + i)
        graphs += gs
        algos += [alg] * len(gs)
    rng = random.Random(a.seed)
    rs = np.random.RandomState(a.seed % (2 ** 32))
    sbm = [sbm_graph(rng, rs) for _ in range(a.graphs_per_algorithm)]
    G.update(graphs=graphs, algos=algos, sbm=sbm, seed=a.seed)
    cells = [(t, e) for t in TASKS for e in ENCODERS]
    per = math.ceil(a.n / len(cells))
    jobs = [(ci, t, e, per) for ci, (t, e) in enumerate(cells)]
    counts, seen, dup = collections.Counter(), set(), 0
    results = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for ci, rows in pool.imap_unordered(_cell, jobs):
            results[ci] = rows
    with DocWriter("graphqa_gen") as w:
        for ci in range(len(jobs)):  # fixed order -> reproducible shards
            for q, ans, meta in results[ci]:
                if (q, ans) in seen:
                    dup += 1
                    continue
                seen.add((q, ans))
                w.write(f"{meta['task']}-{meta['encoder']}-{len(seen)}", ans, meta, instruction=q)
                counts[meta["task"]] += 1
    info = {"rows": len(seen), "duplicates_removed": dup, "per_cell": per, "graphs": len(graphs) + len(sbm),
            "seed": a.seed, "per_task": dict(counts)}
    write_json(STATS / "graphqa_gen_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
