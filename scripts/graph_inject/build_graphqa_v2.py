"""GraphQA v2 for graph_inject: MaximumFlow with printed capacities, val/test without train duplicates.

    .venv/bin/python scripts/graph_inject/build_graphqa_v2.py [--src ...] [--dst ...]

- MaximumFlow: GraphQA's gold implies random integer capacities in 1-10 per edge, but no prompt prints them
  (wiki: graphqa-unsaturated-tasks). Every MaximumFlow row (all splits) gets fresh capacities, uniform in 1-10 with
  rng([0, split, row]), printed as "(u, v, c)" (graph_inject.structure.EDGE_TUPLE parses the third number), and the
  gold is recomputed with networkx on the undirected graph (self-loops dropped).
- Dedup: val/test rows whose original instruction is a train instruction are dropped (GraphQA reuses graphs across
  splits; 61 test / 78 val rows). Train is unchanged. dedup_report.json lists the kept and dropped original ids.
"""
import argparse
import json
import os
import re

import networkx as nx
import numpy as np

SPLITS = ("train", "val", "test")
UNDIRECTED = "(i,j) means that node i and node j are connected with an undirected edge."
WITH_CAPACITY = "(i,j,c) means that node i and node j are connected with an undirected edge of capacity c."
EDGE = re.compile(r"\((\d+), (\d+)\)")
FLOW_QUESTION = re.compile(r"from node (\d+) to node (\d+)")


def with_capacities(row: dict, rng: np.random.Generator) -> dict:
    ins = row["instruction"]
    if UNDIRECTED not in ins:
        raise ValueError("unexpected MaximumFlow header")
    head, rest = ins.split("The edges in G are:", 1)
    edge_text, question = rest.split("\nQ:", 1)
    edges = [(int(u), int(v)) for u, v in EDGE.findall(edge_text)]
    caps = rng.integers(1, 11, size=len(edges))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(re.findall(r"\d+", re.search(r"among nodes ([^\n]*?)\.\n", head).group(1)))))
    for (u, v), c in zip(edges, caps):
        if u != v:
            graph.add_edge(u, v, capacity=int(c))
    s, t = map(int, FLOW_QUESTION.search(question).groups())
    flow = int(nx.maximum_flow_value(graph, s, t)) if nx.has_path(graph, s, t) else 0
    edge_str = " ".join(f"({u}, {v}, {c})" for (u, v), c in zip(edges, caps))
    instruction = head.replace(UNDIRECTED, WITH_CAPACITY) + f"The edges in G are: {edge_str}.\nQ:" + question
    response = re.sub(r"\d+", str(flow), row["response"], count=1)
    return {**row, "instruction": instruction, "response": response}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/graphs/graphqa/hrm-text-grl/standard")
    ap.add_argument("--dst", default="data/graphs/graphqa/hrm-text-grl-v2/standard")
    args = ap.parse_args()
    rows = {k: [json.loads(line) for line in open(os.path.join(args.src, f"{k}.jsonl"))] for k in SPLITS}
    train_instructions = {r["instruction"] for r in rows["train"]}
    os.makedirs(args.dst, exist_ok=True)
    report = {}
    for split_id, k in enumerate(SPLITS):
        kept, dropped, out = [], [], []
        for i, r in enumerate(rows[k]):
            if k != "train" and r["instruction"] in train_instructions:
                dropped.append(i)
                continue
            kept.append(i)
            if r["task"] == "MaximumFlow":
                r = with_capacities(r, np.random.default_rng([0, split_id, i]))
            out.append(r)
        with open(os.path.join(args.dst, f"{k}.jsonl"), "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
        report[k] = {"n_src": len(rows[k]), "n_kept": len(kept), "kept_ids": kept, "dropped_ids": dropped}
        print(f"{k}: {len(rows[k])} -> {len(kept)} rows ({len(dropped)} train duplicates dropped)")
    with open(os.path.join(args.dst, "dedup_report.json"), "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
