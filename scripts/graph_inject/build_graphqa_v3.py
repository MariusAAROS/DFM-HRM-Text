"""GraphQA v3 for graph_inject (dataset graphqa_v3_cot): graphqa_v2_cot with a deduplicated train set and more large
TriangleCounting / MaximumFlow train rows; val/test unchanged (wiki: pages/hrm-graph/graphqa-v3.md).

    .venv/bin/python scripts/graph_inject/build_graphqa_v3.py [--src ...] [--dst ...] [--seed ...]

Train = v2-CoT train
- minus exact duplicate rows (the first copy is kept);
- minus TriangleCounting / MaximumFlow rows whose graph is isomorphic to a val/test graph of the same task;
- plus new rows from GraphQA's own generator (google-research graphqa graph_generator_utils, raw/graphqa_gen; er, ba,
  sbm, sfn, star, path, fresh seeds; complete graphs excluded), prompts from its adjacency encoder with GRL's
  "#### Question\\n" prefix. MaximumFlow capacities and gold as build_graphqa_v2.with_capacities. Full-detail CoT
  (graph_inject.cot) only: a candidate whose prompt + trace exceeds MAX_TOKENS with any model's tokenizer is rejected.
  Quotas per size bin (triangles; augmenting paths of the trace), and no algorithm fills more than half of a bin.
  A candidate is dropped if its graph is isomorphic to a val/test graph of any task, to a train graph of the same task,
  or to an earlier new row of the same task (graphs compared without self-loops and isolated nodes).
New rows carry `gen_algorithm`. val/test are copied byte for byte; build_report.json lists drops, quotas and the mix.
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "data/kg_pretrain/raw/graphqa_gen/google-research"))
from build_graphqa_cot import MAX_TOKENS, MODELS  # noqa: E402
from build_graphqa_v2 import with_capacities  # noqa: E402
from graph_inject.cot import cot_response  # noqa: E402
from graph_inject.structure import parse_graphqa  # noqa: E402
from graph_inject.tokenize import PromptTokenizer, tokenizer_checkpoint  # noqa: E402
from graphqa import graph_generator_utils, graph_text_encoder  # noqa: E402

SPLITS = ("train", "val", "test")
ALGORITHMS = ("er", "ba", "sbm", "sfn", "star", "path")  # GraphQA's generators without "complete"
TC_QUESTION = "Q: How many triangles are in this graph?\nA: "
MF_QUESTION = "Q: What is the maximum capacity of the flow from node %s to node %s?\nA: "
PREFIX = "#### Question\n"  # GRL's prompt (hrm-text-grl)
QUOTAS = {"TriangleCounting": {"6-20": 100, ">20": 400}, "MaximumFlow": {"4-8": 100, ">8": 400}}


def size_bin(task: str, n: int) -> str | None:
    """n = triangles (TriangleCounting) or augmenting paths of the trace (MaximumFlow); bins as the wiki's error audit."""
    lo = 6 if task == "TriangleCounting" else 4
    hi = 20 if task == "TriangleCounting" else 8
    return f"{lo}-{hi}" if lo <= n <= hi else f">{hi}" if n > hi else None


def structure(graph: nx.Graph) -> nx.Graph:
    """The graph compared for isomorphism: no self-loops, no isolated nodes."""
    return nx.Graph((u, v) for u, v in graph.edges() if u != v)


class IsoIndex:
    """Graphs up to isomorphism, bucketed by Weisfeiler-Lehman hash (equal for isomorphic graphs)."""

    def __init__(self):
        self.buckets = defaultdict(list)

    @staticmethod
    def _key(g):
        return g.number_of_nodes(), g.number_of_edges(), nx.weisfeiler_lehman_graph_hash(g)

    def __contains__(self, g) -> bool:
        return any(nx.is_isomorphic(g, h) for h in self.buckets[self._key(g)])

    def add(self, g):
        self.buckets[self._key(g)].append(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/graphs/graphqa/hrm-text-grl-v2-cot/standard")
    ap.add_argument("--dst", default="data/graphs/graphqa/hrm-text-grl-v3-cot/standard")
    ap.add_argument("--seed", type=int, default=20261102)  # GraphQA: 9876/5432/1234, kg_pretrain: 20261022+
    ap.add_argument("--graphs-per-algorithm", type=int, default=4000)
    args = ap.parse_args()
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise ValueError("--dst must differ from --src")
    rows = {k: [json.loads(line) for line in open(os.path.join(args.src, f"{k}.jsonl"))] for k in SPLITS}
    tokenizers = [PromptTokenizer(tokenizer_checkpoint(OmegaConf.load(f"graph_inject/configs/models/{m}.yaml").hf_dir))
                  for m in MODELS]

    def length(row) -> int:
        return max(len(t.encode_prompt(row["condition"], row["instruction"], None).ids)
                   + len(t.encode_response(row["condition"], row["instruction"], row["response"])) for t in tokenizers)

    graph_of = lambda r: structure(parse_graphqa(r["instruction"]).to_networkx())
    heldout_any, heldout_task = IsoIndex(), defaultdict(IsoIndex)
    for k in ("val", "test"):
        for r in rows[k]:
            heldout_any.add(graph_of(r))
            heldout_task[r["task"]].add(graph_of(r))
    instructions = {r["instruction"] for k in SPLITS for r in rows[k]}

    # existing train: drop exact duplicates and same-task isomorphs of val/test graphs
    train, seen, dropped, train_task = [], set(), defaultdict(list), defaultdict(IsoIndex)
    for i, r in enumerate(rows["train"]):
        if r["instruction"] in seen:
            dropped["exact_duplicate"].append(i)
            continue
        seen.add(r["instruction"])
        if r["task"] in QUOTAS:
            g = graph_of(r)
            if g in heldout_task[r["task"]]:
                dropped["isomorphic_to_val_test_same_task"].append(i)
                continue
            train_task[r["task"]].add(g)
        train.append(r)
    print(f"train: {len(rows['train'])} -> {len(train)} v2 rows; dropped "
          + ", ".join(f"{k} {len(v)} ({dict(Counter(rows['train'][i]['task'] for i in v))})" for k, v in dropped.items()))

    # new rows
    new, report = [], {}
    for ti, (task, quota) in enumerate(QUOTAS.items()):
        candidates = []
        for ai, alg in enumerate(ALGORITHMS):
            graphs = graph_generator_utils.generate_graphs(number_of_graphs=args.graphs_per_algorithm, algorithm=alg,
                                                           directed=False, random_seed=args.seed + 100 * ti + ai)
            candidates += [(alg, g) for g in graphs]
        order = np.random.default_rng([args.seed, ti]).permutation(len(candidates))
        pick = random.Random(args.seed + ti)
        filled, mix, reasons, new_task = Counter(), defaultdict(Counter), Counter(), IsoIndex()
        cap = {b: q // 2 for b, q in quota.items()}
        for j in order:
            if all(filled[b] >= q for b, q in quota.items()):
                break
            alg, graph = candidates[j]
            g = structure(graph)
            n_nodes = graph.number_of_nodes()
            if g.number_of_edges() == 0:  # no triangle, no flow (and no edge line in the prompt)
                reasons["no_edges"] += 1
                continue
            if g.number_of_edges() == n_nodes * (n_nodes - 1) // 2:
                reasons["complete"] += 1
                continue
            row = {"instruction": PREFIX + graph_text_encoder.encode_graph(graph, "adjacency"), "response": "",
                   "condition": "direct", "task": task}
            if task == "TriangleCounting":
                n = sum(nx.triangles(g).values()) // 3
                gold = f"{n}."
                row["instruction"] += TC_QUESTION
            else:
                s, t = pick.sample(list(graph.nodes()), k=2)
                row["instruction"] += MF_QUESTION % (s, t)
                row = with_capacities({**row, "response": "0."}, np.random.default_rng([args.seed, ti, int(j)]))
                gold = row["response"]
            response, _ = cot_response(row["instruction"], task, gold, "full")
            if task == "MaximumFlow":
                n = response.count("Path ")
            b = size_bin(task, n)
            if b is None or filled[b] >= quota[b]:
                reasons["bin_full_or_outside"] += 1
                continue
            if mix[b][alg] >= cap[b]:
                reasons["algorithm_cap"] += 1
                continue
            if row["instruction"] in instructions or g in heldout_any:
                reasons["isomorphic_to_val_test"] += 1
                continue
            if g in train_task[task]:
                reasons["isomorphic_to_train"] += 1
                continue
            if g in new_task:
                reasons["isomorphic_to_new"] += 1
                continue
            row = {**row, "response": response, "cot_detail": "full", "cot_gold_replaced": False, "gen_algorithm": alg}
            if length(row) > MAX_TOKENS:
                reasons["too_long_at_full_detail"] += 1
                continue
            new.append(row)
            new_task.add(g)
            instructions.add(row["instruction"])
            filled[b] += 1
            mix[b][alg] += 1
        short = {b: q - filled[b] for b, q in quota.items() if filled[b] < q}
        if short:
            raise RuntimeError(f"{task}: candidates exhausted, bins short by {short}; raise --graphs-per-algorithm")
        report[task] = {"quota": quota, "algorithm_mix": {b: dict(c) for b, c in mix.items()},
                        "rejected": dict(reasons), "candidates": len(candidates)}
        print(f"{task}: +{sum(filled.values())} rows {dict(filled)}; mix { {b: dict(c) for b, c in mix.items()} }; "
              f"rejected {dict(reasons)}")

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "train.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train + new)
    for k in ("val", "test"):
        shutil.copy(os.path.join(args.src, f"{k}.jsonl"), os.path.join(args.dst, f"{k}.jsonl"))
    lengths = defaultdict(list)
    for r in new:
        lengths[r["task"]].append(length(r))
    report |= {"src": args.src, "seed": args.seed, "graphs_per_algorithm": args.graphs_per_algorithm,
               "train_rows": {"v2_kept": len(train), "new": len(new), "total": len(train) + len(new)},
               "v2_train_dropped": {k: v for k, v in dropped.items()},
               "new_tokens": {t: {"mean": float(np.mean(v)), "max": int(max(v)), "sum": int(sum(v))}
                              for t, v in lengths.items()}}
    with open(os.path.join(args.dst, "build_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"train: {len(train)} v2 + {len(new)} new = {len(train) + len(new)} rows; new-row tokens "
          + ", ".join(f"{t} mean {np.mean(v):.0f} max {max(v)}" for t, v in lengths.items()))


if __name__ == "__main__":
    main()
