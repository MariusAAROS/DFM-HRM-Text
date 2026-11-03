"""WebQSP and CWQ (Freebase) -> bench records, from RoG's per-question subgraphs (rmanluo/RoG-webqsp, RoG-cwq).

Tier 1 (wiki: mimir-graph-benchmark): WebQSP trains, CWQ (compositions of WebQSP patterns) is a near-OOD held-out
task. RoG's subgraphs hold ~3-5k triples, so each prompt is cut down to MAX_FACTS:

1. Relation paths: the (relation, direction) sequences of the shortest paths (<= MAX_HOPS) from a question entity to
   each answer entity, as RoG defines its "ground-truth reasoning paths".
2. Exact path set: walking a relation path over the subgraph gives a set of entities (minus the question entities, as
   CVT round trips return them); among the paths whose walk stays inside the gold set, one whose walk is exactly the
   gold set, else a greedy cover. If the cover gives exactly the gold set, the question is `supported` and gets a trace.
   Questions with a constraint the path does not express ("the first", a year filter) walk to extra entities and get
   no trace, which is the honest outcome.
3. Prompt facts: every triple of those walks (or, without an exact set, the shortest-path triples that reach the
   answers), then distractors incident to the question entities and walk nodes, up to MAX_FACTS, shuffled by id.
   A supported question whose walks need more than MAX_WALK_FACTS triples loses its trace (the prompt cannot hold it).

Trace: like MetaQA's, one block per relation path (hop, cited prompt facts, resulting entities), then the union.
Freebase CVT nodes keep their ids (m.0k8nh0b): they are the intermediate nodes of 2-hop facts.

    .venv/bin/python -m graph_bench.sources.rog --dataset webqsp    (and --dataset cwq)
"""
import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict, deque
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

from graph_bench.schema import Record, render_fact, write_records

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/graphs/bench/raw"
OUT = REPO / "data/graphs/bench"
MAX_HOPS = 4
MAX_FACTS = 60
MAX_WALK_FACTS = 40
MAX_LIST = 25
MAX_PATHS = 8
SPLITS = {"train": "train", "validation": "val", "test": "test"}
PREAMBLE = "In a knowledge graph, (s, p, o) means that entity s is linked to entity o by relation p."
_CVT = re.compile(r"^[mg]\.[0-9a-z_]+$")
_LITERAL = re.compile(r"^-?[\d.,:/ -]+$")


def _adjacency(graph):
    out, inn = defaultdict(list), defaultdict(list)
    for i, (s, r, o) in enumerate(graph):
        out[s].append((r, o, i))
        inn[o].append((r, s, i))
    return out, inn


def _shortest_relation_paths(q, targets, out, inn):
    """BFS from q; for every target reached, the relation paths of its shortest paths (capped)."""
    parents = {q: []}
    depth = {q: 0}
    frontier = deque([q])
    while frontier:
        u = frontier.popleft()
        if depth[u] >= MAX_HOPS:
            continue
        for direction, edges in ((1, out[u]), (-1, inn[u])):
            for r, v, _ in edges:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    parents[v] = [(u, r, direction)]
                    frontier.append(v)
                elif depth[v] == depth[u] + 1 and len(parents[v]) < 4:
                    parents[v].append((u, r, direction))

    def rel_paths(v):
        if v == q:
            return [()]
        res = []
        for u, r, d in parents[v]:
            for p in rel_paths(u):
                res.append(p + ((r, d),))
                if len(res) >= MAX_PATHS:
                    return res
        return res

    found = {}
    for t in sorted(targets):  # sorted: set order varies with the per-process string hash seed
        if t in depth and t != q:
            found[t] = rel_paths(t)
    return found


def _walk(q, path, out, inn):
    """Entities reached by following `path` from q, with the triple ids used per hop."""
    frontier, used = {q}, []
    for r, d in path:
        nxt, hop = set(), []
        for u in sorted(frontier):
            for rr, v, i in (out[u] if d == 1 else inn[u]):
                if rr == r:
                    nxt.add(v)
                    hop.append(i)
        used.append(hop)
        frontier = nxt
        if not frontier:
            break
    return frontier, used


def _cover(cands, gold: set):
    """Fewest paths whose walks give the gold set: one exact path if any (shortest, most forward hops first), else a
    greedy cover; the candidates themselves if they cannot cover it (the question is then unsupported anyway)."""
    key = lambda c: (len(c[1]), sum(d == -1 for _, d in c[1]))
    exact = sorted((c for c in cands if c[2] == gold), key=key)
    if exact:
        return exact[:1]
    chosen, covered = [], set()
    for _ in range(len(cands)):
        best = max(cands, key=lambda c: (len(c[2] - covered), -key(c)[0], -key(c)[1]), default=None)
        if best is None or not best[2] - covered:
            break
        chosen.append(best)
        covered |= best[2]
    return chosen if covered == gold else cands


def _is_entity(name: str) -> bool:
    return not _CVT.match(name) and not _LITERAL.match(name)


def build(row: dict, dataset: str, split: str) -> tuple[Record | None, str]:
    graph = [tuple(t) for t in row["graph"]]
    out, inn = _adjacency(graph)
    nodes = set(out) | set(inn)
    qs = [q for q in row["q_entity"] if q in nodes]
    gold = sorted(set(row["a_entity"]) or set(row["answer"]))
    if not qs:
        return None, "no_question_entity_in_graph"
    if not gold:
        return None, "no_answer"
    reachable = [a for a in gold if a in nodes]

    # 1-2: relation paths and the exact path set
    chosen, walk_ids, union, path_ids = [], set(), set(), set()
    for q in qs:
        for t, paths in _shortest_relation_paths(q, set(reachable), out, inn).items():
            for p in paths:
                ids = []
                for u, r, d in _path_edges(q, p, t, out, inn):
                    ids.append(u)
                path_ids.update(ids)
                key = (q, p)
                if key in {(c[0], c[1]) for c in chosen}:
                    continue
                ends, used = _walk(q, p, out, inn)
                ends -= set(qs)  # a question entity is never its own answer (CVT round trips: spouse_s)
                if ends and ends <= set(gold):
                    chosen.append((q, p, ends, used))
    chosen = _cover(chosen, set(gold))
    for q, p, ends, used in chosen:
        union |= ends
        for hop in used:
            walk_ids.update(hop)
    supported = union == set(gold) and len(walk_ids) <= MAX_WALK_FACTS

    # 3: prompt facts
    rng = random.Random(row["id"])
    core = sorted(walk_ids) if supported else sorted(path_ids)[:MAX_WALK_FACTS]
    near = {n for i in core for n in (graph[i][0], graph[i][2])} | set(qs)
    pool = [i for n in sorted(near) for _, _, i in out[n] + inn[n] if i not in set(core)]
    pool = sorted(set(pool))
    fill = rng.sample(pool, min(len(pool), max(0, MAX_FACTS - len(core))))
    ids = list(core) + fill
    rng.shuffle(ids)
    facts = [list(graph[i]) for i in ids]
    prompt_facts = {tuple(f) for f in facts}

    trace = None
    if supported:
        trace = []
        for k, (q, p, ends, used) in enumerate(chosen, 1):
            trace.append(f"Path {k} from {q}: " + " -> ".join(f"{r} ({'forward' if d == 1 else 'backward'})"
                                                             for r, d in p))
            for hop, (r, d) in zip(used, p):
                cited = [graph[i] for i in hop if graph[i] in prompt_facts]
                trace += [f"   {render_fact(list(t))}" for t in cited[:MAX_LIST]]
            shown = sorted(ends)
            trace.append("   -> " + ", ".join(shown[:MAX_LIST]) + (f", ... ({len(shown)} in total)"
                                                                  if len(shown) > MAX_LIST else ""))
        if len(chosen) > 1:
            trace.append("Together: " + ", ".join(sorted(union)))
    ents = sorted({x for f in facts for x in (f[0], f[2]) if _is_entity(x)})
    rec = Record(id=f"{dataset}/{split}/{row['id']}", dataset=dataset, split=split,
                 task=f"{len(chosen[0][1])}hop" if supported else "unsupported", question=row["question"],
                 answers=gold, answer_format="single" if len(gold) == 1 else "set", facts=facts, preamble=PREAMBLE,
                 trace=trace, entities=ents,
                 meta={"source_id": row["id"], "q_entity": qs, "supported": supported,
                       "answers_in_prompt": sum(a in {x for f in facts for x in (f[0], f[2])} for a in gold),
                       "n_answers": len(gold)})
    return rec, "supported" if supported else "unsupported"


def _path_edges(q, path, target, out, inn):
    """Triple ids of one instance of `path` from q that ends at target (DFS)."""
    def dfs(u, k):
        if k == len(path):
            return [] if u == target else None
        r, d = path[k]
        for rr, v, i in (out[u] if d == 1 else inn[u]):
            if rr == r:
                rest = dfs(v, k + 1)
                if rest is not None:
                    return [(i, r, d)] + rest
        return None
    return dfs(q, 0) or []


def _job(args):
    row, dataset, split = args
    try:
        return build(row, dataset, split)
    except RecursionError:
        return None, "recursion"


def _rows(files: list[str], limit: int | None, batch_size: int = 64):
    """Stream the parquet rows: CWQ train is 8.1 GB of Arrow (~30 GB as Python objects), so the table is never
    materialised (reading it whole and forking 32 workers off that heap ran the container out of memory, 2026-11-02)."""
    n = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                if limit is not None and n >= limit:
                    return
                n += 1
                yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["webqsp", "cwq"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT, help="Output root (<out>/<dataset>/<split>.jsonl).")
    args = ap.parse_args()
    # Fork the workers before reading any data, so they do not inherit (and copy-on-write) a large parent heap.
    with Pool(args.workers, maxtasksperchild=200) as pool:
        for src_split, split in SPLITS.items():
            files = sorted(glob.glob(str(RAW / args.dataset / "data" / f"{src_split}-*.parquet")))
            stats = Counter()

            def records():
                jobs = ((r, args.dataset, split) for r in _rows(files, args.limit))
                for rec, reason in pool.imap(_job, jobs, chunksize=4):  # ordered: same output as before
                    stats[reason] += 1
                    if rec is not None:
                        yield rec

            n = write_records(records(), args.out / args.dataset / f"{split}.jsonl")
            print(f"{args.dataset} {split}: {n} records {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
