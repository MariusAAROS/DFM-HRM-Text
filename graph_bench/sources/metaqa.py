"""MetaQA -> bench records with hop-by-hop traces (Tier 1; 1-2 hop train, 3-hop = within-task OOD depth).

Rows are GRL's (data/graphs/metaqa/baseline-grl, single-answer questions, `gold` and `retrieved` subgraphs), so
test is the same 1,197 questions as the earlier MetaQA runs. The Graph-Representation-Learning-for-LLM generator is
re-run with its seed only to recover each row's question record (topic entity, relation chain from the qtype): its
`retrieved` subgraphs iterate Python sets of strings, so they depend on the process hash seed and are not
reproducible (2026-11-02: every gold prompt is identical, 650 of 2,996 retrieved ones are; question, answer and
strategy align on every row). The facts are therefore parsed from the baseline prompt text (9 known relations, so
names with commas parse unambiguously), and the rendered prompt is checked byte-identical to it.

Trace: start at the topic entity and follow the qtype's relation chain over the prompt's edges only, citing each edge
used; the topic is left out of every hop after the first (MetaQA's convention). Kept when the final set equals the gold.

    .venv/bin/python -m graph_bench.sources.metaqa
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from graph_bench.schema import Record, prompt_text, render_fact, write_records

REPO = Path(__file__).resolve().parents[2]
GRL = Path("/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM")
MQA = REPO / "data/graphs/metaqa"
OUT = REPO / "data/graphs/bench/metaqa"
PREAMBLE = "In a knowledge graph, (s, p, o) means that entity s is linked to entity o by relation p."
MAX_LIST = 25

sys.path.insert(0, str(GRL))
from src.datasets.metaqa import dataset_generator as grl_gen  # noqa: E402
from src.datasets.metaqa.kb_index import KBIndex  # noqa: E402
from src.datasets.metaqa.question_types import FORWARD  # noqa: E402


class TracingKBIndex(KBIndex):
    def serialize(self, spine, core, filler, max_triples, rng, core_cap=None):
        text, nnodes, nedges = self._serialize(spine, core, filler, max_triples, rng, core_cap)
        return text, nnodes, nedges

    def _serialize(self, spine, core, filler, max_triples, rng, core_cap=None):
        # GRL's serialize, unchanged except that it keeps the edge list.
        from src.datasets.metaqa.kb_index import PREAMBLE as GRL_PREAMBLE, format_edge
        kept = list(spine[:max_triples])
        cap = min(core_cap if core_cap is not None else max_triples, max_triples)
        room = max(0, cap - len(kept))
        kept += rng.sample(core, min(room, len(core)))
        room = max(0, max_triples - len(kept))
        kept += rng.sample(filler, min(room, len(filler)))
        rng.shuffle(kept)
        self.last_edges = list(kept)
        nodes = {node for subject, _, obj in kept for node in (subject, obj)}
        return GRL_PREAMBLE + "\n".join(format_edge(edge) for edge in kept) + "\n", len(nodes), len(kept)


def grl_rows(kb: TracingKBIndex) -> dict[str, list[dict]]:
    """GRL's main() (default arguments), also keeping each row's record and prompt edges."""
    grl_gen.DATA_DIR = str(MQA)
    rng = random.Random(grl_gen.SEED)
    sizes = {"train": grl_gen.N_TRAIN, "val": grl_gen.N_VAL, "test": grl_gen.N_TEST}
    out = {}
    for split, total in sizes.items():
        records = []
        for hops in grl_gen.HOPS:
            records.extend(grl_gen.load_records(hops, split, total // len(grl_gen.HOPS), rng))
        rng.shuffle(records)
        rows = []
        for record in records:
            for strategy in grl_gen.STRATEGIES:
                built = kb.build_subgraph(record, strategy, max_triples=grl_gen.MAX_TRIPLES,
                                          max_nodes=grl_gen.MAX_NODES, rng=rng)
                if built is None:
                    continue
                text, nnodes, nedges = built
                if strategy == "gold" and not all(a in text for a in record["answers"]):
                    continue
                rows.append({"algorithm": strategy, "question": f"{text}Q: {record['question']}\nA: ",
                             "answer": f"{record['answer_text']}.", "record": record,
                             "edges": list(kb.last_edges)})
        out[split] = rows
    return out


RELATIONS = ("directed by", "written by", "starred actors", "has genre", "in language", "release year", "has tags",
             "has imdb rating", "has imdb votes")


def parse_facts(prompt: str) -> list[list[str]]:
    """'(s, rel, o)' lines of a GRL MetaQA prompt; rel is one of RELATIONS, so s and o may contain ', '."""
    block = prompt.split("The facts are:\n", 1)[1].rsplit("Q: ", 1)[0]
    facts = []
    for line in block.splitlines():
        if not line:
            continue
        body = line[1:-1]
        hits = [(body.index(f", {rel}, "), rel) for rel in RELATIONS if f", {rel}, " in body]
        if len(hits) != 1:
            raise ValueError(f"cannot parse fact {line!r}")
        k, rel = hits[0]
        facts.append([body[:k], rel, body[k + len(rel) + 4:]])
    return facts


def trace(record: dict, facts: list[list[str]]) -> list[str] | None:
    topic, chain, gold = record["topic"], record["chain"], set(record["answers"])
    frontier, lines = {topic}, [f"Start: {topic}"]
    for k, (relation, direction) in enumerate(chain, 1):
        rel = relation.replace("_", " ")
        if direction == FORWARD:
            used = [f for f in facts if f[1] == rel and f[0] in frontier]
            nxt = {f[2] for f in used}
        else:
            used = [f for f in facts if f[1] == rel and f[2] in frontier]
            nxt = {f[0] for f in used}
        # MetaQA leaves the start entity out of every hop after the first ("films that share actors with X" excludes
        # X, as does "writers of films directed by X"), but not out of hop 1 ("who acted in the movie Lemmy" -> Lemmy).
        # This rule reproduces every gold-subgraph row whose prompt is unambiguous (2026-11-02).
        dropped = k >= 2 and topic in nxt
        if dropped:
            nxt.discard(topic)
        if not nxt:
            return None
        lines.append(f"Hop {k}: {rel} ({'forward' if direction == FORWARD else 'backward'})")
        lines += [f"   {render_fact(f)}" for f in used[:MAX_LIST]]
        if dropped:
            lines.append(f"   (leave out the start entity {topic})")
        frontier = nxt
        shown = sorted(frontier)
        lines.append("   -> " + ", ".join(shown[:MAX_LIST]) + (f", ... ({len(shown)} in total)"
                                                              if len(shown) > MAX_LIST else ""))
    return lines if frontier == gold else None


def main():
    import pickle
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-cache", default=None)
    args = ap.parse_args()
    if args.rows_cache and Path(args.rows_cache).exists():
        rows_by_split = pickle.load(open(args.rows_cache, "rb"))
    else:
        kb = TracingKBIndex(str(MQA / "kb.txt"))
        rows_by_split = grl_rows(kb)
        if args.rows_cache:
            pickle.dump(rows_by_split, open(args.rows_cache, "wb"))
    stats = Counter()
    for split, rows in rows_by_split.items():
        baseline = json.load(open(MQA / f"baseline-grl/standard/{split}.json"))
        if len(baseline) != len(rows) or any(
                b["question"].rsplit("Q: ", 1)[1] != r["question"].rsplit("Q: ", 1)[1] or b["answer"] != r["answer"]
                or b["algorithm"] != r["algorithm"] for b, r in zip(baseline, rows)):
            raise SystemExit(f"{split}: re-generated records do not align with data/graphs/metaqa/baseline-grl; stop")
        recs = []
        for i, (b, r) in enumerate(zip(baseline, rows)):
            rec_src = r["record"]
            facts = parse_facts(b["question"])
            tr = trace(rec_src, facts)
            stats[f"{split}_{rec_src['task']}_{'trace' if tr else 'no_trace'}"] += 1
            ents = sorted({x for f in facts for x in (f[0], f[2])})
            rec = Record(id=f"metaqa/{split}/{i}", dataset="metaqa", split=split, task=rec_src["task"],
                         question=rec_src["question"], answers=list(rec_src["answers"]),
                         answer_format="single" if len(rec_src["answers"]) == 1 else "set",
                         facts=facts, preamble=PREAMBLE, trace=tr, entities=ents,
                         meta={"algorithm": r["algorithm"], "topic": rec_src["topic"], "supported": tr is not None,
                               "chain": [[rel, d] for rel, d in rec_src["chain"]]})
            if prompt_text(rec) != b["question"]:
                raise SystemExit(f"{rec.id}: rendered prompt differs from the baseline question; stop")
            recs.append(rec)
        write_records(recs, OUT / f"{split}.jsonl")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
