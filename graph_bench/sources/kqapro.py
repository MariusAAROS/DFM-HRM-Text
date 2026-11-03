"""KQA Pro -> bench records with KoPL execution traces (Tier 1, train; wiki: mimir-graph-benchmark).

Rows are GRL's (data/graphs/kqapro/baseline): Graph-Representation-Learning-for-LLM's generator is re-run with its seed
(42) and a KBIndex subclass that also keeps the structured facts, and every row is checked to be byte-identical to
the baseline file before it is used. So test is the same 1,134 questions as data/graphs/kqapro/hrm-text-grl, and
scores stay comparable with every earlier KQA Pro run.

Trace: the question's KoPL program, executed by the reference engine (kopl 0.0.5) on a KB that holds only the facts
in the prompt, with no concept hierarchy (so FilterConcept only sees "instance of" facts the prompt states). Each step
names its function, the prompt facts it reads, and its result; the trace is kept only when the final result equals
the gold answer and no entity-set step was empty (an empty Find / Relate that still lands on the gold is luck).
meta.supported marks those rows. Measured 2026-11-02 on the `gold` rows of train+test: supported for Basic 723/985,
Comparison 411/777, Verify 281/499, Qualifier 316/751, Multihop 30/98, Count 13/504 (adding the full concept
hierarchy changes these by at most 34). So most Count golds, and about half of the others, cannot be derived from
GRL's 60-fact prompt; report KQA Pro also on the supported subset.

    .venv/bin/python -m graph_bench.sources.kqapro
"""
import ast
import contextlib
import copy
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

from graph_bench.schema import KG_PREAMBLE, QUALIFIER_NOTE, Record, prompt_text, render_fact, write_records
from graph_bench.scoring import normalize

REPO = Path(__file__).resolve().parents[2]
GRL = Path("/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM")
KQA = REPO / "data/graphs/kqapro"
OUT = REPO / "data/graphs/bench/kqapro"
MAX_LIST = 25

sys.path.insert(0, str(GRL))
from src.datasets.kqapro import dataset_generator as grl_gen  # noqa: E402
from src.datasets.kqapro.kb_index import KBIndex, format_qualifiers, format_value  # noqa: E402
from src.datasets.kqapro.question_types import derive_task, program_length  # noqa: E402


class Fact(str):
    """A rendered fact line that remembers what it is: ('instance', eid, cid) / ('attr', eid, i) / ('rel', eid, i)."""
    ref: tuple


def _fact(text: str, ref: tuple) -> Fact:
    f = Fact(text)
    f.ref = ref
    return f


class TracingKBIndex(KBIndex):
    """GRL's KBIndex, whose fact texts carry their KB reference (same text, same RNG calls)."""

    def _triples_for(self, eid, selected, terms, focus):
        entity = self.entities[eid]
        name = entity["name"]
        in_focus = eid in focus
        for cid in entity["instanceOf"]:
            concept = self.concept_name(cid)
            yield in_focus and concept in terms, _fact(f"({name}, instance of, {concept})", ("instance", eid, cid))
        for i, attr in enumerate(entity["attributes"]):
            value = format_value(attr["value"])
            named_key = attr["key"] in terms
            relevant = (named_key or value in terms or any(qk in terms for qk in attr["qualifiers"]))
            yield relevant and (in_focus or named_key), _fact(
                f"({name}, {attr['key']}, {value}){format_qualifiers(attr['qualifiers'])}", ("attr", eid, i))
        for i, rel in enumerate(entity["relations"]):
            if rel["direction"] != "forward" or rel["object"] not in selected:
                continue
            obj = self.node_names[rel["object"]]
            touches_focus = in_focus or rel["object"] in focus
            core = touches_focus and (rel["predicate"] in terms or obj in terms or name in terms)
            yield core, _fact(f"({name}, {rel['predicate']}, {obj}){format_qualifiers(rel['qualifiers'])}",
                              ("rel", eid, i))

    def serialize(self, selected, terms, focus, answer, max_triples, rng):
        # GRL's serialize, unchanged except that it also returns the kept Fact objects.
        must, core, distractors = [], [], []
        for eid in sorted(selected):
            if eid not in self.entities:
                continue
            for is_core, text in self._triples_for(eid, selected, terms, focus):
                if answer and answer in text:
                    must.append((0 if eid in focus else 1 if is_core else 2, eid, text))
                elif is_core:
                    core.append((eid, text))
                else:
                    distractors.append((eid, text))
        must.sort(key=lambda item: item[0])
        kept = [(eid, text) for _, eid, text in must[:max(1, max_triples // 3)]]
        triples = kept + core[:max(0, max_triples - len(kept))]
        budget = max_triples - len(triples)
        if budget > 0:
            triples += rng.sample(distractors, min(budget, len(distractors)))
        rng.shuffle(triples)
        nnodes = len({eid for eid, _ in triples})
        from src.datasets.kqapro.kb_index import PREAMBLE
        text = PREAMBLE + "\n".join(t for _, t in triples) + "\n"
        self.last_facts = [t for _, t in triples]
        return text, nnodes, len(triples)


def grl_rows(kb: TracingKBIndex) -> dict[str, list[dict]]:
    """GRL's main() for --model-type baseline, also keeping each row's record, program and Fact list."""
    from sklearn.model_selection import train_test_split
    rng = random.Random(grl_gen.SEED)
    official_train = json.load(open(KQA / "original/train.json"))
    official_val = json.load(open(KQA / "original/val.json"))
    pool = rng.sample(official_train, min(grl_gen.N_TRAIN_POOL, len(official_train)))
    train_records, val_records = train_test_split(pool, test_size=grl_gen.VAL_FRACTION, random_state=grl_gen.SEED)
    test_records = rng.sample(official_val, min(grl_gen.N_TEST, len(official_val)))
    out = {}
    for split, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        rows = []
        for record in records:
            if not isinstance(record["program"], dict):
                record = {**record, "program": ast.literal_eval(record["program"])}
            task, nsteps = derive_task(record["program"]), program_length(record["program"])
            for strategy, hops in grl_gen.STRATEGIES:
                built = kb.build_subgraph(record, strategy, hops, max_triples=grl_gen.MAX_TRIPLES,
                                          max_nodes=grl_gen.MAX_NODES, rng=rng)
                if built is None:
                    continue
                text, nnodes, nedges = built
                if strategy == "gold" and task not in grl_gen.DERIVED_TASKS and str(record["answer"]) not in text:
                    continue
                rows.append({"algorithm": f"{strategy}_{hops}hop", "task": task, "question": f"{text}Q: "
                             f"{record['question']}\nA: ", "answer": f"{record['answer']}.", "record": record,
                             "facts": list(kb.last_facts), "nsteps": nsteps})
        out[split] = rows
    return out


def mini_kb(kb: TracingKBIndex, facts: list[Fact]) -> dict:
    """KQA Pro-format KB holding only the prompt's facts (both directions of each relation), no concept hierarchy."""
    ents, concepts = {}, {}

    def ent(eid):
        if eid not in ents:
            ents[eid] = {"name": kb.node_names[eid], "instanceOf": [], "attributes": [], "relations": []}
        return ents[eid]

    for f in facts:
        kind, eid, x = f.ref
        src = kb.entities[eid]
        if kind == "instance":
            ent(eid)["instanceOf"].append(x)
            concepts[x] = {"name": kb.concept_name(x), "subclassOf": []}
        elif kind == "attr":
            ent(eid)["attributes"].append(copy.deepcopy(src["attributes"][x]))
        else:
            rel = src["relations"][x]
            ent(eid)["relations"].append({**copy.deepcopy({k: v for k, v in rel.items() if k != "predicate"}),
                                          "relation": rel["predicate"]})
            if rel["object"] in kb.entities:
                ent(rel["object"])["relations"].append({"relation": rel["predicate"], "direction": "backward",
                                                         "object": eid,
                                                         "qualifiers": copy.deepcopy(rel["qualifiers"])})
            else:  # a concept as relation object: a leaf node with a name
                concepts.setdefault(rel["object"], {"name": kb.concept_name(rel["object"]), "subclassOf": []})
    return {"entities": ents, "concepts": concepts}


def _names(engine, ids) -> list[str]:
    return sorted({engine.kb.entities[i]["name"] for i in ids})


def _list(names) -> str:
    if not names:
        return "none"
    if len(names) > MAX_LIST:
        return ", ".join(names[:MAX_LIST]) + f", ... ({len(names)} in total)"
    return ", ".join(names)


class Facts:
    """The prompt's facts, for citing the lines a step reads (rendered exactly as in the prompt)."""

    def __init__(self, facts: list[list[str]]):
        self.facts = facts

    def lines(self, pred) -> list[str]:
        return [render_fact(f) for f in self.facts if pred(f)]


# Functions whose result is read off prompt facts: (subject names, relation / key) -> the fact lines they cite.
def _support(fn: str, inp: list[str], in_names: list[set], out_names: set, facts: Facts) -> list[str]:
    src = in_names[0] if in_names else set()
    if fn == "Relate":
        rel, direction = inp
        if direction == "forward":
            return facts.lines(lambda f: f[1] == rel and f[0] in src and f[2] in out_names)
        return facts.lines(lambda f: f[1] == rel and f[2] in src and f[0] in out_names)
    if fn == "FilterConcept":
        return facts.lines(lambda f: f[1] == "instance of" and f[2] == inp[0] and f[0] in out_names)
    if fn in ("FilterStr", "FilterNum", "FilterYear", "FilterDate"):
        return facts.lines(lambda f: f[1] == inp[0] and f[0] in out_names)
    if fn in ("QueryAttr", "QueryAttrUnderCondition", "QueryAttrQualifier", "SelectAmong"):
        return facts.lines(lambda f: f[1] == inp[0] and f[0] in src)
    if fn == "SelectBetween":
        both = src | (in_names[1] if len(in_names) > 1 else set())
        return facts.lines(lambda f: f[1] == inp[0] and f[0] in both)
    if fn in ("QueryRelation", "QueryRelationQualifier"):
        a, b = src, (in_names[1] if len(in_names) > 1 else set())
        rel = inp[0] if fn == "QueryRelationQualifier" else None
        return facts.lines(lambda f: (rel is None or f[1] == rel) and
                           ((f[0] in a and f[2] in b) or (f[0] in b and f[2] in a)))
    return []


def execute(engine, program: dict, facts: Facts) -> tuple[list[str], object, bool]:
    """Run the program step by step (kopl's dependency inference); one header line per step, the prompt facts it
    reads, and its result. Returns (lines, final result, whether every entity-set step was non-empty)."""
    functions, inputs = program["function"], program["inputs"]
    deps, stack = [], []
    for i, fn in enumerate(functions):
        if fn in ("FindAll", "Find"):
            deps.append([])
            stack.append(i - 1)
        elif fn in ("And", "Or", "SelectBetween", "QueryRelation", "QueryRelationQualifier"):
            deps.append([stack.pop(), i - 1])
        else:
            deps.append([i - 1])
    memory, names, lines, nonempty = [], [], [], True
    for i, (fn, dep, inp) in enumerate(zip(functions, deps, inputs)):
        res = getattr(engine, "QueryName" if fn == "What" else fn)(*[memory[d] for d in dep], *inp)
        memory.append(res)
        in_names = [names[d] for d in dep]
        if isinstance(res, tuple):
            out = _names(engine, res[0])
            names.append(set(out))
            nonempty &= bool(out)
            shown = f"all {len(res[0])} entities" if fn == "FindAll" else _list(out)
        else:
            names.append(set())
            vals = sorted(set(map(str, res))) if isinstance(res, list) else None
            if vals is not None:
                nonempty &= bool(vals)
            shown = (_list(vals) if vals is not None else ("yes" if res is True else "no" if res is False
                                                           else str(res)))
        support = _support(fn, inp, in_names, names[-1] if isinstance(res, tuple) else set(), facts)
        lines.append(f"{i + 1}. {fn}({', '.join(inp)})")
        lines += [f"   {line}" for line in support[:MAX_LIST]]
        lines.append(f"   -> {shown}")
    return lines, memory[-1], nonempty


def _final_text(res) -> str | None:
    if isinstance(res, list):
        vals = sorted(set(map(str, res)))
        return vals[0] if len(vals) == 1 else None
    if isinstance(res, bool):
        return "yes" if res else "no"
    return str(res) if res is not None else None


def main():
    import argparse
    import pickle
    from kopl.kopl import KoPLEngine
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-cache", default=None, help="Pickle of (kb, GRL rows); rebuilding them takes ~6 min.")
    args = ap.parse_args()
    if args.rows_cache and Path(args.rows_cache).exists():
        kb, rows_by_split = pickle.load(open(args.rows_cache, "rb"))
    else:
        print("loading KB and re-generating GRL rows (~6 min) ...")
        kb = TracingKBIndex(str(KQA / "kb.json"))
        rows_by_split = grl_rows(kb)
        if args.rows_cache:
            pickle.dump((kb, rows_by_split), open(args.rows_cache, "wb"))
    stats = Counter()
    for split, rows in rows_by_split.items():
        baseline = json.load(open(KQA / f"baseline/standard/{split}.json"))
        if len(baseline) != len(rows) or any(b["question"] != r["question"] or b["answer"] != r["answer"]
                                             for b, r in zip(baseline, rows)):
            raise SystemExit(f"{split}: re-generated rows differ from data/graphs/kqapro/baseline; stop")
        recs = []
        for i, r in enumerate(rows):
            facts = [[*_split_fact(kb, f)] for f in r["facts"]]
            gold = r["record"]["answer"]
            trace = None
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    engine = KoPLEngine(mini_kb(kb, r["facts"]))
                lines, res, nonempty = execute(engine, r["record"]["program"], Facts(facts))
                ok = _final_text(res) is not None and normalize(_final_text(res)) == normalize(str(gold))
                if ok and not nonempty:
                    stats[f"{split}_empty_step"] += 1  # gold reached through an empty set: luck, not reasoning
                elif ok:
                    trace = lines
            except Exception as e:  # a step the subgraph cannot support (missing name / key)
                stats[f"{split}_exec_error_{type(e).__name__}"] += 1
            stats[f"{split}_trace" if trace else f"{split}_no_trace"] += 1
            rec = Record(id=f"kqapro/{split}/{i}", dataset="kqapro", split=split, task=r["task"],
                         question=r["record"]["question"], answers=[str(gold)], facts=facts, trace=trace,
                         preamble=KG_PREAMBLE + QUALIFIER_NOTE, entities=_entity_names(kb, r["facts"]),
                         meta={"algorithm": r["algorithm"], "nsteps": r["nsteps"], "supported": trace is not None})
            # The rendered prompt must be the GRL prompt the earlier runs used.
            if prompt_text(rec) != r["question"]:
                raise SystemExit(f"{rec.id}: rendered prompt differs from the baseline question; stop")
            recs.append(rec)
        write_records(recs, OUT / f"{split}.jsonl")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


def _split_fact(kb: TracingKBIndex, f: Fact) -> list[str]:
    kind, eid, x = f.ref
    e = kb.entities[eid]
    if kind == "instance":
        return [e["name"], "instance of", kb.concept_name(x)]
    if kind == "attr":
        a = e["attributes"][x]
        return [e["name"], a["key"], format_value(a["value"]), format_qualifiers(a["qualifiers"])[2:-1]]
    rel = e["relations"][x]
    return [e["name"], rel["predicate"], kb.node_names[rel["object"]], format_qualifiers(rel["qualifiers"])[2:-1]]


def _entity_names(kb: TracingKBIndex, facts: list[Fact]) -> list[str]:
    names = set()
    for f in facts:
        kind, eid, x = f.ref
        names.add(kb.entities[eid]["name"])
        if kind == "rel":
            obj = kb.entities[eid]["relations"][x]["object"]
            if obj in kb.entities:
                names.add(kb.node_names[obj])
    return sorted(names)


if __name__ == "__main__":
    main()
