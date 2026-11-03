"""KQA Pro gold-only rebuild: program-driven subgraphs whose facts derive the answer (wiki: kgqa-gold-rebuild).

GRL's `gold` arm (kqapro.py) guarantees that the answer string is in the prompt, not that the KoPL program can reach
it: about half of its rows trace. Here every prompt is built from the program instead, on the same questions (GRL's
sample: 4,000 official-train questions split 90/10 into train / val, 600 official-val questions as test):

1. Spine: run the program on the full KB and keep the facts each step needs for the entities that must appear,
   computed backwards from the last step (Count needs every member, SelectAmong only the winners, a filter only the
   entities that reach the end). "instance of" facts name the concept the program filters on, also when the KB
   reaches it through the concept hierarchy (the prompt has no hierarchy).
2. Distractors, up to MAX_FACTS: other facts of the spine entities, facts of other members of the filtered concepts,
   facts of neighbours; added in chunks, each chunk re-validated.
3. Validation (closed world): the program, run on a KB holding only the prompt facts, must return the gold answer
   with no empty entity set; a distractor chunk that changes that is retried fact by fact and the breaking facts are
   left out. The validated run is the record's trace (the same format as kqapro.py).
Questions whose spine is over MAX_SPINE facts, or whose program does not reproduce the gold on the full KB, are
dropped and counted. test_retrieved.jsonl keeps GRL's retrieved-arm test rows (an eval condition, never trained).

    .venv/bin/python -m graph_bench.sources.kqapro_gold
"""
import ast
import contextlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from graph_bench.schema import KG_PREAMBLE, QUALIFIER_NOTE, Record, write_records
from graph_bench.scoring import normalize
from graph_bench.sources.kqapro import (KQA, OUT, Facts, TracingKBIndex, _entity_names, _fact, _final_text,
                                        _split_fact, derive_task, execute, grl_gen, mini_kb, program_length)

MAX_SPINE = 40
MAX_FACTS = 60
CHUNK = 5
SET_FILTERS = {"FilterConcept", "FilterStr", "FilterNum", "FilterYear", "FilterDate"}
Q_FILTERS = {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate"}


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a)


def dependencies(functions: list[str]) -> list[list[int]]:
    """kopl's argument inference, as in kqapro.execute."""
    deps, stack = [], []
    for i, fn in enumerate(functions):
        if fn in ("FindAll", "Find"):
            deps.append([])
            stack.append(i - 1)
        elif fn in ("And", "Or", "SelectBetween", "QueryRelation", "QueryRelationQualifier"):
            deps.append([stack.pop(), i - 1])
        else:
            deps.append([i - 1])
    return deps


class SpineBuilder:
    def __init__(self, kb: TracingKBIndex, engine):
        self.kb, self.engine = kb, engine
        self.concept_ids = defaultdict(list)
        for cid, c in kb.concepts.items():
            self.concept_ids[c["name"]].append(cid)

    # ---------------------------------------------------------------- fact references
    def attr_refs(self, eid, key, facts=None):
        """('attr', eid, idx) for the attributes of eid with this key (only the given fact objects when set)."""
        ents = self.engine.kb.entities
        return [("attr", eid, i) for i in self.engine.kb.attribute_inv_index[key].get(eid, [])
                if facts is None or any(ents[eid]["attributes"][i] is f for f in facts)]

    def forward_rel_ref(self, subj, pred, obj):
        for i, r in enumerate(self.kb.entities[subj]["relations"]):
            if r["direction"] == "forward" and r["predicate"] == pred and r["object"] == obj:
                return ("rel", subj, i)
        return None

    def instance_ref(self, eid, concept_name):
        direct = set(self.kb.entities[eid]["instanceOf"])
        ids = self.concept_ids[concept_name]
        cid = next((c for c in ids if c in direct), ids[0] if ids else None)
        return ("instance", eid, cid) if cid else None

    # ---------------------------------------------------------------- spine
    def spine(self, program: dict):
        fns, inps = program["function"], program["inputs"]
        deps = dependencies(fns)
        mem = []
        for fn, dep, inp in zip(fns, deps, inps):
            mem.append(getattr(self.engine, "QueryName" if fn == "What" else fn)(*[mem[d] for d in dep], *inp))
        ids = [set(m[0]) if isinstance(m, tuple) else None for m in mem]
        need = [set() for _ in fns]
        refs = set()
        consumer = {d: i for i, dep in enumerate(deps) for d in dep}
        for i in range(len(fns) - 1, -1, -1):
            fn, dep, inp, res = fns[i], deps[i], inps[i], mem[i]
            if ids[i] is not None and i == len(fns) - 1:
                need[i] = set(ids[i])  # a program ending on a set (What / Count handle theirs below)
            if fn in ("Count", "What"):
                need[dep[0]] |= ids[dep[0]]
            elif fn in ("QueryAttr", "QueryAttrUnderCondition", "QueryAttrQualifier"):
                for e in ids[dep[0]]:
                    if self.attr_refs(e, inp[0]):
                        need[dep[0]].add(e)
                        refs.update(self.attr_refs(e, inp[0]))
            elif fn == "SelectAmong":
                winners = set(res)
                for e in ids[dep[0]]:
                    if self.kb.node_names.get(e) in winners and self.attr_refs(e, inp[0]):
                        need[dep[0]].add(e)
                        refs.update(self.attr_refs(e, inp[0]))
            elif fn == "SelectBetween":
                for d in dep:
                    for e in ids[d]:
                        need[d].add(e)
                        refs.update(self.attr_refs(e, inp[0]))
            elif fn in ("QueryRelation", "QueryRelationQualifier"):
                for a in ids[dep[0]]:
                    for b in ids[dep[1]]:
                        for k in self.engine.kb.forward_relation_index[(a, b)]:
                            rel = self.kb.entities[a]["relations"][k]
                            if fn == "QueryRelation" or rel["predicate"] == inp[0]:
                                refs.add(("rel", a, k))
                                need[dep[0]].add(a)
                                need[dep[1]].add(b)
            elif fn == "FilterConcept":
                need[dep[0]] |= need[i]
                for e in need[i]:
                    r = self.instance_ref(e, inp[0])
                    if r:
                        refs.add(r)
            elif fn in SET_FILTERS or fn in Q_FILTERS:
                need[dep[0]] |= need[i]
                facts = res[1] or []
                # the fact a later qualifier filter kept is the one to show
                j = consumer.get(i)
                if j is not None and fns[j] in Q_FILTERS:
                    facts = mem[j][1] or []
                if fn in SET_FILTERS:
                    by_ent = defaultdict(list)
                    for e, f in zip(res[0], res[1] or []):
                        by_ent[e].append(f)
                    for e in need[i]:
                        chosen = [f for f in by_ent[e] if any(f is g for g in facts)] or by_ent[e][:1]
                        refs.update(self.attr_refs(e, inp[0], chosen))
            elif fn == "Relate":
                pred, direction = inp
                src = ids[dep[0]]
                for e in need[i]:
                    for s in sorted(src, key=lambda s: s not in need[dep[0]]):
                        r = (self.forward_rel_ref(s, pred, e) if direction == "forward"
                             else self.forward_rel_ref(e, pred, s))
                        if r:
                            refs.add(r)
                            need[dep[0]].add(s)
                            break
            elif fn == "And":
                for d in dep:
                    need[d] |= need[i]
            elif fn == "Or":
                for e in need[i]:
                    need[dep[0] if e in ids[dep[0]] else dep[1]].add(e)
        # Every Find entity appears in the prompt, so an empty branch reads "exists, nothing matches", not "unknown"
        shown = {r[1] for r in refs} | {self.kb.entities[r[1]]["relations"][r[2]]["object"] for r in refs if r[0] == "rel"}
        for fn, m in zip(fns, mem):
            if fn == "Find":
                for e in m[0]:
                    if e not in shown and e in self.kb.entities:
                        anchor = self.entity_refs(e)[:1]
                        refs.update(anchor)
                        shown.add(e)
        return sorted(refs, key=str), mem

    @staticmethod
    def empty_allowed(program: dict, sizes: list) -> bool:
        """The no-empty-set rule (an empty set that still lands on the gold is luck), except on a branch that feeds a
        non-empty Or: "stations in South Vietnam or owned by CBC" is answered by the CBC branch alone."""
        fns = program["function"]
        deps = dependencies(fns)
        consumer = {d: i for i, dep in enumerate(deps) for d in dep}
        for i, n in enumerate(sizes):
            if n is None or n > 0:
                continue
            j = i
            while j in consumer and fns[consumer[j]] != "Or":
                j = consumer[j]
            if j not in consumer or not sizes[consumer[j]]:
                return False
        return True

    # ---------------------------------------------------------------- distractors
    def candidates(self, spine_refs, program, rng):
        ents = {r[1] for r in spine_refs} | {self.kb.entities[r[1]]["relations"][r[2]]["object"]
                                            for r in spine_refs if r[0] == "rel"}
        ents = {e for e in ents if e in self.kb.entities}
        near, far = [], []
        for e in sorted(ents):
            near += self.entity_refs(e)
        for fn, inp in zip(program["function"], program["inputs"]):
            if fn == "FilterConcept":
                members = sorted(self.engine.kb.concept_to_entity.get(self.concept_ids[inp[0]][0], [])) \
                    if self.concept_ids[inp[0]] else []
                members = [e for e in members if e in self.kb.entities]  # the index also lists sub-concepts
                for e in rng.sample(members, min(8, len(members))):
                    far += self.entity_refs(e)
        nbrs = sorted({o for e in ents for o in self.kb._neighbors.get(e, ()) if o in self.kb.entities} - ents)
        for e in rng.sample(nbrs, min(8, len(nbrs))):
            far += self.entity_refs(e)
        spine = set(spine_refs)
        near = [r for r in dict.fromkeys(near) if r not in spine]
        far = [r for r in dict.fromkeys(far) if r not in spine and r not in set(near)]
        rng.shuffle(near)
        rng.shuffle(far)
        return near[: (MAX_FACTS * 2) // 3] + far

    def entity_refs(self, e):
        ent = self.kb.entities[e]
        out = [("instance", e, c) for c in ent["instanceOf"]]
        out += [("attr", e, i) for i in range(len(ent["attributes"]))]
        out += [("rel", e, i) for i, r in enumerate(ent["relations"])
                if r["direction"] == "forward" and r["object"] in self.kb.node_names]
        return out

    # ---------------------------------------------------------------- validation
    def to_facts(self, refs):
        return [_fact("", r) for r in refs]

    @staticmethod
    def merge_by_name(kb_dict):
        """The model reads names, not ids: entities sharing a name (Chicago the city and the musical) are one node of
        the prompt, so validation runs on the name-merged KB. A distractor that makes the text ambiguous then fails."""
        ents = kb_dict["entities"]
        nid = {eid: "N::" + e["name"] for eid, e in ents.items()}
        merged = {}
        for eid, e in ents.items():
            m = merged.setdefault(nid[eid], {"name": e["name"], "instanceOf": [], "attributes": [], "relations": []})
            m["instanceOf"] += [c for c in e["instanceOf"] if c not in m["instanceOf"]]
            m["attributes"] += e["attributes"]
            m["relations"] += [{**r, "object": nid.get(r["object"], r["object"])} for r in e["relations"]]
        return {"entities": merged, "concepts": kb_dict["concepts"]}

    def check(self, refs, program, gold):
        facts = self.to_facts(refs)
        rows = [_split_fact(self.kb, f) for f in facts]
        try:
            engine = quiet(KoPLEngine, self.merge_by_name(mini_kb(self.kb, facts)))
            lines, res, nonempty = execute(engine, program, Facts(rows))
            mem = []
            for fn, dep, inp in zip(program["function"], dependencies(program["function"]), program["inputs"]):
                mem.append(getattr(engine, "QueryName" if fn == "What" else fn)(*[mem[d] for d in dep], *inp))
            sizes = [len(m[0]) if isinstance(m, tuple) else None for m in mem]
        except Exception:
            return None
        ok = (nonempty or self.empty_allowed(program, sizes)) and _final_text(res) is not None and normalize(_final_text(res)) == normalize(str(gold))
        return (lines, rows) if ok else None


def grl_records():
    """GRL's question sample (dataset_generator.main): the RNG draws happen before any subgraph is built."""
    from sklearn.model_selection import train_test_split
    rng = random.Random(grl_gen.SEED)
    official_train = json.load(open(KQA / "original/train.json"))
    official_val = json.load(open(KQA / "original/val.json"))
    pool = rng.sample(official_train, min(grl_gen.N_TRAIN_POOL, len(official_train)))
    train, val = train_test_split(pool, test_size=grl_gen.VAL_FRACTION, random_state=grl_gen.SEED)
    test = rng.sample(official_val, min(grl_gen.N_TEST, len(official_val)))
    fix = lambda r: r if isinstance(r["program"], dict) else {**r, "program": ast.literal_eval(r["program"])}
    return {"train": [fix(r) for r in train], "val": [fix(r) for r in val], "test": [fix(r) for r in test]}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None, help="Questions per split (trial runs).")
    args = ap.parse_args()
    global KoPLEngine
    from kopl.kopl import KoPLEngine
    from src.datasets.kqapro.kb_index import load_kopl_engine
    kb = TracingKBIndex(str(KQA / "kb.json"))
    engine = quiet(load_kopl_engine, str(KQA / "kb.json"))
    builder = SpineBuilder(kb, engine)
    stats = Counter()
    for split, records in grl_records().items():
        recs = []
        for i, r in enumerate(records[: args.limit]):
            task, prog, gold = derive_task(r["program"]), r["program"], r["answer"]
            rng = random.Random(f"{split}/{i}")
            try:
                spine, mem = builder.spine(prog)
            except Exception as e:
                stats[f"{split}_{task}_full_kb_error_{type(e).__name__}"] += 1
                continue
            if _final_text(mem[-1]) is None or normalize(_final_text(mem[-1])) != normalize(str(gold)):
                stats[f"{split}_{task}_full_kb_mismatch"] += 1
                continue
            if len(spine) > MAX_SPINE:
                stats[f"{split}_{task}_spine_too_large"] += 1
                continue
            if builder.check(spine, prog, gold) is None:
                stats[f"{split}_{task}_spine_invalid"] += 1
                continue
            kept, rejected = list(spine), 0
            cands = builder.candidates(spine, prog, rng)
            for k in range(0, len(cands), CHUNK):
                if len(kept) >= MAX_FACTS:
                    break
                chunk = cands[k:k + CHUNK][: MAX_FACTS - len(kept)]
                if builder.check(kept + chunk, prog, gold):
                    kept += chunk
                    continue
                for c in chunk:  # find the breaking facts
                    if builder.check(kept + [c], prog, gold):
                        kept.append(c)
                    else:
                        rejected += 1
            rng.shuffle(kept)
            lines, rows = builder.check(kept, prog, gold)
            stats[f"{split}_{task}_kept"] += 1
            stats["distractors_rejected"] += rejected
            recs.append(Record(
                id=f"kqapro/{split}/{i}", dataset="kqapro", split=split, task=task, question=r["question"],
                answers=[str(gold)], facts=rows, trace=lines, preamble=KG_PREAMBLE + QUALIFIER_NOTE,
                entities=_entity_names(kb, builder.to_facts(kept)),
                meta={"algorithm": "program_gold", "nsteps": program_length(prog), "spine_facts": len(spine),
                      "facts": len(kept), "supported": True}))
        stats[f"{split}_written"] = write_records(recs, args.out / f"{split}.jsonl")
        print(f"{split}: {len(recs)} of {len(records)} questions", flush=True)
    # the retrieved arm stays GRL's, as an eval condition
    mixed = Path(OUT).parent / "_mixed_20261103/kqapro/test.jsonl"
    retrieved = []
    for line in open(mixed):
        rec = Record.from_dict(json.loads(line))
        if rec.meta.get("algorithm") == "retrieved_2hop":
            rec.split = "test_retrieved"
            rec.meta["answer_in_prompt"] = rec.trace is not None  # derivable from the prompt (traced)
            retrieved.append(rec)
    stats["test_retrieved_written"] = write_records(retrieved, args.out / "test_retrieved.jsonl")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
