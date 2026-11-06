"""Check a build of scripts/osm/build_kgqa.py:
  1. no innermost sub-question (and no question) is shared between train / val / test;
  2. every answer is derivable from the prompt alone: a symbolic solver that reads only the fact lines and the
     question must return exactly the `answers` field;
  3. (reported) accuracy of a shortcut that checks only the outermost hop and accepts any link of the right
     relation type to the previous hop: how much of the set actually needs the inner hops;
  4. (reported) why the shortcut works where it does, judged on the full map (all POIs), not the prompt:
     'dataset: hop redundant' = the outer hop's own conditions already give exactly the answers;
     'dataset: hop removes only isolated POIs' = the POIs failing only the hop have no POI at all in the hop
     relation's range, so the shortcut (which needs some link) rejects them anyway;
     'builder: names shared' = such a POI could be linked, but only with a name shared by several POIs (case-
     insensitive), which the builder never puts in a prompt; 'extraction' = a POI failing only the hop could be
     linked with unique names but is not in the prompt (a builder bug, ~0 after the 2026-11-06 fix).

  .venv/bin/python scripts/osm/check_kgqa.py data/osm/kgqa
"""

import collections

import numpy as np
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osm_kg as K  # noqa: E402
from build_kgqa import ATTR_LABEL, innermost, phrase  # noqa: E402

FACT = re.compile(r"\((.*?), (category|cuisine|outdoor seating|indoor seating|at less than \d+ meters from|near"
                  r"|to the \w+ of|between|at the intersection of|following|inside|at the border of|at the \w+ of), (.*)\)$")


def solve_from_prompt(rec: dict, shortcut: bool = False) -> list[str]:
    body = rec["instruction"].split("The facts are:\n", 1)[1].split("\nQ: ")[0]
    facts = collections.defaultdict(set)
    for line in body.split("\n"):
        m = FACT.match(line)
        assert m, f"unparseable fact: {line}"
        facts[m.group(1)].add((m.group(2), m.group(3)))
    prev = None
    chain = K.parse_question(rec["question"]).chain()
    for lvl in chain[-1:] if shortcut else chain:
        cand = set()
        for ent, fs in facts.items():
            if not all((ATTR_LABEL[k], v.replace("_", " ")) in fs for k, v in lvl.attrs.items()):
                continue
            def hop_ok(c):
                rel = phrase(c, "_")[0]
                return any(f[0] == rel for f in fs) if shortcut else any((rel, p) in fs for p in prev)
            ok = all(hop_ok(c) if K.SUB in c.anchors else phrase(c) in fs for c in lvl.clauses)
            if ok:
                cand.add(ent)
        prev = cand
    return sorted(prev, key=str.lower)


CAUSES = ("needs inner hops", "extraction", "dataset: hop redundant", "dataset: hop removes only isolated POIs",
          "builder: names shared")
_UNIQUE = {}


def _unique(kg) -> np.ndarray:
    if id(kg) not in _UNIQUE:
        low = kg.poi.poi_name.str.lower().str.strip()
        _UNIQUE[id(kg)] = (low.map(low.value_counts()) == 1).to_numpy()
    return _UNIQUE[id(kg)]


def shortcut_cause(kg, rec: dict) -> str:
    """Why the outer-hop shortcut works on a multi-hop row (see the module docstring, point 4)."""
    if solve_from_prompt(rec, shortcut=True) != rec["answers"]:
        return CAUSES[0]
    q = K.parse_question(rec["question"])
    levels = K.solve(kg, q)
    outer = q.chain()[-1]
    hop = next(c for c in outer.clauses if K.SUB in c.anchors)
    own = K.solve_level(kg, outer.attrs, [c for c in outer.clauses if K.SUB not in c.anchors])
    hop_fail = np.setdiff1d(own, levels[-1])
    if len(hop_fail) == 0:
        return CAUSES[2]
    reach = (hop.param if hop.kind == "within" else K.NEAR_M) + 1
    taken = set(levels[-2].tolist()) | set(levels[-1].tolist())
    uniq = _unique(kg)
    names = kg.poi.poi_name.to_numpy()
    anchors = {a for lvl in q.chain() for c in lvl.clauses for a in c.anchors}
    linkable_any = False
    for h in hop_fail:  # can h get a hop link to some POI that is not a valid previous-hop answer?
        for r in kg.tree.query_ball_point(kg.xy[h], reach):
            if r != h and r not in taken and K.holds(kg, hop, np.array([h]), np.array([r]))[0]:
                linkable_any = True
                if uniq[h] and uniq[r] and names[h] not in anchors and names[r] not in anchors:
                    return CAUSES[1]
    return CAUSES[4] if linkable_any else CAUSES[3]


def main(root: str):
    splits = {s: [json.loads(line) for line in open(Path(root) / f"{s}.jsonl")] for s in ("train", "val", "test")}
    leaves = {s: {innermost(r["question"]) for r in rs} for s, rs in splits.items()}
    questions = {s: [r["question"] for r in rs] for s, rs in splits.items()}
    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n_leaf, n_q = len(leaves[a] & leaves[b]), len(set(questions[a]) & set(questions[b]))
        print(f"{a}/{b}: shared innermost sub-questions {n_leaf}, shared questions {n_q}")
        ok &= n_leaf == 0 and n_q == 0
    for s, rs in splits.items():
        dup = len(rs) - len(set(questions[s]))
        bad = [r for r in rs if solve_from_prompt(r) != r["answers"]]
        print(f"{s}: {len(rs):,} rows, {dup} duplicate questions, {len(rs) - len(bad):,} answerable from the prompt "
              f"({len(bad)} not)")
        ok &= dup == 0 and not bad
        by_depth = collections.defaultdict(list)
        for r in rs:
            by_depth[r["depth"]].append(solve_from_prompt(r, shortcut=True) == r["answers"])
        print(f"  outer-hop shortcut accuracy by depth: "
              + ", ".join(f"{d}: {sum(v) / len(v):.3f}" for d, v in sorted(by_depth.items())))
    kg = K.load(multihop=False)
    for s, rs in splits.items():
        cause = collections.Counter()
        for r in rs:
            if r["depth"] > 1:
                cause[(r["depth"], shortcut_cause(kg, r))] += 1
        for d in (2, 3, 4):
            n = sum(v for (dd, _), v in cause.items() if dd == d)
            if n:
                print(f"  {s} depth {d}: " + ", ".join(f"{k} {cause[(d, k)] / n:.1%}" for k in CAUSES))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/osm/kgqa")
