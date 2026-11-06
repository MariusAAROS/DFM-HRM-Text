"""Check a build of scripts/osm/build_kgqa.py:
  1. no innermost sub-question (and no question) is shared between train / val / test;
  2. every answer is derivable from the prompt alone: a symbolic solver that reads only the fact lines and the
     question must return exactly the `answers` field;
  3. (reported) accuracy of a shortcut that checks only the outermost hop and accepts any link of the right
     relation type to the previous hop: how much of the set actually needs the inner hops.

  .venv/bin/python scripts/osm/check_kgqa.py data/osm/kgqa
"""

import collections
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
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/osm/kgqa")
