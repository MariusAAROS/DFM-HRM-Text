"""Lenient (format-only) rescoring of eval reports, applied identically to every model (wiki: mimir-graph-benchmark-results).

The strict scorer (graph_bench.scoring) compares answer strings, which penalises a zero-shot model that never saw the
target format: GraphQA "Yes" against gold "Yes, there is a cycle", CLRS "[9, 7, 7]" against "[9 7 7]", node lists in
numeric rather than string order, KQA Pro dates written 2010/1/1 against 2010-01-01. A row is right under the lenient
rule if it is right under the strict one, or if, after these format-only fixes, it matches:
  - CLRS: the same integers in the same order, whatever the separators;
  - GraphQA yes/no tasks (CycleCheck, EdgeExistence, Reachability): the same yes / no word;
  - GraphQA node lists (ConnectedNodes, DisconnectedNodes): the same set of node ids ("none" for an empty gold);
  - other GraphQA tasks: the first integer equals a single-integer gold;
  - KGQA / CLUTRR: equal after stripping markdown / quotes and normalising Y/M/D dates, or, for multi-answer golds,
    the same set when split on | or ;.
Empty answers (cut off before an Answer: line) stay wrong. Strict remains the main score; this is a robustness column.

    .venv/bin/python -m graph_bench.eval.lenient_score $ROOT/e0/glm53/named_t16k.samples.jsonl $ROOT/e1/mimir15/named.samples.jsonl
"""
import argparse
import json
import re
from collections import defaultdict

from graph_bench.checkers import ints
from graph_bench.scoring import normalize

DATASETS = ("graphqa", "metaqa", "kqapro", "webqsp", "clrs", "cwq", "clutrr")
YES_NO = re.compile(r"\b(yes|no)\b", re.I)
YES_NO_TASKS = ("CycleCheck", "EdgeExistence", "Reachability")
NODE_SET_TASKS = ("ConnectedNodes", "DisconnectedNodes")


def date_norm(s: str) -> str:
    m = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s.strip())
    return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}" if m else s


def clean(s: str) -> str:
    s = re.sub(r"\*\*|`|\\boxed\{|\}$", "", s).strip()
    return date_norm(normalize(s.strip("\"'")))


def lenient(row: dict) -> bool:
    """Row of a .samples.jsonl report (greedy, n = 1)."""
    if row["acc"][0]:
        return True
    pred, gold, dataset, task = row["preds"][0], row["gold"], row["dataset"], row["task"]
    if not pred.strip():
        return False
    if dataset == "clrs":
        return ints(pred) == ints(gold[0])
    if dataset == "graphqa":
        if task in YES_NO_TASKS:
            a, b = YES_NO.search(pred), YES_NO.search(gold[0])
            return bool(a and b) and a[1].lower() == b[1].lower()
        if task in NODE_SET_TASKS:
            g, p = set(ints(gold[0])), ints(pred)
            if not g:
                return normalize(pred) in ("none", "no nodes", "") or not p
            return set(p) == g
        nums, gold_nums = re.findall(r"-?\d+", pred), re.findall(r"-?\d+", gold[0])
        return bool(nums and gold_nums) and nums[0] == gold_nums[0] and len(gold_nums) == 1
    golds = {clean(g) for g in gold}
    if clean(pred) in golds:
        return True
    if len(gold) > 1:
        return {clean(p) for p in re.split(r"\||;", pred) if p.strip()} == golds
    return False


def score_file(path: str) -> dict:
    """Per dataset: strict and lenient accuracy and the share of empty answers; macro over the datasets present."""
    by = defaultdict(lambda: [0, 0, 0, 0])
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g = by[r["dataset"]]
        g[0] += 1
        g[1] += bool(r["acc"][0])
        g[2] += lenient(r)
        g[3] += not r["generations"][0].strip()
    per = {d: {"n": v[0], "strict": v[1] / v[0], "lenient": v[2] / v[0], "empty": v[3] / v[0]} for d, v in by.items()}
    present = [d for d in DATASETS if d in per] or list(per)
    return {"datasets": per,
            "macro_strict": sum(per[d]["strict"] for d in present) / len(present),
            "macro_lenient": sum(per[d]["lenient"] for d in present) / len(present)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help=".samples.jsonl reports")
    for path in ap.parse_args().samples:
        s = score_file(path)
        print(f"{path}\n  macro strict {s['macro_strict']:.3f}  lenient {s['macro_lenient']:.3f}  "
              f"(+{s['macro_lenient'] - s['macro_strict']:.3f})")
        for d, v in s["datasets"].items():
            print(f"  {d:8s} n={v['n']:5d}  strict {v['strict']:.3f}  lenient {v['lenient']:.3f}  empty {v['empty']:.3f}")


if __name__ == "__main__":
    main()
