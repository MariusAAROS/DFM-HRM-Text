"""Per-depth OSM scores, split by whether the question needs its inner hops (wiki: mimir-graph-benchmark).

Reads eval_vllm .samples.jsonl files of data/graphs/bench/osm/<split>.jsonl and joins meta.needs_inner_hops by id.
"chain" = rows the outer-hop shortcut gets wrong (the model must follow the hops); "shortcut" = rows answerable
from the last hop alone (depths 2-4 only). Greedy runs: first sample of each row.

    .venv/bin/python -m graph_bench.eval.osm_breakdown $ROOT/e*/*/osm_t3k.samples.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from graph_bench.schema import read_records

BENCH = Path(__file__).resolve().parents[2] / "data/graphs/bench/osm"


def breakdown(samples: Path) -> dict:
    rows = [json.loads(line) for line in open(samples, encoding="utf-8")]
    split = rows[0]["id"].split("/")[1]
    meta = {r.id: r.meta for r in read_records(BENCH / f"{split}.jsonl")}
    cells = defaultdict(list)
    for r in rows:
        m = meta[r["id"]]
        acc, f1 = float(r["acc"][0]), float(r["f1"][0])
        keys = ["all", r["task"]]
        if m["depth"] > 1:
            kind = "chain" if m["needs_inner_hops"] else "shortcut"
            keys += [f"2-4hop/{kind}", f"{r['task']}/{kind}"]
        for k in keys:
            cells[k].append((acc, f1))
    return {k: {"n": len(v), "acc": sum(a for a, _ in v) / len(v), "f1": sum(f for _, f in v) / len(v)}
            for k, v in cells.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="Also write {run: breakdown} here.")
    args = ap.parse_args()
    out = {}
    for p in args.samples:
        name = f"{p.parent.parent.name}/{p.parent.name}/{p.name.split('.')[0]}"  # <arm>/<model>/<report>
        out[name] = breakdown(p)
    keys = ["all", "1hop", "2hop", "3hop", "4hop", "2-4hop/chain", "2-4hop/shortcut"]
    print("| run | " + " | ".join(keys) + " |")
    print("| --- |" + " --- |" * len(keys))
    for name, b in out.items():
        print(f"| {name} | " + " | ".join(f"{b[k]['acc']:.3f} / {b[k]['f1']:.3f}" if k in b else "-" for k in keys)
              + " |")
    print("cells: acc / set-F1; n per cell: " + ", ".join(f"{k} {next(iter(out.values()))[k]['n']}" for k in keys
                                                           if k in next(iter(out.values()))))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
