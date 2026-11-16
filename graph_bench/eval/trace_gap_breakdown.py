"""Where a trace-trained arm (E2 / E3) loses or gains against the answer-only arm (E1), row by row.

Joins the `.samples.jsonl` files of the arms (same rows, same order) with the bench records and prints:
1. the macro gap split by cause: KGQA rows whose gold path is / is not in the prompt (`meta.supported`), hop counts
   absent from the SFT data, CLRS algorithms whose trace is mostly repeated lines, CLUTRR, GraphQA;
2. KGQA accuracy by what the prompt allows: the gold path is in the facts (reasoning) / no path but every answer is
   in the facts / not every answer is in the facts (nothing to derive: a right answer is a guess);
3. CWQ by its base question: every CWQ question extends a WebQSP question, and for 60% of the CWQ test rows that
   question is in WebQSP train, which the arms train on (in part, and not the same part). Rows whose base question is
   a WebQSP val / test question are the ones no arm has trained facts for: compare CWQ across arms on them;
4. per dataset: share of generations with text before the `Answer:` line (does the arm still write a trace);
5. per CLRS algorithm: share of repeated trace lines in train against the accuracy of each arm.
Wiki: mimir-graph-trace-sft-analysis.

    .venv/bin/python -m graph_bench.eval.trace_gap_breakdown --names E1,E2,E3 \\
        $B/e1/mimir15/named.samples.jsonl $B/e2/mimir15/named_t3k.samples.jsonl $B/e3/mimir15/named_t3k.samples.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from graph_bench.schema import read_records
from graph_bench.scoring import cut_at_end_of_turn, has_answer_line

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data/graphs/bench"
KGQA = ("metaqa", "kqapro", "webqsp", "cwq")
REPEAT_THRESHOLD = 0.9


def repeat_share(trace: list[str]) -> float:
    states = [line.split(": ", 1)[-1] for line in trace]
    return sum(a == b for a, b in zip(states, states[1:])) / max(1, len(states) - 1)


def clrs_repeat_by_algo() -> dict[str, float]:
    shares = defaultdict(list)
    for rec in read_records(BENCH / "clrs/train.jsonl"):
        shares[rec.task].append(repeat_share(rec.trace))
    return {algo: sum(v) / len(v) for algo, v in shares.items()}


def derivable(rec) -> str:
    """What the prompt of a KGQA record allows (WebQSP / CWQ: meta.answers_in_prompt of n_answers; the retrieved
    MetaQA / KQA Pro files: meta.answer_in_prompt)."""
    if rec.meta.get("supported", rec.trace is not None):
        return "path in facts"
    if "n_answers" in rec.meta:
        full = rec.meta["answers_in_prompt"] == rec.meta["n_answers"]
    else:
        full = bool(rec.meta.get("answer_in_prompt"))
    return "no path, answers in facts" if full else "not all answers in facts"


def writes_trace(generation: str) -> bool:
    """Text before the (last) Answer: line."""
    lines = [line for line in cut_at_end_of_turn(generation).strip().split("\n") if line.strip()]
    return len(lines) > 1 or (len(lines) == 1 and not lines[0].lstrip().lower().startswith("answer"))


def category(row: dict, rec, repeated: set[str]) -> str:
    dataset = row["dataset"]
    if dataset in KGQA:
        if derivable(rec) != "path in facts":
            return "KGQA: gold path not in the prompt"
        if dataset == "cwq" or row["task"] in ("3hop", "4hop"):
            return "KGQA: path in prompt, composition not in SFT data (MetaQA 3-hop, CWQ)"
        return "KGQA: WebQSP, path in prompt" if dataset == "webqsp" else "KGQA: MetaQA 1-2 hop / KQA Pro, path in prompt"
    if dataset == "clrs":
        kind = "repeated-line traces" if row["task"] in repeated else "other algorithms"
        return f"CLRS {'OOD size' if 'ood' in row['source'] else 'test'}: {kind}"
    return "CLUTRR (held out)" if dataset == "clutrr" else "GraphQA"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="answer-only arm first, then the trace arms (same rows, same order)")
    ap.add_argument("--names", default=None, help="Column names, comma-separated, one per samples file.")
    args = ap.parse_args()
    arms = [[json.loads(line) for line in open(p, encoding="utf-8")] for p in args.samples]
    assert all(len(a) == len(arms[0]) for a in arms), "arms differ in row count"
    sources = [next(a[i]["source"] for a in arms if "source" in a[i]) for i in range(len(arms[0]))]
    recs = {(src, r.id): r for src in set(sources) for r in read_records(BENCH / f"{src}.jsonl")}
    repeat = clrs_repeat_by_algo()
    webqsp_train = {r.meta["source_id"] for r in read_records(BENCH / "webqsp/train.jsonl")}
    repeated = {algo for algo, share in repeat.items() if share >= REPEAT_THRESHOLD}
    per_dataset = Counter(row["dataset"] for row in arms[0])
    n_datasets = len(per_dataset)

    cats = defaultdict(lambda: {"n": 0, "ok": [0] * len(arms), "macro": [0.0] * len(arms), "no_answer": [0] * len(arms)})
    clrs = defaultdict(lambda: {"n": 0, "ok": [0] * len(arms)})
    kgqa = defaultdict(lambda: {"n": 0, "ok": [0] * len(arms)})
    traces = defaultdict(lambda: {"n": 0, "ok": [0] * len(arms)})
    cwq = defaultdict(lambda: {"n": 0, "ok": [0] * len(arms)})
    for i, src in enumerate(sources):
        rows = [a[i] for a in arms]
        assert all(r["id"] == rows[0]["id"] and r["gold"] == rows[0]["gold"] for r in rows), f"row {i} differs"
        base = dict(rows[0], source=src)
        rec = recs[(src, base["id"])]
        c = cats[category(base, rec, repeated)]
        tallies = [(traces[base["dataset"]], [writes_trace(r["generations"][0]) for r in rows])]
        if base["dataset"] in KGQA:
            hit = [bool(r["acc"][0]) for r in rows]
            tallies += [(kgqa[(d, derivable(rec))], hit) for d in (base["dataset"], "all 4")]
            if base["dataset"] == "cwq":
                exposed = rec.meta["source_id"].split("_")[0] in webqsp_train
                group = "base question in WebQSP train" if exposed else "base question not in WebQSP train"
                tallies += [(cwq[group], hit), (cwq[f"{group} / {derivable(rec)}"], hit)]
            if base["dataset"] == "webqsp" and derivable(rec) == "path in facts":
                tallies.append((traces["webqsp, path in facts"], tallies[0][1]))
        for t, values in tallies:
            t["n"] += 1
            for k, v in enumerate(values):
                t["ok"][k] += v
        c["n"] += 1
        for k, row in enumerate(rows):
            ok = bool(row["acc"][0])
            c["ok"][k] += ok
            c["macro"][k] += (ok - bool(rows[0]["acc"][0])) / per_dataset[base["dataset"]] / n_datasets
            c["no_answer"][k] += not has_answer_line(row["generations"][0])
        if base["dataset"] == "clrs":
            t = clrs[(src.split("/")[1], base["task"])]
            t["n"] += 1
            for k, row in enumerate(rows):
                t["ok"][k] += bool(row["acc"][0])

    names = args.names.split(",") if args.names else [f"arm{k}" for k in range(len(arms))]
    assert len(names) == len(arms), "--names: one per samples file"
    print(f"{'category':76s}     n  " + "  ".join(f"{n:>5s}" for n in names)
          + "  | macro vs arm0: " + "  ".join(names[1:]) + "  | no Answer line: " + "  ".join(names[1:]))
    for name, c in sorted(cats.items()):
        print(f"{name:76s} {c['n']:5d}  " + "  ".join(f"{ok / c['n']:.3f}" for ok in c["ok"])
              + "  |  " + "  ".join(f"{m:+.4f}" for m in c["macro"][1:])
              + "  |  " + "  ".join(f"{x / c['n']:.3f}" for x in c["no_answer"][1:]))
    print("macro total vs arm0: " + "  ".join(f"{sum(c['macro'][k] for c in cats.values()):+.4f}" for k in range(1, len(arms))))

    for title, table in (("KGQA accuracy by what the prompt allows", kgqa), ("CWQ accuracy by its base question", cwq),
                         ("share of generations with a trace", traces)):
        print(f"\n{title:60s}     n  " + "  ".join(f"{n:>5s}" for n in names))
        for name, t in sorted(table.items(), key=str):
            label = " / ".join(name) if isinstance(name, tuple) else name
            print(f"{label:60s} {t['n']:5d}  " + "  ".join(f"{ok / t['n']:.3f}" for ok in t["ok"]))

    print(f"\n{'CLRS algorithm':32s} repeated lines | test: " + " ".join(names) + " | OOD: " + " ".join(names))
    for algo, share in sorted(repeat.items(), key=lambda kv: -kv[1]):
        cells = []
        for split in ("test", "test_ood"):
            t = clrs.get((split, algo))
            cells.append("  ".join(f"{ok / t['n']:.2f}" for ok in t["ok"]) if t else "-")
        print(f"{algo:32s} {share:.2f}           | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    main()
