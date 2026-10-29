"""Results table of graph_inject runs: test greedy_acc (overall, without MaximumFlow, per task) at the best val epoch.

    .venv/bin/python -m graph_inject.summarize [--dataset graphqa | metaqa | kqapro] [--root <out_root>] [--per_task]
        [--test_ids <dedup_report.json>]

--test_ids re-scores every run on a subset of its test samples, from the per-sample results in eval_test.json: a JSON
list of test ids, or build_graphqa_v2.py's dedup_report.json (its test kept_ids: GraphQA without train duplicates).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from graph_inject.generate import UNANSWERABLE_TASKS
from graph_inject.train import CONFIG_DIR
from omegaconf import OmegaConf


def subset(report: dict, keep: set[int]) -> dict:
    """greedy_acc, greedy_acc_answerable and per-task accuracy over the samples whose id is in `keep`."""
    rows = [r for r in report["samples"] if r["id"] in keep]
    tasks = defaultdict(list)
    for r in rows:
        tasks[r["task"]].append(r["greedy_acc"])
    answerable = [r["greedy_acc"] for r in rows if r["task"] not in UNANSWERABLE_TASKS]
    return {"greedy_acc": sum(r["greedy_acc"] for r in rows) / len(rows),
            "greedy_acc_answerable": sum(answerable) / len(answerable),
            "task_accuracy": {t: sum(v) / len(v) for t, v in sorted(tasks.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="graphqa")
    ap.add_argument("--root", help="default: the dataset's out_root")
    ap.add_argument("--per_task", action="store_true")
    ap.add_argument("--test_ids", help="JSON list of test ids, or a dedup_report.json: score only those samples")
    args = ap.parse_args()
    args.root = args.root or OmegaConf.load(os.path.join(CONFIG_DIR, "datasets", f"{args.dataset}.yaml")).out_root

    keep = None
    if args.test_ids:
        with open(args.test_ids) as f:
            ids = json.load(f)
        keep = set(ids["test"]["kept_ids"] if isinstance(ids, dict) else ids)
    runs = defaultdict(dict)  # method -> model -> [reports over seeds]
    for path in sorted(glob.glob(os.path.join(args.root, "*", "*_s*", "eval_test.json"))):
        model = path.split(os.sep)[-3]
        method, seed = os.path.basename(os.path.dirname(path)).rsplit("_s", 1)
        with open(path) as f:
            report = json.load(f)
        runs[method].setdefault(model, []).append(subset(report, keep) if keep is not None else report)
    models = sorted({m for v in runs.values() for m in v})

    def cell(reports, key):
        vals = [r[key] for r in reports]
        mean = sum(vals) / len(vals)
        return f"{mean:.3f}" + (f" (n={len(vals)})" if len(vals) > 1 else "")

    print("| method | " + " | ".join(f"{m} acc | {m} acc w/o MaxFlow" for m in models) + " |")
    print("| --- |" + " ---: | ---: |" * len(models))
    for method, by_model in sorted(runs.items()):
        cells = []
        for m in models:
            reps = by_model.get(m)
            cells += [cell(reps, "greedy_acc"), cell(reps, "greedy_acc_answerable")] if reps else ["", ""]
        print(f"| {method} | " + " | ".join(cells) + " |")

    if args.per_task:
        for m in models:
            tasks = sorted({t for v in runs.values() for r in v.get(m, []) for t in r["task_accuracy"]})
            print(f"\n{m}\n| method | " + " | ".join(tasks) + " |\n| --- |" + " ---: |" * len(tasks))
            for method, by_model in sorted(runs.items()):
                reps = by_model.get(m, [])
                if reps:
                    accs = [sum(r["task_accuracy"][t] for r in reps) / len(reps) for t in tasks]
                    print(f"| {method} | " + " | ".join(f"{a:.2f}" for a in accs) + " |")


if __name__ == "__main__":
    main()
