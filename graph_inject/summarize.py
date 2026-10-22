"""Results table of graph_inject runs: test greedy_acc (overall, without MaximumFlow, per task) at the best val epoch.

    .venv/bin/python -m graph_inject.summarize [--root /work/dfm/marius-ortega/dfm-hrm-graphs/graph_inject]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from graph_inject.train import CONFIG_DIR
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=OmegaConf.load(os.path.join(CONFIG_DIR, "base.yaml")).out_root)
    ap.add_argument("--per_task", action="store_true")
    args = ap.parse_args()

    runs = defaultdict(dict)  # method -> model -> [reports over seeds]
    for path in sorted(glob.glob(os.path.join(args.root, "*", "*_s*", "eval_test.json"))):
        model = path.split(os.sep)[-3]
        method, seed = os.path.basename(os.path.dirname(path)).rsplit("_s", 1)
        with open(path) as f:
            runs[method].setdefault(model, []).append(json.load(f))
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
