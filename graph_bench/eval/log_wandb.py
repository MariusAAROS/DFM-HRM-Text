"""Log finished eval_vllm reports to W&B (wiki: mimir-graph-benchmark). eval_vllm.py itself only writes JSON.

One W&B run per report: project Mimir-Graph-RLVR (the whole benchmark: E0-E4), job_type eval, group --group, by
default mimir-bench-<arm> for a report at <root>/<arm>/<model>/<cond>.json; name <arm>/<model>/<cond>.

Metrics, per eval file ("split": <dataset dir>/<file stem>, e.g. metaqa/test_retrieved), per dataset, and per
dataset/task: acc, f1, pass@k (n > 1), answer_rate (share of first samples with an "Answer:" line), count; plus
macro_acc over datasets. They go to the run summary (runs table), to the history at step 0 as split/<split>/<metric>
and dataset/<name>/<metric> (so the workspace gets panels; use bar charts to compare runs), and to a "metrics" table.
Config: the report's settings. A "samples" table holds the first --table-rows samples.

Reports written before eval_vllm recorded each row's source file: the split is rebuilt from the row order (rows follow
the input files in order, min(limit, rows) each; an anonymised report's files are one per dataset).

A report already logged is marked by <report>.wandb (run path, then the logger version); a v1 run (summary only) is
resumed and completed in place, not duplicated.

    .venv/bin/python -m graph_bench.eval.log_wandb $ROOT/e0/*/*.json
    .venv/bin/python -m graph_bench.eval.log_wandb --watch $ROOT/e0 $ROOT/e1 $ROOT/e2   # until killed
"""
import argparse
import json
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import wandb

PROJECT = "Mimir-Graph-RLVR"
VERSION = "v2"


def source_name(path: str) -> str:  # = eval_vllm.source_name (not imported: it pulls in the scorer and anonymiser)
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def rebuild_sources(report: dict, samples: list[dict]) -> list[str]:
    """Source file of each sample for reports without a "source" field."""
    files = report["data"]
    if len(files) == 1:
        return [source_name(files[0])] * len(samples)
    by_ds = defaultdict(list)
    for f in files:
        by_ds[Path(f).parent.name].append(f)
    if all(len(v) == 1 for v in by_ds.values()) and all(s["dataset"] in by_ds for s in samples):  # one file each
        return [source_name(by_ds[s["dataset"]][0]) for s in samples]
    if report.get("anonymized"):
        raise ValueError("anonymised report with two files of one dataset: dropped rows make the order ambiguous")
    sizes = [sum(1 for _ in open(f, encoding="utf-8")) for f in files]
    limits = [n for n in range(1, max(sizes) + 1) if sum(min(n, s) for s in sizes) == len(samples)]
    if not limits:
        raise ValueError(f"{len(samples)} samples do not match files of {sizes} rows")
    out = []
    for f, s in zip(files, sizes):
        out += [source_name(f)] * min(limits[0], s)
    for s, src in zip(samples, out):  # row order check: each sample's dataset matches its file's
        if s["dataset"] != Path(src).parent.name and Path(src).parent.name in by_ds:
            raise ValueError(f"sample {s['id']} ({s['dataset']}) does not fall in {src}")
    return out


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def metrics(samples: list[dict], sources: list[str]) -> dict[str, dict]:
    groups = defaultdict(list)
    for s, src in zip(samples, sources):
        groups[f"split/{src}"].append(s)
        groups[f"dataset/{s['dataset']}"].append(s)
        groups[f"task/{s['dataset']}/{s['task']}"].append(s)
    out = {}
    for name, rs in sorted(groups.items()):
        n = rs[0]["n"]
        m = {"count": len(rs), "acc": sum(r["acc"][0] for r in rs) / len(rs),
             "f1": sum(r["f1"][0] for r in rs) / len(rs),
             "answer_rate": sum("Answer:" in r["generations"][0] for r in rs) / len(rs)}
        if n > 1:
            for k in (1, 4, 8, 16, 32):
                if k <= n:
                    m[f"pass@{k}"] = sum(pass_at_k(n, sum(r["acc"]), k) for r in rs) / len(rs)
        out[name] = m
    datasets = [v["acc"] for k, v in out.items() if k.startswith("dataset/")]
    out["macro_acc"] = {"value": sum(datasets) / len(datasets) if datasets else 0.0}
    return out


def log_report(path: Path, group: str | None, table_rows: int) -> str | None:
    marker = path.with_suffix(".wandb")
    old = marker.read_text().split() if marker.exists() else []
    if old[1:2] == [VERSION]:
        return None
    report = json.loads(path.read_text())
    samples_path = path.with_suffix(".samples.jsonl")
    samples = [json.loads(line) for line in open(samples_path, encoding="utf-8")]
    sources = [s["source"] for s in samples] if samples and "source" in samples[0] else rebuild_sources(report, samples)
    m = metrics(samples, sources)
    model_name, arm = path.parent.name, path.parent.parent.name
    group = group or f"mimir-bench-{arm}"
    config = {k: v for k, v in report.items() if k != "summary"} | {"report": str(path), "model_name": model_name,
                                                                    "condition": path.stem, "arm": arm}
    init = dict(project=PROJECT, group=group, job_type="eval", name=f"{arm}/{model_name}/{path.stem}",
                tags=[group, arm, model_name, path.stem], config=config)
    if old:  # v1 run: resume it
        init |= {"id": old[0].split("/")[-1], "resume": "must"}
    run = wandb.init(**init, reinit="finish_previous")
    flat = {"macro_acc": m.pop("macro_acc")["value"]}
    for name, v in m.items():
        flat.update({f"{name}/{k}": x for k, x in v.items()})
    table = wandb.Table(columns=["level", "name", "count", "acc", "f1", "answer_rate"]
                        + sorted({k for v in m.values() for k in v if k.startswith("pass@")}))
    for name, v in m.items():
        level, _, key = name.partition("/")
        table.add_data(level, key, v["count"], v["acc"], v["f1"], v["answer_rate"],
                       *[v.get(c) for c in table.columns[6:]])
    payload = {k: x for k, x in flat.items() if not k.startswith("task/")} | {"metrics": table}
    if not old and table_rows:
        st = wandb.Table(columns=["id", "split", "dataset", "task", "acc", "gold", "pred", "generation", "prompt_tail"])
        for s, src in list(zip(samples, sources))[:table_rows]:
            st.add_data(s["id"], src, s["dataset"], s["task"], s["acc"][0], json.dumps(s["gold"], ensure_ascii=False),
                        str(s["preds"][0]), s["generations"][0], s["prompt_tail"])
        payload["samples"] = st
    run.log(payload)
    run.summary.update(flat)
    run_path = run.path
    run.finish()
    marker.write_text(f"{run_path}\n{VERSION}\n")
    return run_path


def reports(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        cands = sorted(p.glob("*/*.json")) if p.is_dir() else ([p] if p.exists() else [])
        out += [c for c in cands if not c.name.endswith(".samples.json")]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="Report JSON files, or dirs holding <model>/<cond>.json.")
    ap.add_argument("--group", default=None, help="Default: mimir-bench-<arm dir>.")
    ap.add_argument("--table-rows", type=int, default=300)
    ap.add_argument("--watch", action="store_true", help="Poll the dirs every 60 s and log new reports.")
    args = ap.parse_args()
    while True:
        for path in reports(args.paths):
            try:
                run_path = log_report(path, args.group, args.table_rows)
            except Exception as e:  # one bad report must not stop the watcher
                print(f"{path}: {type(e).__name__}: {e}", flush=True)
                continue
            if run_path:
                print(f"{path} -> {run_path}", flush=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
