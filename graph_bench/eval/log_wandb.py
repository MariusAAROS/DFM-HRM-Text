"""Log finished eval_vllm reports to W&B (wiki: mimir-graph-benchmark). eval_vllm.py itself only writes JSON.

One W&B run per report: project Mimir-Graph-RLVR (the whole benchmark: E0-E4), job_type eval, group --group, by
default mimir-bench-<arm> for a report at <root>/<arm>/<model>/<cond>.json; name <arm>/<model>/<cond>. Summary: acc / f1 / pass@k per dataset and dataset/task, macro_acc; config: the
report's settings; a table of the first --table-rows samples (prompt tail, generation, pred, gold, acc).
A report already logged is marked by a <report>.wandb file holding the run path, and skipped.

    .venv/bin/python -m graph_bench.eval.log_wandb $ROOT/e0/*/*.json
    .venv/bin/python -m graph_bench.eval.log_wandb --watch $ROOT/e0 $ROOT/e1 $ROOT/e2   # until killed
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
import wandb  # noqa: E402

PROJECT = "Mimir-Graph-RLVR"


def log_report(path: Path, group: str | None, table_rows: int) -> str | None:
    marker = path.with_suffix(".wandb")
    if marker.exists():
        return None
    report = json.loads(path.read_text())
    model_name, arm = path.parent.name, path.parent.parent.name
    group = group or f"mimir-bench-{arm}"
    summary = {}
    for name, s in report["summary"].items():
        if isinstance(s, dict):
            summary.update({f"{name}/{k}": v for k, v in s.items()})
        else:
            summary[name] = s
    config = {k: v for k, v in report.items() if k != "summary"} | {"report": str(path), "model_name": model_name,
                                                                    "condition": path.stem, "arm": arm}
    run = wandb.init(project=PROJECT, group=group, job_type="eval", name=f"{arm}/{model_name}/{path.stem}",
                     tags=[group, arm, model_name, path.stem], config=config, reinit="finish_previous")
    run.summary.update(summary)
    samples = path.with_suffix(".samples.jsonl")
    if samples.exists() and table_rows:
        table = wandb.Table(columns=["id", "dataset", "task", "acc", "gold", "pred", "generation", "prompt_tail"])
        with open(samples, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= table_rows:
                    break
                r = json.loads(line)
                table.add_data(r["id"], r["dataset"], r["task"], r["acc"][0], json.dumps(r["gold"], ensure_ascii=False),
                               str(r["preds"][0]), r["generations"][0], r["prompt_tail"])
        run.log({"samples": table})
    run_path = run.path
    run.finish()
    marker.write_text(run_path + "\n")
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
            run_path = log_report(path, args.group, args.table_rows)
            if run_path:
                print(f"{path} -> {run_path}", flush=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
