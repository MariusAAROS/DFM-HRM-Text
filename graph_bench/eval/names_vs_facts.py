"""Does a model answer from the facts it is given or from entity names? Mimir E1 vs Qwen3.5-4B E1, mean over seeds
(wiki: mimir-graph-vs-baselines, "Names or facts?").

  1. named test, KGQA rows by what the facts allow (path in facts / no path, answers in facts / not all answers in facts)
  2. named test, by hop count (WebQSP, CWQ, MetaQA) and chain length (CLUTRR)
  3. the same questions with names and anonymised (MetaQA, KQA Pro, CWQ test; WebQSP's anonymised rows are all
     dropped because the question loses its entity), paired by (source, id)

Row sources: a .samples.jsonl row has no source field, and the *_retrieved files reuse the ids of the plain test
files, so the named rows take their source from E3e1's report (same rows, same order, asserted). The anonymised rows
all come from */test files, so their source is the id prefix.

    .venv/bin/python -m graph_bench.eval.names_vs_facts
"""
import argparse
import json

import numpy as np
import pandas as pd

from graph_bench.eval.trace_gap_breakdown import BENCH, KGQA, derivable
from graph_bench.schema import read_records

ROOT = "/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench/"
SOURCE_REPORT = "e3e1/mimir15/named_t3k.samples.jsonl"  # has a source field per row
MODELS = {"Mimir E1": ["e1/mimir15", "e1/mimir15_s1", "e1/mimir15_s2"],
          "Qwen3.5-4B E1": ["e1/qwen35_4b_lr5e-5", "e1/qwen35_4b_lr5e-5_s1", "e1/qwen35_4b_lr5e-5_s2"]}

_records = {}


def record(src: str, rid: str):
    if src not in _records:
        _records[src] = {r.id: r for r in read_records(BENCH / f"{src}.jsonl")}
    return _records[src][rid]


def load(kind: str) -> dict:
    """{model: DataFrame(source, id, dataset, task, allows, acc = mean over seeds)}"""
    ref = [json.loads(line) for line in open(ROOT + SOURCE_REPORT, encoding="utf-8")]
    out = {}
    for model, dirs in MODELS.items():
        seeds = [[json.loads(line) for line in open(f"{ROOT}{d}/{kind}.samples.jsonl", encoding="utf-8")] for d in dirs]
        rows = seeds[0]
        assert all([r["id"] for r in s] == [r["id"] for r in rows] for s in seeds), f"{model} {kind}: rows differ by seed"
        ids = [r["id"].removesuffix("#anon") for r in rows]
        if kind == "named":
            assert ids == [r["id"] for r in ref], f"{model}: named rows differ from {SOURCE_REPORT}"
            srcs = [r["source"] for r in ref]
        else:
            srcs = [i.rsplit("/", 1)[0] for i in ids]
        out[model] = pd.DataFrame({
            "source": srcs, "id": ids, "dataset": [r["dataset"] for r in rows], "task": [r["task"] for r in rows],
            "allows": [derivable(record(s, i)) if r["dataset"] in KGQA else None for s, i, r in zip(srcs, ids, rows)],
            "acc": np.mean([[float(r["acc"][0]) for r in s] for s in seeds], axis=0)})
    return out


def side_by_side(frames: dict, keys: list) -> pd.DataFrame:
    (a_name, a), (b_name, b) = frames.items()
    ga, gb = a.groupby(keys)["acc"].mean(), b.groupby(keys)["acc"].mean()
    return pd.DataFrame({"n": a.groupby(keys).size(), a_name: ga, b_name: gb, "Δ": ga - gb}).round(3)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    named, anon = load("named"), load("anon")
    print("== 1. named test, KGQA rows by what the facts allow")
    print(side_by_side({m: d[d.dataset.isin(KGQA)] for m, d in named.items()}, ["dataset", "allows"]))
    print("\n== 2. named test, by hops / chain length")
    keep = ["webqsp", "cwq", "metaqa", "clutrr"]
    print(side_by_side({m: d[d.dataset.isin(keep)] for m, d in named.items()}, ["dataset", "task"]))
    print("\n== 3. the same questions with names and anonymised")
    rows = []
    for model in MODELS:
        a = anon[model].set_index(["source", "id"])
        n = named[model].set_index(["source", "id"]).loc[a.index]
        groups = [(("all", "all"), a.index)] + [(k, g.index) for k, g in a.groupby(["dataset", "allows"])]
        for (ds, allows), idx in groups:
            rows.append({"model": model, "dataset": ds, "allows": allows, "n": len(idx),
                         "named": n.loc[idx, "acc"].mean(), "anon": a.loc[idx, "acc"].mean()})
    p = pd.DataFrame(rows)
    p["anon - named"] = p["anon"] - p["named"]
    print(p.pivot_table(index=["dataset", "allows"], columns="model", values=["n", "named", "anon", "anon - named"]).round(3))


if __name__ == "__main__":
    main()
