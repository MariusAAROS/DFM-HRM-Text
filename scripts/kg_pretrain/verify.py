"""Triple store checks (plan, verification step 5). Writes stats/verify_<source>.json.

  1. Every s / o id resolves in entities.parquet, and every r in relations.parquet.
  2. The split counts match holdout.json.
  3. No leakage: for up to --sample held-out (valid/test) entity triples, the object label does not appear on
     the subject's "- <relation>: ..." line in its rendered documents. It can only appear when another train
     triple has the same subject, relation and object label; those cases are counted separately.
  4. --benchmark TSV (s, r, o): those triples, where present in the store, are all split == "test".

    python -I scripts/kg_pretrain/verify.py conceptnet
    python -I scripts/kg_pretrain/verify.py wikidata --benchmark data/kg_pretrain/kg/wikidata5m/benchmark_triples.tsv
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANONICAL, KG, STATS, triple_shards, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--sample", type=int, default=10_000)
    ap.add_argument("--benchmark", type=Path)
    a = ap.parse_args()
    d = KG / a.source
    ent_ids = pq.read_table(d / "entities.parquet", columns=["entity_id"])["entity_id"]
    rel = pq.read_table(d / "relations.parquet")
    rel_label = dict(zip(rel["relation_id"].to_pylist(), rel["label"].to_pylist()))
    rel_ids = pa.array(list(rel_label))
    out = {"source": a.source}

    # 1 + 2
    splits, unresolved = collections.Counter(), collections.Counter()
    held = []
    for p in triple_shards(a.source):
        t = pq.read_table(p)
        for sp, n in zip(*[x.to_pylist() for x in pc.value_counts(t["split"]).flatten()]):
            splits[sp] += n
        unresolved["s"] += int(pc.sum(pc.invert(pc.is_in(t["s"], value_set=ent_ids))).as_py() or 0)
        o = t["o"].drop_null()
        unresolved["o"] += int(pc.sum(pc.invert(pc.is_in(o, value_set=ent_ids))).as_py() or 0)
        unresolved["r"] += int(pc.sum(pc.invert(pc.is_in(t["r"], value_set=rel_ids))).as_py() or 0)
        h = t.filter(pc.and_(pc.not_equal(t["split"], "train"), pc.is_valid(t["o"])))
        held += list(zip(h["s"].to_pylist(), h["r"].to_pylist(), h["o"].to_pylist()))
    out["splits"] = dict(splits)
    out["unresolved_ids"] = dict(unresolved)
    hj = d / "holdout.json"
    if hj.exists():
        counts = json.loads(hj.read_text())["counts"]
        out["holdout_json_matches"] = {k: v for k, v in counts.items() if k in ("train", "valid", "test")} == dict(splits)

    # 3
    random.seed(0)
    sample = random.sample(held, min(a.sample, len(held)))
    by_subject = collections.defaultdict(list)
    for s, r, o in sample:
        by_subject[s].append((r, o))
    labels_needed = pa.array(sorted({x for s, _, o in sample for x in (s, o)}))
    ent = pq.read_table(d / "entities.parquet", columns=["entity_id", "label"])
    ent = ent.filter(pc.is_in(ent["entity_id"], value_set=labels_needed))
    label = dict(zip(ent["entity_id"].to_pylist(), ent["label"].to_pylist()))
    # train objects with the same (s, r) and the same label explain a legitimate appearance
    same_label_train = set()
    subj = pa.array(sorted(by_subject))
    train_rows = []
    for p in triple_shards(a.source):
        t = pq.read_table(p)
        t = t.filter(pc.and_(pc.is_in(t["s"], value_set=subj), pc.equal(t["split"], "train")))
        train_rows += [x for x in zip(t["s"].to_pylist(), t["r"].to_pylist(), t["o"].to_pylist()) if x[2] is not None]
    all_ent = pq.read_table(d / "entities.parquet", columns=["entity_id", "label"])  # read once
    objs = pa.array(sorted({o for _, _, o in train_rows}), pa.string())
    e2 = all_ent.filter(pc.is_in(all_ent["entity_id"], value_set=objs))
    ol = dict(zip(e2["entity_id"].to_pylist(), e2["label"].to_pylist()))
    del all_ent, e2
    for s, r, o in train_rows:
        if o in ol:
            same_label_train.add((s, rel_label[r], ol[o]))
    docs = collections.defaultdict(list)
    for p in sorted((CANONICAL / a.source).glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                i = line.rfind('"entity_id": "')  # cheap pre-check; parse only the sampled subjects
                if i < 0:
                    continue
                e = line[i + 14:line.index('"', i + 14)]
                if e in by_subject:
                    docs[e].append(json.loads(line)["text"])
    leaks = explained = checked = 0
    examples = []
    for s, pairs in by_subject.items():
        lines = [ln for t in docs.get(s, []) for ln in t.split("\n")]
        for r, o in pairs:
            ol, rl = label.get(o), rel_label[r]
            if ol is None:
                continue
            checked += 1
            prefix = f"- {rl}: "
            hit = any(ln.startswith(prefix) and ol in ln[len(prefix):].split("; ") for ln in lines)
            if hit:
                if (s, rl, ol) in same_label_train:
                    explained += 1
                else:
                    leaks += 1
                    if len(examples) < 5:
                        examples.append((s, r, o, ol))
    out["leak_check"] = {"held_out_entity_triples": len(held), "checked": checked, "leaks": leaks,
                         "explained_by_same_label_train_fact": explained, "examples": examples}

    # 4
    if a.benchmark:
        bench = {tuple(l.rstrip("\n").split("\t")) for l in open(a.benchmark, encoding="utf-8")}
        found = collections.Counter()
        bs = pa.array(sorted({s for s, _, _ in bench} | {o for _, _, o in bench}))
        for p in triple_shards(a.source):
            t = pq.read_table(p)
            t = t.filter(pc.is_in(t["s"], value_set=bs))
            for s, r, o, sp in zip(*(t[c].to_pylist() for c in ("s", "r", "o", "split"))):
                if (s, r, o) in bench or (o, r, s) in bench:
                    found[sp] += 1
        out["benchmark_in_store"] = {"benchmark_triples": len(bench), "found_by_split": dict(found),
                                     "ok": set(found) <= {"test"}}
    write_json(STATS / f"verify_{a.source}.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
