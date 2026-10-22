"""Assign the valid/test holdout of a triple store (data/kg_pretrain/kg/<source>/triples-*.parquet).

Rule (hash mode, the default):
  - Only entity-object triples are candidates.
  - A triple is held out by a seeded hash of (min(s, o), r, max(s, o)), where r drops any "|..." direction
    suffix. Both directions of an edge (PrimeKG lists every edge twice) therefore get the same decision.
  - A candidate is kept only if s and o both have degree >= 3 (counted over all triples), so held-out
    entities stay seen in train.
  - The split is fixed once and stored in the `split` column. Every training view reads split == "train".

--keep-existing: keep splits already present (Wikidata5M ships official valid/test) and only add
--force-test triples.
--force-test FILE: TSV of s<TAB>r<TAB>o (e.g. the Wikidata5M valid+test triples) marked "test" in either
direction, so the benchmark never leaks through a superset KG.

    python -I scripts/kg_pretrain/holdout.py conceptnet
    python -I scripts/kg_pretrain/holdout.py wikidata --force-test data/kg_pretrain/kg/wikidata5m/benchmark_triples.tsv
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KG, stable_hash, triple_shards, write_json  # noqa: E402

BUCKETS = 1_000_000
G: dict = {}  # shared with forked workers


def _degrees(path: Path) -> collections.Counter:
    t = pq.read_table(path, columns=["s", "o"])
    c = collections.Counter()
    for col in ("s", "o"):
        for d in pc.value_counts(t[col].drop_null()).to_pylist():
            c[d["values"]] += d["counts"]
    return c


def _assign(path: Path) -> dict:
    a = G["args"]
    t = pq.read_table(path)
    s, r, o, split = (t[c].to_pylist() for c in ("s", "r", "o", "split"))
    deg, force = G["deg"], G["force"]
    lo_valid, lo_test = int(a.valid * BUCKETS), int((a.valid + a.test) * BUCKETS)
    out, counts = [], collections.Counter()
    for si, ri, oi, sp in zip(s, r, o, split):
        if not a.keep_existing:
            sp = "train"
        if oi is not None:
            if force and ((si, ri, oi) in force or (oi, ri, si) in force):
                sp = "test"
                counts["forced_test"] += 1
            elif sp == "train" and not a.keep_existing:
                a_, b_ = (si, oi) if si <= oi else (oi, si)
                # A "|direction" suffix (PrimeKG: indication|drug|disease) is ignored, so both directions agree.
                h = stable_hash(a_, ri.split("|", 1)[0], b_, seed=a.seed) % BUCKETS
                if h < lo_test and deg.get(si, 0) >= 3 and deg.get(oi, 0) >= 3:
                    sp = "valid" if h < lo_valid else "test"
        out.append(sp)
        counts[sp] += 1
    t = t.set_column(t.schema.get_field_index("split"), "split", pa.array(out, pa.string()))
    tmp = path.with_suffix(".tmp")
    pq.write_table(t, tmp, compression="zstd")
    tmp.replace(path)
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--valid", type=float, default=0.001)
    ap.add_argument("--test", type=float, default=0.001)
    ap.add_argument("--seed", default="kg-holdout-2026")
    ap.add_argument("--keep-existing", action="store_true")
    ap.add_argument("--force-test", type=Path)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    shards = triple_shards(a.source)
    assert shards, f"no triple shards for {a.source}"
    workers = min(a.workers, len(shards))

    deg = collections.Counter()
    if not a.keep_existing:
        with mp.get_context("fork").Pool(workers) as pool:
            for c in pool.imap_unordered(_degrees, shards):
                deg.update(c)
    force = set()
    if a.force_test:
        with open(a.force_test, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 3:
                    force.add(tuple(p))
    G.update(args=a, deg=deg, force=force)
    total = collections.Counter()
    with mp.get_context("fork").Pool(workers) as pool:
        for c in pool.imap_unordered(_assign, shards):
            total.update(c)
    info = {"source": a.source, "mode": "keep-existing" if a.keep_existing else "hash",
            "seed": a.seed, "valid_frac": a.valid, "test_frac": a.test,
            "rule": "entity-object triples; hash(min(s,o), r, max(s,o)) bucket; deg(s), deg(o) >= 3",
            "force_test_file": str(a.force_test) if a.force_test else None,
            "force_test_triples": len(force), "counts": dict(total)}
    write_json(KG / a.source / "holdout.json", info)
    print(info)


if __name__ == "__main__":
    main()
