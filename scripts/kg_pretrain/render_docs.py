"""Render a triple store into entity-centric documents (canonical + data_io converted views).

Only split == "train" triples are rendered. One document per subject; documents longer than --max-chars
are split on fact boundaries with the header (label, description) repeated.
Relations are ordered by --first, then by global frequency; values keep the store order.

    python -I scripts/kg_pretrain/render_docs.py hetionet --inverse --types-line
    python -I scripts/kg_pretrain/render_docs.py wikidata --first P31,P279 --workers 48
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import sys
from itertools import groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANONICAL, CONVERTED, KG, MAX_CHARS, DocWriter, clear_dir, entity_doc_chunks, triple_shards  # noqa: E402

G: dict = {}  # loaded before fork; arrow buffers are shared copy-on-write


def _labels_for(ids: pa.Array) -> dict:
    """Map the given entity ids to (label, description, types); the hash set is built on the (small) id side."""
    ent = G["entities"]
    j = ent.filter(pc.is_in(ent["entity_id"], value_set=pc.unique(ids.drop_null())))
    return {e: (l, d, t) for e, l, d, t in zip(j["entity_id"].to_pylist(), j["label"].to_pylist(),
                                                j["description"].to_pylist(), j["types"].to_pylist())}


def _render_shard(args) -> dict:
    k, paths = args
    a, rel, rank = G["args"], G["relations"], G["rank"]
    t = pa.concat_tables(pq.read_table(p) for p in paths)
    t = t.filter(pc.equal(t["split"], "train"))
    if G["skip"]:
        t = t.filter(pc.invert(pc.is_in(t["r"], value_set=pa.array(sorted(G["skip"]), pa.string()))))
    lab = _labels_for(pa.concat_arrays([t["s"].combine_chunks(), t["o"].combine_chunks()]))
    facts = collections.defaultdict(list)  # s -> [(r, value)]
    dropped = 0
    for s, r, o, lit in zip(t["s"].to_pylist(), t["r"].to_pylist(), t["o"].to_pylist(), t["o_literal"].to_pylist()):
        if o is not None:
            if o not in lab:
                dropped += 1
                continue
            v = lab[o][0]
        else:
            v = lit
        facts[s].append((rank.get(r, 1 << 30), rel[r][0], v))
        if a.inverse and o is not None and rel[r][1]:
            facts[o].append((rank.get(r, 1 << 30) + 0.5, rel[r][1], lab.get(s, (s,))[0]))
    if a.inverse:  # incoming edges add subjects that need labels too
        missing = [e for e in facts if e not in lab]
        if missing:
            lab.update(_labels_for(pa.array(missing, pa.string())))
    n_docs = n_chunks = few = 0
    with DocWriter(a.source, prefix=f"part{k:05d}", fresh=False) as w:
        for s in sorted(facts):
            if s not in lab:
                dropped += len(facts[s])
                continue
            label, desc, types = lab[s]
            fl = sorted(facts[s], key=lambda x: x[0])
            if len(fl) < a.min_facts:
                few += 1
                continue
            grouped = []
            if a.types_line and types:
                grouped.append(("type", list(types)))
            for (_, rlabel), grp in groupby(fl, key=lambda x: (x[0], x[1])):
                vals = list(dict.fromkeys(v for _, _, v in grp))
                grouped.append((rlabel, vals))
            chunks = entity_doc_chunks(label, desc, grouped, a.max_chars)
            for i, text in enumerate(chunks):
                w.write(f"{s}#{i}" if len(chunks) > 1 else s, text,
                        {"entity_id": s, "n_facts": len(fl), "chunk": i, "n_chunks": len(chunks)})
            n_docs += 1
            n_chunks += len(chunks)
    return {"entities": n_docs, "rows": n_chunks, "dropped_unlabeled": dropped, "skipped_few_facts": few}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--first", default="", help="comma-separated relation ids rendered first, in this order")
    ap.add_argument("--inverse", action="store_true", help="also render incoming edges with inverse_label")
    ap.add_argument("--types-line", action="store_true", help="render entity types as a first '- type:' line")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--min-facts", type=int, default=1, help="skip subjects with fewer rendered facts")
    ap.add_argument("--skip-relations", default="", help="comma-separated relation ids not rendered (store keeps them)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--group", type=int, default=1, help="triple shards per render task")
    a = ap.parse_args()

    shards = triple_shards(a.source)
    assert shards, f"no triple shards for {a.source}"
    if a.inverse:  # incoming edges of a subject live in other shards: render everything in one task
        a.group = len(shards)
    rel_t = pq.read_table(KG / a.source / "relations.parquet")
    relations = {r: (l, inv) for r, l, inv in zip(rel_t["relation_id"].to_pylist(), rel_t["label"].to_pylist(),
                                                   rel_t["inverse_label"].to_pylist())}
    freq = collections.Counter()
    for p in shards:
        for d in pc.value_counts(pq.read_table(p, columns=["r"])["r"]).to_pylist():
            freq[d["values"]] += d["counts"]
    first = [r for r in a.first.split(",") if r]
    order = first + [r for r, _ in freq.most_common() if r not in first]
    G.update(args=a, relations=relations, skip={r for r in a.skip_relations.split(",") if r}, rank={r: i for i, r in enumerate(order)},
             entities=pq.read_table(KG / a.source / "entities.parquet"))

    clear_dir(CANONICAL / a.source)
    clear_dir(CONVERTED / f"kg-{a.source}")
    tasks = [(i, shards[i * a.group:(i + 1) * a.group]) for i in range((len(shards) + a.group - 1) // a.group)]
    total = collections.Counter()
    if len(tasks) == 1:
        total.update(_render_shard(tasks[0]))
    else:
        with mp.get_context("fork").Pool(min(a.workers, len(tasks))) as pool:
            for c in pool.imap_unordered(_render_shard, tasks):
                total.update(c)
    print({"source": a.source, **total})


if __name__ == "__main__":
    main()
