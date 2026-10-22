"""Wikidata5M (transductive) -> triple store kg/wikidata5m with the official train/valid/test splits.

Entity label = first alias, description = the Wikipedia paragraph from wikidata5m_text.txt.
The valid+test triples are also written to kg/wikidata5m/benchmark_triples.tsv; holdout.py --force-test
marks them "test" in the Wikidata (and YAGO) stores.
"""
from __future__ import annotations

import collections
import gzip
import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KG, RAW, STATS, TripleStoreWriter, write_entities, write_json, write_relations  # noqa: E402


def tar_lines(path: Path, member: str):
    with tarfile.open(path, "r:gz") as tf:
        fb = tf.extractfile(member)
        yield from io.TextIOWrapper(fb, encoding="utf-8")


def relabel():
    """Replace alias labels with Wikidata's English labels: the first alias is often not the name
    ("united stated", a Tamil transliteration of Singapore). Relations get Wikidata property labels.
    Aliases stay as the fallback for items missing from kg/wikidata (blocklisted classes)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    d = KG / "wikidata5m"
    ent = pq.read_table(d / "entities.parquet")
    wd = pq.read_table(KG / "wikidata" / "entities.parquet", columns=["entity_id", "label"])
    wd = wd.filter(pc.is_in(wd["entity_id"], value_set=ent["entity_id"]))
    wd_label = dict(zip(wd["entity_id"].to_pylist(), wd["label"].to_pylist()))
    ids, old = ent["entity_id"].to_pylist(), ent["label"].to_pylist()
    new = [wd_label.get(e, l) for e, l in zip(ids, old)]
    ent = ent.set_column(ent.schema.get_field_index("label"), "label", pa.array(new, pa.string()))
    pq.write_table(ent, d / "entities.parquet", compression="zstd")
    plabel = {}
    with open(RAW / "wikidata" / "properties.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p, _, label = (line.rstrip("\n").split("\t") + [""])[:3]
            if label.startswith('"'):
                plabel[p.rsplit("/", 1)[-1].rstrip(">")] = label[1:label.rfind('"')]
    rel = pq.read_table(d / "relations.parquet")
    rids = rel["relation_id"].to_pylist()
    rel = rel.set_column(rel.schema.get_field_index("label"), "label",
                         pa.array([plabel.get(r, l) for r, l in zip(rids, rel["label"].to_pylist())], pa.string()))
    pq.write_table(rel, d / "relations.parquet", compression="zstd")
    info = {"entities": len(ids), "wikidata_label": len(wd_label), "changed": sum(a != b for a, b in zip(old, new)),
            "relations_with_wikidata_label": sum(r in plabel for r in rids), "relations": len(rids)}
    write_json(STATS / "wikidata5m_relabel.json", info)
    print(info)


def main():
    if "--relabel" in sys.argv:
        return relabel()
    raw = RAW / "wikidata5m"
    labels, rel_labels = {}, {}
    for line in tar_lines(raw / "wikidata5m_alias.tar.gz", "wikidata5m_entity.txt"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1].strip():
            labels[p[0]] = p[1].strip()
    for line in tar_lines(raw / "wikidata5m_alias.tar.gz", "wikidata5m_relation.txt"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            rel_labels[p[0]] = p[1].strip()
    desc = {}
    with gzip.open(raw / "wikidata5m_text.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            q, _, text = line.rstrip("\n").partition("\t")
            if text.strip():
                desc[q] = text.strip()

    store = TripleStoreWriter("wikidata5m", n_shards=16)
    counts, drops = collections.Counter(), collections.Counter()
    used_rel = set()
    with open(KG / "wikidata5m" / "benchmark_triples.tsv", "w", encoding="utf-8") as bench:
        for split, name in (("train", "train"), ("valid", "valid"), ("test", "test")):
            for line in tar_lines(raw / "wikidata5m_transductive.tar.gz", f"wikidata5m_transductive_{name}.txt"):
                p = line.rstrip("\n").split("\t")
                if len(p) != 3:
                    drops["malformed"] += 1
                    continue
                s, r, o = p
                if s not in labels or o not in labels or r not in rel_labels:
                    drops["unlabeled"] += 1
                    continue
                store.add(s, r, o, split=split)
                used_rel.add(r)
                counts[split] += 1
                if split != "train":
                    bench.write(f"{s}\t{r}\t{o}\n")
    n = store.close()
    write_entities("wikidata5m", ({"entity_id": q, "label": l, "description": desc.get(q)} for q, l in labels.items()))
    write_relations("wikidata5m", {r: {"label": rel_labels[r]} for r in sorted(used_rel)})
    info = {"triples": n, "splits": dict(counts), "entities": len(labels), "with_description": len(desc),
            "relations": len(used_rel), "dropped": dict(drops)}
    write_json(STATS / "wikidata5m_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
