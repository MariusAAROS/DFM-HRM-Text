"""T-REx (Wikipedia abstracts aligned with Wikidata triples) -> kg-trex documents + aligned sentence pairs.

Document: "<title>\\n<abstract>\\n\\nFacts:\\n- subject | predicate | object" for the aligned triples.
Entities use their Wikidata English label (kg/wikidata/entities.parquet), so a fact never reads
"He | member of sports team | ..."; the surface form is the fallback (dates, items missing from kg/wikidata).
Predicate labels come from raw/wikidata/properties.tsv.
Triples in the Wikidata5M valid/test set (kg/wikidata5m/benchmark_triples.tsv, either direction) are left
out of the fact list. Abstracts longer than MAX_DOC_CHARS are cut at a sentence boundary, keeping only the
facts of the kept sentences.
kg/trex/aligned-*.parquet keeps (doc, sentence, triples) for later sentence <-> triples tasks.
"""
from __future__ import annotations

import collections
import json
import multiprocessing as mp
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANONICAL, CONVERTED, KG, RAW, STATS, DocWriter, clear_dir, write_json  # noqa: E402

ZIP = RAW / "trex" / "T-REx.zip"
MAX_DOC_CHARS = 12000
G: dict = {}


def qid(uri: str | None) -> str | None:
    return uri.rsplit("/", 1)[-1] if uri else None


def _file(args) -> dict:
    i, member = args
    plabel, bench, ent = G["plabel"], G["bench"], G["entities"]
    c = collections.Counter()
    pairs = {"doc": [], "sentence": [], "triples": []}
    with zipfile.ZipFile(ZIP) as z:
        docs = json.loads(z.read(member))
    qids = {qid(t[k].get("uri")) for d in docs for t in d["triples"] for k in ("subject", "object")}
    sub = ent.filter(pc.is_in(ent["entity_id"], value_set=pa.array(sorted(q for q in qids if q))))
    elabel = dict(zip(sub["entity_id"].to_pylist(), sub["label"].to_pylist()))
    with DocWriter("trex", prefix=f"part{i:05d}", fresh=False) as w:
        for d in docs:
            text, title = (d.get("text") or "").strip(), (d.get("title") or "").strip()
            if not text:
                c["empty"] += 1
                continue
            bounds = json.loads(d["sentences_boundaries"]) if isinstance(d["sentences_boundaries"], str) else d["sentences_boundaries"]
            limit = len(text)
            if len(text) > MAX_DOC_CHARS:
                ends = [e for _, e in bounds if e <= MAX_DOC_CHARS]
                limit = ends[-1] if ends else MAX_DOC_CHARS
                text = text[:limit].strip()
                c["truncated"] += 1
            facts, by_sent = [], collections.defaultdict(list)
            for t in d["triples"]:
                s, p, o = t["subject"], t["predicate"], t["object"]
                pid = qid(p.get("uri"))
                if pid not in plabel:
                    c["unknown_predicate"] += 1
                    continue
                sq, oq = qid(s.get("uri")), qid(o.get("uri"))
                if (sq, pid, oq) in bench or (oq, pid, sq) in bench:
                    c["benchmark_triple"] += 1
                    continue
                ssf = elabel.get(sq) or s.get("surfaceform") or (title if s.get("uri") == d.get("uri") else None)
                osf = elabel.get(oq) or o.get("surfaceform")
                if not ssf or not osf:
                    c["no_surface_form"] += 1
                    continue
                sid = t.get("sentence_id")
                if sid is not None and sid < len(bounds) and bounds[sid][1] > limit:
                    continue
                fact = f"{ssf} | {plabel[pid]} | {osf}"
                if fact not in facts:
                    facts.append(fact)
                    if sid is not None and sid < len(bounds):
                        by_sent[sid].append([ssf, plabel[pid], osf, sq, pid, oq])
            body = f"{title}\n{text}" if title and not text.startswith(title) else text
            if facts:
                body += "\n\nFacts:\n" + "\n".join(f"- {f}" for f in facts)
            w.write(qid(d.get("uri")) or d.get("docid"), body, {"title": title, "n_facts": len(facts)})
            c["docs"] += 1
            c["facts"] += len(facts)
            for sid, ts in by_sent.items():
                a, b = bounds[sid]
                pairs["doc"].append(qid(d.get("uri")))
                pairs["sentence"].append(d["text"][a:b])
                pairs["triples"].append(json.dumps(ts, ensure_ascii=False))
    pq.write_table(pa.table(pairs), KG / "trex" / f"aligned-{i:05d}.parquet", compression="zstd")
    return dict(c)


def main():
    plabel = {}
    with open(RAW / "wikidata" / "properties.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p, _, label = (line.rstrip("\n").split("\t") + [""])[:3]
            if label.startswith('"'):
                plabel[p.rsplit("/", 1)[-1].rstrip(">")] = label[1:label.rfind('"')]
    bench = set()
    with open(KG / "wikidata5m" / "benchmark_triples.tsv", encoding="utf-8") as f:
        for line in f:
            bench.add(tuple(line.rstrip("\n").split("\t")))
    (KG / "trex").mkdir(parents=True, exist_ok=True)
    for p in (KG / "trex").glob("aligned-*.parquet"):
        p.unlink()
    clear_dir(CANONICAL / "trex")
    clear_dir(CONVERTED / "kg-trex")
    with zipfile.ZipFile(ZIP) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".json"))
    G.update(plabel=plabel, bench=bench,
             entities=pq.read_table(KG / "wikidata" / "entities.parquet", columns=["entity_id", "label"]))
    total = collections.Counter()
    with mp.get_context("fork").Pool(32) as pool:
        for c in pool.imap_unordered(_file, list(enumerate(members))):
            total.update(c)
    info = {"files": len(members), **total}
    write_json(STATS / "trex_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
