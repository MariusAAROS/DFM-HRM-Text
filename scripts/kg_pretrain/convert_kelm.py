"""KELM generated corpus (Wikidata triples verbalized as sentences) -> kg-kelm documents + aligned pairs.

Each item renders as the generated sentence followed by its facts:
    Valentin Lavigne played for FC Lorient between 2014 and 2016.
    Facts: Valentin Lavigne | member of sports team | FC Lorient; (Valentin Lavigne, FC Lorient) | start time | ...
Items are sorted by subject and packed, in that order, into documents of at most MAX_CHARS (most subjects
have a single item, so per-subject documents would be tiny).
Fact values are detokenized (clean_value). Dropped: sentences with the generator's "⁇" (unknown character) artifact, and items that contain a
Wikidata5M valid/test triple. Those are matched on English labels, from kg/wikidata entities and
raw/wikidata/properties.tsv, so this runs after the Wikidata pass 2.
kg/kelm/aligned.parquet keeps (subject, sentence, triples) for later sentence <-> triples tasks.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KG, RAW, STATS, DocWriter, pack_texts, write_json  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.replace(" ( ", " (").replace(" )", ")").split()).lower()


def benchmark_label_triples() -> set:
    bench = [l.rstrip("\n").split("\t") for l in open(KG / "wikidata5m" / "benchmark_triples.tsv", encoding="utf-8")]
    ids = sorted({x for s, _, o in bench for x in (s, o)})
    ent = pq.read_table(KG / "wikidata" / "entities.parquet", columns=["entity_id", "label"])
    ent = ent.filter(pc.is_in(ent["entity_id"], value_set=pa.array(ids)))
    lab = dict(zip(ent["entity_id"].to_pylist(), ent["label"].to_pylist()))
    plab = {}
    with open(RAW / "wikidata" / "properties.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p, _, label = (line.rstrip("\n").split("\t") + [""])[:3]
            if label.startswith('"'):
                plab[p.rsplit("/", 1)[-1].rstrip(">")] = label[1:label.rfind('"')]
    out = set()
    for s, r, o in bench:
        if s in lab and o in lab and r in plab:
            out.add((norm(lab[s]), norm(plab[r]), norm(lab[o])))
            out.add((norm(lab[o]), norm(plab[r]), norm(lab[s])))
    return out


_DETOK = [(re.compile(r"\( (.*?) \)"), r"(\1)"), (re.compile(r" (['’]s)\b"), r"\1"), (re.compile(r" -- "), " – "),
          (re.compile(r" ([,.;:])"), r"\1")]


def clean_value(x: str) -> str:
    """KELM values are space-tokenized: "Keith ( given name )", "Men 's", "+2116", "00  2000"."""
    x = " ".join(x.split())
    for pat, rep in _DETOK:
        x = pat.sub(rep, x)
    m = re.fullmatch(r"00 (?:00 )?(-?\d{1,4})", x)  # unknown day/month
    if m:
        return m.group(1)
    if re.fullmatch(r"[+]\d+(\.\d+)?", x):
        return x[1:]
    return x


def render_fact(t: list[str]) -> str:
    if len(t) == 3:
        return " | ".join(clean_value(x) for x in t)
    s, o, q, v = (clean_value(x) for x in t[:4])
    return f"({s}, {o}) | {q} | {v}"


def main():
    bench = benchmark_label_triples()
    items, c = [], collections.Counter()
    aligned = {"subject": [], "sentence": [], "triples": []}
    with open(RAW / "kelm" / "kelm_generated_corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sent, triples = (r.get("gen_sentence") or "").strip(), r.get("triples") or []
            if not sent or not triples:
                c["empty"] += 1
                continue
            if "⁇" in sent:
                c["unk_artifact"] += 1
                continue
            if any(len(t) == 3 and (norm(t[0]), norm(t[1]), norm(t[2])) in bench for t in triples):
                c["benchmark_triple"] += 1
                continue
            subj = triples[0][0].strip()
            items.append((subj, f"{sent}\nFacts: " + "; ".join(render_fact(t) for t in triples)))
            aligned["subject"].append(subj)
            aligned["sentence"].append(sent)
            aligned["triples"].append(json.dumps(triples, ensure_ascii=False))
    c["items"] = len(items)
    (KG / "kelm").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(aligned), KG / "kelm" / "aligned.parquet", compression="zstd")
    del aligned
    items.sort(key=lambda x: x[0])
    n = 0
    with DocWriter("kelm") as w:
        for keys, text in pack_texts(items, sep="\n\n"):
            w.write(f"{n}", text, {"subjects": sorted(set(keys)), "n_items": len(keys)})
            n += 1
    info = {"docs": n, "benchmark_label_triples": len(bench), **c}
    write_json(STATS / "kelm_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
