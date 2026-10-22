"""YAGO 4.6 -> triple store kg/yago (facts + beyond-wikipedia + taxonomy).

Entity ids are YAGO local names (yago:Belgium -> Belgium); other prefixes are kept (schema:Country).
Labels and descriptions are the English rdfs:label / schema:description from every file.
Skipped predicates: labels/names/descriptions, owl:sameAs (used for the Wikidata id map), siteLinks, images,
URLs. Literals are rendered (dates, numbers with their unit: "+31112"^^yago:United_States_dollar ->
"31112 United States dollar"). Non-English strings, URIs and geometries are dropped.
Also writes kg/yago/benchmark_triples.tsv: the Wikidata5M valid/test triples mapped via owl:sameAs and the
schema's ys:fromProperty, for holdout.py --force-test.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KG, RAW, STATS, TripleStoreWriter, write_entities, write_json, write_relations  # noqa: E402
from wikidata_pass1 import clean  # noqa: E402

Y = RAW / "yago"
FACT_FILES = ["yago-4.6-taxonomy.zip", "yago-4.6-facts.zip", "yago-4.6-beyond-wikipedia.zip"]
LABEL_FILES = FACT_FILES + ["yago-4.6-labels.zip", "yago-4.6-beyond-wikipedia-labels.zip"]
SKIP = {"rdfs:label", "schema:description", "schema:alternateName", "schema:name", "owl:sameAs",
        "yago:siteLinks", "schema:image", "schema:mainEntityOfPage", "schema:url", "rdfs:comment",
        "schema:sameAs", "schema:logo", "skos:prefLabel", "rdfs:seeAlso"}
REL_LABELS = {"rdf:type": "type", "rdfs:subClassOf": "subclass of"}
DATE = re.compile(r"^([+-]?)(\d+)(?:-(\d\d))?(?:-(\d\d))?")


def stream(zipname: str, grep: str | None = None):
    cmd = f"unzip -p '{Y / zipname}'"
    if grep:
        cmd += f" | LC_ALL=C grep -P '{grep}'"
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace", bufsize=1 << 22)
    yield from p.stdout
    p.wait()


def split_line(line: str):
    if line.startswith("@") or not line.strip():
        return None
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], "\t".join(parts[2:-1])


def local(tok: str) -> str:
    return tok[5:] if tok.startswith("yago:") else tok


def name_from_id(tok: str) -> str:
    name = tok.split(":", 1)[-1]
    name = re.sub(r"_u([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), name)
    return name.replace("_", " ")


def rel_label(p: str) -> str:
    if p in REL_LABELS:
        return REL_LABELS[p]
    name = p.split(":", 1)[-1]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def literal(obj: str) -> str | None:
    end = obj.rfind('"')
    if end <= 0:
        return None
    lex, suffix = clean(obj[1:end]), obj[end + 1:]
    if suffix.startswith("@"):
        return lex if suffix == "@en" else None
    if not suffix:
        return lex[:300] or None
    dt = suffix[2:]
    if dt in ("xsd:anyURI", "geo:wktLiteral", "rdf:langString"):
        return None
    if dt.startswith("xsd:date") or dt.startswith("xsd:gYear"):
        m = DATE.match(lex)
        if not m:
            return None
        sign, y, mo, d = m.groups()
        y = int(y)
        if y == 0:
            return None
        ys = f"{y} BC" if sign == "-" else str(y)
        if sign == "-" or not mo:
            return ys
        return f"{ys}-{mo}" if not d else f"{ys}-{mo}-{d}"
    if dt.startswith("xsd:"):
        return lex.lstrip("+") or None
    return f"{lex.lstrip('+')} {name_from_id(dt)}"  # unit datatype, e.g. yago:United_States_dollar


def main():
    labels, desc, qid = {}, {}, {}
    lab_re = r"\t(rdfs:label|schema:description|rdfs:comment)\t\".*\"@en\t\.$|\towl:sameAs\twd:Q"
    for f in LABEL_FILES:
        for line in stream(f, lab_re):
            t = split_line(line)
            if not t:
                continue
            s, p, o = t
            s = local(s)
            if p == "owl:sameAs":
                qid[o[3:]] = s
            elif p == "rdfs:label":
                labels.setdefault(s, clean(o[1:-4]))
            else:
                desc.setdefault(s, clean(o[1:-4]))
        print(f"{f}: labels {len(labels)} desc {len(desc)} sameAs {len(qid)}", flush=True)

    store = TripleStoreWriter("yago", n_shards=64, flush_rows=500_000)
    rels, ents, drops = {}, set(), collections.Counter()
    for f in FACT_FILES:
        for line in stream(f):
            t = split_line(line)
            if not t:
                continue
            s, p, o = t
            if p in SKIP:
                continue
            s = local(s)
            if s not in labels:
                drops["unlabeled_subject"] += 1
                continue
            if o.startswith('"'):
                v = literal(o)
                if v is None:
                    drops["literal"] += 1
                    continue
                store.add(s, p, o_literal=v)
            else:
                o = local(o)
                if o not in labels:
                    drops["unlabeled_object"] += 1
                    continue
                store.add(s, p, o)
                ents.add(o)
            ents.add(s)
            if p not in rels:
                rels[p] = {"label": rel_label(p)}
        print(f"{f}: triples {store.count} {dict(drops)}", flush=True)
    n = store.close()
    write_entities("yago", ({"entity_id": e, "label": labels[e], "description": desc.get(e)} for e in sorted(ents)))
    write_relations("yago", rels)

    # Wikidata5M benchmark triples in YAGO ids
    fromprop = collections.defaultdict(set)
    path = None
    for line in stream("yago-4.6-schema.zip"):
        m = re.search(r"sh:path (\S+) ;", line)
        if m:
            path = m.group(1)
        m = re.search(r"ys:fromProperty (.+?) ;", line)
        if m and path:
            for wp in re.findall(r"wdt:(P\d+)", m.group(1)):
                fromprop[wp].add(path)
    mapped = 0
    with open(KG / "wikidata5m" / "benchmark_triples.tsv", encoding="utf-8") as fin, \
            open(KG / "yago" / "benchmark_triples.tsv", "w", encoding="utf-8") as fout:
        for line in fin:
            s, r, o = line.rstrip("\n").split("\t")
            if s in qid and o in qid:
                for yp in fromprop.get(r, ()):
                    fout.write(f"{qid[s]}\t{yp}\t{qid[o]}\n")
                    mapped += 1
    info = {"triples": n, "entities": len(ents), "relations": len(rels), "labels": len(labels),
            "wikidata_sameas": len(qid), "benchmark_triples_mapped": mapped, "dropped": dict(drops)}
    write_json(STATS / "yago_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
