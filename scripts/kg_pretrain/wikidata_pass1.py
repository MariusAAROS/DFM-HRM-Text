"""Wikidata truthy N-Triples pass 1 worker: stdin lines -> subject-sharded TSV (run by wikidata.sh).

Reads the grep-prefiltered stream (wdt: statements, English rdfs:label / schema:description) and writes
under <out>/:
    labels/w<W>.tsv           L|D \t Qid \t text           (English label / description)
    p31/w<W>.tsv              Qid \t class Qid             (instance of, for the class blocklist)
    facts/s<K>/w<W>.tsv       Qid \t Pid \t kind \t value  (kind E = entity Qid, T time, N number, S string, M @en text)
K = Q number % n_shards, so every subject lands in one shard.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ENT = "<http://www.wikidata.org/entity/Q"
WDT = "<http://www.wikidata.org/prop/direct/P"
LABEL = "<http://www.w3.org/2000/01/rdf-schema#label>"
DESC = "<http://schema.org/description>"
XSD = "^^<http://www.w3.org/2001/XMLSchema#"
_ESC = re.compile(r'\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)')
_SIMPLE = {"t": "\t", "n": "\n", "r": "\r", '"': '"', "\\": "\\", "'": "'", "b": "\b", "f": "\f"}


def unescape(s: str) -> str:
    if "\\" not in s:
        return s
    def rep(m):
        g = m.group(1)
        if g[0] in "uU" and len(g) > 1:
            try:
                return chr(int(g[1:], 16))
            except ValueError:
                return g
        return _SIMPLE.get(g, g)
    return _ESC.sub(rep, s)


def clean(s: str) -> str:
    return unescape(s).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def main():
    out, worker, n_shards = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    for d in ("labels", "p31"):
        (out / d).mkdir(parents=True, exist_ok=True)
    lab = open(out / "labels" / f"w{worker}.tsv", "w", encoding="utf-8", buffering=1 << 22)
    p31 = open(out / "p31" / f"w{worker}.tsv", "w", encoding="utf-8", buffering=1 << 22)
    facts = []
    for k in range(n_shards):
        (out / "facts" / f"s{k:03d}").mkdir(parents=True, exist_ok=True)
        facts.append(open(out / "facts" / f"s{k:03d}" / f"w{worker}.tsv", "w", encoding="utf-8", buffering=1 << 20))
    for line in sys.stdin:
        if not line.startswith(ENT):
            continue
        s_end = line.index(">", len(ENT))
        q = line[len(ENT) - 1:s_end]
        rest = line[s_end + 2:]
        if rest.startswith(WDT):
            p_end = rest.index(">")
            pid = rest[len(WDT) - 1:p_end]
            obj = rest[p_end + 2:].rstrip()
            if obj.endswith(" ."):
                obj = obj[:-2]
            if obj.startswith(ENT):
                o = obj[len(ENT) - 1:-1]
                facts[int(q[1:]) % n_shards].write(f"{q}\t{pid}\tE\t{o}\n")
                if pid == "P31":
                    p31.write(f"{q}\t{o}\n")
            elif obj.startswith('"'):
                end = obj.rindex('"')
                lex, suffix = obj[1:end], obj[end + 1:]
                if suffix.startswith(XSD):
                    kind = "T" if suffix.startswith(XSD + "dateTime") else "N" if suffix.startswith(XSD + "decimal") else None
                elif suffix == "@en":
                    kind = "M"
                elif suffix == "":
                    kind = "S"
                else:
                    kind = None  # other languages, geo literals, ...
                if kind:
                    facts[int(q[1:]) % n_shards].write(f"{q}\t{pid}\t{kind}\t{clean(lex)}\n")
            continue
        if rest.startswith(LABEL) or rest.startswith(DESC):
            obj = rest.split("> ", 1)[1].rstrip()
            if obj.endswith('"@en .'):
                tag = "L" if rest.startswith(LABEL) else "D"
                lab.write(f"{tag}\t{q}\t{clean(obj[1:-6])}\n")
    for f in (lab, p31, *facts):
        f.close()


if __name__ == "__main__":
    main()
