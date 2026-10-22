"""ConceptNet 5.7 assertions -> triple store kg/conceptnet.

Keeps English-English edges with weight >= --min-weight. POS/sense suffixes are dropped (/c/en/dog/n -> dog),
duplicate (s, r, o) are merged, and /r/ExternalURL is skipped.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATS, TripleStoreWriter, write_entities, write_json, write_relations  # noqa: E402

REL = {
    "RelatedTo": "related to", "FormOf": "form of", "IsA": "is a", "PartOf": "part of", "HasA": "has",
    "UsedFor": "used for", "CapableOf": "capable of", "AtLocation": "found at", "Causes": "causes",
    "HasSubevent": "has subevent", "HasFirstSubevent": "begins with", "HasLastSubevent": "ends with",
    "HasPrerequisite": "requires", "HasProperty": "has property", "MotivatedByGoal": "motivated by",
    "ObstructedBy": "obstructed by", "Desires": "desires", "CreatedBy": "created by", "Synonym": "synonym",
    "Antonym": "antonym", "DistinctFrom": "distinct from", "DerivedFrom": "derived from", "SymbolOf": "symbol of",
    "DefinedAs": "defined as", "MannerOf": "manner of", "LocatedNear": "located near", "HasContext": "context",
    "SimilarTo": "similar to", "EtymologicallyRelatedTo": "etymologically related to",
    "EtymologicallyDerivedFrom": "etymologically derived from", "CausesDesire": "makes you want",
    "MadeOf": "made of", "ReceivesAction": "can be", "InstanceOf": "instance of", "Entails": "entails",
    "NotDesires": "does not desire", "NotUsedFor": "not used for", "NotCapableOf": "not capable of",
    "NotHasProperty": "does not have property",
}


def rel_label(r: str) -> str:
    name = r[len("/r/"):]
    if name in REL:
        return REL[name]
    name = name.split("/")[-1]  # dbpedia/genre -> genre
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def concept(uri: str) -> str | None:
    p = uri.split("/")
    return p[3] if len(p) > 3 and p[2] == "en" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-weight", type=float, default=1.0)
    a = ap.parse_args()
    src = RAW / "conceptnet" / "conceptnet-assertions-5.7.0.csv.gz"
    seen, ents, rels = set(), set(), {}
    drops = collections.Counter()
    store = TripleStoreWriter("conceptnet")
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            _, r, s, o, meta = line.rstrip("\n").split("\t")
            if not (s.startswith("/c/en/") and o.startswith("/c/en/")):
                drops["non_english"] += 1
                continue
            if r == "/r/ExternalURL":
                drops["external_url"] += 1
                continue
            if json.loads(meta).get("weight", 1.0) < a.min_weight:
                drops["low_weight"] += 1
                continue
            s, o = concept(s), concept(o)
            if not s or not o or s == o or (s, r, o) in seen:
                drops["duplicate_or_self"] += 1
                continue
            seen.add((s, r, o))
            ents.update((s, o))
            rels.setdefault(r, {"label": rel_label(r)})
            store.add(s, r, o)
    n = store.close()
    write_entities("conceptnet", {e: {"label": e.replace("_", " ")} for e in sorted(ents)})
    write_relations("conceptnet", rels)
    info = {"triples": n, "entities": len(ents), "relations": len(rels), "dropped": dict(drops)}
    write_json(STATS / "conceptnet_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
