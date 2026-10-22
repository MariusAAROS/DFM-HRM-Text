"""ATOMIC-2020 -> triple store kg/atomic2020, keeping the official splits (train / dev -> valid / test).

Heads and tails are free-text events or phrases; the text itself is the entity id.
Tails "none" and empty tails are dropped. The test split is stored but never rendered.
"""
from __future__ import annotations

import collections
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATS, TripleStoreWriter, write_entities, write_json, write_relations  # noqa: E402

# Relation phrasing follows the natural-language templates of the COMET-ATOMIC 2020 paper.
REL = {
    "xAttr": "PersonX is seen as", "xEffect": "as a result, PersonX", "xIntent": "because PersonX wanted",
    "xNeed": "before, PersonX needed", "xReact": "as a result, PersonX feels", "xReason": "because",
    "xWant": "as a result, PersonX wants", "oEffect": "as a result, others", "oReact": "as a result, others feel",
    "oWant": "as a result, others want", "isAfter": "happens after", "isBefore": "happens before",
    "HinderedBy": "can be hindered by", "isFilledBy": "the blank can be filled by", "ObjectUse": "used for",
    "MadeUpOf": "made up of", "HasSubEvent": "includes", "AtLocation": "found at", "CapableOf": "capable of",
    "Causes": "causes", "CausesDesire": "makes you want", "CreatedBy": "created by", "Desires": "desires",
    "HasA": "has", "HasFirstSubevent": "begins with", "HasLastSubevent": "ends with",
    "HasPrerequisite": "requires", "HasProperty": "has property", "InstanceOf": "instance of",
    "MadeOf": "made of", "MotivatedByGoal": "motivated by", "NotDesires": "does not desire",
    "PartOf": "part of", "ReceivesAction": "can be", "UsedFor": "used for",
}
SPLITS = {"train.tsv": "train", "dev.tsv": "valid", "test.tsv": "test"}


def main():
    zpath = RAW / "atomic2020" / "atomic2020_data-feb2021.zip"
    store = TripleStoreWriter("atomic2020")
    seen, ents, rels = set(), set(), {}
    counts, drops = collections.Counter(), collections.Counter()
    with zipfile.ZipFile(zpath) as z:
        for name, split in SPLITS.items():
            with z.open(f"atomic2020_data-feb2021/{name}") as fb:
                for line in io.TextIOWrapper(fb, encoding="utf-8"):
                    p = line.rstrip("\n").split("\t")
                    if len(p) != 3:
                        drops["malformed"] += 1
                        continue
                    h, r, t = (x.strip() for x in p)
                    if not h or not t or t.lower() == "none":
                        drops["none_tail"] += 1
                        continue
                    if r not in REL:
                        drops[f"unknown_rel:{r}"] += 1
                        continue
                    if (h, r, t) in seen:
                        drops["duplicate"] += 1
                        continue
                    seen.add((h, r, t))
                    ents.update((h, t))
                    rels.setdefault(r, {"label": REL[r]})
                    store.add(h, r, t, split=split)
                    counts[split] += 1
    n = store.close()
    write_entities("atomic2020", {e: {"label": e} for e in sorted(ents)})
    write_relations("atomic2020", rels)
    info = {"triples": n, "splits": dict(counts), "entities": len(ents), "relations": len(rels),
            "dropped": dict(drops)}
    write_json(STATS / "atomic2020_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
