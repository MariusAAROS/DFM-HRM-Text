"""PrimeKG (Harvard Dataverse kg.csv) -> triple store kg/primekg.

kg.csv lists most edges in both directions. Relation ids are "<relation>|<x_type>|<y_type>", so each
direction gets its own label (indication: drug -> "indicated for", disease -> "treated by"). holdout.py
ignores the suffix, so both directions of an edge share one split.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATS, TripleStoreWriter, write_entities, write_json, write_relations  # noqa: E402

PROTEIN = "gene/protein"
# (display_relation, x_type) -> label; x_type None = any direction.
LABELS = {
    ("parent-child", None): "parent-child",
    ("expression present", "anatomy"): "expresses", ("expression present", PROTEIN): "expressed in",
    ("expression absent", "anatomy"): "does not express", ("expression absent", PROTEIN): "not expressed in",
    ("interacts with", None): "interacts with",
    ("contraindication", "drug"): "contraindicated for", ("contraindication", "disease"): "contraindicated drug",
    ("phenotype present", "disease"): "has phenotype", ("phenotype present", "effect/phenotype"): "phenotype of",
    ("phenotype absent", "disease"): "lacks phenotype", ("phenotype absent", "effect/phenotype"): "absent in",
    ("associated with", None): "associated with",
    ("synergistic interaction", None): "synergistic interaction with",
    ("side effect", "drug"): "side effect", ("side effect", "effect/phenotype"): "side effect of",
    ("carrier", "drug"): "carrier protein", ("carrier", PROTEIN): "carrier for drug",
    ("enzyme", "drug"): "metabolizing enzyme", ("enzyme", PROTEIN): "metabolizes drug",
    ("target", "drug"): "target", ("target", PROTEIN): "targeted by",
    ("transporter", "drug"): "transporter", ("transporter", PROTEIN): "transports drug",
    ("linked to", None): "linked to",
    ("indication", "drug"): "indicated for", ("indication", "disease"): "treated by",
    ("off-label use", "drug"): "off-label use for", ("off-label use", "disease"): "off-label treatment",
    ("ppi", None): "interacts with protein",
}


def label_for(display: str, x_type: str) -> str:
    return LABELS.get((display, x_type)) or LABELS[(display, None)]


def main():
    raw = RAW / "primekg"
    ents, rels = {}, {}
    store = TripleStoreWriter("primekg")
    with open(raw / "kg.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            x, y = row["x_index"], row["y_index"]
            for idx, name, typ in ((x, row["x_name"], row["x_type"]), (y, row["y_name"], row["y_type"])):
                if idx not in ents:
                    ents[idx] = {"label": name, "types": [typ.replace("_", " ")]}
            rid = f"{row['relation']}|{row['x_type']}|{row['y_type']}"
            if rid not in rels:
                rels[rid] = {"label": label_for(row["display_relation"], row["x_type"])}
            store.add(x, rid, y)
    n = store.close()
    write_entities("primekg", ents)
    write_relations("primekg", rels)
    info = {"triples": n, "entities": len(ents), "relations": len(rels)}
    write_json(STATS / "primekg_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
