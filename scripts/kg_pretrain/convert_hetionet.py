"""Hetionet v1.0 -> triple store kg/hetionet.

Each edge is stored once in its native direction. Rendering uses inverse labels (render_docs.py --inverse),
so a gene document also lists the compounds that bind it.
"""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATS, TripleStoreWriter, write_entities, write_json, write_relations  # noqa: E402

# metaedge abbreviation -> (label, inverse label); see hetionet's metagraph.
METAEDGES = {
    "AdG": ("downregulates", "downregulated in"), "AeG": ("expresses", "expressed in"),
    "AuG": ("upregulates", "upregulated in"), "CbG": ("binds", "bound by"),
    "CcSE": ("causes side effect", "side effect of"), "CdG": ("downregulates", "downregulated by"),
    "CpD": ("palliates", "palliated by"), "CrC": ("resembles", "resembles"), "CtD": ("treats", "treated by"),
    "CuG": ("upregulates", "upregulated by"), "DaG": ("associated gene", "associated disease"),
    "DdG": ("downregulates", "downregulated in"), "DlA": ("localizes to", "site of"),
    "DpS": ("presents symptom", "symptom of"), "DrD": ("resembles", "resembles"),
    "DuG": ("upregulates", "upregulated in"), "GcG": ("covaries with", "covaries with"),
    "GiG": ("interacts with", "interacts with"), "GpBP": ("participates in", "involves gene"),
    "GpCC": ("participates in", "involves gene"), "GpMF": ("participates in", "involves gene"),
    "GpPW": ("participates in", "involves gene"), "Gr>G": ("regulates", "regulated by"),
    "PCiC": ("includes", "member of class"),
}


def main():
    raw = RAW / "hetionet"
    ents = {}
    with open(raw / "hetionet-v1.0-nodes.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ents[row["id"]] = {"label": row["name"], "types": [row["kind"].lower()]}
    store = TripleStoreWriter("hetionet")
    with gzip.open(raw / "hetionet-v1.0-edges.sif.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, me, o = line.rstrip("\n").split("\t")
            store.add(s, me, o)
    n = store.close()
    write_entities("hetionet", ents)
    write_relations("hetionet", {k: {"label": l, "inverse_label": inv} for k, (l, inv) in METAEDGES.items()})
    info = {"triples": n, "entities": len(ents), "relations": len(METAEDGES)}
    write_json(STATS / "hetionet_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
