"""Wikidata pass 2: pass-1 TSVs -> triple store kg/wikidata (run by wikidata.sh).

  - Entities: items with an English label whose P31 avoids wikidata_blocklist.json.
    types = their P31 classes.
  - Relations: properties of type WikibaseItem, Quantity, Time or Monolingualtext, plus STRING_ALLOW
    (labels and datatypes from raw/wikidata/properties.tsv, a SPARQL export). Most String properties are
    codes (Commons category, phone number, URL formatter) and are dropped. Quantity properties with an
    allowed-units constraint (raw/wikidata/property_units.tsv) are dropped too: truthy dumps keep the
    amount but not its unit, so "area: 38.22" is ambiguous. Unitless counts (population) stay.
  - Triples: the subject is a kept entity, and the property type matches the value kind. Entity objects must
    be kept entities. Literals are rendered:
      - time "+1830-10-04T00:00:00Z" -> "1830-10-04"; "-01-01T00:00:00Z" dates -> year only (truthy carries
        no precision); negative years -> "N BC";
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KG, RAW, STATS, TRIPLE_SCHEMA, write_entities, write_json, write_relations  # noqa: E402

KIND_FOR_TYPE = {"WikibaseItem": "E", "Quantity": "N", "Time": "T", "String": "S", "Monolingualtext": "M"}
TIME = re.compile(r"^([+-]?)(\d+)-(\d\d)-(\d\d)T(\d\d:\d\d:\d\d)Z$")
MAX_STRING = 300
# String properties with readable values; every other String property is a code or identifier.
STRING_ALLOW = {"P281", "P395", "P473", "P487", "P1545", "P1814", "P2125"}
DENY = {"P8687"}  # social media followers: volatile counts
G: dict = {}


def read_tsv(paths, names) -> pd.DataFrame:
    frames = [pd.read_csv(p, sep="\t", header=None, names=names, dtype=str, quoting=csv.QUOTE_NONE,
                          na_filter=False, engine="c") for p in paths if p.stat().st_size]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=names)


def qnum(col: pd.Series) -> np.ndarray:
    return col.str.slice(1).astype(np.int64).to_numpy()


def is_member(nums: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_set, nums)
    idx[idx == len(sorted_set)] = 0
    return sorted_set[idx] == nums


def fmt_time(v: str) -> str | None:
    m = TIME.match(v)
    if not m:
        return None
    sign, year, month, day, clock = m.groups()
    y = int(year)
    if y == 0:
        return None
    ys = f"{y} BC" if sign == "-" else str(y)
    if month == "00" or (month == "01" and day in ("00", "01") and clock == "00:00:00"):
        return ys
    if day == "00":
        return f"{ys}-{month}"
    return f"{ys}-{month}-{day}" if sign != "-" else f"{ys}, {month}-{day}"


def fmt_number(v: str) -> str | None:
    v = v.lstrip("+")
    if v.endswith(".0"):
        v = v[:-2]
    return v or None


def _shard(k: int) -> dict:
    pass1, ents, ptype = G["pass1"], G["entity_nums"], G["ptype"]
    df = read_tsv(sorted((pass1 / "facts" / f"s{k:03d}").glob("w*.tsv")), ["s", "p", "kind", "v"])
    c = collections.Counter(read=len(df))
    if df.empty:
        return dict(c)
    keep = is_member(qnum(df["s"]), ents)
    c["drop_subject"] = int((~keep).sum())
    df = df[keep]
    want = df["p"].map(ptype)
    ok = want.eq(df["kind"])
    c["drop_property_type"] = int((~ok).sum())
    df = df[ok]
    e = df["kind"].eq("E").to_numpy()
    o_ok = np.ones(len(df), dtype=bool)
    if e.any():
        o_ok[e] = is_member(qnum(df["v"][e]), ents)
    c["drop_object"] = int((~o_ok).sum())
    df = df[o_ok]
    lit = df["v"].where(df["kind"].ne("E"), None)
    kinds = df["kind"].to_numpy()
    vals = lit.to_numpy(dtype=object)
    for i in np.flatnonzero(kinds != "E"):
        kd, v = kinds[i], vals[i]
        vals[i] = fmt_time(v) if kd == "T" else fmt_number(v) if kd == "N" else (v[:MAX_STRING] or None)
    lit_ok = (kinds == "E") | pd.notna(vals)
    c["drop_literal_format"] = int((~lit_ok).sum())
    df = df[lit_ok]
    vals = vals[lit_ok]
    is_e = df["kind"].eq("E").to_numpy()
    out = pd.DataFrame({"s": df["s"].to_numpy(), "r": df["p"].to_numpy(),
                        "o": np.where(is_e, df["v"].to_numpy(), None),
                        "o_literal": np.where(is_e, None, vals)})
    n0 = len(out)
    out = out.drop_duplicates().sort_values("s", kind="stable")
    out["split"] = "train"
    t = pa.Table.from_pandas(out, schema=TRIPLE_SCHEMA, preserve_index=False)
    c["drop_duplicate"] = n0 - t.num_rows
    c["kept"] = t.num_rows
    for kd, n in collections.Counter(df["kind"]).items():
        c[f"kind_{kd}"] = n
    pq.write_table(t.cast(TRIPLE_SCHEMA), KG / "wikidata" / f"triples-{k:05d}.parquet", compression="zstd")
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=256)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--blocklist", type=Path, default=Path(__file__).with_name("wikidata_blocklist.json"))
    a = ap.parse_args()
    out = KG / "wikidata"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("triples-*.parquet"):
        p.unlink()

    # properties
    props = {}
    with open(RAW / "wikidata" / "properties.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p, typ, label = (line.rstrip("\n").split("\t") + [""])[:3]
            pid, typ = p.rsplit("/", 1)[-1].rstrip(">"), typ.rsplit("#", 1)[-1].rstrip(">")
            label = label[1:label.rfind('"')] if label.startswith('"') else ""
            props[pid] = (typ, label)
    with_units = set()
    with open(RAW / "wikidata" / "property_units.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            with_units.add(line.split("\t", 1)[0].rsplit("/", 1)[-1].rstrip(">"))
    ptype = {p: KIND_FOR_TYPE[t] for p, (t, l) in props.items()
             if t in KIND_FOR_TYPE and l and p not in DENY and p not in with_units
             and (t != "String" or p in STRING_ALLOW)}
    write_relations("wikidata", {p: {"label": props[p][1], "datatype": props[p][0]} for p in sorted(ptype)})

    # classes, blocklist
    blocked_classes = set(json.loads(a.blocklist.read_text())["classes"])
    p31 = read_tsv(sorted((a.pass1 / "p31").glob("w*.tsv")), ["q", "cls"])
    blocked = set(p31.loc[p31["cls"].isin(blocked_classes), "q"])
    class_counts = p31["cls"].value_counts()
    # q number -> [class ids]. A numpy sort + run boundaries; pyarrow's hash_list aggregation aborted
    # (std::length_error) on the full 125M rows.
    qn = qnum(p31["q"])
    order = np.argsort(qn, kind="stable")
    qs, cls = qn[order], p31["cls"].to_numpy()[order]
    del p31, qn, order
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    ends = np.r_[starts[1:], len(qs)]
    types = {int(qs[s]): cls[s:e].tolist() for s, e in zip(starts, ends)}
    del qs, cls, starts, ends

    # labels
    lab = read_tsv(sorted((a.pass1 / "labels").glob("w*.tsv")), ["tag", "q", "text"])
    labels = lab[lab["tag"].eq("L")].drop_duplicates("q").set_index("q")["text"]
    desc = lab[lab["tag"].eq("D")].drop_duplicates("q").set_index("q")["text"].to_dict()
    del lab
    keep = labels[~labels.index.isin(blocked) & labels.str.len().gt(0)]
    n_ent = write_entities("wikidata", ({"entity_id": q, "label": l, "description": desc.get(q),
                                         "types": types.get(int(q[1:]))} for q, l in keep.items()))
    entity_nums = np.sort(np.fromiter((int(q[1:]) for q in keep.index), dtype=np.int64, count=len(keep)))
    labelled = len(labels)
    del labels, desc, types, keep

    G.update(pass1=a.pass1, entity_nums=entity_nums, ptype=ptype)
    total = collections.Counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for c in pool.imap_unordered(_shard, range(a.shards)):
            total.update(c)
    names = pq.read_table(out / "entities.parquet", columns=["entity_id", "label"])
    name = dict(zip(names["entity_id"].to_pylist(), names["label"].to_pylist()))
    info = {"labelled_items": labelled, "blocked_items": len(blocked), "entities": n_ent,
            "relations": len(ptype), "triples": dict(total),
            "top_classes": [(q, name.get(q, ""), int(n)) for q, n in class_counts.head(80).items()],
            "blocked_classes": {q: int(class_counts.get(q, 0)) for q in sorted(blocked_classes)}}
    write_json(STATS / "wikidata_store.json", info)
    print({k: v for k, v in info.items() if k not in ("top_classes", "blocked_classes")})


if __name__ == "__main__":
    main()
