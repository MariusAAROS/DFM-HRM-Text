"""Build a train/val/test KGQA version of the OSM multi-hop dataset (data/osm/df_*.parquet).

Each sample = question + a question-specific set of knowledge-graph facts + named answers, in the same
JSONL format as the other graph datasets (instruction / response / condition / task, GRL "#### Question"
prompt, answers '|'-joined so eval_graphqa.py's set metrics apply).

Facts (closed world: every true fact between the described entities is listed, nothing else):
  * the gold reasoning path: every final answer and, at each inner hop, the POI that supports it;
  * hard distractors at every hop: POIs with the right attributes that fail exactly one clause, POIs that
    satisfy every clause but have a wrong attribute, and (hops 2+) POIs that satisfy everything except the hop
    relation and are linked only to an invalid previous-hop POI;
  * for each described POI: its attributes (category, cuisine, seating) and every relation that holds to the
    question's named anchors and to the described POIs of the previous hop.

Filters: the outer attribute filter (df_multihop features/values) is stated in the question text; every anchor
links to a uniquely named entity; the rule-based solution (scripts/osm/osm_kg.py) equals
the gold answer set exactly (so facts and labels agree); 1..--max_answers answers; prompt <= --max_tokens.

Splits are grouped by the innermost sub-question, so no test question shares a hop with a train question.
Groups of the 800 df_dump_question questions go to test.

  .venv/bin/python scripts/osm/build_kgqa.py --out data/osm/kgqa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osm_kg as K  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TOKENIZER = REPO / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json"
HEADER = ("#### Question\nIn a knowledge graph, (s, p, o) means that entity s is linked to entity o by relation p.\n"
          "The facts are:\n")
ATTR_LABEL = {"category": "category", "cooking_type": "cuisine", "outdoor_seating": "outdoor seating",
              "indoor_seating": "indoor seating"}

# ------------------------------------------------------------------------------------------- worker

KG: K.OSM | None = None
TOK = None
ARGS = None
UNIQUE: np.ndarray | None = None  # row -> name is unique in df_osm


def init_worker(args):
    global KG, TOK, ARGS, UNIQUE
    KG = K.load(multihop=False)
    ARGS = args
    lower = KG.poi.poi_name.str.lower().str.strip()  # eval_graphqa normalizes case: 'Café NATA' == 'Café Nata'
    counts = lower.map(lower.value_counts())
    UNIQUE = (counts == 1).to_numpy() & ~KG.poi.poi_name.str.contains(r"\|", regex=True).to_numpy()
    from tokenizers import Tokenizer
    TOK = Tokenizer.from_file(str(TOKENIZER))


def phrase(c: K.Clause, sub_name: str | None = None) -> tuple[str, str]:
    """(relation phrase, object) as worded in the questions."""
    a = c.anchors
    obj = lambda i: sub_name if a[i] == K.SUB else a[i]  # noqa: E731
    if c.kind in ("near_towards", "between", "intersection"):
        rel, sep = {"near_towards": ("near", "towards"), "between": ("between", "and"),
                    "intersection": ("at the intersection of", "and")}[c.kind]
        return rel, f"{obj(0)} {sep} {obj(1)}"
    return {
        "within": f"at less than {c.param} meters from", "near": "near", "dir_of_poi": f"to the {c.param} of",
        "following": "following", "inside": "inside", "border": "at the border of",
        "dir_in_area": f"at the {c.param} of",
    }[c.kind], obj(0)


def clause_truth(lvl: K.Level, idx: np.ndarray, sub_rows) -> np.ndarray:
    """(len(idx), n_clauses) truth table; the hop clause is evaluated against the full previous answer set."""
    return np.stack([K.holds(KG, c, idx, sub_rows) for c in lvl.clauses], axis=1) if lvl.clauses else \
        np.ones((len(idx), 0), bool)


def pair_holds(c: K.Clause, p: int, q: int) -> bool:
    return bool(K.holds(KG, c, np.array([p]), np.array([q]))[0])


def build_sample(item):
    qid, query, feats, vals, gold = item
    rng = np.random.default_rng(int(hashlib.md5(str(qid).encode()).hexdigest()[:8], 16))
    try:
        q = K.parse_question(query)
    except Exception:
        return qid, "parse_error", None
    if q.attrs != dict(zip(feats, vals)):  # constraint not stated in the text ('pois', seating=garden/street/...)
        return qid, "attrs_not_in_text", None
    chain = q.chain()
    poi_anchors = [a for l in chain for c in l.clauses for a in c.anchors
                   if a != K.SUB and K.RELATION_ANCHOR[c.kind] == "poi"]
    if any(a not in KG.by_name for a in poi_anchors) or any(
            a != K.SUB and K.RELATION_ANCHOR[c.kind] != "poi" and a not in
            (KG.street_lines if K.RELATION_ANCHOR[c.kind] == "street" else KG.area_geom)
            for l in chain for c in l.clauses for a in c.anchors):
        return qid, "unlinked_anchor", None
    if any(len(KG.by_name[a]) > 1 for a in poi_anchors):
        return qid, "ambiguous_anchor", None
    if not 1 <= len(gold) <= ARGS.max_answers:
        return qid, "answer_count", None
    levels = K.solve(KG, q)
    if set(KG.poi.poi_id.to_numpy()[levels[-1]].tolist()) != set(np.atleast_1d(gold).tolist()):
        return qid, "rules_disagree_with_gold", None
    if any(len(x) > ARGS.max_level_size for x in levels[:-1]):
        return qid, "inner_level_too_large", None
    names = KG.poi.poi_name.to_numpy()
    if not UNIQUE[levels[-1]].all():
        return qid, "ambiguous_answer_name", None

    # ---- gold path: final answers, then for each POI of hop i a supporting POI of hop i-1 (nearest valid one)
    path = [None] * len(chain)
    path[-1] = levels[-1]
    support = {}
    for i in range(len(chain) - 1, 0, -1):
        hop = next(c for c in chain[i].clauses if K.SUB in c.anchors)
        prev = levels[i - 1][UNIQUE[levels[i - 1]]]
        keep = []
        for p in path[i]:
            ok = [r for r in prev if pair_holds(hop, p, r)]
            if not ok:
                return qid, "ambiguous_inner_name", None
            best = min(ok, key=lambda r: np.linalg.norm(KG.xy[r] - KG.xy[p]))
            support[p] = best
            keep.append(best)
        path[i - 1] = np.unique(keep)

    # ---- distractors per hop
    anchor_set = set(poi_anchors)
    described = []
    for i, lvl in enumerate(chain):
        full = set(levels[i].tolist())
        sub_full = levels[i - 1] if i else None
        centre = KG.xy[path[i]]
        pool = np.unique(np.concatenate(KG.tree.query_ball_point(centre, ARGS.distractor_radius)))
        pool = pool[[(r not in full) and UNIQUE[r] and names[r] not in anchor_set for r in pool]]
        attr_ok = K.attribute_mask(KG, lvl.attrs)[pool]
        same_cat = KG.poi.category.to_numpy()[pool] == lvl.attrs.get("category")
        truth = clause_truth(lvl, pool, sub_full)
        n_sat = truth.sum(1)
        n = truth.shape[1]
        near_miss = pool[attr_ok & (n_sat == n - 1)] if n else pool[:0]
        attr_miss = pool[~attr_ok & same_cat & (n_sat == n)]
        attr_miss = attr_miss if len(attr_miss) else pool[~attr_ok & (n_sat == n)]
        pick = list(rng.permutation(near_miss)[:ARGS.near_miss])
        pick += list(rng.permutation(attr_miss)[:ARGS.attr_miss])
        if len(pick) < ARGS.near_miss + ARGS.attr_miss and n >= 2:  # fill with 2-clause misses
            far = pool[attr_ok & (n_sat == n - 2)]
            pick += list(rng.permutation(far)[:ARGS.near_miss + ARGS.attr_miss - len(pick)])
        described.append(np.unique(np.concatenate([path[i], np.array(pick, dtype=int)])))

        # hop distractors: everything at this hop holds except the hop relation (checked against the full previous
        # answer set), and they are linked to an *invalid* previous-hop POI that we describe. Without them a
        # model can skip the inner hops (pick any entity with a link of the right relation type).
        # Candidates come from the FULL map (not only near the path). Each is linked to any invalid POI in range
        # of the hop relation, preferring one with the previous hop's attributes (harder), else any category
        # (the shortcut accepts any link, so that already defeats it).
        if i and ARGS.hop_miss:
            hop_j = next(j for j, c in enumerate(lvl.clauses) if K.SUB in c.anchors)
            hop_c = lvl.clauses[hop_j]
            cand = np.flatnonzero(K.attribute_mask(KG, lvl.attrs))
            cand = cand[[(r not in full) and UNIQUE[r] and names[r] not in anchor_set for r in cand]]
            for j, c in enumerate(lvl.clauses):
                if j != hop_j and len(cand):
                    cand = cand[K.holds(KG, c, cand, sub_full)]
            if len(cand):
                cand = cand[~K.holds(KG, hop_c, cand, sub_full)]
            prev_lvl, prev_full = chain[i - 1], set(levels[i - 1].tolist())
            reach = (hop_c.param if hop_c.kind == "within" else K.NEAR_M) + 1
            prev_attr = K.attribute_mask(KG, prev_lvl.attrs)
            added = 0
            for h in rng.permutation(cand):
                near = np.array(KG.tree.query_ball_point(KG.xy[h], reach), dtype=int)
                near = near[[(r != h) and (r not in prev_full) and (r not in full) and UNIQUE[r]
                             and names[r] not in anchor_set for r in near]]
                near = np.concatenate([rng.permutation(near[prev_attr[near]]), rng.permutation(near[~prev_attr[near]])])
                link = next((r for r in near if pair_holds(hop_c, h, r)), None)
                if link is None:
                    continue
                described[i] = np.union1d(described[i], [h])
                described[i - 1] = np.union1d(described[i - 1], [link])
                added += 1
                if added >= ARGS.hop_miss:
                    break

    # ---- facts
    poi = KG.poi
    facts = set()
    for i, lvl in enumerate(chain):
        keys = ["category"] + [k for k in lvl.attrs if k != "category"]
        rows = described[i]
        truth = clause_truth(lvl, rows, levels[i - 1] if i else None)
        for k, r in enumerate(rows):
            for key in keys:
                v = poi[key].iat[r]
                if isinstance(v, str):
                    facts.add((names[r], ATTR_LABEL[key], v.replace("_", " ")))
            for j, c in enumerate(lvl.clauses):
                if K.SUB in c.anchors:  # one fact per described previous-hop POI it relates to
                    for qrow in described[i - 1]:
                        if pair_holds(c, r, qrow):
                            facts.add((names[r],) + phrase(c, names[qrow]))
                elif truth[k, j]:
                    facts.add((names[r],) + phrase(c))
    facts = sorted(facts)
    rng.shuffle(facts)
    if len(facts) > ARGS.max_facts:
        return qid, "too_many_facts", None
    question = re.sub(r"\s+", " ", query).strip()
    instruction = HEADER + "".join(f"({s}, {p}, {o})\n" for s, p, o in facts) + f"Q: {question}\nA: "
    n_tok = len(TOK.encode(instruction).ids)
    if n_tok > ARGS.max_tokens:
        return qid, "too_many_tokens", None
    ans = sorted(names[levels[-1]].tolist(), key=str.lower)
    return qid, "ok", {
        "instruction": instruction,
        "response": " | ".join(ans) + ".",
        "condition": "direct",
        "task": f"{len(chain)}hop",
        "qid": int(qid),
        "depth": len(chain),
        "question": question,
        "answers": ans,
        "answer_poi_ids": [int(x) for x in sorted(np.atleast_1d(gold).tolist())],
        "n_facts": len(facts),
        "n_entities": int(sum(len(d) for d in described)),
        "prompt_tokens": n_tok,
    }


# ------------------------------------------------------------------------------------------- driver


def innermost(q: str) -> str:
    inner = re.findall(r"\[([^\[\]]*)\]", q)
    return re.sub(r"\s+", " ", (inner[-1] if inner else q)).strip()


def assign_splits(mh: pd.DataFrame, dump_idx, val_frac: float, test_frac: float) -> pd.Series:
    leaf = mh.query_texte.map(innermost)
    h = leaf.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    split = np.where(h < test_frac, "test", np.where(h < test_frac + val_frac, "val", "train"))
    split = pd.Series(split, index=mh.index)
    forced = set(leaf.loc[[i for i in dump_idx if i in leaf.index]])
    split[leaf.isin(forced)] = "test"
    return split, leaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "osm" / "kgqa"))
    ap.add_argument("--train_per_depth", type=int, default=5000)
    ap.add_argument("--val_per_depth", type=int, default=250)
    ap.add_argument("--test_per_depth", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--max_answers", type=int, default=10)
    ap.add_argument("--max_level_size", type=int, default=50)
    ap.add_argument("--near_miss", type=int, default=3)
    ap.add_argument("--attr_miss", type=int, default=1)
    ap.add_argument("--hop_miss", type=int, default=2)
    ap.add_argument("--distractor_radius", type=float, default=1500.0)
    ap.add_argument("--max_facts", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cols = ["query_texte", "features", "values", "answers", "template"]
    mh = pd.read_parquet(K.DATA_DIR / "df_multihop.parquet", columns=cols)
    mh = mh[~mh.duplicated("query_texte")]
    mh["depth"] = mh.template.map(len) - 1
    dump = pd.read_parquet(K.DATA_DIR / "df_dump_question.parquet")
    mh["split"], mh["leaf"] = assign_splits(mh, dump.index, args.val_frac, args.test_frac)
    print(f"{len(mh):,} unique questions, {mh.leaf.nunique():,} leaf groups; split shares "
          f"{mh.split.value_counts(normalize=True).round(3).to_dict()}  ({time.time() - t0:.0f}s)", flush=True)

    quota = {"train": args.train_per_depth, "val": args.val_per_depth, "test": args.test_per_depth}
    rng = np.random.default_rng(args.seed)
    stats = Counter()
    records = {s: [] for s in quota}
    with mp.get_context("fork").Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool:
        for split, n_want in quota.items():
            for depth in range(1, 5):
                sub = mh[(mh.split == split) & (mh.depth == depth)]
                if split == "test":  # eval-subset questions first
                    first = sub.index.intersection(dump.index)
                    order = np.concatenate([first, rng.permutation(sub.index.difference(first))])
                else:
                    order = rng.permutation(sub.index)
                got = []
                for start in range(0, len(order), args.chunk):
                    ids = order[start:start + args.chunk]
                    items = [(i, sub.at[i, "query_texte"], list(sub.at[i, "features"]), list(sub.at[i, "values"]),
                              sub.at[i, "answers"]) for i in ids]
                    for qid, status, rec in pool.imap(build_sample, items, chunksize=16):
                        stats[(split, depth, status)] += 1
                        if rec is not None and len(got) < n_want:
                            rec["in_dump_eval"] = bool(qid in dump.index)
                            got.append(rec)
                    print(f"  {split} depth {depth}: {len(got):,}/{n_want:,} after {start + len(ids):,} candidates "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    if len(got) >= n_want:
                        break
                records[split] += got

    for split, recs in records.items():
        rng.shuffle(recs)
        with open(out / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = pd.Series(stats).rename_axis(["split", "depth", "status"]).rename("n").reset_index()
    st.to_csv(out / "build_stats.csv", index=False)
    summary = {s: {"n": len(r), "by_depth": dict(Counter(x["depth"] for x in r)),
                   "prompt_tokens_mean": float(np.mean([x["prompt_tokens"] for x in r])) if r else 0,
                   "prompt_tokens_max": int(max(x["prompt_tokens"] for x in r)) if r else 0,
                   "facts_mean": float(np.mean([x["n_facts"] for x in r])) if r else 0,
                   "answers_mean": float(np.mean([len(x["answers"]) for x in r])) if r else 0,
                   "single_answer_share": float(np.mean([len(x["answers"]) == 1 for x in r])) if r else 0,
                   "from_dump_eval": int(sum(x["in_dump_eval"] for x in r))}
               for s, r in records.items()}
    (out / "build_config.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    print(st.groupby("status").n.sum().sort_values(ascending=False).to_string())
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
