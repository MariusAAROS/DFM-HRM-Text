"""OSM Paris spatial KGQA (data/osm/kgqa, wiki: osm-spatial-kgqa) -> bench records, for zero-shot transfer evals.

Held-out downstream task (wiki: mimir-graph-benchmark): geospatial multi-hop questions over Paris POIs, a domain and
relation vocabulary (distances, directions, streets, areas) that no train set of the benchmark covers. Never trained.

The source rows already hold the GRL prompt ("#### Question\\n<preamble>\\nThe facts are:\\n...\\nQ: <q>\\nA: "). The
graph block is kept verbatim as `context` (so prompt_text reproduces the source prompt exactly, checked per row), not
re-parsed into triples: POI names contain commas. So --anonymize does not apply to this set.

answer_format is `set` (1-10 POI names). meta carries depth, qid, in_dump_eval and needs_inner_hops (False when the
outer-hop shortcut of scripts/osm/check_kgqa.py already returns the gold set, i.e. the chain is not needed; always
True at depth 1 by convention), so scores can be split by whether the question really needs its inner hops.

    .venv/bin/python -m graph_bench.sources.osm
"""
import json
import sys
from pathlib import Path

from graph_bench.schema import GRL_PREFIX, Record, prompt_text, write_records

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/osm/kgqa"
OUT = REPO / "data/graphs/bench/osm"
SPLITS = ("val", "test")

sys.path.insert(0, str(REPO / "scripts/osm"))
from check_kgqa import solve_from_prompt  # noqa: E402


def convert(row: dict, split: str, i: int) -> Record:
    ins = row["instruction"]
    assert ins.startswith(GRL_PREFIX), row["qid"]
    q_at = ins.rfind("\nQ: ")
    context = ins[len(GRL_PREFIX):q_at + 1]
    needs = True if row["depth"] == 1 else solve_from_prompt(row, shortcut=True) != row["answers"]
    rec = Record(id=f"osm/{split}/{i}", dataset="osm", split=split, task=row["task"], question=row["question"],
                 answers=row["answers"], answer_format="set", context=context, preamble="",
                 meta={"qid": row["qid"], "depth": row["depth"], "in_dump_eval": row["in_dump_eval"],
                       "needs_inner_hops": needs, "n_facts": row["n_facts"]})
    assert GRL_PREFIX + prompt_text(rec) == ins, f"prompt mismatch on qid {row['qid']}"
    return rec


def main():
    for split in SPLITS:
        with open(SRC / f"{split}.jsonl", encoding="utf-8") as f:
            recs = [convert(json.loads(line), split, i) for i, line in enumerate(f) if line.strip()]
        n = write_records(recs, OUT / f"{split}.jsonl")
        by = {}
        for r in recs:
            d = by.setdefault(r.task, [0, 0])
            d[0] += 1
            d[1] += r.meta["needs_inner_hops"]
        print(f"{split}: {n} rows; per task (rows, needs_inner_hops): {dict(sorted(by.items()))}")


if __name__ == "__main__":
    main()
