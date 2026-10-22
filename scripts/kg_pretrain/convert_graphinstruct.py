"""GraphWiz GraphInstruct + GraphInstruct-RFT-72K -> canonical/converted kg-graphinstruct (native Q -> A, cot).

RFT holds several sampled reasoning paths per query and largely contains GraphInstruct, so the union is
deduplicated on (query, answer). -Test and -DPO are not used.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATS, DocWriter, write_json  # noqa: E402

FILES = {"GraphInstruct": "GraphInstruct/GraphInstruct.json", "GraphInstruct-RFT": "GraphInstruct-RFT-72K/GraphInstruct-RFT.json"}


def main():
    seen, counts = set(), collections.Counter()
    queries = set()
    with DocWriter("graphinstruct") as w:
        for origin, rel in FILES.items():
            with open(RAW / "graphinstruct" / rel, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    q, ans = r["query"].strip(), r["answer"].strip()
                    if not q or not ans:
                        counts[f"{origin}:empty"] += 1
                        continue
                    if (q, ans) in seen:
                        counts[f"{origin}:duplicate"] += 1
                        continue
                    seen.add((q, ans))
                    queries.add(q)
                    w.write(f"{origin}-{i}", ans, {"task": r.get("task"), "origin": origin},
                            instruction=q, condition="cot")
                    counts[f"{origin}:kept"] += 1
    info = {"rows": len(seen), "unique_queries": len(queries), "counts": dict(counts)}
    write_json(STATS / "graphinstruct_store.json", info)
    print(info)


if __name__ == "__main__":
    main()
