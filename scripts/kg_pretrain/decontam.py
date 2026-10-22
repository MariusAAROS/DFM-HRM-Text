"""Drop QA rows whose instruction matches an evaluation prompt in data/graphs (val/test splits).

Normalization: the "#### Question\\n" prefix of the GRL prompt variant is removed, then whitespace is
collapsed. Applies to the QA-native folders (kg-graphqa_gen, kg-graphinstruct) in place, in both the
converted and the canonical views. Writes stats/decontam.json.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANONICAL, CONVERTED, REPO, STATS, write_json  # noqa: E402

TARGETS = {"kg-graphqa_gen": "graphqa_gen", "kg-graphinstruct": "graphinstruct"}


def norm(s: str) -> str:
    s = s.removeprefix("#### Question\n")
    return " ".join(s.split())


def eval_prompts() -> tuple[set, dict]:
    prompts, files = set(), {}
    for root, _, fs in os.walk(REPO / "data" / "graphs"):
        if "/sampled" in root or "/tokenized" in root or "/tok_in" in root:
            continue
        for f in fs:
            if f.endswith(".jsonl") and any(k in f for k in ("val", "test")):
                p = Path(root) / f
                n = 0
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        r = json.loads(line)
                        if r.get("instruction"):
                            prompts.add(norm(r["instruction"]))
                            n += 1
                files[str(p.relative_to(REPO))] = n
    return prompts, files


def filter_folder(conv: Path, can: Path, prompts: set) -> dict:
    removed = kept = 0
    for p in sorted(conv.glob("*.jsonl.gz")):
        cp = can / p.name
        with gzip.open(p, "rt", encoding="utf-8") as f, gzip.open(cp, "rt", encoding="utf-8") as g:
            rows = [(json.loads(a), b) for a, b in zip(f, g)]
        keep = [(r, c) for r, c in rows if norm(r["instruction"]) not in prompts]
        removed += len(rows) - len(keep)
        kept += len(keep)
        if len(keep) != len(rows):
            for path, lines in ((p, [json.dumps(r, ensure_ascii=False) + "\n" for r, _ in keep]),
                                (cp, [c for _, c in keep])):
                tmp = path.with_suffix(".tmp")
                with gzip.open(tmp, "wt", encoding="utf-8") as out:
                    out.writelines(lines)
                tmp.replace(path)
    return {"removed": removed, "kept": kept}


def main():
    prompts, files = eval_prompts()
    out = {"eval_prompts": len(prompts), "eval_files": files}
    for folder, source in TARGETS.items():
        if (CONVERTED / folder).exists():
            out[folder] = filter_folder(CONVERTED / folder, CANONICAL / source, prompts)
            print(folder, out[folder])
    write_json(STATS / "decontam.json", out)


if __name__ == "__main__":
    main()
