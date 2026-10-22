"""Per-source statistics of the converted (data_io) views, plus a summary table.

Counts rows and characters exactly. Tokens are estimated from a random sample of --sample rows: the
tokens/char ratio of the HRM-Text BPE tokenizer and the Gemma tokenizer of DFM-Mimir v1.5 (template tokens
excluded), multiplied by the total characters. Writes stats/<source>.json and stats/summary.md.

    python -I scripts/kg_pretrain/stats.py            # every converted/kg-* folder
    python -I scripts/kg_pretrain/stats.py kg-trex
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONVERTED, REPO, STATS, write_json  # noqa: E402

TOKENIZERS = {
    "hrm_bpe": REPO / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json",
    "gemma_mimir15": Path("/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5/tokenizer.json"),
}
# Estimates from corpus_graphes_llm.csv (Tokens_estimes), for comparison.
CSV_ESTIMATE = {
    "kg-yago": 2.004e9, "kg-wikidata": 3.0e9, "kg-kelm": 0.9e9, "kg-trex": 0.45e9, "kg-conceptnet": 0.34e9,
    "kg-atomic2020": 0.01995e9, "kg-graphinstruct": 0.087e9, "kg-wikidata5m": 0.75e9, "kg-primekg": 0.0486e9,
    "kg-hetionet": 0.027e9,
}


def scan(folder: Path, sample: int, seed: int = 0):
    rng = random.Random(seed)
    rows = chars = inst_rows = empty = 0
    reservoir = []
    for p in sorted(folder.glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text = r["instruction"] + r["response"]
                rows += 1
                chars += len(text)
                inst_rows += bool(r["instruction"])
                empty += not r["response"].strip()
                if len(reservoir) < sample:
                    reservoir.append(r)
                else:
                    j = rng.randrange(rows)
                    if j < sample:
                        reservoir[j] = r
    return rows, chars, inst_rows, empty, reservoir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--sample", type=int, default=10_000)
    a = ap.parse_args()
    folders = [CONVERTED / f for f in a.folders] or sorted(p for p in CONVERTED.glob("kg-*") if p.is_dir())
    toks = {k: Tokenizer.from_file(str(p)) for k, p in TOKENIZERS.items() if p.exists()}
    results = []
    for folder in folders:
        rows, chars, inst_rows, empty, sample = scan(folder, a.sample)
        if not rows:
            continue
        s_chars = sum(len(r["instruction"]) + len(r["response"]) for r in sample)
        est = {}
        for name, tok in toks.items():
            n = 0
            for i in range(0, len(sample), 1000):
                batch = sample[i:i + 1000]
                enc = tok.encode_batch([r["instruction"] for r in batch] + [r["response"] for r in batch],
                                       add_special_tokens=False)
                n += sum(len(e.ids) for e in enc)
            est[name] = {"tokens_per_char": n / s_chars, "est_tokens": round(chars * n / s_chars),
                         "sample_mean_tokens_per_row": n / len(sample)}
        info = {"folder": folder.name, "rows": rows, "chars": chars, "rows_with_instruction": inst_rows,
                "empty_responses": empty, "shards": len(list(folder.glob("*.jsonl.gz"))), "tokens": est,
                "csv_estimate_tokens": CSV_ESTIMATE.get(folder.name)}
        write_json(STATS / f"{folder.name}.json", info)
        results.append(info)
        print(folder.name, rows, chars, {k: v["est_tokens"] for k, v in est.items()}, flush=True)
    if not a.folders:
        lines = ["| folder | rows | chars | est. tokens HRM BPE | est. tokens Gemma | CSV estimate |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in results:
            t = r["tokens"]
            fmt = lambda x: f"{x / 1e6:,.1f}M" if x else "-"
            lines.append(f"| {r['folder']} | {r['rows']:,} | {r['chars'] / 1e6:,.0f}M | "
                         f"{fmt(t.get('hrm_bpe', {}).get('est_tokens'))} | {fmt(t.get('gemma_mimir15', {}).get('est_tokens'))} | "
                         f"{fmt(r['csv_estimate_tokens'])} |")
        tot = lambda k: sum(r["tokens"].get(k, {}).get("est_tokens", 0) for r in results)
        lines.append(f"| **total** | {sum(r['rows'] for r in results):,} | {sum(r['chars'] for r in results) / 1e6:,.0f}M | "
                     f"{tot('hrm_bpe') / 1e6:,.1f}M | {tot('gemma_mimir15') / 1e6:,.1f}M | |")
        (STATS / "summary.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
