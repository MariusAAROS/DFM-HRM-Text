"""Does GLM-5.3's token cap change its behaviour? Pilot before rerunning the cut-off rows at a higher cap
(wiki: mimir-graph-benchmark-results, GLM-5.3 cut-offs).

The SDU server applies the cap as max_completion_tokens (hidden reasoning + answer); the prompt never states it. If the
model cannot see the cap, a row that finished at 16k gives the same output at 32k (up to serving noise), and only the
cut-off rows need rerunning. If it can see it (e.g. a budget injected by the server), every row must be rerun.

  prepare  sample rows that finished in the 16k run (n per dataset, seed 0) and write their bench records as
           <out>/data/<dataset>/<split>.jsonl, so eval_api keys them by the same (source, id)
  compare  each pilot run (16k again = serving-noise reference, 32k) against the original 16k run: answer agreement,
           completion-token ratio, accuracy, budget mentions in the reasoning; applies the rule fixed before the run:
           the cap is invisible if agreement(32k) >= agreement(16k rerun) - 0.03 and the median token ratio of the
           32k run is within +-10% of the rerun's.
  seed     write a partial file for the rerun: the original run's rows that finished (last attempt per key, restricted
           to the final report's rows), so eval_api's resume logic calls only the cut-off rows.

    .venv/bin/python -m graph_bench.eval.glm_cap_pilot prepare --per-dataset 20 --out $ROOT/e0/glm53/cap_pilot
    .venv/bin/python -m graph_bench.eval.glm_cap_pilot compare --out $ROOT/e0/glm53/cap_pilot
    .venv/bin/python -m graph_bench.eval.glm_cap_pilot seed --run named_t16k --to named_t32k
"""
import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from graph_bench.scoring import normalize

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data/graphs/bench"
GLM = Path("/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench/e0/glm53")
BUDGET = re.compile(r"token budget|tokens? (left|remaining)|max(imum)? (completion )?tokens|running out of (space|tokens)"
                    r"|length limit|huge token|limited tokens|budget", re.I)


def last_attempts(partial: Path) -> dict:
    rows = {}
    for line in open(partial, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["source"], r["id"]] = r
    return rows


def samples(path: Path) -> dict:
    return {(r["source"], r["id"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def prepare(args):
    orig = samples(GLM / f"{args.run}.samples.jsonl")
    by_ds = defaultdict(list)
    for key, r in orig.items():
        if r.get("finish") != "length" and r["generations"][0].strip():
            by_ds[r["dataset"]].append(key)
    rng = random.Random(args.seed)
    picked = defaultdict(set)
    for ds in sorted(by_ds):
        for src, rid in rng.sample(sorted(by_ds[ds]), min(args.per_dataset, len(by_ds[ds]))):
            picked[src].add(rid)
    n = 0
    for src, ids in sorted(picked.items()):
        out = Path(args.out) / "data" / f"{src}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(BENCH / f"{src}.jsonl", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip() and json.loads(line)["id"] in ids:
                    fout.write(line)
                    n += 1
    assert n == sum(map(len, picked.values())), "some picked ids not found in their bench file"
    print(f"{n} rows over {len(picked)} files -> {args.out}/data")


def compare(args):
    out = Path(args.out)
    orig_s, orig_p = samples(GLM / f"{args.run}.samples.jsonl"), last_attempts(GLM / f"{args.run}.partial.jsonl")
    result = {}
    for cond in ("t16k", "t32k"):
        s, p = samples(out / f"pilot_{cond}.samples.jsonl"), last_attempts(out / f"pilot_{cond}.partial.jsonl")
        keys = sorted(s)
        same = [normalize(s[k]["preds"][0]) == normalize(orig_s[k]["preds"][0]) for k in keys]
        ratio = [p[k]["usage"]["completion_tokens"] / orig_p[k]["usage"]["completion_tokens"] for k in keys
                 if p[k].get("usage", {}).get("completion_tokens") and orig_p[k].get("usage", {}).get("completion_tokens")]
        result[cond] = {"n": len(keys), "answer_agreement": sum(same) / len(keys),
                        "median_token_ratio": statistics.median(ratio),
                        "acc": sum(bool(s[k]["acc"][0]) for k in keys) / len(keys),
                        "acc_original": sum(bool(orig_s[k]["acc"][0]) for k in keys) / len(keys),
                        "cut_off": sum(s[k].get("finish") == "length" for k in keys),
                        "budget_mentions": sum(bool(BUDGET.search(p[k].get("reasoning") or "")) for k in keys),
                        "budget_mentions_original": sum(bool(BUDGET.search(orig_p[k].get("reasoning") or "")) for k in keys)}
    r16, r32 = result["t16k"], result["t32k"]
    invisible = (r32["answer_agreement"] >= r16["answer_agreement"] - 0.03
                 and abs(r32["median_token_ratio"] - r16["median_token_ratio"]) <= 0.10 * r16["median_token_ratio"])
    result["cap_invisible"] = invisible
    result["rule"] = ("invisible if agreement(32k) >= agreement(16k rerun) - 0.03 and |ratio(32k) - ratio(16k rerun)| "
                      "<= 0.10 * ratio(16k rerun), both against the original 16k run")
    (out / "verdict.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def seed(args):
    final = samples(GLM / f"{args.run}.samples.jsonl")
    attempts = last_attempts(GLM / f"{args.run}.partial.jsonl")
    target = GLM / f"{args.to}.partial.jsonl"
    if target.exists():
        raise SystemExit(f"{target} exists: not overwritten")
    kept = 0
    with open(target, "w", encoding="utf-8") as f:
        for key in final:
            r = attempts[key]
            if r.get("finish") != "length" and not r.get("error"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    print(f"{target}: {kept} finished rows of {len(final)} reused; {len(final) - kept} left to call")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "compare", "seed"])
    ap.add_argument("--run", default="named_t16k", help="Original GLM run (report stem in e0/glm53).")
    ap.add_argument("--out", default=str(GLM / "cap_pilot"))
    ap.add_argument("--per-dataset", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--to", help="seed: stem of the new run whose partial file is written.")
    args = ap.parse_args()
    {"prepare": prepare, "compare": compare, "seed": seed}[args.cmd](args)


if __name__ == "__main__":
    main()
