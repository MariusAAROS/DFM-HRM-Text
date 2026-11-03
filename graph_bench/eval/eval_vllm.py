"""Generate with vLLM on bench records and score with graph_bench.scoring (arms E0-E4; wiki: mimir-graph-benchmark-plan).

One driver for every model vLLM serves: Mimir (vLLM >= 0.27 runs hrm_text natively), Qwen3, Gemma, ... (Ouro needs an
older vLLM env; IFM checkpoints go through HF generate, not this script).

Prompt styles:
  sft       the user message is exactly the SFT instruction (GRL prompt, export_hrm.py), so a model fine-tuned on
            hrm-text-grl rows is evaluated on its training prompt. Chat template applied.
  instruct  the RL / zero-shot message (export_verl.user_message): asks for step-by-step reasoning and a final
            "Answer:" line. Chat template applied.
  base      no chat template: --shots worked examples from the train split of the same dataset, then the record,
            for pretraining-only models (HRM-Text-1B, Ouro base, ...).
--prefill appends text to the rendered prompt (e.g. "<|channel>thought\\n" for Mimir's thought channel). vLLM computes
every prompt token bidirectionally for hrm_text, prefill included.

pass@k: --n samples per record at --temperature; acc of the first sample is reported as acc@1 when n > 1 and
greedy decoding is only used with --n 1 --temperature 0.

    /home/ucloud/miniforge3/bin/python -m graph_bench.eval.eval_vllm --model danish-foundation-models/DFM-Mimir-v1.5 \\
        --style instruct --prefill $'<|channel>thought\\n' --max-tokens 1024 \\
        --out reports/mimir15_zeroshot.json data/graphs/bench/cwq/test.jsonl data/graphs/bench/clutrr/test.jsonl
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_bench.anonymize import anonymize  # noqa: E402
from graph_bench.export_verl import user_message  # noqa: E402
from graph_bench.schema import GRL_PREFIX, direct_response, prompt_text, read_records  # noqa: E402
from graph_bench.scoring import pass_at_k, score_record  # noqa: E402


def build_prompt(rec, style: str, tokenizer, shots: list = (), chat_kwargs: dict | None = None) -> str:
    if style == "base":
        demos = "".join(prompt_text(s) + direct_response(s) + "\n\n" for s in shots)
        return demos + prompt_text(rec)
    content = GRL_PREFIX + prompt_text(rec) if style == "sft" else user_message(rec)
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, **(chat_kwargs or {}))


def load(paths: list[str], limit: int | None, anon: bool, seed: int):
    recs, dropped = [], 0
    for path in paths:
        rs = read_records(path)
        if limit is not None and len(rs) > limit:
            rs = random.Random(seed).sample(rs, limit)
        for r in rs:
            if anon:
                r, grounded = anonymize(r)
                if not grounded:
                    dropped += 1
                    continue
            recs.append(r)
    return recs, dropped


def summarize(rows: list[dict], ks: list[int]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
        groups[f"{row['dataset']}/{row['task']}"].append(row)
    out = {}
    for name, rs in sorted(groups.items()):
        n = rs[0]["n"]
        s = {"count": len(rs), "acc": sum(r["acc"][0] for r in rs) / len(rs),
             "f1": sum(r["f1"][0] for r in rs) / len(rs)}
        for k in ks:
            if k <= n:
                s[f"pass@{k}"] = sum(pass_at_k(n, sum(r["acc"]), k) for r in rs) / len(rs)
        out[name] = s
    datasets = [v for k, v in out.items() if "/" not in k]
    out["macro_acc"] = sum(d["acc"] for d in datasets) / len(datasets) if datasets else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="Bench JSONL files (one dataset/split each).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Report JSON; per-sample generations go next to it (.samples.jsonl).")
    ap.add_argument("--style", choices=["sft", "instruct", "base"], default="instruct")
    ap.add_argument("--prefill", default="", help="Text appended after the rendered prompt.")
    ap.add_argument("--enable-thinking", choices=["auto", "true", "false"], default="auto",
                    help="chat_template_kwargs enable_thinking (Qwen3 / Gemma-4 style templates).")
    ap.add_argument("--shots", type=int, default=3, help="Worked examples for --style base.")
    ap.add_argument("--shots-from", default=None, help="Train JSONL to draw base-style shots from (per dataset: "
                    "the record's dataset must match).")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="Random subset per input file.")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass-k", type=int, nargs="*", default=[1, 4, 16, 64])
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    recs, dropped = load(args.data, args.limit, args.anonymize, args.seed)
    print(f"{len(recs)} records ({dropped} dropped: question lost its entity after anonymisation)")
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager,
              kv_cache_dtype=args.kv_cache_dtype, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    chat_kwargs = {} if args.enable_thinking == "auto" else {"enable_thinking": args.enable_thinking == "true"}
    shots_by_ds = defaultdict(list)
    if args.style == "base" and args.shots_from:
        for s in read_records(args.shots_from):
            shots_by_ds[s.dataset].append(s)
    rng = random.Random(args.seed)
    prompts = []
    for r in recs:
        pool = shots_by_ds.get(r.dataset, [])
        shots = rng.sample(pool, min(args.shots, len(pool))) if args.style == "base" else []
        prompts.append(build_prompt(r, args.style, tok, shots, chat_kwargs) + args.prefill)
    too_long = sum(len(tok(p, add_special_tokens=False)["input_ids"]) + args.max_tokens > args.max_model_len
                   for p in prompts)
    if too_long:
        print(f"WARNING: {too_long} prompts + max_tokens exceed --max-model-len; vLLM will reject them")

    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                            seed=args.seed)
    outputs = llm.generate([{"prompt": p} for p in prompts], params)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".samples.jsonl"), "w", encoding="utf-8") as f:
        for rec, prompt, out in zip(recs, prompts, outputs):
            texts = [o.text for o in out.outputs]
            scores = [score_record(t, rec) for t in texts]
            row = {"id": rec.id, "dataset": rec.dataset, "task": rec.task, "n": len(texts),
                   "acc": [s["acc"] for s in scores], "f1": [s["f1"] for s in scores]}
            rows.append(row)
            f.write(json.dumps({**row, "gold": rec.answers, "preds": [s["pred"] for s in scores],
                                "generations": texts, "prompt_tail": prompt[-300:]}, ensure_ascii=False) + "\n")
    report = {"model": args.model, "style": args.style, "prefill": args.prefill, "n": args.n,
              "temperature": args.temperature, "anonymized": args.anonymize, "data": args.data,
              "summary": summarize(rows, args.pass_k)}
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report["summary"].items() if "/" not in k}, indent=1))


if __name__ == "__main__":
    main()
