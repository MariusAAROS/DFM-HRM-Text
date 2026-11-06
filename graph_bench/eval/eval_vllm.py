"""Generate with vLLM on bench records and score with graph_bench.scoring (arms E0-E4; wiki: mimir-graph-benchmark).

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

MIN_ROOM = 16  # generation tokens below which a prompt is not run (scores 0)


def build_prompt(rec, style: str, tokenizer, shots: list = (), chat_kwargs: dict | None = None) -> str:
    if style == "base":
        demos = "".join(prompt_text(s) + direct_response(s) + "\n\n" for s in shots)
        return demos + prompt_text(rec)
    content = GRL_PREFIX + prompt_text(rec) if style == "sft" else user_message(rec)
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, **(chat_kwargs or {}))


def source_name(path: str) -> str:
    """<dataset dir>/<file stem>, e.g. metaqa/test_retrieved: the eval split (ids do not tell gold from retrieved)."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def load(paths: list[str], limit: int | None, anon: bool, seed: int):
    recs, sources, dropped = [], [], 0
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
            sources.append(source_name(path))
    return recs, sources, dropped


def summarize(rows: list[dict], ks: list[int]) -> dict:
    """Keys: <dataset>, <dataset>/<task>, file/<dataset dir>/<file stem> (one per input file); macro_acc over datasets."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
        groups[f"{row['dataset']}/{row['task']}"].append(row)
        if "source" in row:
            groups[f"file/{row['source']}"].append(row)
    out = {}
    for name, rs in sorted(groups.items()):
        n = rs[0]["n"]
        s = {"count": len(rs), "acc": sum(r["acc"][0] for r in rs) / len(rs),
             "f1": sum(r["f1"][0] for r in rs) / len(rs)}
        for k in ks:
            if k <= n:
                s[f"pass@{k}"] = sum(pass_at_k(n, sum(r["acc"]), k) for r in rs) / len(rs)
        out[name] = s
    datasets = [v for k, v in out.items() if "/" not in k]  # file/... keys have a "/" too
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
    ap.add_argument("--dtype", default="bfloat16", help="float32 for numerics audits (parity gate referee).")
    ap.add_argument("--limit", type=int, default=None, help="Random subset per input file.")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass-k", type=int, nargs="*", default=[1, 4, 16, 64])
    ap.add_argument("--allow-tokenizer-mismatch", action="store_true",
                    help="Run even if transformers tokenizes prompts differently from the model's tokenizer.json.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    recs, sources, dropped = load(args.data, args.limit, args.anonymize, args.seed)
    print(f"{len(recs)} records ({dropped} dropped: question lost its entity after anonymisation)")
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len, seed=args.seed,
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
    tok_json = Path(args.model) / "tokenizer.json"
    if tok_json.exists():  # local dirs only: HF's tokenizer must tokenize like tokenizer.json (data_io, native engine)
        from tokenizers import Tokenizer
        raw = Tokenizer.from_file(str(tok_json))
        bad = sum(tok(p, add_special_tokens=False)["input_ids"] != raw.encode(p, add_special_tokens=False).ids
                  for p in prompts[:50])
        if bad and not args.allow_tokenizer_mismatch:
            raise SystemExit(f"{bad}/50 prompts tokenize differently in transformers than in tokenizer.json (Mimir's "
                             "fix_mistral_regex flag?). Use a graph_bench.eval.hf_view dir, or "
                             "--allow-tokenizer-mismatch.")
    # Token ids, not text: vLLM tokenizes a text prompt with add_special_tokens=True, which puts a second <bos> before
    # the chat template's own (Gemma tokenizers add one), and that shifts every prediction (parity gate, 2026-11-03).
    # (style base has no template, so there the tokenizer must add the <bos>)
    prompt_ids = [tok(p, add_special_tokens=args.style == "base")["input_ids"] for p in prompts]
    # A prompt + max_tokens over the context gets the room that is left; with less than MIN_ROOM it is not generated
    # and scores 0 (kept in the denominator, so every model is scored on the same rows).
    room = [min(args.max_tokens, args.max_model_len - len(ids)) for ids in prompt_ids]
    shortened = sum(MIN_ROOM <= r < args.max_tokens for r in room)
    skipped = [r < MIN_ROOM for r in room]
    print(f"{shortened} prompts get fewer than --max-tokens (context), {sum(skipped)} do not fit and score 0")

    def params(r):
        return SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=r, seed=args.seed)
    todo = [i for i, s in enumerate(skipped) if not s]
    done = llm.generate([{"prompt_token_ids": prompt_ids[i]} for i in todo], [params(room[i]) for i in todo])
    outputs = [None] * len(recs)
    for i, out in zip(todo, done):
        outputs[i] = out

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".samples.jsonl"), "w", encoding="utf-8") as f:
        for rec, source, prompt, out in zip(recs, sources, prompts, outputs):
            texts = [o.text for o in out.outputs] if out is not None else [""] * args.n
            scores = [score_record(t, rec) for t in texts]
            row = {"id": rec.id, "dataset": rec.dataset, "task": rec.task, "source": source, "n": len(texts),
                   "acc": [s["acc"] for s in scores], "f1": [s["f1"] for s in scores]}
            rows.append(row)
            f.write(json.dumps({**row, "gold": rec.answers, "preds": [s["pred"] for s in scores],
                                "generations": texts, "prompt_tail": prompt[-300:]}, ensure_ascii=False) + "\n")
    report = {"model": args.model, "style": args.style, "prefill": args.prefill, "n": args.n,
              "temperature": args.temperature, "anonymized": args.anonymize, "data": args.data,
              "max_tokens": args.max_tokens, "max_model_len": args.max_model_len, "enforce_eager": args.enforce_eager,
              "shortened": shortened, "skipped_no_room": sum(skipped),
              "summary": summarize(rows, args.pass_k)}
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report["summary"].items() if "/" not in k}, indent=1))


if __name__ == "__main__":
    main()
