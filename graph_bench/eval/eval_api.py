"""Score an OpenAI-compatible API model on bench records (reference rows for E0-E4; wiki: mimir-graph-benchmark).

The API counterpart of eval_vllm.py, for models we can only call (GLM-5.3 on the SDU server): the same records and
row subsets (eval_vllm.load, seed 0), the same instruct prompt (export_verl.user_message), the same scoring
(scoring.score_record) and the same report / .samples.jsonl format, so log_wandb.py and osm_breakdown.py read it
unchanged. Greedy (temperature 0), one sample. The score reads message.content only; the hidden reasoning is kept
in the samples file, not scored.

GLM-5.3 on the SDU server (checked 2026-11-09): thinking cannot be turned off (chat_template_kwargs, thinking.type
and /nothink are all ignored), max_tokens is ignored, max_completion_tokens caps reasoning + answer together, and
reasoning_effort accepts only "low". Benchmark setting (user, 2026-11-09): --max-tokens 3072 --reasoning-effort low,
i.e. the other models' total budget, split by GLM between hidden reasoning and answer.

Rows are keyed by (source file, id): ids repeat across files (kqapro/test and kqapro/test_retrieved share 292 ids
in the named set), and keying by id alone scored one response against both rows (fixed 2026-11-09).

Long runs are resumable: every finished row is appended to <out>.partial.jsonl and skipped on restart; the report is
written once every row is done. Rows that still fail after --retries score 0 and stay in the denominator
(report: api_errors). Credentials come from --env (default: the repo's .env, GLM_SERVER / GLM_API_KEY); the key is
never written anywhere.

    /home/ucloud/miniforge3/bin/python -m graph_bench.eval.eval_api --model zai-org/GLM-5.3 --max-tokens 3072 \\
        --limit 1000 --out $ROOT/e0/glm53/named_t3k.json data/graphs/bench/graphqa/test.jsonl ...
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from graph_bench.eval.eval_vllm import load, summarize  # noqa: E402
from graph_bench.export_verl import user_message  # noqa: E402
from graph_bench.scoring import score_record  # noqa: E402


def read_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip().removeprefix("export ").strip()] = v.strip().strip("'\"")
    return env


async def call(client, sem, args, rec) -> dict:
    extra = {} if args.enable_thinking == "auto" else {
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking == "true"}}
    kw = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    err = None
    for attempt in range(args.retries + 1):
        try:
            async with sem:
                # max_completion_tokens, not max_tokens: the SDU server ignores max_tokens, and this cap covers hidden
                # reasoning + visible answer together (checked 2026-11-09)
                r = await client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": user_message(rec)}],
                    temperature=args.temperature, max_completion_tokens=args.max_tokens, seed=args.seed,
                    extra_body=extra, **kw)
            ch = r.choices[0]
            msg = ch.message
            usage = r.usage.model_dump() if r.usage else {}
            return {"text": msg.content or "", "reasoning": getattr(msg, "reasoning_content", None)
                    or (msg.model_extra or {}).get("reasoning_content") or (msg.model_extra or {}).get("reasoning"),
                    "finish": ch.finish_reason, "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}}
        except Exception as e:  # noqa: BLE001 - every API failure is retried, then recorded
            err = f"{type(e).__name__}: {str(e)[:300]}"
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                break  # a request error will not fix itself
            await asyncio.sleep(min(60.0, 2.0 ** attempt) * (1 + random.random()))
    return {"text": "", "reasoning": None, "finish": None, "usage": {}, "error": err}


async def run(args, recs, sources, partial: Path, done: dict):
    from openai import AsyncOpenAI
    env = read_env(Path(args.env))
    client = AsyncOpenAI(base_url=env[args.server_var], api_key=env[args.key_var], timeout=args.timeout,
                         max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    todo = [(r, s) for r, s in zip(recs, sources) if (s, r.id) not in done]
    print(f"{len(recs)} records, {len(done)} already done, {len(todo)} to call", flush=True)
    t0, n_done = time.time(), 0
    with open(partial, "a", encoding="utf-8") as f:
        async def one(rec, src):
            nonlocal n_done
            out = await call(client, sem, args, rec)
            row = {"id": rec.id, "source": src, **out}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[src, rec.id] = row
            n_done += 1
            if n_done % 100 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"{n_done}/{len(todo)} in {el / 60:.1f} min ({n_done / el:.2f} rows/s)", flush=True)
        await asyncio.gather(*(one(r, s) for r, s in todo))
    await client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="Bench JSONL files (one dataset/split each).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Report JSON; per-sample generations go next to it (.samples.jsonl).")
    ap.add_argument("--env", default=str(REPO / ".env"))
    ap.add_argument("--server-var", default="GLM_SERVER")
    ap.add_argument("--key-var", default="GLM_API_KEY")
    ap.add_argument("--enable-thinking", choices=["auto", "true", "false"], default="auto",
                    help="chat_template_kwargs enable_thinking. GLM-5.3 on the SDU server ignores it (thinking stays on).")
    ap.add_argument("--reasoning-effort", default=None, help="e.g. low (the only value GLM-5.3 on SDU accepts).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=None, help="Random subset per input file (as eval_vllm).")
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    recs, sources, dropped = load(args.data, args.limit, args.anonymize, args.seed)
    print(f"{len(recs)} records ({dropped} dropped: question lost its entity after anonymisation)", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.jsonl")
    done = {}
    if partial.exists():
        for line in open(partial, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):  # failed rows are retried on restart
                    done[row["source"], row["id"]] = row
    asyncio.run(run(args, recs, sources, partial, done))

    rows, errors, cut, usage = [], 0, 0, {"prompt_tokens": 0, "completion_tokens": 0}
    with open(out_path.with_suffix(".samples.jsonl"), "w", encoding="utf-8") as f:
        for rec, source in zip(recs, sources):
            g = done[source, rec.id]
            errors += bool(g.get("error"))
            cut += g.get("finish") == "length"
            for k in usage:
                usage[k] += g.get("usage", {}).get(k) or 0
            s = score_record(g["text"], rec)
            row = {"id": rec.id, "dataset": rec.dataset, "task": rec.task, "source": source, "n": 1,
                   "acc": [s["acc"]], "f1": [s["f1"]]}
            rows.append(row)
            extra = {k: g[k] for k in ("reasoning", "finish", "error") if g.get(k)}
            f.write(json.dumps({**row, "gold": rec.answers, "preds": [s["pred"]], "generations": [g["text"]],
                                "prompt_tail": user_message(rec)[-300:], **extra}, ensure_ascii=False) + "\n")
    server = read_env(Path(args.env))[args.server_var]
    report = {"model": args.model, "style": "instruct", "prefill": "", "n": 1, "temperature": args.temperature,
              "anonymized": args.anonymize, "data": args.data, "max_tokens": args.max_tokens, "max_model_len": None,
              "enforce_eager": None, "shortened": 0, "skipped_no_room": 0, "backend": "openai-api",
              "server_host": urlparse(server).netloc, "enable_thinking": args.enable_thinking,
              "reasoning_effort": args.reasoning_effort, "token_cap": "max_completion_tokens (reasoning + answer)",
              "api_errors": errors, "finish_length": cut,
              "with_reasoning": sum(bool(done[s, r.id].get("reasoning")) for r, s in zip(recs, sources)),
              "usage": usage, "summary": summarize(rows, [1])}
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report["summary"].items() if "/" not in k}, indent=1))
    print(f"api_errors {errors}, finish_length {cut}, with_reasoning {report['with_reasoning']}, usage {usage}")


if __name__ == "__main__":
    main()
