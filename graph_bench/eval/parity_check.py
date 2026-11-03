"""vLLM-vs-native parity for the §7a gate (wiki: mimir-graph-benchmark-plan, RL stack).

Two checks, both CPU-only once their inputs exist:

  tokens   Does the prompt vLLM receives (HF tokenizer + chat template, eval_vllm.py --style sft) tokenize to the same
           ids as the SFT prompt built by data_io (scripts/tokenize_chat_template.py, Gemma template)? Mismatched
           prompts would make a vLLM score of an SFT model meaningless.
               .venv/bin/python -m graph_bench.eval.parity_check tokens --tokenizer <hf dir> \\
                   --template data_io/chat_templates/gemma4_native_chat.jinja data/graphs/bench/metaqa/test.jsonl

  outputs  Per-sample agreement of two greedy runs on the same rows: a native report (scripts/graphs/eval_graphqa.py
           --out, "samples" with pred / greedy_acc) and a vLLM report (eval_vllm.py .samples.jsonl). Rows are
           matched by order (eval_graphqa's ids are row indices) and must share the gold answer.
               .venv/bin/python -m graph_bench.eval.parity_check outputs native.json vllm.samples.jsonl
Gate: identical prompt ids on every row; greedy_acc within 1 pt and >= 95% identical predictions (bf16 noise flips a
few argmaxes, as in the HF-import parity: 202/203).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_bench.schema import GRL_PREFIX, prompt_text, read_records  # noqa: E402
from graph_bench.scoring import normalize  # noqa: E402


def check_tokens(args) -> int:
    import jinja2
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import tokenize_chat_template as tct  # data_io's SFT rendering

    hf = AutoTokenizer.from_pretrained(args.tokenizer)
    template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(Path(args.template).read_text())
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    bad = 0
    recs = [r for path in args.data for r in read_records(path)][: args.limit]
    for rec in recs:
        msgs = [{"role": "user", "content": GRL_PREFIX + prompt_text(rec)}]
        sft_ids = raw.encode(tct.render(template, msgs, [], True, False), add_special_tokens=False).ids
        hf_text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        hf_ids = hf(hf_text, add_special_tokens=False)["input_ids"]
        if sft_ids != hf_ids:
            bad += 1
            if bad <= 3:
                i = next((k for k, (a, b) in enumerate(zip(sft_ids, hf_ids)) if a != b), min(len(sft_ids), len(hf_ids)))
                print(f"MISMATCH {rec.id} at token {i}: sft {sft_ids[i:i + 8]} vs hf {hf_ids[i:i + 8]}")
    print(f"tokens: {len(recs) - bad}/{len(recs)} prompts identical")
    return int(bad > 0)


def check_outputs(args) -> int:
    native = json.loads(Path(args.native).read_text())["samples"]
    with open(args.vllm) as f:
        vllm = [json.loads(line) for line in f if line.strip()]
    if len(native) != len(vllm):
        raise SystemExit(f"row counts differ: native {len(native)} vs vllm {len(vllm)}")
    same_pred = acc_native = acc_vllm = 0
    for nat, vl in zip(native, vllm):
        gold = vl["gold"][0]
        if normalize(nat["gold"]) != normalize(gold):
            raise SystemExit(f"row {nat['id']}: gold differs ({nat['gold']!r} vs {gold!r}); not the same rows")
        same_pred += normalize(nat["pred"]) == normalize(vl["preds"][0])
        acc_native += nat["greedy_acc"]
        acc_vllm += vl["acc"][0]
    n = len(native)
    print(f"outputs: n={n} identical predictions {same_pred / n:.3f}; greedy_acc native {acc_native / n:.4f} "
          f"vs vLLM {acc_vllm / n:.4f}")
    ok = same_pred / n >= 0.95 and abs(acc_native - acc_vllm) / n <= 0.01
    print("PASS" if ok else "FAIL")
    return int(not ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("data", nargs="+")
    t.add_argument("--tokenizer", required=True, help="Local HF model dir (tokenizer.json + chat template).")
    t.add_argument("--template", default="data_io/chat_templates/gemma4_native_chat.jinja")
    t.add_argument("--limit", type=int, default=200)
    o = sub.add_parser("outputs")
    o.add_argument("native")
    o.add_argument("vllm")
    args = ap.parse_args()
    sys.exit(check_tokens(args) if args.cmd == "tokens" else check_outputs(args))


if __name__ == "__main__":
    main()
