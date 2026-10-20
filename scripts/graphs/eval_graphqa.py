"""Exact-match evaluation of a trained HRM-Text checkpoint on GraphQA-style JSONL (also MetaQA, KQA Pro, pointer).

Ported from HRM-Text-for-Graphs/scripts/eval_graphqa.py. Answers are short and deterministic ("Yes.",
a node id, a small list). Three exact-match metrics, all from one greedy generation:
  greedy_acc  normalize(pred) == normalize(gold): the whole answer, case / whitespace / trailing-period
              insensitive, nothing generated after it. The headline metric (per-task: task_accuracy).
  greedy_em   pred.startswith(gold) on raw strings: whitespace sensitive, ignores what follows.
              (Called tf_exact_match before 2026-10-20; it is not teacher-forced.)
  tf_em       teacher-forced exact match, GRL's val/exact_match (argmax == label at every answer token,
              stop token and anything after excluded). Under greedy decoding this is exactly "the
              generated token ids start with the gold answer's training token ids", so it is read off
              the generation; it needs --max_generation >= the answer length (rows that do not fit are
              counted wrong and reported as tf_em_unmeasured).
Set-based precision / recall / F1 / hits@1 over '|'-joined answers are reported for the multi-answer
variants (e.g. metaqa-gold-5); on single-answer data they reduce to exact match. Rows with a `task`
field also get per-task greedy_acc.

Generation uses simple_inference_engine.inference_generate (batched greedy decoding with a KV cache;
prompt prefill is bidirectional like the PrefixLM training prefix). Prompts are tokenized exactly as in
training (scripts/graphs/check_tokenization_parity.py verifies this).

The recursion depth (H_cycles / L_cycles) is read from the checkpoint's all_config.yaml.
--H_cycles / --L_cycles override it at inference (no weight depends on them), e.g. --L_cycles 1 tests
whether a model trained at L = 6 actually uses its extra L steps.

Usage:
    .venv/bin/python scripts/graphs/eval_graphqa.py \
        --ckpt_path <ckpt_dir> --data data/graphs/graphqa/hrm-text/standard/test.jsonl \
        --use_ema --out <ckpt_dir>/eval_test.json
    (logs to W&B project HRM-Graph, group = the checkpoint's wandb_group; --no_wandb to skip)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import wandb
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from simple_inference_engine import inference_generate, inference_load_checkpoint  # noqa: E402


def normalize(text: str) -> str:
    """Whitespace/case/trailing-punctuation-insensitive normalization."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .")


def answer_list(text: str) -> list[str]:
    """Split a '|'-joined answer string into normalized answers, order preserved."""
    return [a for a in (normalize(part) for part in text.split("|")) if a]


def set_scores(pred: str, label: str) -> tuple[float, float, float, float]:
    """Precision / recall / F1 / hits@1 between two separator-joined answer lists."""
    gold, got = set(answer_list(label)), answer_list(pred)
    if not gold:
        return 0.0, 0.0, 0.0, 0.0
    hit = len(gold & set(got))
    precision = hit / len(got) if got else 0.0
    recall = hit / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return precision, recall, f1, float(bool(got) and got[0] in gold)


def read_config(ckpt_path: str) -> dict:
    cfg_file = os.path.join(ckpt_path, "all_config.yaml")
    if not os.path.exists(cfg_file):
        return {}
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Directory with fsdp2_*/unsharded_* checkpoints + all_config.yaml.")
    ap.add_argument("--data", required=True, help="JSONL (instruction/response/condition).")
    ap.add_argument("--ckpt_epoch", type=int, default=None, help="Epoch to load (default: latest).")
    ap.add_argument("--ckpt_tag", default=None, help="Explicit checkpoint tag, e.g. step_250 (overrides --ckpt_epoch).")
    ap.add_argument("--use_ema", action="store_true", help="Use EMA weights (recommended).")
    ap.add_argument("--max_generation", type=int, default=96, help="Max new tokens for the answer (GraphQA gold answers reach 62 tokens).")
    ap.add_argument("--max_tokens", type=int, default=4096, help="KV-cache length: prompt + generation upper bound.")
    ap.add_argument("--batch_size", type=int, default=64, help="Concurrent generations.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of eval samples.")
    ap.add_argument("--out", default=None, help="Where to write the JSON report.")
    ap.add_argument("--H_cycles", type=int, default=None, help="Override the trained H_cycles at inference.")
    ap.add_argument("--L_cycles", type=int, default=None, help="Override the trained L_cycles at inference.")
    # wandb (on by default; the queues pass --no_wandb and log reports via log_graphqa_evals_to_wandb.py)
    ap.add_argument("--wandb_project", default="HRM-Graph", help="W&B project name.")
    ap.add_argument("--wandb_group", default=None, help="W&B group (default: the checkpoint's wandb_group).")
    ap.add_argument("--no_wandb", action="store_true", help="Do not log to W&B.")
    ap.add_argument("--wandb_entity", default=None, help="W&B entity (team or user). Uses default if omitted.")
    ap.add_argument("--wandb_name", default=None, help="W&B run name. Auto-generated from ckpt if omitted.")
    ap.add_argument("--wandb_tags", nargs="*", default=None, help="Optional W&B tags (e.g. --wandb_tags ablation ema).")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} eval samples from {args.data}")

    overrides = {k: v for k, v in (("H_cycles", args.H_cycles), ("L_cycles", args.L_cycles)) if v is not None}
    ckpt = inference_load_checkpoint(args.ckpt_path, args.ckpt_epoch, args.use_ema, ckpt_tag=args.ckpt_tag,
                                     arch_overrides=overrides)
    train_cfg = read_config(args.ckpt_path)
    trained_H, trained_L = train_cfg.get("arch", {}).get("H_cycles"), train_cfg.get("arch", {}).get("L_cycles")
    H_cycles, L_cycles = overrides.get("H_cycles", trained_H), overrides.get("L_cycles", trained_L)

    # ---- wandb init ---------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_name or os.path.basename(os.path.normpath(args.ckpt_path))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group or train_cfg.get("wandb_group"),
            job_type="eval",
            tags=args.wandb_tags,
            config={
                "ckpt_path": args.ckpt_path,
                "ckpt_epoch": args.ckpt_epoch,
                "ckpt_tag": args.ckpt_tag,
                "data": args.data,
                "use_ema": args.use_ema,
                "max_generation": args.max_generation,
                "limit": args.limit,
                "H_cycles": H_cycles,
                "L_cycles": L_cycles,
                "trained_H_cycles": trained_H,
                "trained_L_cycles": trained_L,
                "ratio_L_over_H": (L_cycles / H_cycles) if (H_cycles and L_cycles) else None,
            },
        )

    prompts = ((idx, (r.get("condition", "direct"), r["instruction"])) for idx, r in enumerate(rows))
    preds: dict[int, str] = {}
    pred_ids: dict[int, list[int]] = {}
    for idx, pred, ids in inference_generate(ckpt, prompts, max_tokens=args.max_tokens, max_generation=args.max_generation,
                                             batch_size=args.batch_size, return_tokens=True):
        preds[idx] = pred
        pred_ids[idx] = ids.tolist()
        if len(preds) % 500 == 0:
            print(f"  [{len(preds)}/{len(rows)}] generated", flush=True)

    # Gold answer tokens exactly as training supervised them (stop token excluded): the chat-template tokenizer
    # trims the response, the HRM (Rust) template does not.
    chat = ckpt.tokenizer_info.get("template_mode") == "jinja_chat_template"
    def gold_ids(gold: str) -> list[int]:
        return ckpt.tokenizer(gold.strip() if chat else gold, add_special_tokens=False)["input_ids"]

    results = []
    correct = em_correct = tf_correct = tf_unmeasured = 0
    per_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    sum_p = sum_r = sum_f1 = sum_hits1 = 0.0
    for idx, row in enumerate(rows):
        pred, gold = preds[idx], row["response"]
        is_correct = normalize(pred) == normalize(gold)
        em_match = pred.startswith(gold)
        gold_tok = gold_ids(gold)
        tf_match = pred_ids[idx][:len(gold_tok)] == gold_tok
        tf_unmeasured += int(len(gold_tok) > args.max_generation)
        correct += int(is_correct)
        em_correct += int(em_match)
        tf_correct += int(tf_match)
        if "task" in row:
            per_task[row["task"]][0] += int(is_correct)
            per_task[row["task"]][1] += 1
        p, r, f1, hits1 = set_scores(pred, gold)
        sum_p += p
        sum_r += r
        sum_f1 += f1
        sum_hits1 += hits1
        results.append({"id": idx, "task": row.get("task"), "gold": gold, "pred": pred,
                        "greedy_acc": is_correct, "greedy_em": em_match, "tf_em": tf_match,
                        "precision": p, "recall": r, "f1": f1, "hits_at_1": hits1})

    n = len(rows)
    greedy_acc = correct / n if n else 0.0
    greedy_em = em_correct / n if n else 0.0
    tf_em = tf_correct / n if n else 0.0
    if tf_unmeasured:
        print(f"WARNING: {tf_unmeasured} gold answers are longer than --max_generation {args.max_generation}; "
              "tf_em counts them wrong")
    task_accuracy = {t: c / k for t, (c, k) in sorted(per_task.items())}
    precision = sum_p / n if n else 0.0
    recall = sum_r / n if n else 0.0
    f1 = sum_f1 / n if n else 0.0
    hits_at_1 = sum_hits1 / n if n else 0.0
    report = {
        "ckpt_path": args.ckpt_path,
        "ckpt_tag": args.ckpt_tag,
        "data": args.data,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "trained_H_cycles": trained_H,
        "trained_L_cycles": trained_L,
        "ratio_L_over_H": (L_cycles / H_cycles) if (H_cycles and L_cycles) else None,
        "n": n,
        "correct": correct,  # greedy_acc count
        "greedy_acc": greedy_acc,
        "greedy_em": greedy_em,
        "tf_em": tf_em,
        "tf_em_unmeasured": tf_unmeasured,
        "max_generation": args.max_generation,
        "task_accuracy": task_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits_at_1": hits_at_1,
        "use_ema": args.use_ema,
        "samples": results,
    }

    print(f"H_cycles={H_cycles} L_cycles={L_cycles}  "
          f"greedy_acc = {greedy_acc:.4f} ({correct}/{n})  greedy_em = {greedy_em:.4f}  tf_em = {tf_em:.4f}  "
          f"P = {precision:.4f}  R = {recall:.4f}  F1 = {f1:.4f}  hits@1 = {hits_at_1:.4f}")

    # ---- wandb log ----------------------------------------------------------
    if use_wandb:
        wandb.log({"eval/greedy_acc": greedy_acc, "eval/greedy_em": greedy_em, "eval/tf_em": tf_em, "eval/correct": correct,
                   "eval/n": n, "eval/precision": precision, "eval/recall": recall, "eval/f1": f1,
                   "eval/hits_at_1": hits_at_1, **{f"eval/task/{t}": a for t, a in task_accuracy.items()}})

        # Per-sample table for drill-down in the wandb UI
        table = wandb.Table(columns=["id", "task", "gold", "pred", "greedy_acc", "greedy_em", "tf_em",
                                     "precision", "recall", "f1", "hits_at_1"])
        for r in results:
            table.add_data(r["id"], r["task"], r["gold"], r["pred"], r["greedy_acc"], r["greedy_em"], r["tf_em"],
                           r["precision"], r["recall"], r["f1"], r["hits_at_1"])
        wandb.log({"eval/samples": table})
        wandb.finish()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
