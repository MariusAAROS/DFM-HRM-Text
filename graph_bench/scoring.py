"""Shared scorer for every model and backend (native engine, vLLM, HF, VeRL reward).

normalize / answer_list / set_scores are eval_graphqa.py's definitions, copied so this module imports neither torch
nor the repo (VeRL reward workers run in another env); tests/test_graph_bench.py checks they still agree.

`acc` extends eval_graphqa's greedy_acc to the answer formats of schema.py:
  single   normalize(pred) equals one alias      (= greedy_acc on single-answer data)
  set      the normalized answer sets are equal   (P / R / F1 / hits@1 are the partial-credit view)
  ordered  the normalized lists are equal
  checker  checkers.CHECKERS[name](pred, payload)
The prediction is the text after the last "Answer:" when there is one (chain-of-thought outputs), after a closing
think tag when there is one (thinking models), else the whole generation.
"""
import math
import re

from graph_bench.checkers import CHECKERS

ANSWER_MARKER = "Answer:"
THINK_CLOSERS = ("</think>", "<channel|>")
_ANSWER_LINE = re.compile(r"(?im)^[ \t]*answer[ \t]*:[ \t]*(.*)$")


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


def extract_answer(text: str) -> str:
    """The answer part of a generation: after the last "Answer:" line, else after a think closer, else all."""
    matches = list(_ANSWER_LINE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    for closer in THINK_CLOSERS:
        if closer in text:
            text = text.rsplit(closer, 1)[-1]
    return text.strip()


def _ordered_list(text: str) -> list[str]:
    return [a for a in (normalize(p) for p in re.split(r",|\|", text)) if a]


def score(pred_text: str, answers: list[str], answer_format: str = "single", checker: dict | None = None) -> dict:
    """Metrics of one generation against one record's gold (see the module docstring)."""
    pred = extract_answer(pred_text)
    if answer_format == "single":
        acc = normalize(pred) in {normalize(a) for a in answers}
        p, r, f1, hits1 = set_scores(pred, answers[0]) if not acc else (1.0, 1.0, 1.0, 1.0)
    elif answer_format == "set":
        gold = "|".join(answers)
        acc = set(answer_list(pred)) == set(answer_list(gold))
        p, r, f1, hits1 = set_scores(pred, gold)
    elif answer_format == "ordered":
        acc = _ordered_list(pred) == [normalize(a) for a in answers]
        p = r = f1 = hits1 = float(acc)
    elif answer_format == "checker":
        if not checker or checker.get("name") not in CHECKERS:
            raise ValueError(f"unknown checker {checker!r}")
        acc = bool(CHECKERS[checker["name"]](pred, checker))
        p = r = f1 = hits1 = float(acc)
    else:
        raise ValueError(f"unknown answer_format {answer_format!r}")
    return {"acc": bool(acc), "precision": p, "recall": r, "f1": f1, "hits_at_1": hits1, "pred": pred}


def score_record(pred_text: str, rec) -> dict:
    return score(pred_text, rec.answers, rec.answer_format, rec.checker)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
