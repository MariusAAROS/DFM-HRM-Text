"""Greedy KV-cache generation and GraphQA metrics for a GraphInjectedLM.

Same decoding as simple_inference_engine.inference_generate (batch-1 prefill of each prompt into a free
cache slot, then batched one-token decodes with _batched_decode), written synchronously and with one
addition: the prefill batch carries the method's graph keys. Decode steps carry none, so methods act on
prompt tokens only, as in training (response tokens have no graph element and no soft slot).

Metrics are eval_graphqa.py's (normalize / set_scores imported from it): greedy_acc, greedy_em, tf_em,
P/R/F1/hits@1, per-task greedy_acc, plus greedy_acc without MaximumFlow, whose capacities are not in the
GraphQA prompt text (unanswerable, ~9% of test).
"""
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils._pytree as pytree

from graph_inject.data import EncodedSample
from graph_inject.methods.base import CollateItem, GraphMethod
from simple_inference_engine import _batched_decode

_spec = importlib.util.spec_from_file_location(
    "eval_graphqa", Path(__file__).resolve().parents[1] / "scripts" / "graphs" / "eval_graphqa.py")
eval_graphqa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_graphqa)  # pyright: ignore[reportOptionalMemberAccess]

UNANSWERABLE_TASKS = ("MaximumFlow",)
ANSWER_MARKER = "Answer:"


def _prefill_batch(sample: EncodedSample, method: GraphMethod, cache_slice, device) -> dict:
    ids = torch.from_numpy(sample.prompt.ids).to(device)
    graph = method.collate([CollateItem(sample.sample, sample.features, sample.prompt, offset=0)], rng=None)
    return {"inputs": ids[None], "position_ids": torch.arange(ids.shape[0], device=device), "cache": cache_slice,
            "cache_lengths": torch.zeros(1, dtype=torch.int32, device=device),
            **{k: v.to(device) for k, v in graph.items()}}


@torch.inference_mode()
def generate(model, carry, method: GraphMethod, samples: list[EncodedSample], stop_id: int,
             max_generation: int = 96, batch_size: int = 16) -> list[list[int]]:
    """Greedy generations (token ids incl. the stop token if reached), in sample order."""
    device = next(model.parameters()).device
    max_tokens = max(len(s.prompt.ids) for s in samples) + max_generation + 1
    cache = model.create_cache(max_batch_size=batch_size, max_seq_len=max_tokens, dtype=torch.bfloat16, device=device)
    cache_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
    last = torch.zeros(batch_size, dtype=torch.long, device=device)
    slot_sample = [-1] * batch_size
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    stopped = np.ones(batch_size, dtype=bool)
    results: dict[int, list[int]] = {}
    queue = list(range(len(samples)))[::-1]

    while True:
        for i in np.nonzero(stopped)[0]:
            if slot_sample[i] != -1:
                results[slot_sample[i]] = generated[i]
                slot_sample[i] = -1
            if not queue:
                continue
            j = queue.pop()
            s = samples[j]
            batch = _prefill_batch(s, method, pytree.tree_map(lambda x: x[i: i + 1], cache), device)
            token = int(model(carry=carry, batch=batch)[-1][0, -1].argmax())
            last[i], cache_lengths[i] = token, len(s.prompt.ids)
            slot_sample[i], generated[i] = j, [token]
            stopped[i] = token == stop_id or max_generation == 1
        if stopped.all():
            if not queue:
                break
            continue
        tokens = _batched_decode(model, carry, last, cache, cache_lengths).argmax(-1)
        cache_lengths.add_(1).clamp_max_(max_tokens - 1)
        last = tokens
        tokens_np = tokens.cpu().numpy()
        for i in np.nonzero(~stopped)[0]:
            generated[i].append(int(tokens_np[i]))
            stopped[i] = tokens_np[i] == stop_id or len(generated[i]) >= max_generation
    for i, j in enumerate(slot_sample):  # prompts that stopped at their prefill token
        if j != -1:
            results[j] = generated[i]
    return [results[j] for j in range(len(samples))]


def final_answer(text: str, gold: str) -> str:
    """Chain-of-thought targets end in "Answer: <x>": greedy_acc compares only <x> (tf_em stays on the full text)."""
    return text.rsplit(ANSWER_MARKER, 1)[-1] if ANSWER_MARKER in gold else text


def score(samples: list[EncodedSample], generations: list[list[int]], decode, stop_id: int) -> dict:
    """eval_graphqa.py's metrics; `decode(ids)` turns generated ids (stop token removed) into text."""
    per_task = defaultdict(lambda: [0, 0])
    rows, sums = [], defaultdict(float)
    for s, ids in zip(samples, generations):
        text_ids = ids[:-1] if ids and ids[-1] == stop_id else ids
        pred, gold = decode(text_ids), s.sample.response
        acc = eval_graphqa.normalize(final_answer(pred, gold)) == eval_graphqa.normalize(final_answer(gold, gold))
        em = pred.startswith(gold)
        tf = ids[:len(s.gold_ids)] == s.gold_ids
        p, r, f1, hits1 = eval_graphqa.set_scores(pred, gold)
        per_task[s.sample.task][0] += int(acc)
        per_task[s.sample.task][1] += 1
        for k, v in (("greedy_acc", acc), ("greedy_em", em), ("tf_em", tf), ("precision", p), ("recall", r),
                     ("f1", f1), ("hits_at_1", hits1)):
            sums[k] += float(v)
        rows.append({"id": s.sample.idx, "task": s.sample.task, "gold": gold, "pred": pred,
                     "greedy_acc": acc, "greedy_em": em, "tf_em": tf})
    n = len(samples)
    answerable = [r for r in rows if r["task"] not in UNANSWERABLE_TASKS]
    return {"n": n, **{k: v / n for k, v in sums.items()},
            "greedy_acc_answerable": sum(r["greedy_acc"] for r in answerable) / max(len(answerable), 1),
            "task_accuracy": {t: c / k for t, (c, k) in sorted(per_task.items())},
            "samples": rows}
