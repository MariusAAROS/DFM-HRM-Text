"""VeRL custom reward for the graph sets: the evaluation metric itself (graph_bench.scoring), not a proxy.

ground_truth is the JSON written by graph_bench.export_verl ({answers, answer_format, checker}). The completion must
end its reasoning with an "Answer:" line; without one the reward is 0, so dumping text cannot score.

Reward (GRAPH_REWARD env var): "acc" (default) is 1/0 on the record's acc; "f1" gives set-answer records their
set-F1 as partial credit (acc elsewhere). The returned dict also carries acc and f1 for logging and for
algorithm.filter_groups.metric=acc (DAPO dynamic sampling).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for graph_bench in VeRL's reward workers

from graph_bench.scoring import _ANSWER_LINE, score  # noqa: E402

MODE = os.environ.get("GRAPH_REWARD", "acc")


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not _ANSWER_LINE.search(solution_str):
        return {"score": 0.0, "acc": 0.0, "f1": 0.0, "has_answer": 0.0}
    m = score(solution_str, gt["answers"], gt["answer_format"], gt.get("checker"))
    value = m["f1"] if MODE == "f1" and gt["answer_format"] == "set" else float(m["acc"])
    return {"score": value, "acc": float(m["acc"]), "f1": m["f1"], "has_answer": 1.0}
