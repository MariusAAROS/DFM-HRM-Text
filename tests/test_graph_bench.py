"""graph_bench: scorer (and its agreement with eval_graphqa.py), checkers, anonymisation, rendering, VeRL reward.

CPU only; the eval_graphqa agreement test imports scripts/graphs/eval_graphqa.py (torch, wandb).
"""
import importlib.util
import json
import random
from pathlib import Path

import pytest

from graph_bench import checkers, scoring
from graph_bench.anonymize import anonymize
from graph_bench.schema import Record, cot_response, direct_response, prompt_text

REPO = Path(__file__).resolve().parents[1]


def _rec(**kw) -> Record:
    base = dict(id="t/0", dataset="toy", split="test", task="2hop", question="who directed the films written by Ann",
                answers=["Bob"], facts=[["Film A", "written by", "Ann"], ["Film A", "directed by", "Bob"]],
                entities=["Ann", "Bob", "Film A"])
    base.update(kw)
    return Record(**base)


def test_scoring_matches_eval_graphqa():
    spec = importlib.util.spec_from_file_location("eval_graphqa", REPO / "scripts/graphs/eval_graphqa.py")
    eg = importlib.util.module_from_spec(spec)
    pytest.importorskip("torch")
    spec.loader.exec_module(eg)
    rng = random.Random(0)
    words = ["Yes.", " yes ", "No.", "0, 3, 2.", "Food, Inc..", "a | b", "A|B|c", "", "  4.", "x  y."]
    for _ in range(300):
        p, g = rng.choice(words), rng.choice(words)
        assert scoring.normalize(p) == eg.normalize(p)
        assert scoring.answer_list(p) == eg.answer_list(p)
        assert scoring.set_scores(p, g) == eg.set_scores(p, g)


def test_single_answer_acc_is_greedy_acc():
    assert scoring.score(" bob. ", ["Bob"])["acc"]
    assert not scoring.score("Bob Smith", ["Bob"])["acc"]
    assert scoring.score("Robert", ["Bob", "Robert"])["acc"]  # aliases


def test_extract_answer():
    cot = "Start: Ann.\nAnswer: wrong.\n(Film A, directed by, Bob)\nAnswer: Bob."
    assert scoring.extract_answer(cot) == "Bob."
    assert scoring.extract_answer("<think>hmm</think> Bob") == "Bob"
    assert scoring.extract_answer("thinking...<channel|>Bob.") == "Bob."
    assert scoring.score(cot, ["Bob"])["acc"]


def test_special_tokens():
    # VeRL's reward manager keeps special tokens (hrm_verl_ext); vLLM eval drops them. Both must score the same.
    assert scoring.extract_answer("x\nAnswer: Paris<turn|>") == "Paris"
    assert scoring.extract_answer("x\nAnswer: Paris<turn|><eos>") == "Paris"
    assert scoring.extract_answer("<|channel>thought\nx<channel|>Answer: Paris<turn|>") == "Paris"
    assert scoring.extract_answer("thinking...<channel|>Bob.<turn|>") == "Bob."
    assert scoring.extract_answer("Answer: Bob.<|box_end|>") == "Bob."
    assert scoring.score("Answer: A | B<turn|>", ["a", "b"], "set")["acc"]
    assert not scoring.has_answer_line("Paris<turn|>Answer: Lyon")  # text after the end of turn is ignored


def test_set_and_ordered():
    m = scoring.score("Answer: b | A", ["a", "b"], "set")
    assert m["acc"] and m["f1"] == 1.0
    m = scoring.score("Answer: a | c", ["a", "b"], "set")
    assert not m["acc"] and m["f1"] == pytest.approx(0.5)
    assert scoring.score("1, 2, 3", ["1", "2", "3"], "ordered")["acc"]
    assert not scoring.score("1, 3, 2", ["1", "2", "3"], "ordered")["acc"]


def test_pass_at_k():
    assert scoring.pass_at_k(10, 0, 4) == 0.0
    assert scoring.pass_at_k(10, 10, 1) == 1.0
    assert scoring.pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert scoring.pass_at_k(4, 1, 4) == 1.0


def test_checkers():
    dag = {"n": 4, "edges": [[0, 1], [0, 2], [2, 3]]}
    assert checkers.topological_sort("0, 2, 3, 1", dag)
    assert not checkers.topological_sort("2, 0, 3, 1", dag)
    path = {"n": 4, "edges": [[0, 1], [1, 2], [2, 3], [0, 3]]}
    assert checkers.hamilton_path("1 -> 0 -> 3 -> 2", path)
    assert not checkers.hamilton_path("0, 2, 1, 3", path)
    sp = {"source": 0, "target": 3, "edges": [[0, 1, 1], [1, 3, 1], [0, 3, 5]], "optimal": 2}
    assert checkers.simple_path("0,1,3", sp) and not checkers.simple_path("0,3", sp)
    tri = {"n": 4, "edges": [[0, 1], [1, 2], [0, 2], [2, 3]], "optimal": 3, "property": "clique"}
    assert checkers.node_set("[0, 1, 2]", tri) and not checkers.node_set("1, 2, 3", tri)
    cover = {"n": 3, "edges": [[0, 1], [1, 2]], "optimal": 1, "property": "vertex_cover"}
    assert checkers.node_set("1", cover) and not checkers.node_set("0", cover)
    match = {"edges": [[0, 2], [1, 3], [0, 3]], "optimal": 2}
    assert checkers.matching("(0, 2), (1, 3)", match) and not checkers.matching("0 3", match)
    tsp = {"n": 3, "edges": [[0, 1, 1], [1, 2, 2], [0, 2, 3]], "optimal": 6}
    assert checkers.tour("0, 1, 2", tsp) and checkers.tour("0 2 1 0", tsp)
    assert checkers.number("The flow is 1,200.", {"value": 1200})
    assert scoring.score("Answer: 0, 2, 3, 1", ["0, 1, 2, 3"], "checker",
                         {"name": "topological_sort", **dag})["acc"]


def test_rendering():
    rec = _rec(trace=["Start: Ann.", "(Film A, written by, Ann) gives Film A.", "(Film A, directed by, Bob) gives Bob."])
    text = prompt_text(rec)
    assert text.startswith("In a knowledge graph") and "The facts are:\n(Film A, written by, Ann)\n" in text
    assert text.endswith("Q: who directed the films written by Ann\nA: ")
    assert direct_response(rec) == "Bob."
    assert cot_response(rec).endswith("gives Bob.\nAnswer: Bob.")
    q = _rec(facts=[["Ann", "award received", "Prize X", "point in time: 1969"]])
    assert "qualifiers, given in brackets" in prompt_text(q) and "(Ann, award received, Prize X) [point in time: 1969]" \
        in prompt_text(q)
    s = _rec(answers=["b", "a"], answer_format="set")
    assert direct_response(s) == "b | a."


def test_anonymize():
    rec = _rec(trace=["(Film A, directed by, Bob) gives Bob."])
    anon, grounded = anonymize(rec)
    assert grounded and anon.id.endswith("#anon")
    assert "Ann" not in prompt_text(anon) and "Bob" not in prompt_text(anon) and "Film A" not in prompt_text(anon)
    assert anon.answers[0] in prompt_text(anon)            # answer still in the graph
    assert "directed by" in prompt_text(anon)               # relations untouched
    assert anon.trace[0].endswith(f"gives {anon.answers[0]}.")
    assert anonymize(rec)[0] == anon                        # reproducible
    ungrounded, ok = anonymize(_rec(question="who directed the films written by Annie"))
    assert not ok


def test_reward():
    from graph_bench.rl.reward import compute_score
    gt = json.dumps({"answers": ["a", "b"], "answer_format": "set", "checker": None})
    assert compute_score("toy", "reasoning\nAnswer: a | b", gt)["score"] == 1.0
    assert compute_score("toy", "a | b", gt)["score"] == 0.0          # no Answer line
    assert compute_score("toy", "x\nAnswer: a", gt)["acc"] == 0.0
    assert compute_score("toy", "x\nAnswer: a | b<turn|>", gt)["score"] == 1.0
    assert compute_score("toy", "<|channel>thought\nx<channel|>Answer: a | b<turn|>", gt)["score"] == 1.0


def test_kqapro_gold_validation_rules():
    from graph_bench.sources.kqapro_gold import SpineBuilder
    kb = {"entities": {"Q1": {"name": "Chicago", "instanceOf": ["C1"], "attributes": [], "relations": []},
                       "Q2": {"name": "Chicago", "instanceOf": ["C2"], "attributes": [],
                              "relations": [{"relation": "r", "direction": "forward", "object": "Q3", "qualifiers": {}}]},
                       "Q3": {"name": "Fosse", "instanceOf": [], "attributes": [], "relations": []}},
          "concepts": {}}
    merged = SpineBuilder.merge_by_name(kb)["entities"]
    assert set(merged) == {"N::Chicago", "N::Fosse"}            # homonyms are one prompt node
    assert merged["N::Chicago"]["instanceOf"] == ["C1", "C2"]
    assert merged["N::Chicago"]["relations"][0]["object"] == "N::Fosse"
    prog = {"function": ["Find", "Relate", "Find", "Relate", "Or", "Count"], "inputs": [[]] * 6}
    assert SpineBuilder.empty_allowed(prog, [0, 0, 1, 1, 1, None])  # empty branch of a non-empty Or
    assert not SpineBuilder.empty_allowed(prog, [0, 0, 0, 0, 0, None])
    prog = {"function": ["Find", "Relate", "Find", "Relate", "And", "Count"], "inputs": [[]] * 6}
    assert not SpineBuilder.empty_allowed(prog, [1, 1, 1, 1, 0, None])  # Count 0 through an empty And
