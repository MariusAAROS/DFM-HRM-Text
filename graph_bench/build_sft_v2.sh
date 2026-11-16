#!/usr/bin/env bash
# SFT data of the relaunch (wiki: mimir-graph-trace-sft-analysis), next to E1's train_answer.jsonl / E2's train_cot.jsonl:
#   sft/train_cot_c.jsonl     E2c: E2's 17,916 rows, CLRS traces with consecutive equal states merged
#   sft/train_answer_p.jsonl  E1p: E1's 17,917 rows + the WebQSP train questions without a trace whose answers are all
#                             in the prompt (answer line)
#   sft/train_cot_p.jsonl     E2p: E2c's rows + the same WebQSP questions, answer line only (mixed target)
# The exporter arguments are E1 / E2's (this input order, --cap 3700 --seed 0: reproduces train_answer.jsonl byte for
# byte, checked 2026-11-16) plus the two new flags. The one row E2 lost (a CLRS Kruskal trace over 4,096 tokens,
# train_cot.jsonl.orig17917) stays out of the cot files. Then the checks; any failure exits non-zero.
# Usage: bash graph_bench/build_sft_v2.sh
set -euo pipefail
cd /work/dfm/marius-ortega/DFM-HRM-Text
B=data/graphs/bench
PY=/home/ucloud/miniforge3/bin/python
TOK=/work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5-hfview
IN="$B/graphqa/train.jsonl $B/metaqa/train.jsonl $B/kqapro/train.jsonl $B/webqsp/train.jsonl $B/clrs/train.jsonl"
ARGS="--prompt instruct --traced-only --cap 3700 --seed 0 --extra $B/sft/replay.jsonl"
export HF_HUB_OFFLINE=1

.venv/bin/python -m graph_bench.export_hrm --target cot --collapse-repeats $ARGS --out $B/sft/train_cot_c.jsonl $IN
.venv/bin/python -m graph_bench.export_hrm --target answer --answerable-untraced $ARGS --out $B/sft/train_answer_p.jsonl $IN
.venv/bin/python -m graph_bench.export_hrm --target cot --collapse-repeats --answerable-untraced $ARGS \
  --out $B/sft/train_cot_p.jsonl $IN

$PY - <<PYEOF
import json, re
from collections import Counter
from transformers import AutoTokenizer

S = "$B/sft/"
load = lambda name: [json.loads(line) for line in open(S + name, encoding="utf-8")]
dump = lambda name, rows: open(S + name, "w", encoding="utf-8").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
e1, e2, e2_full = load("train_answer.jsonl"), load("train_cot.jsonl"), load("train_cot.jsonl.orig17917")
key = lambda r: r["instruction"]
assert len({key(r) for r in e2_full}) == len(e2_full), "instructions are not unique"
lost = {key(r) for r in e2_full} - {key(r) for r in e2}
assert len(lost) == 1 and {key(r) for r in e1} == {key(r) for r in e2_full}

# the row E2 lost stays out of the cot files
for name in ("train_cot_c.jsonl", "train_cot_p.jsonl"):
    dump(name, [r for r in load(name) if key(r) not in lost])
c, e1p, e2p = load("train_cot_c.jsonl"), load("train_answer_p.jsonl"), load("train_cot_p.jsonl")

# E2c: E2's rows in E2's order; only CLRS responses differ, and only by merged states
assert [key(r) for r in c] == [key(r) for r in e2], "train_cot_c: rows or order differ from train_cot"
states = lambda text: [line.split(": ", 1)[1] for line in text.split("\n") if re.match(r"State \d+: ", line)]
changed = Counter()
for new, old in zip(c, e2):
    if new["response"] == old["response"]:
        continue
    changed[new["dataset"]] += 1
    assert new["dataset"] == "clrs", f"a {new['dataset']} response changed"
    assert new["response"].rsplit("\n", 1)[1] == old["response"].rsplit("\n", 1)[1], "CLRS answer line changed"
    merged = [s for i, s in enumerate(states(old["response"])) if i == 0 or s != states(old["response"])[i - 1]]
    assert states(new["response"]) == merged, "CLRS states are not the merged old states"
for r in c:
    st = states(r["response"]) if r["dataset"] == "clrs" else []
    assert all(a != b for a, b in zip(st, st[1:])), "consecutive equal states left"
print("train_cot_c:", len(c), "rows; changed responses:", dict(changed),
      "; CLRS response chars", sum(len(r["response"]) for r in e2 if r["dataset"] == "clrs"), "->",
      sum(len(r["response"]) for r in c if r["dataset"] == "clrs"))

# plus files: the old questions + the same new WebQSP rows, answer line only, answers in the prompt, not val / test
old_q = {key(r) for r in e1}
new_a = [r for r in e1p if key(r) not in old_q]
new_c = [r for r in e2p if key(r) not in old_q]
assert {key(r) for r in e1p} - {key(r) for r in new_a} == old_q
assert {key(r) for r in e2p} - {key(r) for r in new_c} == old_q - lost
assert {key(r) for r in new_a} == {key(r) for r in new_c} and len(new_a) == len(new_c)
by_q = {key(r): r for r in c}
for r in e2p:  # every old row of E2p is E2c's row
    assert key(r) in {key(n) for n in new_c} or r["response"] == by_q[key(r)]["response"]
held = set()
for split in ("val", "test"):
    held |= {json.loads(line)["question"] for line in open("$B/webqsp/%s.jsonl" % split, encoding="utf-8")}
for a, cot in zip(sorted(new_a, key=key), sorted(new_c, key=key)):
    assert a["dataset"] == "webqsp" and a["response"] == cot["response"] and a["response"].startswith("Answer: ")
    assert "\n" not in a["response"]
    facts = a["instruction"].split("The facts are:\n", 1)[1]
    for ans in a["response"][len("Answer: "):].rstrip(".").split(" | "):
        assert ans in facts, f"answer {ans!r} not in the prompt"
    question = re.search(r"\nQ: (.*)\n", a["instruction"]).group(1)
    assert question not in held, f"held-out question in train: {question}"
tok = AutoTokenizer.from_pretrained("$TOK")
render = lambda r: tok.apply_chat_template([{"role": "user", "content": r["instruction"]},
                                            {"role": "assistant", "content": r["response"]}], tokenize=False)
longest = max(len(tok(render(r), add_special_tokens=False).input_ids) for r in new_a)
assert 100 < longest <= 4000, f"longest new row: {longest} tokens"  # the trainer refuses rows over 4,096
print("train_answer_p:", len(e1p), "rows; train_cot_p:", len(e2p), "rows; new WebQSP rows:", len(new_a),
      "; longest new row", longest, "tokens")
print("by dataset (train_cot_p):", dict(Counter(r["dataset"] for r in e2p)))
print("OK")
PYEOF
