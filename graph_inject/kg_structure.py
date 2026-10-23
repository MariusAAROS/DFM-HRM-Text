"""Knowledge-graph structure recovered from MetaQA / KQA Pro prompt text (GRL's serialization).

The prompts come from Graph-Representation-Learning-for-LLM src/datasets/{metaqa,kqapro}/kb_index.py:

    #### Question
    In a knowledge graph, (s, p, o) means that entity s is linked to entity o by relation p ...
    The facts are:
    (El Topo, directed by, Alejandro Jodorowsky)
    (My Week with Marilyn, cast member, Judi Dench) [character role: Sybil Thorndike]
    Q: who directed films for the writer of Santa Sangre
    A:

Entity names and relations may contain ", " ("Yesterday, Today and Tomorrow", "languages spoken, written or
signed"), so a fact is split where the middle field is a known relation (kg_vocab: kb.txt / kb.json); among
several candidate splits the longest relation wins.

Graph (a PromptGraph, numbered as for GraphQA: node v -> v, edge e -> nnodes + e):
  nodes  distinct entity / value strings (subjects, objects, qualifier values), in order of first appearance
  edges  one per fact (s, o); one per qualifier "k: v" as (s, v), so hyper-relational facts hang off the subject
Characters: a fact line belongs to its fact edge, a "k: v" qualifier to its qualifier edge, and entity names
to their node (later spans override earlier ones). Node names found on the Q: line (exact, longest first, at
word boundaries) are node mentions too.
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from graph_inject.structure import PromptGraph

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "graphs"
PREAMBLE_START = "In a knowledge graph"
FACT_LINE = re.compile(r"^\((.*)\)(?: \[(.*)\])?$")
Q_LINE = re.compile(r"\nQ: [^\n]*")


@dataclass
class KGFact:
    subject: str
    relation: str
    obj: str
    qualifiers: list[tuple[str, str]] = field(default_factory=list)

    def serialize(self) -> str:
        quals = f" [{'; '.join(f'{k}: {v}' for k, v in self.qualifiers)}]" if self.qualifiers else ""
        return f"({self.subject}, {self.relation}, {self.obj}){quals}"


@lru_cache(maxsize=None)
def kg_vocab(dataset: str) -> tuple[frozenset[str], frozenset[str]]:
    """(relations and attribute keys, qualifier keys) as they appear in the prompts."""
    if dataset == "metaqa":
        with open(DATA_DIR / "metaqa" / "kb.txt", encoding="utf-8") as f:
            rels = {line.split("|")[1].replace("_", " ") for line in f if line.count("|") == 2}
        return frozenset(rels), frozenset()
    if dataset == "kqapro":
        with open(DATA_DIR / "kqapro" / "kb.json", encoding="utf-8") as f:
            kb = json.load(f)
        rels, quals = {"instance of"}, set()
        for e in kb["entities"].values():
            for a in e["attributes"]:
                rels.add(a["key"])
                quals.update(a["qualifiers"])
            for r in e["relations"]:
                rels.add(r["predicate"])
                quals.update(r["qualifiers"])
        return frozenset(rels), frozenset(quals)
    raise ValueError(f"no KG vocabulary for {dataset}")


def split_fact(body: str, relations: frozenset[str]) -> tuple[str, str, str]:
    parts = body.split(", ")
    if len(parts) == 3 and parts[1] in relations:
        return parts[0], parts[1], parts[2]
    candidates = [(i, j) for i in range(1, len(parts) - 1) for j in range(i + 1, len(parts))
                  if ", ".join(parts[i:j]) in relations]
    if not candidates:
        raise ValueError(f"no known relation in fact ({body})")
    i, j = max(candidates, key=lambda c: (len(", ".join(parts[c[0]:c[1]])), -c[0]))
    return ", ".join(parts[:i]), ", ".join(parts[i:j]), ", ".join(parts[j:])


def split_qualifiers(text: str, keys: frozenset[str]) -> list[tuple[str, str]]:
    """"k: v; k2: v2" -> [(k, v), (k2, v2)]; a "; " not followed by a known key is part of the value."""
    out: list[list[str]] = []
    for piece in text.split("; "):
        key = piece.split(": ", 1)[0]
        if ": " in piece and (key in keys or not out):
            out.append([key, piece[len(key) + 2:]])
        else:
            out[-1][1] += "; " + piece
    return [(k, v) for k, v in out]


def parse_kg_facts(text: str, dataset: str) -> PromptGraph:
    relations, qual_keys = kg_vocab(dataset)
    start = text.find(PREAMBLE_START)
    q_line = Q_LINE.search(text)
    if start < 0 or q_line is None:
        raise ValueError("no knowledge-graph preamble or Q: line in prompt")
    q_start = q_line.start() + 1

    names: dict[str, int] = {}
    node = lambda name: names.setdefault(name, len(names))  # noqa: E731
    edges, edge_spans, node_spans, facts = [], [], [], []
    pos = text.index("\n", text.index("The facts are:", start)) + 1
    while pos < q_start:
        end = text.index("\n", pos)
        line = text[pos:end]
        if not line:  # KQA Pro: 4% of prompts have no facts at all
            pos = end + 1
            continue
        m = FACT_LINE.match(line)
        if m is None:
            raise ValueError(f"not a fact line: {line!r}")
        s, p, o = split_fact(m.group(1), relations)
        quals = split_qualifiers(m.group(2), qual_keys) if m.group(2) is not None else []
        fact = KGFact(s, p, o, quals)
        if fact.serialize() != line:
            raise ValueError(f"fact does not re-serialize: {line!r}")
        facts.append(fact)
        su, ob = node(s), node(o)
        edge_spans.append((pos, end, len(edges)))
        edges.append((su, ob))
        node_spans.append((pos + 1, pos + 1 + len(s), su))
        o_start = pos + 1 + len(s) + 2 + len(p) + 2
        node_spans.append((o_start, o_start + len(o), ob))
        q = pos + len(f"({s}, {p}, {o}) [")  # first qualifier char
        for k, v in quals:
            qv = node(v)
            edge_spans.append((q, q + len(k) + 2 + len(v), len(edges)))
            edges.append((su, qv))
            node_spans.append((q + len(k) + 2, q + len(k) + 2 + len(v), qv))
            q += len(k) + 2 + len(v) + 2
        pos = end + 1

    node_spans += question_mentions(text, q_start, q_line.end(), names)
    return PromptGraph(nnodes=len(names), edges=edges, node_spans=node_spans, edge_spans=edge_spans,
                       description_span=(start, q_start), question_start=q_start,
                       node_names=list(names), facts=facts)


def question_mentions(text: str, start: int, end: int, names: dict[str, int]) -> list[tuple[int, int, int]]:
    """Exact, non-overlapping node-name matches on the Q: line, longest names first, at word boundaries."""
    taken = [False] * (end - start)
    spans = []
    for name in sorted((n for n in names if len(n) >= 2), key=len, reverse=True):
        for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text[start:end]):
            if not any(taken[m.start():m.end()]):
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
                spans.append((start + m.start(), start + m.end(), names[name]))
    return spans
