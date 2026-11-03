"""Entity-anonymised variant of a record: every label in `entities` becomes an opaque id ("E17") everywhere.

The gap between a model's score on the named and the anonymised version of a set measures how much it answers from
parametric knowledge rather than from the graph in the prompt (wiki: mimir-graph-benchmark, challenge 4).
Relations and literals keep their text, so the reasoning the question asks for is unchanged.

Labels are replaced longest first on word boundaries, in facts (subject and entity objects only), question, answers
and trace. The id permutation is seeded by the record id, so the variant is reproducible. A question that contains
no id after replacement names its entity by a surface form other than the label, so it no longer points into the
graph; the caller counts and drops those.
"""
import random
import re
from dataclasses import replace

from graph_bench.schema import Record


def _pattern(labels: list[str]) -> re.Pattern:
    alts = sorted({lab for lab in labels if lab}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(a) for a in alts) + r")(?!\w)")


def anonymize(rec: Record) -> tuple[Record, bool]:
    """(anonymised record, whether the question still mentions an entity id)."""
    labels = sorted(set(rec.entities or []))
    if not labels:
        return rec, True
    ids = list(range(len(labels)))
    random.Random(rec.id).shuffle(ids)
    mapping = {lab: f"E{i}" for lab, i in zip(labels, ids)}
    pat = _pattern(labels)

    def sub(text: str) -> str:
        return pat.sub(lambda m: mapping[m.group(1)], text)

    facts = None
    if rec.facts is not None:
        facts = [[mapping.get(f[0], f[0]), f[1], mapping.get(f[2], f[2]), *f[3:]] for f in rec.facts]
    question = sub(rec.question)
    out = replace(rec, id=rec.id + "#anon", facts=facts, question=question,
                  context=sub(rec.context) if rec.context is not None else None,
                  answers=[mapping.get(a, sub(a)) for a in rec.answers],
                  trace=[sub(t) for t in rec.trace] if rec.trace else None,
                  entities=sorted(mapping.values()), meta={**rec.meta, "anonymized": True})
    return out, bool(re.search(r"(?<!\w)E\d+(?!\w)", question))
