"""One record format for every graph / KG reasoning set of the Mimir benchmark (wiki: mimir-graph-benchmark).

Converters in graph_bench/sources/ write `Record`s to data/graphs/bench/<dataset>/<split>.jsonl. Exporters render them:
export_hrm.py -> data_io JSONL (instruction / response / condition / task), export_verl.py -> VeRL parquet.

A record carries its graph either as `facts` (triples, rendered "(s, r, o)" one per line, optional qualifier text as
a 4th field) or as free `context` text (algorithmic sets whose graphs are not triples). `entities` lists the labels
anonymize.py may rename; literals (dates, numbers) are never in it.

answer_format decides how a prediction is compared (scoring.py):
  single   one answer; `answers` holds its accepted aliases
  set      unordered answers, rendered " | "-joined
  ordered  a sequence, rendered ", "-joined, compared in order
  checker  any answer that the named checker accepts (checkers.py); answers[0] is one reference solution
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ANSWER_FORMATS = ("single", "set", "ordered", "checker")
KG_PREAMBLE = ("In a knowledge graph, (s, p, o) means that entity s is linked to entity o by relation p, and (s, k, v) "
               "means that entity s has attribute k with value v.")
QUALIFIER_NOTE = " Facts may carry qualifiers, given in brackets."
SET_SEP = " | "
ORDERED_SEP = ", "
GRL_PREFIX = "#### Question\n"


@dataclass
class Record:
    id: str
    dataset: str
    split: str
    task: str
    question: str
    answers: list[str]
    answer_format: str = "single"
    facts: list[list[str]] | None = None
    context: str | None = None
    preamble: str = KG_PREAMBLE
    trace: list[str] | None = None
    checker: dict | None = None
    entities: list[str] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.answer_format not in ANSWER_FORMATS:
            raise ValueError(f"{self.id}: answer_format {self.answer_format!r} not in {ANSWER_FORMATS}")
        if not self.answers:
            raise ValueError(f"{self.id}: no gold answer")
        if self.answer_format == "checker" and not self.checker:
            raise ValueError(f"{self.id}: answer_format checker needs a checker payload")

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None}, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        return cls(**d)


def render_fact(fact: list[str]) -> str:
    s, r, o = fact[:3]
    qual = f" [{fact[3]}]" if len(fact) > 3 and fact[3] else ""
    return f"({s}, {r}, {o}){qual}"


def graph_text(rec: Record) -> str:
    """The graph block of the prompt, ending in a newline (empty for closed-book records)."""
    if rec.context is not None:
        return rec.context.rstrip("\n") + "\n"
    if rec.facts is None:
        return ""
    has_qual = any(len(f) > 3 and f[3] for f in rec.facts)
    head = rec.preamble + (QUALIFIER_NOTE if has_qual and rec.preamble == KG_PREAMBLE else "")
    # An empty fact list still prints its (empty) block, as GRL's generators do ("The facts are:\n\n").
    return head + "\nThe facts are:\n" + ("".join(render_fact(f) + "\n" for f in rec.facts) or "\n")


def prompt_text(rec: Record) -> str:
    """GraphQA-baseline layout ("<graph>Q: <question>\\nA: "), as in GRL's KQA Pro / MetaQA generators."""
    return f"{graph_text(rec)}Q: {rec.question}\nA: "


def answer_text(rec: Record) -> str:
    if rec.answer_format == "set":
        return SET_SEP.join(rec.answers)
    if rec.answer_format == "ordered":
        return ORDERED_SEP.join(rec.answers)
    return rec.answers[0]


def direct_response(rec: Record) -> str:
    return f"{answer_text(rec)}."


def cot_response(rec: Record) -> str:
    """Trace lines, then "Answer: <answer>." (the graphqa_v2_cot convention; scoring reads only the last answer)."""
    if not rec.trace:
        raise ValueError(f"{rec.id}: no trace")
    return "\n".join(rec.trace) + f"\nAnswer: {direct_response(rec)}"


def read_records(path: str | Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return [Record.from_dict(json.loads(line)) for line in f if line.strip()]


def write_records(records, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_json() + "\n")
            n += 1
    return n
