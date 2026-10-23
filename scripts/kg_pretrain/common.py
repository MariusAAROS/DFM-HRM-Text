"""Shared pieces of the KG pretraining corpus pipeline (see wiki/pages/hrm-graph/kg-pretraining-corpus.md).

Layout under data/kg_pretrain/:
    raw/<source>/                 downloads + SOURCE.json (fetch.sh)
    kg/<source>/                  triple store: entities.parquet, relations.parquet, triples-NNNNN.parquet
    canonical/<source>/           {"id", "source", "text", "meta"} gz JSONL shards
    converted/kg-<source>/        {"condition", "instruction", "response"} gz JSONL shards (data_io input)
    stats/<source>.json

Triple store schema:
    entities:  entity_id str, label str, description str|null, types list<str>
    relations: relation_id str, label str, inverse_label str|null, datatype str|null
    triples:   s str, r str, o str|null (entity id), o_literal str|null (rendered value), split str
               exactly one of o / o_literal is set; split is train|valid|test (holdout.py).
Triple shards are partitioned by subject (stable hash of s), so all facts of a subject sit in one shard.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("KG_ROOT", REPO / "data" / "kg_pretrain"))
RAW, KG, CANONICAL, CONVERTED, STATS = (ROOT / d for d in ("raw", "kg", "canonical", "converted", "stats"))

MAX_CHARS = 6000  # ~1.5k BPE tokens per document chunk
ROWS_PER_SHARD = 100_000

ENTITY_SCHEMA = pa.schema([("entity_id", pa.string()), ("label", pa.string()),
                           ("description", pa.string()), ("types", pa.list_(pa.string()))])
RELATION_SCHEMA = pa.schema([("relation_id", pa.string()), ("label", pa.string()),
                             ("inverse_label", pa.string()), ("datatype", pa.string())])
TRIPLE_SCHEMA = pa.schema([("s", pa.string()), ("r", pa.string()), ("o", pa.string()),
                           ("o_literal", pa.string()), ("split", pa.string())])


def stable_hash(*parts: str, seed: str = "") -> int:
    """64-bit hash that is identical across processes and runs (Python's hash() is salted)."""
    h = hashlib.blake2b(digest_size=8, key=seed.encode()[:64])
    for p in parts:
        h.update(p.encode())
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little")


def subject_shard(s: str, n_shards: int) -> int:
    return stable_hash(s) % n_shards


# ---------------------------------------------------------------- JSONL shards


class ShardWriter:
    """Writes rows to <dir>/part-NNNNN.jsonl.gz, rotating every `rows_per_shard` rows."""

    def __init__(self, out_dir: Path, rows_per_shard: int = ROWS_PER_SHARD, prefix: str = "part"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard, self.prefix = rows_per_shard, prefix
        self.shard, self.in_shard, self.rows, self.fh = 0, 0, 0, None

    def _open(self):
        path = self.out_dir / f"{self.prefix}-{self.shard:05d}.jsonl.gz"
        self.fh = gzip.open(path, "wt", encoding="utf-8", compresslevel=6)

    def write(self, row: dict):
        if self.fh is None:
            self._open()
        elif self.in_shard >= self.rows_per_shard:
            self.fh.close()
            self.shard, self.in_shard = self.shard + 1, 0
            self._open()
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.in_shard += 1
        self.rows += 1

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def clear_dir(path: Path, pattern: str = "*.jsonl.gz"):
    """Remove previous outputs so a re-run never leaves stale shards behind."""
    path = Path(path)
    if path.exists():
        for p in path.glob(pattern):
            p.unlink()


class DocWriter:
    """Writes the canonical and the data_io (converted) view of every document together."""

    def __init__(self, source: str, converted_name: str | None = None, prefix: str = "part",
                 rows_per_shard: int = ROWS_PER_SHARD, fresh: bool = True):
        self.source = source
        can_dir, conv_dir = CANONICAL / source, CONVERTED / f"kg-{converted_name or source}"
        if fresh:
            clear_dir(can_dir, f"{prefix}-*.jsonl.gz")
            clear_dir(conv_dir, f"{prefix}-*.jsonl.gz")
        self.canonical = ShardWriter(can_dir, rows_per_shard, prefix)
        self.converted = ShardWriter(conv_dir, rows_per_shard, prefix)

    def write(self, doc_id: str, text: str, meta: dict | None = None, *, instruction: str = "",
              condition: str = "direct"):
        """Plain document: instruction="" and the text is the response. QA rows pass `instruction`."""
        assert text.strip(), doc_id
        can = {"id": f"{self.source}:{doc_id}", "source": self.source, "text": text, "meta": meta or {}}
        if instruction:
            can["instruction"] = instruction
            can["condition"] = condition
        self.canonical.write(can)
        self.converted.write({"condition": condition, "instruction": instruction, "response": text})

    def close(self):
        self.canonical.close()
        self.converted.close()
        return self.canonical.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ---------------------------------------------------------------- triple store


class TripleStoreWriter:
    """Buffers triples per subject shard and flushes them as parquet row groups."""

    def __init__(self, source: str, n_shards: int = 1, flush_rows: int = 1_000_000):
        self.dir = KG / source
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in self.dir.glob("triples-*.parquet"):
            p.unlink()
        self.n_shards, self.flush_rows = n_shards, flush_rows
        self.buf = [[] for _ in range(n_shards)]
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.count = 0

    def add(self, s: str, r: str, o: str | None = None, o_literal: str | None = None, split: str = "train"):
        assert (o is None) != (o_literal is None), (s, r, o, o_literal)
        k = subject_shard(s, self.n_shards) if self.n_shards > 1 else 0
        self.buf[k].append((s, r, o, o_literal, split))
        self.count += 1
        if len(self.buf[k]) >= self.flush_rows:
            self._flush(k)

    def _flush(self, k: int):
        rows = self.buf[k]
        if not rows:
            return
        s, r, o, lit, split = zip(*rows)
        table = pa.table({"s": s, "r": r, "o": o, "o_literal": lit, "split": split}, schema=TRIPLE_SCHEMA)
        if k not in self.writers:
            self.writers[k] = pq.ParquetWriter(self.dir / f"triples-{k:05d}.parquet", TRIPLE_SCHEMA,
                                               compression="zstd")
        self.writers[k].write_table(table)
        self.buf[k] = []

    def close(self):
        for k in range(self.n_shards):
            self._flush(k)
        for w in self.writers.values():
            w.close()
        return self.count


def write_entities(source: str, rows: dict[str, dict] | Iterable[dict]):
    """rows: entity_id -> {label, description, types} or an iterable of full dicts."""
    if isinstance(rows, dict):
        rows = ({"entity_id": k, **v} for k, v in rows.items())
    cols = {"entity_id": [], "label": [], "description": [], "types": []}
    for r in rows:
        cols["entity_id"].append(r["entity_id"])
        cols["label"].append(r["label"])
        cols["description"].append(r.get("description") or None)
        cols["types"].append(list(r.get("types") or []))
    (KG / source).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, schema=ENTITY_SCHEMA), KG / source / "entities.parquet", compression="zstd")
    return len(cols["entity_id"])


def write_relations(source: str, rows: dict[str, dict]):
    """rows: relation_id -> {label, inverse_label?, datatype?}"""
    cols = {"relation_id": [], "label": [], "inverse_label": [], "datatype": []}
    for rid, r in rows.items():
        cols["relation_id"].append(rid)
        cols["label"].append(r["label"])
        cols["inverse_label"].append(r.get("inverse_label"))
        cols["datatype"].append(r.get("datatype"))
    (KG / source).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, schema=RELATION_SCHEMA), KG / source / "relations.parquet", compression="zstd")
    return len(cols["relation_id"])


def triple_shards(source: str) -> list[Path]:
    return sorted((KG / source).glob("triples-*.parquet"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- rendering


def entity_doc_chunks(label: str, description: str | None, facts: list[tuple[str, list[str]]],
                      max_chars: int = MAX_CHARS) -> list[str]:
    """Render an entity as text, split on fact boundaries; the header repeats in every chunk.

        Copenhagen
        capital city of Denmark
        - instance of: city; capital
        - country: Denmark
    """
    header = label if not description else f"{label}\n{description}"
    if len(header) > max_chars // 2:
        header = header[: max_chars // 2].rsplit(" ", 1)[0] + " ..."
    budget = max_chars - len(header) - 1
    lines: list[str] = []
    for rel, values in facts:
        # A relation with many values is split over several lines that each fit the budget.
        cur: list[str] = []
        cur_len = len(rel) + 4
        for v in values:
            if len(v) > budget - len(rel) - 4:
                v = v[: budget - len(rel) - 8] + " ..."
            if cur and cur_len + len(v) + 2 > budget:
                lines.append(f"- {rel}: " + "; ".join(cur))
                cur, cur_len = [], len(rel) + 4
            cur.append(v)
            cur_len += len(v) + 2
        if cur:
            lines.append(f"- {rel}: " + "; ".join(cur))
    chunks, cur, cur_len = [], [], 0
    for line in lines:
        if cur and cur_len + len(line) + 1 > budget:
            chunks.append(header + "\n" + "\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append(header + "\n" + "\n".join(cur))
    return chunks


def pack_texts(items: Iterable[tuple[str, str]], max_chars: int = MAX_CHARS, sep: str = "\n\n"
               ) -> Iterator[tuple[list[str], str]]:
    """Greedily pack (key, text) items into documents of at most max_chars. Yields (keys, text)."""
    keys, parts, n = [], [], 0
    for key, text in items:
        if parts and n + len(sep) + len(text) > max_chars:
            yield keys, sep.join(parts)
            keys, parts, n = [], [], 0
        keys.append(key)
        parts.append(text)
        n += len(text) + (len(sep) if len(parts) > 1 else 0)
    if parts:
        yield keys, sep.join(parts)
