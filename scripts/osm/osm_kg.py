"""Helpers for notebooks/osm.ipynb and scripts/osm/build_kgqa.py: load the OSM multi-hop dataset, decode geometries, parse questions,
and evaluate the spatial relations so that questions can be answered from the entity tables (KGQA check).

Tables (data/osm/):
    df_osm            POIs (poi_id 0..17654, name, category, cuisine, seating, Lambert-93 x/y)
    df_street         street polylines (LineString / MultiLineString)
    df_area           districts, arrondissements, parks (Polygon)
    df_multihop       1.18M questions: nested text, answer poi_ids, per-level clause counts (`template`)
    df_dump_question  800-question eval subset, 200 per depth

Relation rules were fitted on the gold answers of 1-hop questions (see the notebook, section 6).
Geometry is in metres (Lambert-93, EPSG:2154); no shapely needed.
"""

from __future__ import annotations

import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "osm"

# --------------------------------------------------------------------------------------------- WKB


def wkb(buf: bytes):
    """Decode WKB into ('Point', xy) / ('LineString', (n,2)) / ('Polygon', [rings]) / ('Multi*', [geoms])."""
    b = bytes(buf)
    off = 0

    def read(fmt: str, endian: str):
        nonlocal off
        vals = struct.unpack_from(endian + fmt, b, off)
        off += struct.calcsize(endian + fmt)
        return vals

    def geom():
        nonlocal off
        endian = "<" if b[off] == 1 else ">"
        off += 1
        (gtype,) = read("I", endian)
        if gtype == 1:
            return ("Point", np.array(read("2d", endian)))
        if gtype == 2:
            (n,) = read("I", endian)
            return ("LineString", np.array(read(f"{2 * n}d", endian)).reshape(-1, 2))
        if gtype == 3:
            (k,) = read("I", endian)
            rings = []
            for _ in range(k):
                (n,) = read("I", endian)
                rings.append(np.array(read(f"{2 * n}d", endian)).reshape(-1, 2))
            return ("Polygon", rings)
        if gtype in (4, 5, 6):
            (k,) = read("I", endian)
            name = {4: "MultiPoint", 5: "MultiLineString", 6: "MultiPolygon"}[gtype]
            return (name, [geom() for _ in range(k)])
        raise ValueError(f"unsupported WKB type {gtype}")

    return geom()


def polylines(g) -> list[np.ndarray]:
    """All polylines of a geometry (polygon rings included)."""
    kind, c = g
    if kind == "LineString":
        return [c]
    if kind == "Polygon":
        return list(c)
    if kind.startswith("Multi"):
        return [line for sub in c for line in polylines(sub)]
    return []


def polygons(g) -> list[list[np.ndarray]]:
    kind, c = g
    if kind == "Polygon":
        return [c]
    if kind == "MultiPolygon":
        return [sub[1] for sub in c]
    return []


def dist_to_polyline(P: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Distance from points P (n,2) to polyline L (m,2)."""
    if len(L) == 1:
        return np.linalg.norm(P - L[0], axis=1)
    A, B = L[:-1][None], L[1:][None]
    p = P[:, None]
    AB = B - A
    t = np.clip(((p - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(A + t[..., None] * AB - p, axis=-1).min(1)


def dist_to_lines(P: np.ndarray, lines: list[np.ndarray]) -> np.ndarray:
    if len(P) == 0:
        return np.zeros(0)
    return np.min([dist_to_polyline(P, line) for line in lines], axis=0)


def points_inside(P: np.ndarray, g) -> np.ndarray:
    res = np.zeros(len(P), bool)
    for rings in polygons(g):
        r = MplPath(rings[0]).contains_points(P)
        for hole in rings[1:]:
            r &= ~MplPath(hole).contains_points(P)
        res |= r
    return res


# ------------------------------------------------------------------------------------------ loading


@dataclass
class OSM:
    poi: pd.DataFrame
    street: pd.DataFrame
    area: pd.DataFrame
    dump: pd.DataFrame
    multihop: pd.DataFrame | None = None
    xy: np.ndarray = field(init=False)
    pos: dict = field(init=False)  # poi_id -> row
    by_name: dict = field(init=False)  # poi_name -> rows
    street_lines: dict = field(init=False)  # street_name -> [polylines] (all segments with that name)
    area_geom: dict = field(init=False)  # area_name -> geometry
    tree: cKDTree = field(init=False)

    def __post_init__(self):
        self.xy = self.poi[["x", "y"]].to_numpy()
        self.pos = {p: i for i, p in enumerate(self.poi.poi_id.to_numpy())}
        self.by_name = {k: np.asarray(v) for k, v in self.poi.groupby("poi_name").indices.items()}
        self.street_lines = defaultdict(list)
        for name, g in zip(self.street.street_name, self.street.geometry):
            self.street_lines[name] += polylines(wkb(g))
        self.street_lines = dict(self.street_lines)
        self.area_geom = {n: wkb(g) for n, g in zip(self.area.area_name, self.area.geometry)}
        self.tree = cKDTree(self.xy)

    def rows(self, poi_ids) -> np.ndarray:
        return np.array([self.pos[p] for p in np.atleast_1d(poi_ids)], dtype=int)

    def names(self, poi_ids) -> list[str]:
        return self.poi.poi_name.to_numpy()[self.rows(poi_ids)].tolist()


def load(data_dir: Path = DATA_DIR, multihop: bool = True) -> OSM:
    data_dir = Path(data_dir)
    return OSM(
        poi=pd.read_parquet(data_dir / "df_osm.parquet"),
        street=pd.read_parquet(data_dir / "df_street.parquet"),
        area=pd.read_parquet(data_dir / "df_area.parquet").reset_index(drop=True),
        dump=pd.read_parquet(data_dir / "df_dump_question.parquet"),
        multihop=pd.read_parquet(data_dir / "df_multihop.parquet") if multihop else None,
    )


# ------------------------------------------------------------------------------------------ parsing

_REL = (
    r"(?:near|between|at the intersection of|at less than \d+ meters from|to the (?:north|south|east|west) of"
    r"|following|inside|at the border of|at the (?:north|south|east|west) of)"
)
_REL_START = re.compile(r" (?=" + _REL + r" )")
_REL_SPLIT = re.compile(r" and (?=" + _REL + r" )")
_REL_HEAD = re.compile("(" + _REL + r") +(.*)$")
SUB = "<SUB>"  # placeholder for a nested sub-question

# kind -> (surface phrase, anchor type, rule fitted on gold answers)
RELATIONS = {
    "within": ("at less than N meters from X", "POI / sub-question", "d(p, X) < N  (N = 100, 300, 500, 1000)"),
    "near": ("near X", "POI / sub-question", "d(p, X) < 1000 m"),
    "near_towards": ("near X towards Y", "POI, POI", "d(p, X) < 1000 m and angle(X->p, X->Y) <= 90 deg"),
    "dir_of_poi": ("to the {N,S,E,W} of X", "POI / sub-question", "d(p, X) < 1000 m and bearing X->p within 45 deg of the direction"),
    "between": ("between X and Y", "POI, POI", "d(p, segment XY) < 200 m"),
    "intersection": ("at the intersection of S1 and S2", "street, street", "d(p, crossing point of S1 and S2) < 1000 m"),
    "following": ("following S", "street", "d(p, S) < 50 m"),
    "inside": ("inside A", "area", "p in polygon A"),
    "border": ("at the border of A", "area", "d(p, boundary of A) < 200 m"),
    "dir_in_area": ("at the {N,S,E,W} of A", "area", "p in A and p on that side of the bbox centre of A"),
}
RELATION_ANCHOR = {  # which table each anchor slot links to
    "within": "poi", "near": "poi", "near_towards": "poi", "dir_of_poi": "poi", "between": "poi",
    "intersection": "street", "following": "street", "inside": "area", "border": "area", "dir_in_area": "area",
}
NEAR_M, BETWEEN_M, FOLLOW_M, BORDER_M, INTER_M, TOWARDS_DEG = 1000, 200, 50, 200, 1000, 90


@dataclass
class Clause:
    kind: str
    anchors: tuple  # names (str) or SUB
    param: object = None  # metres for `within`, direction for dir_*

    def __str__(self):
        a = " | ".join(self.anchors)
        return f"{self.kind}{'(' + str(self.param) + ')' if self.param is not None else ''}: {a}"


def to_clause(rel: str, arg: str) -> Clause:
    arg = arg.strip()
    if rel.startswith("at less than"):
        return Clause("within", (arg,), int(rel.split()[3]))
    if rel == "near":
        if " towards " in arg:
            return Clause("near_towards", tuple(arg.split(" towards ", 1)))
        return Clause("near", (arg,))
    if rel.startswith("to the"):
        return Clause("dir_of_poi", (arg,), rel.split()[2])
    if rel == "between":
        return Clause("between", tuple(arg.split(" and ", 1)))
    if rel == "at the intersection of":
        return Clause("intersection", tuple(a.strip() for a in arg.split(" and ", 1)))
    if rel == "following":
        return Clause("following", (arg,))
    if rel == "inside":
        return Clause("inside", (arg,))
    if rel == "at the border of":
        return Clause("border", (arg,))
    if rel.startswith("at the"):
        return Clause("dir_in_area", (arg,), rel.split()[2])
    raise ValueError(rel)


def split_clauses(text: str) -> tuple[str, list[Clause]]:
    """'restaurant italian food near X and following S' -> ('restaurant italian food', [near X, following S])."""
    text = re.sub(r"\s+", " ", text.strip().rstrip("."))
    m = _REL_START.search(text)
    if not m:
        return text, []
    clauses = []
    for part in _REL_SPLIT.split(text[m.start() + 1 :]):
        mm = _REL_HEAD.match(part)
        if mm:
            clauses.append(to_clause(mm.group(1), mm.group(2)))
    return text[: m.start()], clauses


SEATING = {"outdoor": "outdoor_seating", "indoor": "indoor_seating"}


def parse_head(head: str) -> dict:
    """'restaurant indian food without outdoor seating' -> {'category': 'restaurant', 'cooking_type': 'indian',
    'outdoor_seating': 'no'}."""
    h = head.lower().strip()
    out = {"category": h.split()[0]}
    m = re.match(r"\S+ (.+?) food\b", h)
    if m:
        out["cooking_type"] = m.group(1)
    for neg, kind in re.findall(r"(with|without) (outdoor|indoor) seating", h):
        out[SEATING[kind]] = "no" if neg == "without" else "yes"
    # outdoor_seating values other than yes/no: 'on sidewalk', 'with terrace', 'in pedestrian zone', ...
    m = re.search(r"\b(?:on|with|in) (sidewalk|terrace|veranda|parklet|pedestrian zone)\b", h)
    if m:
        out["outdoor_seating"] = m.group(1).replace(" ", "_")
    return out


@dataclass
class Level:
    head: str
    attrs: dict
    clauses: list[Clause]
    sub: "Level | None" = None

    @property
    def depth(self) -> int:
        return 1 + (self.sub.depth if self.sub else 0)

    def chain(self) -> list["Level"]:
        """Innermost first: the order in which the question is solved."""
        return (self.sub.chain() if self.sub else []) + [self]


def _match_bracket(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        depth += {"[": 1, "]": -1}.get(s[j], 0)
        if depth == 0:
            return j
    raise ValueError("unbalanced brackets")


def parse_question(query_texte: str) -> Level:
    """Parse the nested `query_texte` into a chain of levels (outermost returned)."""
    q = re.sub(r"\s+", " ", query_texte.strip())
    while q.startswith("[") and _match_bracket(q, 0) == len(q) - 1:
        q = q[1:-1].strip()
    i = q.find("[")
    if i < 0:
        head, clauses = split_clauses(q)
        return Level(head, parse_head(head), clauses)
    j = _match_bracket(q, i)
    head, clauses = split_clauses(q[:i].rstrip() + " " + SUB)
    after = q[j + 1 :].strip()
    if after.startswith("and "):
        clauses += split_clauses("x " + after[4:])[1]
    return Level(head, parse_head(head), clauses, sub=parse_question(q[i + 1 : j]))


# ---------------------------------------------------------------------------------------- semantics


def _sector(v: np.ndarray, direction: str) -> np.ndarray:
    ref = {"east": 0, "north": 90, "west": 180, "south": -90}[direction]
    ang = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return np.abs((ang - ref + 180) % 360 - 180) <= 45


def anchor_rows(kg: OSM, name: str, sub_rows: np.ndarray | None) -> np.ndarray:
    if name == SUB:
        return sub_rows
    return kg.by_name[name]  # KeyError -> unlinked anchor


def holds(kg: OSM, c: Clause, idx: np.ndarray, sub_rows: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask over candidate POI rows `idx`: does clause `c` hold? POI anchors with an ambiguous
    name hold if any POI of that name satisfies the rule."""
    P = kg.xy[idx]
    if len(idx) == 0:
        return np.zeros(0, bool)
    if c.kind in ("within", "near", "dir_of_poi"):
        A = kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]
        radius = c.param if c.kind == "within" else NEAR_M
        if c.kind != "dir_of_poi":
            d, _ = cKDTree(A).query(P, distance_upper_bound=radius)
            return d < radius
        ok = np.zeros(len(idx), bool)
        for k, near in enumerate(cKDTree(A).query_ball_point(P, radius)):
            if near:
                ok[k] = _sector(P[k] - A[near], c.param).any()
        return ok
    if c.kind == "near_towards":
        ok = np.zeros(len(idx), bool)
        for u in kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]:
            for w in kg.xy[anchor_rows(kg, c.anchors[1], sub_rows)]:
                v, t = P - u, w - u
                cos = (v @ t) / np.maximum(np.linalg.norm(v, axis=1) * np.linalg.norm(t), 1e-9)
                ok |= (np.linalg.norm(v, axis=1) < NEAR_M) & (cos >= np.cos(np.radians(TOWARDS_DEG)) - 1e-9)
        return ok
    if c.kind == "between":
        ok = np.zeros(len(idx), bool)
        for u in kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]:
            for w in kg.xy[anchor_rows(kg, c.anchors[1], sub_rows)]:
                ok |= dist_to_polyline(P, np.stack([u, w])) < BETWEEN_M
        return ok
    if c.kind == "intersection":
        return np.linalg.norm(P - crossing_point(kg, *c.anchors), axis=1) < INTER_M
    if c.kind == "following":
        return dist_to_lines(P, kg.street_lines[c.anchors[0]]) < FOLLOW_M
    g = kg.area_geom[c.anchors[0]]
    if c.kind == "inside":
        return points_inside(P, g)
    if c.kind == "border":
        return dist_to_lines(P, polylines(g)) < BORDER_M
    if c.kind == "dir_in_area":
        pts = np.concatenate(polylines(g))
        mid = (pts.min(0) + pts.max(0)) / 2
        side = {"east": P[:, 0] > mid[0], "west": P[:, 0] < mid[0], "north": P[:, 1] > mid[1], "south": P[:, 1] < mid[1]}
        return points_inside(P, g) & side[c.param]
    raise ValueError(c.kind)


def crossing_point(kg: OSM, s1: str, s2: str) -> np.ndarray:
    """Midpoint of the closest pair of vertices of two streets (they share a vertex in 99% of cases)."""
    L1, L2 = np.concatenate(kg.street_lines[s1]), np.concatenate(kg.street_lines[s2])
    d = cKDTree(L2).query(L1)[0]
    i = int(d.argmin())
    j = cKDTree(L2).query(L1[i])[1]
    return (L1[i] + L2[j]) / 2


def attribute_mask(kg: OSM, attrs: dict) -> np.ndarray:
    m = np.ones(len(kg.poi), bool)
    for col, val in attrs.items():
        m &= kg.poi[col].to_numpy() == val
    return m


def solve_level(kg: OSM, attrs: dict, clauses: list[Clause], sub_rows: np.ndarray | None = None) -> np.ndarray:
    """POI rows satisfying the attribute filter and every clause. Named POI anchors are not answers."""
    idx = np.flatnonzero(attribute_mask(kg, attrs))
    for c in clauses:
        if len(idx) == 0:
            break
        idx = idx[holds(kg, c, idx, sub_rows)]
    anchors = {a for c in clauses if RELATION_ANCHOR[c.kind] == "poi" for a in c.anchors if a != SUB}
    if anchors:
        idx = idx[~np.isin(kg.poi.poi_name.to_numpy()[idx], list(anchors))]
    return idx


def solve(kg: OSM, q: Level) -> list[np.ndarray]:
    """Answer a parsed question from the tables alone, innermost level first. Returns the answer rows of
    every level (innermost first); the last entry is the answer to the question."""
    out, sub = [], None
    for lvl in q.chain():
        sub = solve_level(kg, lvl.attrs, lvl.clauses, sub)
        out.append(sub)
    return out


def compare(pred_ids, gold_ids) -> dict:
    p, g = set(np.atleast_1d(pred_ids).tolist()), set(np.atleast_1d(gold_ids).tolist())
    if p == g:
        verdict = "exact"
    elif p > g:
        verdict = "superset"
    elif p < g:
        verdict = "subset"
    else:
        verdict = "other"
    jac = len(p & g) / len(p | g) if p | g else 1.0
    return {"verdict": verdict, "jaccard": jac, "recall": len(p & g) / len(g) if g else 1.0}


# ------------------------------------------------------------------------------ diagnostics & graph


def clause_metric(kg: OSM, c: Clause, rows: np.ndarray, sub_rows: np.ndarray | None = None) -> np.ndarray:
    """The continuous quantity each rule thresholds, for POI rows `rows` (used to fit / show the rules).
    Ambiguous anchor names take the most favourable POI of that name."""
    P = kg.xy[rows]
    if c.kind in ("within", "near"):
        return cKDTree(kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]).query(P)[0]
    if c.kind == "dir_of_poi":  # angular deviation from the stated direction, nearest anchor instance
        A = kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]
        v = P - A[cKDTree(A).query(P)[1]]
        ref = {"east": 0, "north": 90, "west": 180, "south": -90}[c.param]
        return np.abs((np.degrees(np.arctan2(v[:, 1], v[:, 0])) - ref + 180) % 360 - 180)
    if c.kind == "near_towards":  # angle between X->p and X->Y
        best = np.full(len(rows), 180.0)
        for u in kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]:
            for w in kg.xy[anchor_rows(kg, c.anchors[1], sub_rows)]:
                v, t = P - u, w - u
                cos = (v @ t) / np.maximum(np.linalg.norm(v, axis=1) * np.linalg.norm(t), 1e-9)
                best = np.minimum(best, np.degrees(np.arccos(np.clip(cos, -1, 1))))
        return best
    if c.kind == "between":
        best = np.full(len(rows), np.inf)
        for u in kg.xy[anchor_rows(kg, c.anchors[0], sub_rows)]:
            for w in kg.xy[anchor_rows(kg, c.anchors[1], sub_rows)]:
                best = np.minimum(best, dist_to_polyline(P, np.stack([u, w])))
        return best
    if c.kind == "intersection":
        return np.linalg.norm(P - crossing_point(kg, *c.anchors), axis=1)
    if c.kind == "following":
        return dist_to_lines(P, kg.street_lines[c.anchors[0]])
    if c.kind == "border":
        return dist_to_lines(P, polylines(kg.area_geom[c.anchors[0]]))
    if c.kind in ("inside", "dir_in_area"):
        return holds(kg, c, rows, sub_rows).astype(float)
    raise ValueError(c.kind)


def _densify(line: np.ndarray, step: float) -> np.ndarray:
    seg = np.diff(line, axis=0)
    n = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) / step).astype(int), 1)
    pts = [line[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))]
    return np.concatenate(pts + [line[-1:]])


def build_edges(kg: OSM) -> pd.DataFrame:
    """Materialise the binary spatial relations as KG edges and count them. Ternary relations (between,
    near..towards, intersection) and directions are computed from coordinates at query time instead."""
    rows = []
    counts = kg.tree.count_neighbors(kg.tree, [100, 300, 500, 1000]) - len(kg.xy)  # ordered pairs, no self
    for r, n in zip([100, 300, 500, 1000], counts):
        rows.append(("POI", f"within_{r}m", "POI", n // 2))
    # POI -- following --> street (d < 50 m): densify streets every 10 m, radius query
    names, pts = [], []
    for k, (name, lines) in enumerate(kg.street_lines.items()):
        for line in lines:
            d = _densify(line, 10.0)
            pts.append(d)
            names.append(np.full(len(d), k))
    pts, names = np.concatenate(pts), np.concatenate(names)
    pairs = set()
    for i, near in enumerate(cKDTree(pts).query_ball_point(kg.xy, FOLLOW_M)):
        pairs.update((i, int(s)) for s in set(names[near].tolist()))
    rows.append(("POI", "following", "street", len(pairs)))
    inside_n, border_n = 0, 0
    for g in kg.area_geom.values():
        pts_all = np.concatenate(polylines(g))
        lo, hi = pts_all.min(0) - BORDER_M, pts_all.max(0) + BORDER_M
        cand = np.flatnonzero(((kg.xy >= lo) & (kg.xy <= hi)).all(1))
        inside_n += points_inside(kg.xy[cand], g).sum()
        border_n += (dist_to_lines(kg.xy[cand], polylines(g)) < BORDER_M).sum()
    rows.append(("POI", "inside", "area", inside_n))
    rows.append(("POI", "border", "area", border_n))
    return pd.DataFrame(rows, columns=["head", "relation", "tail", "edges"])


def query_graph(q: Level):
    """networkx DiGraph of a parsed question: one variable node per level (?x1 innermost), one node per named
    anchor, edges labelled with the relation. The question asks for the outermost variable."""
    import networkx as nx

    G = nx.DiGraph()
    chain = q.chain()
    for i, lvl in enumerate(chain, 1):
        var = f"?x{i}"
        G.add_node(var, kind="variable", label=f"?x{i}\n{lvl.head}", level=i)
        for c in lvl.clauses:
            rel = c.kind + (f" {c.param}" if c.param is not None else "")
            roles = {"near_towards": ["near", "towards"], "between": ["between", "and"],
                     "intersection": ["crossing of", "and"]}.get(c.kind)
            for k, a in enumerate(c.anchors):
                rel_k = roles[k] if roles else rel
                if a == SUB:
                    G.add_edge(var, f"?x{i - 1}", label=rel_k)
                else:
                    G.add_node(a, kind=RELATION_ANCHOR[c.kind], label=a, level=i)
                    G.add_edge(var, a, label=rel_k)
    return G


def support_path(kg: OSM, levels: list[np.ndarray]) -> list[np.ndarray]:
    """Backtrack from the final answers: for every answer of level i keep the nearest POI of level i-1 (the
    one that makes its hop relation true). Returns the rows on the gold reasoning path, innermost first."""
    keep = [levels[-1]]
    for prev in reversed(levels[:-1]):
        _, j = cKDTree(kg.xy[prev]).query(kg.xy[keep[0]])
        keep.insert(0, np.unique(prev[np.atleast_1d(j)]))
    return keep
