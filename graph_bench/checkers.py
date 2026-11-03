"""Verifiers for tasks with several valid answers (NLGraph, GraphInstruct, GraphArena, CLRS-Text).

A checker takes the extracted answer text and the record's `checker` payload, and returns whether the answer is a
valid (and, where the task asks for it, optimal) solution. Node lists are read as the integers in the answer, in
order, so "0, 3, 2", "0 -> 3 -> 2" and "[0, 3, 2]" are the same answer. Payloads are plain JSON (they travel through
parquet as the VeRL ground truth).
"""
import re
from collections import defaultdict

_INT = re.compile(r"-?\d+")


def ints(text: str) -> list[int]:
    return [int(x) for x in _INT.findall(text)]


def _adjacency(edges, directed: bool) -> dict[int, set[int]]:
    adj = defaultdict(set)
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        if not directed:
            adj[v].add(u)
    return adj


def topological_sort(pred: str, p: dict) -> bool:
    """A permutation of the n nodes in which every directed edge u -> v has u before v."""
    order = ints(pred)
    if sorted(order) != list(range(p["n"])):
        return False
    pos = {v: i for i, v in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in p["edges"])


def hamilton_path(pred: str, p: dict) -> bool:
    """Visits every node exactly once along existing (undirected unless p['directed']) edges."""
    path = ints(pred)
    if sorted(path) != list(range(p["n"])):
        return False
    adj = _adjacency(p["edges"], p.get("directed", False))
    return all(b in adj[a] for a, b in zip(path, path[1:]))


def simple_path(pred: str, p: dict) -> bool:
    """A path from p['source'] to p['target'] along existing edges; with p['optimal'] its weight (or hop count)
    must equal p['optimal']. Edges are [u, v] or [u, v, w]."""
    path = ints(pred)
    if len(path) < 1 or path[0] != p["source"] or path[-1] != p["target"] or len(set(path)) != len(path):
        return False
    weight = {}
    for e in p["edges"]:
        w = e[2] if len(e) > 2 else 1
        weight[(e[0], e[1])] = w
        if not p.get("directed", False):
            weight[(e[1], e[0])] = w
    if any((a, b) not in weight for a, b in zip(path, path[1:])):
        return False
    if "optimal" in p:
        return sum(weight[(a, b)] for a, b in zip(path, path[1:])) == p["optimal"]
    return True


def node_set(pred: str, p: dict) -> bool:
    """A node set with property p['property'] (clique / independent_set / vertex_cover) of size p['optimal']."""
    nodes = set(ints(pred))
    if len(nodes) != p["optimal"] or not nodes <= set(range(p["n"])):
        return False
    edges = {frozenset(e[:2]) for e in p["edges"]}
    prop = p["property"]
    if prop == "clique":
        ns = sorted(nodes)
        return all(frozenset((a, b)) in edges for i, a in enumerate(ns) for b in ns[i + 1:])
    if prop == "independent_set":
        return not any(e <= nodes for e in edges)
    if prop == "vertex_cover":
        return all(e & nodes for e in edges)
    raise ValueError(f"unknown property {prop!r}")


def matching(pred: str, p: dict) -> bool:
    """Pairs (read as consecutive integers) that are existing edges, share no node, and number p['optimal']."""
    xs = ints(pred)
    if len(xs) % 2:
        return False
    pairs = list(zip(xs[::2], xs[1::2]))
    edges = {frozenset(e[:2]) for e in p["edges"]}
    used = [v for pair in pairs for v in pair]
    return (len(pairs) == p["optimal"] and len(set(used)) == len(used)
            and all(frozenset(pair) in edges for pair in pairs))


def tour(pred: str, p: dict) -> bool:
    """A Hamiltonian cycle (closing edge implied) over the weighted complete graph with total weight p['optimal']."""
    path = ints(pred)
    if path and len(path) == p["n"] + 1 and path[0] == path[-1]:
        path = path[:-1]
    if sorted(path) != list(range(p["n"])):
        return False
    weight = {}
    for u, v, w in p["edges"]:
        weight[(u, v)] = weight[(v, u)] = w
    cycle = list(zip(path, path[1:] + path[:1]))
    if any(e not in weight for e in cycle):
        return False
    return sum(weight[e] for e in cycle) == p["optimal"]


def number(pred: str, p: dict) -> bool:
    """The first number in the answer equals p['value'] (counts, distances, flow values, GED)."""
    xs = re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    return bool(xs) and abs(float(xs[0]) - float(p["value"])) < 1e-6


CHECKERS = {
    "topological_sort": topological_sort,
    "hamilton_path": hamilton_path,
    "simple_path": simple_path,
    "node_set": node_set,
    "matching": matching,
    "tour": tour,
    "number": number,
}
