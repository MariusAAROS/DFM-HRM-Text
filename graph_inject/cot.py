"""Chain-of-thought targets for every GraphQA task (dataset graphqa_v2_cot, wiki: graphqa-cot).

Every step of a trace is a copy, a lookup, or one small addition; the last line is "Answer: <gold>", and
generate.final_answer scores only that part. Traces follow the prompt's edge order where the gold does (counts,
appearance-ordered lists, union-find) and increasing node order elsewhere (BFS, augmenting paths), so they are
deterministic. The gold semantics they reproduce (checked on every v2 row):

  EdgeCount          self-loop tuples count as edges
  NodeDegree         a self-loop counts twice
  ConnectedNodes     neighbours in order of first appearance in the edge list (the question says "alphabetical";
                     3 rows follow no rule: their gold is replaced by this order, see cot_response)
  DisconnectedNodes  every other node that is not a neighbour, in string order ("10, 11, 2")
  TriangleCounting / MaximumFlow   self-loops ignored

Two tasks have coarser fallbacks (`detail`), used only when the full trace does not fit the 4,096-token window:
  TriangleCounting   full: list the common larger neighbours of every edge | edge_counts: only their number |
                     node_counts: one count per node
  MaximumFlow        full: residual updates in both directions | forward_updates | no_updates
Hidden steps that remain: finding an augmenting path, and the intersections of the coarser triangle details.
"""
from collections import deque

from graph_inject.structure import EDGE_TUPLE, PromptGraph, parse_graphqa

DETAILS = {"TriangleCounting": ("full", "edge_counts", "node_counts"),
           "MaximumFlow": ("full", "forward_updates", "no_updates")}


def _join(xs) -> str:
    return ", ".join(map(str, xs))


def _norm(text: str) -> str:
    return text.strip().lower().rstrip(" .")


def _node_list(nodes) -> str:
    return f"{_join(nodes)}." if nodes else "No nodes."


def _appearance_neighbors(edges, v) -> list[int]:
    out = []
    for a, b in edges:
        for x, y in ((a, b), (b, a)):
            if x == v and y not in out:
                out.append(y)
    return out


def _sorted_adjacency(g: PromptGraph) -> dict[int, list[int]]:
    adj = {u: set() for u in range(g.nnodes)}
    for a, b in g.edges:
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return {u: sorted(ws) for u, ws in adj.items()}


def _incident(g: PromptGraph, v: int) -> tuple[list[tuple[int, int]], str]:
    inc = [(a, b) for a, b in g.edges if v in (a, b)]
    return inc, f"Edges with node {v}: " + (", ".join(f"({a}, {b})" for a, b in inc) or "none") + "."


def node_count(g: PromptGraph, q):
    lines = ["Count the nodes: " + ", ".join(f"{v} → {i + 1}" for i, v in enumerate(range(g.nnodes))) + "."]
    return lines, f"{g.nnodes}."


def edge_count(g: PromptGraph, q):
    lines = ["Count the edges: " + ", ".join(f"({a}, {b}) → {i + 1}" for i, (a, b) in enumerate(g.edges)) + "."]
    return lines, f"{len(g.edges)}."


def node_degree(g: PromptGraph, q):
    v, deg, parts = q[0], 0, []
    for a, b in g.edges:
        if a == b == v:
            parts.append(f"({a}, {b}) → self-loop counts twice, {deg} + 2 = {deg + 2}")
            deg += 2
        elif v in (a, b):
            deg += 1
            parts.append(f"({a}, {b}) → {deg}")
    lines = [f"Edges with node {v}: " + (", ".join(parts) if parts else "none, degree 0") + "."]
    return lines, f"{deg}."


def edge_existence(g: PromptGraph, q):
    u, v = (q + q)[:2]
    _, edges_line = _incident(g, u)
    nbrs = _appearance_neighbors(g.edges, u)
    yes = v in nbrs
    lines = [edges_line, f"Neighbors of node {u}: {_join(nbrs) or 'none'}.",
             f"Node {v} is {'' if yes else 'not '}a neighbor of node {u}."]
    return lines, "Yes." if yes else "No."


def connected_nodes(g: PromptGraph, q):
    v = q[0]
    _, edges_line = _incident(g, v)
    nbrs = _appearance_neighbors(g.edges, v)
    return [edges_line, f"Neighbors in order of appearance: {_join(nbrs) or 'none'}."], _node_list(nbrs)


def disconnected_nodes(g: PromptGraph, q):
    v = q[0]
    _, edges_line = _incident(g, v)
    nbrs = _appearance_neighbors(g.edges, v)
    order = sorted(range(g.nnodes), key=str)
    checks = [f"{x} itself" if x == v else f"{x} neighbor" if x in nbrs else f"{x} keep" for x in order]
    kept = [x for x in order if x != v and x not in nbrs]
    lines = [edges_line, f"Neighbors of node {v}: {_join(nbrs) or 'none'}.",
             f"Nodes in alphabetical order: {_join(order)}.", f"Check: {_join(checks)}.",
             f"Kept: {_join(kept) or 'none'}."]
    return lines, _node_list(kept)


def _bfs(g: PromptGraph, s: int, t: int) -> tuple[list[str], int | None]:
    """Layered BFS from s, neighbours in increasing order, stopping as soon as t is found."""
    if s == t:
        return [f"Node {s} is the start: distance 0."], 0
    adj, dist, layer = _sorted_adjacency(g), {s: 0}, [s]
    lines = [f"Distance 0: {s}."]
    while layer:
        nxt = []
        for u in layer:
            new = [w for w in adj[u] if w not in dist]
            for w in new:
                dist[w] = dist[u] + 1
            nxt += new
            lines.append(f"Node {u} (distance {dist[u]}): neighbors {_join(adj[u]) or 'none'} → new: {_join(new) or 'none'}.")
            if t in dist:
                lines.append(f"Node {t} is reached at distance {dist[t]}.")
                return lines, dist[t]
        layer = nxt
    lines.append(f"No new nodes: node {t} cannot be reached from node {s}.")
    return lines, None


def reachability(g: PromptGraph, q):
    lines, d = _bfs(g, q[0], (q + q)[1])
    return lines, "No." if d is None else "Yes."


def shortest_path(g: PromptGraph, q):
    s, t = (q + q)[:2]
    lines, d = _bfs(g, s, t)
    return lines, f"There is no path from node {s} to node {t}." if d is None else f"{d}."


def cycle_check(g: PromptGraph, q):
    comp = list(range(g.nnodes))
    lines = ["Each node starts in its own component, named by its number. Go through the edges:"]
    for a, b in g.edges:
        if a == b:
            lines.append(f"({a}, {b}): a self-loop, which is a cycle.")
            return lines, "Yes, there is a cycle."
        ca, cb = comp[a], comp[b]
        if ca == cb:
            lines.append(f"({a}, {b}): node {a} is in {ca}, node {b} is in {cb}: same component, this edge closes a cycle.")
            return lines, "Yes, there is a cycle."
        lo, hi = min(ca, cb), max(ca, cb)
        moved = [x for x in range(g.nnodes) if comp[x] == hi]
        for x in moved:
            comp[x] = lo
        lines.append(f"({a}, {b}): node {a} is in {ca}, node {b} is in {cb}: merge, {_join(moved)} now in {lo}.")
    lines.append("No edge closed a cycle.")
    return lines, "No, there is no cycle."


def triangle_counting(g: PromptGraph, q, detail: str = "full"):
    adj = _sorted_adjacency(g)
    up = {u: [w for w in adj[u] if w > u] for u in range(g.nnodes)}
    lines = ["Neighbors with a larger number:"] + [f"{u}: {_join(up[u]) or 'none'}" for u in range(g.nnodes)]
    total = 0
    if detail == "node_counts":
        lines.append("For each node u, the triangles u < v < w:")
        for u in range(g.nnodes):
            c = sum(1 for v in up[u] for w in up[u] if w > v and w in up[v])
            total += c
            lines.append(f"{u}: {c}, total {total}")
    else:
        lines.append("For each edge (u, v) with u < v, the common larger neighbors w > v (each triangle once):")
        for u in range(g.nnodes):
            for v in up[u]:
                common = [w for w in up[u] if w > v and w in up[v]]
                total += len(common)
                listed = f"{_join(common) or 'none'} → " if detail == "full" else ""
                lines.append(f"({u}, {v}): {listed}{len(common)}, total {total}")
    lines.append(f"Total: {total}.")
    return lines, f"{total}."


def maximum_flow(g: PromptGraph, q, detail: str = "full", caps: list[int] | None = None):
    """Edmonds-Karp on the undirected graph: each edge of capacity c has residual c in both directions, and pushing b
    along x→y moves b of residual from x→y to y→x."""
    s, t = q[:2]
    residual: dict[tuple[int, int], int] = {}
    for (a, b), c in zip(g.edges, caps):
        if a != b:
            residual[(a, b)] = residual.get((a, b), 0) + c
            residual[(b, a)] = residual.get((b, a), 0) + c
    adj, flow = _sorted_adjacency(g), 0
    lines = [f"Push flow along shortest augmenting paths from node {s} to node {t}; every edge starts with its "
             f"capacity as residual in both directions."]
    while True:
        parent, queue = {s: None}, deque([s])
        while queue and t not in parent:
            u = queue.popleft()
            for w in adj[u]:
                if w not in parent and residual[(u, w)] > 0:
                    parent[w] = u
                    queue.append(w)
        if t not in parent:
            break
        path = [t]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        steps = list(zip(path[::-1], path[::-1][1:]))
        b = min(residual[e] for e in steps)
        line = (f"Path {len(lines)}: {' → '.join(map(str, path[::-1]))}, residuals "
                f"{_join(residual[e] for e in steps)}, bottleneck {b}, flow {flow} + {b} = {flow + b}.")
        updates = []
        for x, y in steps:
            if detail != "no_updates":
                updates.append(f"({x}→{y}) {residual[(x, y)]} - {b} = {residual[(x, y)] - b}")
            if detail == "full":
                updates.append(f"({y}→{x}) {residual[(y, x)]} + {b} = {residual[(y, x)] + b}")
            residual[(x, y)] -= b
            residual[(y, x)] += b
        lines.append(line + (f" Update: {', '.join(updates)}." if updates else ""))
        flow += b
    lines.append(f"No augmenting path from node {s} to node {t} is left.")
    return lines, f"{flow}."


TASKS = {"NodeCount": node_count, "EdgeCount": edge_count, "NodeDegree": node_degree,
         "EdgeExistence": edge_existence, "ConnectedNodes": connected_nodes, "DisconnectedNodes": disconnected_nodes,
         "Reachability": reachability, "ShortestPath": shortest_path, "CycleCheck": cycle_check,
         "TriangleCounting": triangle_counting, "MaximumFlow": maximum_flow}


def cot_response(instruction: str, task: str, gold: str, detail: str = "full") -> tuple[str, bool]:
    """The CoT target of one GraphQA row, and whether its gold was replaced.

    The trace's answer must equal the gold (normalized). The only allowed difference: ConnectedNodes rows whose gold
    lists the same neighbours in an order that matches no rule; the trace's appearance order replaces it."""
    g = parse_graphqa(instruction)
    q = g.question_nodes()
    kwargs = {}
    if task in DETAILS:
        if detail not in DETAILS[task]:
            raise ValueError(f"{task}: unknown detail {detail}")
        kwargs["detail"] = detail
    elif detail != "full":
        raise ValueError(f"{task} has only the full trace")
    if task == "MaximumFlow":
        start, end = g.description_span[0], g.question_start
        kwargs["caps"] = [int(m.group(3)) for m in EDGE_TUPLE.finditer(instruction, start, end)]
        if len(kwargs["caps"]) != len(g.edges) or None in kwargs["caps"]:
            raise ValueError("MaximumFlow without one capacity per edge")
    lines, answer = TASKS[task](g, q, **kwargs)
    replaced = _norm(answer) != _norm(gold)
    if replaced and not (task == "ConnectedNodes" and sorted(_norm(answer).split(", ")) == sorted(_norm(gold).split(", "))):
        raise ValueError(f"{task}: trace answer {answer!r} != gold {gold!r}")
    return "\n".join(lines + [f"Answer: {answer}"]), replaced
