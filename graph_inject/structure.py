"""Graph structure recovered from GraphQA prompt text, and the graph element of every character.

Ported from Graph-Representation-Learning-for-LLM, branch graph-enc (commit a3d3418,
src/datasets/graphqa/structure.py): the regexes and wl_labels are unchanged; the single-graph parser
additionally records the character span of every edge tuple and of the graph description.

GraphQA prompts use the adjacency encoder with integer node names:

    #### Question
    In an undirected graph, (i,j) means that ... G describes a graph among nodes 0, 1, ..., and 8.
    The edges in G are: (0, 1) (1, 2) ... (7, 8).
    Q: Is there a path from node 0 to node 3?
    A:

Graph elements are numbered nodes first, then edges: node v -> v, edge e -> nnodes + e, none -> -1.
A node mention is a node-list entry, the first two fields of an edge tuple, or an integer on the Q: line.
The other characters of an edge tuple ("(", ", ", ")") belong to the edge.
"""
import re
from dataclasses import dataclass

import networkx as nx
import numpy as np

GRAPH_HEADER = re.compile(r"In an? (?:un)?directed graph")
NODE_LIST = re.compile(r"among nodes ([^\n]*?)\.\n")
EDGE_TUPLE = re.compile(r"\((\d+), (\d+)(?:, (\d+))?\)")
Q_LINE = re.compile(r"Q: [^\n]*")
INTEGER = re.compile(r"\d+")

NO_ELEMENT = -1


@dataclass
class GraphQAGraph:
    nnodes: int
    edges: list[tuple[int, int]]
    node_spans: list[tuple[int, int, int]]   # (char_start, char_end, node)
    edge_spans: list[tuple[int, int, int]]   # (char_start, char_end, edge index), the whole "(u, v)" tuple
    description_span: tuple[int, int]        # from "In an undirected graph" to the end of the edge line
    question_start: int                      # start of the "Q:" line

    @property
    def nelements(self) -> int:
        return self.nnodes + len(self.edges)

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.nnodes))
        graph.add_edges_from(self.edges)
        return graph

    def char_elements(self, text_len: int) -> np.ndarray:
        """Graph element of every character of the prompt (NO_ELEMENT outside the graph)."""
        elem = np.full(text_len, NO_ELEMENT, dtype=np.int64)
        for start, end, e in self.edge_spans:
            elem[start:end] = self.nnodes + e
        for start, end, v in self.node_spans:  # node digits inside a tuple override the edge
            elem[start:end] = v
        return elem


def parse_graphqa(text: str) -> GraphQAGraph:
    """Parse the single graph of a GraphQA prompt."""
    header = GRAPH_HEADER.search(text)
    if header is None:
        raise ValueError("no graph header in prompt")
    node_list = NODE_LIST.search(text, header.start())
    if node_list is None:
        raise ValueError("no node list in prompt")
    node_spans = [(node_list.start(1) + m.start(), node_list.start(1) + m.end(), int(m.group()))
                  for m in INTEGER.finditer(node_list.group(1))]
    nnodes = len(node_spans)
    if [v for _, _, v in node_spans] != list(range(nnodes)):
        raise ValueError("node list is not 0..n-1")

    q_line = Q_LINE.search(text, node_list.end())
    if q_line is None:
        raise ValueError("no Q: line in prompt")
    edges, edge_spans = [], []
    for m in EDGE_TUPLE.finditer(text, node_list.end(), q_line.start()):
        edge_spans.append((m.start(), m.end(), len(edges)))
        edges.append((int(m.group(1)), int(m.group(2))))
        node_spans.append((m.start(1), m.end(1), int(m.group(1))))
        node_spans.append((m.start(2), m.end(2), int(m.group(2))))
    node_spans.extend((q_line.start() + m.start(), q_line.start() + m.end(), int(m.group()))
                      for m in INTEGER.finditer(q_line.group()))
    bad = [v for _, _, v in node_spans if v >= nnodes]
    if bad:
        raise ValueError(f"node ids {bad} exceed the {nnodes}-node list")
    return GraphQAGraph(nnodes=nnodes, edges=edges, node_spans=node_spans, edge_spans=edge_spans,
                        description_span=(header.start(), q_line.start()), question_start=q_line.start())


def wl_labels(nnodes: int, edges) -> list[int]:
    """Ordered 1-WL colour refinement run to convergence (GRL graph-enc, unchanged).

    Every node starts with label 0. Each round, a node's signature is its current label followed by the
    sorted multiset of its neighbours' labels; new labels are the ranks of the signatures in sorted order.
    Labels are canonical (equal for isomorphic inputs). Refinement stops when the class count stops growing.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(nnodes))
    graph.add_edges_from(edges)
    labels = [0] * nnodes
    n_classes = 1
    for _ in range(nnodes):
        sigs = [(labels[v], tuple(sorted(labels[u] for u in graph.neighbors(v)))) for v in range(nnodes)]
        rank = {sig: i for i, sig in enumerate(sorted(set(sigs)))}
        labels = [rank[s] for s in sigs]
        if len(rank) == n_classes:
            break
        n_classes = len(rank)
    return labels
