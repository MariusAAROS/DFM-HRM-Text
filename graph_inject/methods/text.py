"""Hard prompting: the graph is read as text, only the serialization changes. Combined with LoRA, the
`adjacency` encoding is the "Less is More" (arXiv 2508.10115) baseline.

  adjacency  the GraphQA prompt as is: "(i,j) means ... The edges in G are: (0, 1) (1, 2) ..."
  incident   Talk like a Graph (Fatemi et al., ICLR 2024) incident encoder: "Node 0 is connected to nodes 1, 2."
             Copied from the google-research encoders as ported in Graph-Representation-Learning-for-LLM
             src/datasets/graphqa/graph_text_encoders.py (incident_encoder, create_node_string).
  colors     Colorful Talks (Zangari et al., Findings of ACL 2026, arXiv 2602.10386), C-OWL variant: ordered 1-WL
             labels, min-max normalized to a hue in [0, 180] degrees and named, listed in a <<COLORS>> block after
             the adjacency text. No public code: the template follows the paper's Figure 3; the colour-name
             list is ours (hue-ordered names over 0-180 degrees).
"""
from graph_inject.methods.base import GraphMethod, GraphQASample
from graph_inject.structure import GraphQAGraph, wl_labels

# Hue-ordered colour names from red (0 deg) to cyan (180 deg).
HUE_COLORS = ("red", "scarlet", "vermilion", "orange", "amber", "gold", "yellow", "lime", "chartreuse",
              "green", "emerald", "jade", "mint", "teal", "turquoise", "cyan")


def create_node_string(nnodes: int) -> str:
    return "".join(f"{i}, " for i in range(nnodes - 1)) + f"and {nnodes - 1}"


def incident_encoder(graph: GraphQAGraph) -> str:
    output = "G describes a graph among nodes %s.\n" % create_node_string(graph.nnodes)
    if graph.edges:
        output += "In this graph:\n"
    nx_graph = graph.to_networkx()
    for source in nx_graph.nodes():
        targets = [str(t) for t in nx_graph.neighbors(source)]
        if len(targets) > 1:
            output += "Node %s is connected to nodes %s.\n" % (source, ", ".join(targets))
        elif len(targets) == 1:
            output += "Node %d is connected to node %s.\n" % (source, targets[0])
    return output


def color_block(graph: GraphQAGraph) -> str:
    labels = wl_labels(graph.nnodes, graph.edges)
    lo, hi = min(labels), max(labels)
    names = [HUE_COLORS[round((lab - lo) / (hi - lo) * (len(HUE_COLORS) - 1)) if hi > lo else 0] for lab in labels]
    pairs = ", ".join(f"Color({v}):{c}" for v, c in enumerate(names))
    return ("Colors are given the template: Color(i):c, meaning that node i has color c.\n"
            f"<<COLORS>>\n{pairs}.\n")


class TextMethod(GraphMethod):
    def rewrite_prompt(self, sample: GraphQASample) -> str:
        enc = self.cfg.encoding
        text, g = sample.instruction, sample.graph
        start, end = g.description_span
        if enc == "adjacency":
            return text
        if enc == "incident":
            return text[:start] + incident_encoder(g) + text[end:]
        if enc == "colors":
            return text[:end] + color_block(g) + text[end:]
        raise ValueError(f"unknown text encoding {enc}")
