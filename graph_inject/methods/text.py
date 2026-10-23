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

Knowledge graphs (MetaQA, KQA Pro; graph.node_names is set):
  incident   no KG encoder exists in Talk like a Graph; ours groups the facts by subject, in order of first
             appearance, merging the objects of one relation (outgoing edges only, the graph is directed):
             "Santa Sangre: written by Claudio Argento, Roberto Leoni; has tags circus." Qualifiers stay in
             brackets after their object.
  colors     the same C-OWL block with entity names for node ids, 1-WL on the undirected entity graph
             (relation labels ignored, as in the paper).
"""
from graph_inject.methods.base import GraphMethod, GraphQASample
from graph_inject.structure import PromptGraph, wl_labels

# Hue-ordered colour names from red (0 deg) to cyan (180 deg).
HUE_COLORS = ("red", "scarlet", "vermilion", "orange", "amber", "gold", "yellow", "lime", "chartreuse",
              "green", "emerald", "jade", "mint", "teal", "turquoise", "cyan")


def create_node_string(nnodes: int) -> str:
    return "".join(f"{i}, " for i in range(nnodes - 1)) + f"and {nnodes - 1}"


def incident_encoder(graph: PromptGraph) -> str:
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


KG_INCIDENT_PREAMBLE = (
    "In a knowledge graph, the facts are grouped by entity: \"s: p o1, o2; q o3.\" means that entity s is linked "
    "to entities o1 and o2 by relation p, and to entity o3 by relation q.\n")
KG_INCIDENT_ATTRIBUTES = " A relation may also be an attribute of s, with value o. Facts may carry qualifiers, given in brackets."


def kg_incident_encoder(graph: PromptGraph, preamble: str) -> str:
    by_subject: dict[str, dict[str, list[str]]] = {}
    for f in graph.facts:
        obj = f.obj + (f" [{'; '.join(f'{k}: {v}' for k, v in f.qualifiers)}]" if f.qualifiers else "")
        objs = by_subject.setdefault(f.subject, {}).setdefault(f.relation, [])
        if obj not in objs:  # GRL's subgraphs repeat some facts
            objs.append(obj)
    pre = KG_INCIDENT_PREAMBLE
    if "attribute" in preamble:  # KQA Pro
        pre = pre[:-1] + KG_INCIDENT_ATTRIBUTES + "\n"
    lines = [f"{s}: " + "; ".join(f"{p} {', '.join(objs)}" for p, objs in rels.items()) + "."
             for s, rels in by_subject.items()]
    return pre + "The facts are:\n" + "".join(line + "\n" for line in lines)


def color_block(graph: PromptGraph) -> str:
    labels = wl_labels(graph.nnodes, graph.edges)
    if not labels:  # KQA Pro prompts without facts
        return ""
    lo, hi = min(labels), max(labels)
    names = [HUE_COLORS[round((lab - lo) / (hi - lo) * (len(HUE_COLORS) - 1)) if hi > lo else 0] for lab in labels]
    if graph.node_names is None:
        pairs = ", ".join(f"Color({v}):{c}" for v, c in enumerate(names))
        return ("Colors are given the template: Color(i):c, meaning that node i has color c.\n"
                f"<<COLORS>>\n{pairs}.\n")
    pairs = ", ".join(f"Color({e}):{c}" for e, c in zip(graph.node_names, names))
    return ("Colors are given the template: Color(e):c, meaning that entity e has color c.\n"
            f"<<COLORS>>\n{pairs}.\n")


class TextMethod(GraphMethod):
    def rewrite_prompt(self, sample: GraphQASample) -> str:
        enc = self.cfg.encoding
        text, g = sample.instruction, sample.graph
        start, end = g.description_span
        if enc == "adjacency":
            return text
        if enc == "incident" and g.node_names is not None:
            return text[:start] + kg_incident_encoder(g, text[start:end]) + text[end:]
        if enc == "incident":
            return text[:start] + incident_encoder(g) + text[end:]
        if enc == "colors":
            return text[:end] + color_block(g) + text[end:]
        raise ValueError(f"unknown text encoding {enc}")
