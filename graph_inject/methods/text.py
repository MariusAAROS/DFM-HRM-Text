"""Hard prompting: the graph is read as text, only the serialization changes, + LoRA. The `adjacency` encoding
(GraphQA's Talk-like-a-Graph adjacency text, integer ids, direct answers) is the reference; it is the LoRA arm of
"Less is More" (arXiv 2508.10115), not that paper's recipe, which trains on instructive chain-of-thought solutions.

  adjacency  the GraphQA prompt as is: "(i,j) means ... The edges in G are: (0, 1) (1, 2) ..."
  incident   Talk like a Graph (Fatemi et al., ICLR 2024) incident encoder: "Node 0 is connected to nodes 1, 2."
             Copied from the google-research encoders as ported in Graph-Representation-Learning-for-LLM
             src/datasets/graphqa/graph_text_encoders.py (incident_encoder, create_node_string). Neighbours come
             in edge-list order (TLAG: the generator graph's insertion order); not checked against GraphQA's own
             incident files, which are not on disk.
  colors     Colorful Talks (Zangari et al., Findings of ACL 2026, arXiv 2602.10386), C-OWL variant (colours only;
             the paper's CL-OWL adds the WL labels): ordered 1-WL labels, min-max normalized to a hue in [0, 180]
             degrees and named, listed in a <<COLORS>> block after the adjacency text, with the paper's Figure 5
             instructions. Naming is the official code (angelozangari/CL-OWL @ 1f22ea7, algos.py
             _compute_colors_for_iteration + rgb_to_color_name) on the 0-255 RGB its thresholds are written for
             (the repo divides by 255 first, which would name everything "lime"); only red, crimson, orange, lime,
             green and teal are reachable, exactly the names of the paper's Figure 5 example.

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

COLOR_INSTRUCTIONS = (
    "`<<COLORS>>` shows colors assigned based on WL label similarity. Colors capture the structural role of each "
    "node based on its neighborhood. Colors are given the template: Color(i)=c, meaning that node i has color c. "
    "Colors can be used as a similarity metric among nodes (nodes with similar colors are more similar).\n")


def cl_owl_color(normalized: float) -> str:
    """CL-OWL algos.py: WL label normalized to [0, 1] -> hue in [0, 180] -> simplified HSV-to-RGB -> colour name."""
    hue = normalized * 180
    if hue <= 60:
        r, g, b = 255, int(hue * 255 / 60), 0
    elif hue <= 120:
        r, g, b = int((120 - hue) * 255 / 60), 255, 0
    else:
        r, g, b = 0, 255, int((hue - 120) * 255 / 60)
    if b > 50 and r < 200 and g < 200:
        raise ValueError(f"Unexpected color outside WL spectrum: RGB({r},{g},{b})")
    if r > 240 and g < 50 and b < 50:
        return "red"
    elif r > 200 and g < 120 and b < 50:
        return "crimson"
    elif r > 200 and g > 100 and b < 50:
        return "orange"
    elif r > 200 and g > 180 and b < 50:
        return "amber"
    elif r > 200 and g > 200 and b < 50:
        return "yellow"
    elif r > 100 and g > 200 and b < 50:
        return "lime"
    elif r < 100 and g > 200 and b < 50:
        return "green"
    elif r < 50 and g > 200 and b > 100:
        return "teal"
    elif r < 50 and g > 200 and b > 200:
        return "cyan"
    return "orange" if r > g else "lime"


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


def color_block(graph: PromptGraph, limit: int | None = None) -> str:
    """The colours of all nodes, or of the first `limit` nodes (labels still from the whole graph)."""
    labels = wl_labels(graph.nnodes, graph.edges)
    if not labels:  # KQA Pro prompts without facts
        return ""
    lo, hi = min(labels), max(labels)
    names = [cl_owl_color((lab - lo) / max(1, hi - lo)) for lab in labels][:limit]
    if graph.node_names is None:
        pairs = ", ".join(f"Color({v}):{c}" for v, c in enumerate(names))
        return f"{COLOR_INSTRUCTIONS}<<COLORS>>\n{pairs}\n"
    pairs = ", ".join(f"Color({e}):{c}" for e, c in zip(graph.node_names, names))
    kg = COLOR_INSTRUCTIONS.replace("Color(i)=c, meaning that node i", "Color(e)=c, meaning that entity e")
    return f"{kg.replace('among nodes', 'among entities')}<<COLORS>>\n{pairs}\n"


# Prompt-token budget: every model's RoPE table has 4096 positions (max_seq_len), minus 96 generated tokens and the
# chat-template tokens. Only text-colors on KQA Pro exceeds it (0.6% of prompts, up to 4,496 HRM tokens).
PROMPT_TOKEN_BUDGET = 3990


class TextMethod(GraphMethod):
    def setup(self, tokenizer, embed_tokens) -> None:
        self._tokenizer = tokenizer

    def _fit_colors(self, text: str, end: int, g: PromptGraph) -> str:
        """The prompt with the colour block, truncated to the first nodes that fit in PROMPT_TOKEN_BUDGET."""
        out = text[:end] + color_block(g) + text[end:]
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return out
        count = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
        n, excess = g.nnodes, count(out) - PROMPT_TOKEN_BUDGET
        while excess > 0 and n > 0:
            per_node = count(color_block(g)) / g.nnodes
            n = max(0, n - int(excess / per_node) - 1)
            out = text[:end] + color_block(g, n) + text[end:]
            excess = count(out) - PROMPT_TOKEN_BUDGET
        return out

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
            return self._fit_colors(text, end, g)
        raise ValueError(f"unknown text encoding {enc}")
