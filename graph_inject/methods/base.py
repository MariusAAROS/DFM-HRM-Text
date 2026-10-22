"""GraphMethod: the interface every injection method implements, and shared batching helpers.

A method touches the pipeline at four points:

  rewrite_prompt(sample)       hard prompting: change the text the LLM reads (default: unchanged)
  num_soft_tokens(sample)      soft prompting: placeholder slots inserted before the prompt content
  featurize(sample)            per-sample graph features, computed once on CPU (numpy)
  collate(items, rng)          per-batch tensors, keys prefixed "graph_"; rng is None at eval (deterministic)
  inject(embeddings, graph)    on GPU, between the token embedding and the HRM: overwrite soft slots
                               and/or add per-token vectors

Positions in the graph batch index the *flattened* token sequence: [T] in a packed training batch,
[1 * L] at a single-prompt prefill, so inject works on embeddings.reshape(-1, D) in both cases.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_inject.structure import GraphQAGraph, NO_ELEMENT
from graph_inject.tokenize import EncodedPrompt


@dataclass
class GraphQASample:
    idx: int
    condition: str
    instruction: str
    response: str
    task: str
    graph: GraphQAGraph


@dataclass
class CollateItem:
    sample: GraphQASample
    features: dict[str, np.ndarray]
    prompt: EncodedPrompt
    offset: int  # position of the prompt's first token in the flattened batch


class GraphMethod(nn.Module):
    needs_elements = False  # True: the prompt must be the original text, aligned token -> graph element

    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.embed_rms = embed_rms

    def setup(self, tokenizer, embed_tokens: nn.Module) -> None:
        """Optional access to the tokenizer and (frozen) token embedding, once, before training."""

    def rewrite_prompt(self, sample: GraphQASample) -> str:
        return sample.instruction

    def num_soft_tokens(self, sample: GraphQASample) -> int:
        return 0

    def featurize(self, sample: GraphQASample) -> dict[str, np.ndarray]:
        return {}

    def collate(self, items: list[CollateItem], rng: np.random.Generator | None) -> dict[str, Tensor]:
        return {}

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        return embeddings


class ToEmbeddingScale(nn.Module):
    """RMS-normalize soft tokens and rescale them to the token-embedding RMS (embed_tokens multiplies by
    1/init_std, ~39 for hidden 1536, so raw projector outputs would be far off-scale). The gain is learned."""

    def __init__(self, embed_rms: float):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(embed_rms)))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],)) * self.gain


def soft_positions(items: list[CollateItem]) -> Tensor:
    return torch.from_numpy(np.concatenate([it.offset + it.prompt.soft_slots for it in items])).long()


def write_soft_tokens(embeddings: Tensor, positions: Tensor, tokens: Tensor) -> Tensor:
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    return flat.index_copy(0, positions, tokens.to(flat.dtype)).view_as(embeddings)


def add_to_tokens(embeddings: Tensor, positions: Tensor, vectors: Tensor) -> Tensor:
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    return flat.index_add(0, positions, vectors.to(flat.dtype)).view_as(embeddings)


def stack_node_features(items: list[CollateItem], key: str, rng: np.random.Generator | None,
                        sign_flip_cols: slice | None = None) -> Tensor:
    """Concatenate per-graph node features; at train time flip the sign of each eigenvector column at random."""
    feats = []
    for it in items:
        x = it.features[key].copy()
        if rng is not None and sign_flip_cols is not None:
            n_cols = x[:, sign_flip_cols].shape[1]
            x[:, sign_flip_cols] *= rng.choice([-1.0, 1.0], size=n_cols).astype(np.float32)
        feats.append(x)
    return torch.from_numpy(np.concatenate(feats))


def batched_edge_index(items: list[CollateItem]) -> tuple[Tensor, Tensor]:
    """PyG-style batched graphs: undirected edge_index [2, 2E] with node offsets, and the node -> graph vector."""
    edges, batch, offset = [], [], 0
    for g_idx, it in enumerate(items):
        e = np.asarray(it.sample.graph.edges, dtype=np.int64).reshape(-1, 2) + offset
        edges.append(np.concatenate([e, e[:, ::-1]]))
        batch.append(np.full(it.sample.graph.nnodes, g_idx, dtype=np.int64))
        offset += it.sample.graph.nnodes
    return torch.from_numpy(np.concatenate(edges).T.copy()), torch.from_numpy(np.concatenate(batch))


def element_tokens(items: list[CollateItem]) -> dict[str, Tensor]:
    """Flat positions of graph-element tokens and their global node / edge index (-1 if not that kind)."""
    pos, node, edge, edge_nodes = [], [], [], []
    node_off = edge_off = 0
    for it in items:
        g, el = it.sample.graph, it.prompt.elements
        at = np.nonzero(el != NO_ELEMENT)[0]
        pos.append(it.offset + at)
        is_node = el[at] < g.nnodes
        node.append(np.where(is_node, el[at] + node_off, -1))
        edge.append(np.where(is_node, -1, el[at] - g.nnodes + edge_off))
        edge_nodes.append(np.asarray(g.edges, dtype=np.int64).reshape(-1, 2) + node_off)
        node_off += g.nnodes
        edge_off += len(g.edges)
    cat = lambda xs: torch.from_numpy(np.concatenate(xs)).long()  # noqa: E731
    return {"graph_tok_pos": cat(pos), "graph_tok_node": cat(node), "graph_tok_edge": cat(edge),
            "graph_edge_nodes": cat(edge_nodes)}
