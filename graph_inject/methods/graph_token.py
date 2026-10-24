"""GraphToken (Perozzi et al., 2024, arXiv 2402.05862): graph-level soft tokens from a GNN.

P = W Pool(GNN(G)) gives k soft tokens written into the k placeholder slots in front of the prompt text.
No official code (TF-GNN, not released); the GNN is PyG's GIN, node features are RWSE + LapPE (GraphQA nodes have
no attributes).

Two set-ups:
  graphtoken / graphtoken-frozen   the graph text is kept: "Graph Tokens + Text", the arm of Less is More
                                   (arXiv 2508.10115), + LoRA or with a frozen LLM. Readout [mean || sum] pooling.
  graphtoken-faithful              GraphToken's own setting: frozen LLM, no graph text (graph_text: false; the
                                   question stays), learned identity features (IDX, idx_dim) next to the PEs, and
                                   readout "graph+query": [mean || sum || h_a || h_b], a and b the first two nodes
                                   named in the question (zeros if absent): the paper's node- and edge-level readouts
                                   (the node's / the two nodes' representations, + pooling) for every task at once.
                                   Not GraphToken: AdamW at method_lr instead of Lion at 0.05.
"""
import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.models import GIN

from graph_inject.features import structural_features
from graph_inject.methods.base import (GraphMethod, ToEmbeddingScale, batched_edge_index, node_offsets,
                                       soft_positions, stack_node_features, write_soft_tokens)


class GraphTokenMethod(GraphMethod):
    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        self.k = cfg.num_tokens
        self.query = cfg.get("readout", "graph") == "graph+query"
        idx_dim = cfg.get("idx_dim", 0)
        self.idx = nn.Embedding(cfg.idx_max, idx_dim) if idx_dim else None
        in_dim = cfg.rwse_steps + cfg.lap_k + idx_dim
        self.gnn = GIN(in_dim, cfg.gnn_hidden, num_layers=cfg.gnn_layers, norm="layer_norm")
        readout_dim = (4 if self.query else 2) * cfg.gnn_hidden
        self.readout = nn.Sequential(nn.LayerNorm(readout_dim), nn.Linear(readout_dim, self.k * hidden_size))
        self.scale = ToEmbeddingScale(embed_rms)

    def num_soft_tokens(self, sample) -> int:
        return self.k

    def featurize(self, sample):
        g = sample.graph
        if self.idx is not None and g.nnodes > self.cfg.idx_max:
            raise ValueError(f"{g.nnodes} nodes > idx_max {self.cfg.idx_max}")
        query = np.full(2, -1, dtype=np.int64)
        q = g.question_nodes()[:2]
        query[:len(q)] = q
        return {"x": structural_features(g, self.cfg.rwse_steps, self.cfg.lap_k), "query": query}

    def collate(self, items, rng: np.random.Generator | None):
        edge_index, batch = batched_edge_index(items)
        out = {"graph_soft_pos": soft_positions(items),
               "graph_x": stack_node_features(items, "x", rng, sign_flip_cols=slice(self.cfg.rwse_steps, None)),
               "graph_edge_index": edge_index, "graph_batch": batch}
        if self.idx is not None:
            out["graph_node_idx"] = torch.from_numpy(np.concatenate([np.arange(it.sample.graph.nnodes) for it in items]))
        if self.query:
            out["graph_query"] = torch.from_numpy(np.stack(
                [np.where(it.features["query"] >= 0, it.features["query"] + off, -1)
                 for it, off in zip(items, node_offsets(items))]))
        return out

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_soft_pos" not in graph:
            return embeddings
        x = graph["graph_x"]
        if self.idx is not None:
            x = torch.cat([x, self.idx(graph["graph_node_idx"])], dim=-1)
        h = self.gnn(x, graph["graph_edge_index"])
        n_graphs = graph["graph_soft_pos"].numel() // self.k
        parts = [global_mean_pool(h, graph["graph_batch"], n_graphs), global_add_pool(h, graph["graph_batch"], n_graphs)]
        if self.query:
            q = graph["graph_query"]                                          # [n_graphs, 2] global node, -1: none
            h0 = torch.cat([h, h.new_zeros(1, h.shape[1])])                   # -1 -> a zero row (also: no nodes)
            parts.append(h0[torch.where(q >= 0, q, h.shape[0])].flatten(1))
        pooled = torch.cat(parts, dim=-1)
        tokens = self.scale(self.readout(pooled).view(n_graphs * self.k, self.hidden_size))
        return write_soft_tokens(embeddings, graph["graph_soft_pos"], tokens)
