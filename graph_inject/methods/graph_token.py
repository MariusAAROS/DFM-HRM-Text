"""GraphToken (Perozzi et al., 2024, arXiv 2402.05862): graph-level soft tokens from a GNN.

P = W Pool(GNN(G)) gives k soft tokens written into the k placeholder slots in front of the (unchanged) text.
With k > 1 and LoRA this is also the "graph pooling tokens" setting (Grover et al., arXiv 2604.00342).
No official PyTorch code was found; the GNN is PyG's GIN, node features are RWSE + LapPE (GraphQA nodes have
no attributes), and readout is [mean || sum] pooling followed by a linear map to k x hidden.
"""
import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.models import GIN

from graph_inject.features import structural_features
from graph_inject.methods.base import (GraphMethod, ToEmbeddingScale, batched_edge_index, soft_positions,
                                       stack_node_features, write_soft_tokens)


class GraphTokenMethod(GraphMethod):
    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        self.k = cfg.num_tokens
        in_dim = cfg.rwse_steps + cfg.lap_k
        self.gnn = GIN(in_dim, cfg.gnn_hidden, num_layers=cfg.gnn_layers, norm="layer_norm")
        self.readout = nn.Sequential(nn.LayerNorm(2 * cfg.gnn_hidden), nn.Linear(2 * cfg.gnn_hidden, self.k * hidden_size))
        self.scale = ToEmbeddingScale(embed_rms)

    def num_soft_tokens(self, sample) -> int:
        return self.k

    def featurize(self, sample):
        return {"x": structural_features(sample.graph, self.cfg.rwse_steps, self.cfg.lap_k)}

    def collate(self, items, rng: np.random.Generator | None):
        edge_index, batch = batched_edge_index(items)
        return {"graph_soft_pos": soft_positions(items),
                "graph_x": stack_node_features(items, "x", rng, sign_flip_cols=slice(self.cfg.rwse_steps, None)),
                "graph_edge_index": edge_index, "graph_batch": batch}

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_soft_pos" not in graph:
            return embeddings
        h = self.gnn(graph["graph_x"], graph["graph_edge_index"])
        n_graphs = graph["graph_soft_pos"].numel() // self.k
        pooled = torch.cat([global_mean_pool(h, graph["graph_batch"], n_graphs),
                            global_add_pool(h, graph["graph_batch"], n_graphs)], dim=-1)
        tokens = self.scale(self.readout(pooled).view(n_graphs * self.k, self.hidden_size))
        return write_soft_tokens(embeddings, graph["graph_soft_pos"], tokens)
