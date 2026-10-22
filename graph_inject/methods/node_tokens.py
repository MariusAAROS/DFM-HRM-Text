"""LLaGA-style node tokens (Chen et al., ICML 2024; official code VITA-Group/LLaGA @ 288240f), no GNN.

Hop-field overview (HO) template, as in LLaGA: per node v the features x_v, A_hat x, A_hat^2 x
(A_hat = D^-1/2 A D^-1/2, utils/data_process.py generate_multi_hop_x) each become one token through the
"2-layer-mlp" projector (model/llaga_arch.py build_graph_projector: Linear, GELU, Linear). So a graph with n
nodes adds n * (hops + 1) soft tokens, node-major, in front of the (unchanged) text.

LLaGA's x_v is a text embedding of the node description; GraphQA nodes are bare integers, so x_v is the
LLM's own (frozen) embedding of the node name (mean over its digit tokens) concatenated with RWSE + LapPE.
"""
import numpy as np
import torch
from torch import Tensor, nn

from graph_inject.features import hop_features, normalized_adjacency, structural_features
from graph_inject.methods.base import GraphMethod, ToEmbeddingScale, soft_positions, stack_node_features, write_soft_tokens

MAX_NODES = 64


class NodeTokensMethod(GraphMethod):
    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        self.hops = cfg.hops
        in_dim = hidden_size + cfg.rwse_steps + cfg.lap_k
        self.projector = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.scale = ToEmbeddingScale(embed_rms)
        self.register_buffer("name_embedding", torch.zeros(MAX_NODES, hidden_size), persistent=False)

    @torch.no_grad()
    def setup(self, tokenizer, embed_tokens: nn.Module) -> None:
        for v in range(MAX_NODES):
            ids = torch.tensor(tokenizer(str(v), add_special_tokens=False)["input_ids"], device=self.name_embedding.device)
            emb = embed_tokens(ids).float().mean(0)
            self.name_embedding[v] = emb / emb.pow(2).mean().sqrt()  # unit RMS, like the structural block

    def num_soft_tokens(self, sample) -> int:
        return sample.graph.nnodes * (self.hops + 1)

    def featurize(self, sample):
        g = sample.graph
        if g.nnodes > MAX_NODES:
            raise ValueError(f"{g.nnodes} nodes > {MAX_NODES}")
        return {"pe": structural_features(g, self.cfg.rwse_steps, self.cfg.lap_k), "a_hat": normalized_adjacency(g),
                "node_ids": np.arange(g.nnodes)}

    def collate(self, items, rng: np.random.Generator | None):
        return {"graph_soft_pos": soft_positions(items),
                "graph_pe": stack_node_features(items, "pe", rng, sign_flip_cols=slice(self.cfg.rwse_steps, None)),
                "graph_a_hat": torch.block_diag(*[torch.from_numpy(it.features["a_hat"]) for it in items]),
                "graph_node_ids": torch.from_numpy(np.concatenate([it.features["node_ids"] for it in items])).long()}

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_soft_pos" not in graph:
            return embeddings
        x = torch.cat([self.name_embedding[graph["graph_node_ids"]], graph["graph_pe"]], dim=-1)
        hops = hop_features(x, graph["graph_a_hat"], self.hops)          # [N, hops + 1, F], node-major
        tokens = self.scale(self.projector(hops.reshape(-1, hops.shape[-1])))
        return write_soft_tokens(embeddings, graph["graph_soft_pos"], tokens)
