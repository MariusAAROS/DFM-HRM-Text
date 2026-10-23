"""LLaGA-style node tokens (Chen et al., ICML 2024; official code VITA-Group/LLaGA @ 288240f), no GNN.

Hop-field overview (HO) template, as in LLaGA: per node v the features x_v, A_hat x, A_hat^2 x
(A_hat = D^-1/2 A D^-1/2, utils/data_process.py generate_multi_hop_x) each become one token through the
"2-layer-mlp" projector (model/llaga_arch.py build_graph_projector: Linear, GELU, Linear). So a graph with n
nodes adds n * (hops + 1) soft tokens, node-major, in front of the (unchanged) text.

LLaGA's x_v is a text embedding of the node description; here x_v is the LLM's own (frozen) embedding of the
node name (mean over its tokens: the digits of v on GraphQA, the entity name on MetaQA / KQA Pro), at unit RMS,
concatenated with RWSE + LapPE.
"""
import numpy as np
import torch
from torch import Tensor, nn

from graph_inject.features import hop_features, normalized_adjacency, structural_features
from graph_inject.methods.base import (GraphMethod, ToEmbeddingScale, node_offsets, soft_positions, stack_node_features,
                                       write_soft_tokens)


class NodeTokensMethod(GraphMethod):
    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        self.hops = cfg.hops
        in_dim = hidden_size + cfg.rwse_steps + cfg.lap_k
        self.projector = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.scale = ToEmbeddingScale(embed_rms)
        self._name_ids: dict[str, np.ndarray] = {}

    def setup(self, tokenizer, embed_tokens: nn.Module) -> None:
        self._tokenizer = tokenizer
        self._embed = [embed_tokens]  # a list: not a submodule, its weights stay out of the method's parameters

    def _ids(self, name: str) -> np.ndarray:
        if name not in self._name_ids:
            self._name_ids[name] = np.asarray(self._tokenizer(name, add_special_tokens=False)["input_ids"], dtype=np.int64)
        return self._name_ids[name]

    def num_soft_tokens(self, sample) -> int:
        return sample.graph.nnodes * (self.hops + 1)

    def featurize(self, sample):
        g = sample.graph
        names = g.node_names if g.node_names is not None else [str(v) for v in range(g.nnodes)]
        ids = [self._ids(n) for n in names]
        return {"pe": structural_features(g, self.cfg.rwse_steps, self.cfg.lap_k), "a_hat": normalized_adjacency(g),
                "name_ids": np.concatenate(ids or [np.zeros(0, np.int64)]),
                "name_node": np.repeat(np.arange(g.nnodes), [len(i) for i in ids])}

    def collate(self, items, rng: np.random.Generator | None):
        return {"graph_soft_pos": soft_positions(items),
                "graph_pe": stack_node_features(items, "pe", rng, sign_flip_cols=slice(self.cfg.rwse_steps, None)),
                "graph_a_hat": torch.block_diag(*[torch.from_numpy(it.features["a_hat"]) for it in items]),
                "graph_name_ids": torch.from_numpy(np.concatenate([it.features["name_ids"] for it in items])),
                "graph_name_node": torch.from_numpy(np.concatenate(
                    [it.features["name_node"] + off for it, off in zip(items, node_offsets(items))]))}

    @torch.no_grad()
    def name_embedding(self, ids: Tensor, node: Tensor, nnodes: int) -> Tensor:
        """[nnodes, D]: mean frozen token embedding of each node name, at unit RMS (the scale of the PE block)."""
        emb = self._embed[0](ids).float()
        mean = emb.new_zeros(nnodes, emb.shape[-1]).index_add_(0, node, emb)
        mean /= torch.bincount(node, minlength=nnodes).clamp_min(1)[:, None]
        return mean / mean.pow(2).mean(-1, keepdim=True).sqrt()

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_soft_pos" not in graph:
            return embeddings
        x = torch.cat([self.name_embedding(graph["graph_name_ids"], graph["graph_name_node"], graph["graph_pe"].shape[0]),
                       graph["graph_pe"]], dim=-1)
        hops = hop_features(x, graph["graph_a_hat"], self.hops)          # [N, hops + 1, F], node-major
        tokens = self.scale(self.projector(hops.reshape(-1, hops.shape[-1])))
        return write_soft_tokens(embeddings, graph["graph_soft_pos"], tokens)
