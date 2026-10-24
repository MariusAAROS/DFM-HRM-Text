"""Absolute (additive) graph encoding: h_i = E(s_i) + phi(c(i)), no extra tokens, attention untouched.

phi depends only on the graph element c(i) of token i. Every token of the same element, at every mention
(node list, edge tuples, question), gets the same vector; template tokens get nothing.

  node v         phi(v) = W_N f(x_v),                   f = MLP on the node features x_v
  edge (u, w)    phi(e) = W_E g([f(x_u) || f(x_w)])      the "(", ",", ")" tokens of the tuple

W_N and W_E start at zero, so step 0 is exactly the text model. Node features:

  structural   RWSE + LapPE (LapPE signs flipped at random in training)   -> "abs-pe", the proposed method
  random       xi_v ~ N(0, I), same dimension, redrawn for every sample  -> "abs-id", the identity control:
               consistent within a prompt (co-reference), no structure. Deterministic per sample at eval.
               Random, not learned per index: not GraphToken's IDX features (graphtoken-faithful has those).

Close to GRL graph-enc's HRMGraphICL arms (node-identity / WL embeddings behind a scalar gate), which stayed
at a gate of ~0.002 under full fine-tuning. Differences: zero-init projections instead of a scalar gate,
edges get their own vector, LoRA, and the graph modules have their own (higher) learning rate.
"""
import zlib

import numpy as np
import torch
from torch import Tensor, nn

from graph_inject.features import structural_features
from graph_inject.methods.base import GraphMethod, add_to_tokens, element_tokens, stack_node_features


class AbsolutePEMethod(GraphMethod):
    needs_elements = True

    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        in_dim, h = cfg.rwse_steps + cfg.lap_k, cfg.encoder_hidden
        self.node_encoder = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, h))
        self.edge_encoder = nn.Sequential(nn.Linear(2 * h, h), nn.GELU())
        self.node_out = nn.Linear(h, hidden_size, bias=False)
        self.edge_out = nn.Linear(h, hidden_size, bias=False)
        nn.init.zeros_(self.node_out.weight)
        nn.init.zeros_(self.edge_out.weight)

    def featurize(self, sample):
        if self.cfg.node_features == "structural":
            return {"x": structural_features(sample.graph, self.cfg.rwse_steps, self.cfg.lap_k)}
        if self.cfg.node_features == "random":
            return {}  # drawn in collate
        raise ValueError(f"unknown node_features {self.cfg.node_features}")

    def _random_features(self, items, rng: np.random.Generator | None) -> Tensor:
        dim = self.cfg.rwse_steps + self.cfg.lap_k
        feats = []
        for it in items:
            gen = rng if rng is not None else np.random.default_rng(zlib.crc32(it.sample.instruction.encode()))
            feats.append(gen.standard_normal((it.sample.graph.nnodes, dim)).astype(np.float32))
        return torch.from_numpy(np.concatenate(feats))

    def collate(self, items, rng: np.random.Generator | None):
        if self.cfg.node_features == "structural":
            x = stack_node_features(items, "x", rng, sign_flip_cols=slice(self.cfg.rwse_steps, None))
        else:
            x = self._random_features(items, rng)
        return {"graph_x": x, **element_tokens(items)}

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_tok_pos" not in graph:
            return embeddings
        f = self.node_encoder(graph["graph_x"])                                   # [N, h]
        node_phi = self.node_out(f)                                                # [N, D]
        u, w = graph["graph_edge_nodes"].unbind(-1)
        edge_phi = self.edge_out(self.edge_encoder(torch.cat([f[u], f[w]], dim=-1)))  # [E, D]
        tok_node, tok_edge = graph["graph_tok_node"], graph["graph_tok_edge"]
        is_node = tok_node >= 0
        phi = torch.where(is_node[:, None], node_phi[tok_node.clamp_min(0)], edge_phi[tok_edge.clamp_min(0)])
        return add_to_tokens(embeddings, graph["graph_tok_pos"], phi)
