"""Relative graph attention bias (GTLM / Graphormer spatial encoding): a learned per-layer, per-head scalar
beta_l[h, bucket(c(i), c(j))] added to the attention score of every pair of graph-element tokens. No extra tokens,
embeddings untouched; see attention_bias.py for the buckets and the attention paths.

  distance   bucket = Levi-graph distance 0..8 between the two elements, > 8, unreachable    -> "rel-bias"
  element    bucket = same element / other element: node identity only, no structure           -> "rel-bias-elem",
             the identity control (the counterpart of abs-id)

Pairs where either token has no graph element (template, response) get no bias. No public GTLM code was found;
the buckets follow Graphormer's spatial encoding on the Levi graph so that edge tokens get distances too.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_inject.attention_bias import (DISTANCE_BUCKETS, ELEMENT_BUCKETS, element_buckets, install_relative_bias,
                                         packed_block_mask)
from graph_inject.methods.base import GraphMethod

EL_MULTIPLE, NDOC_MULTIPLE = 64, 8  # padded table shapes: few FlexAttention recompiles


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


class RelativeBiasMethod(GraphMethod):
    needs_elements = True

    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        self.num_buckets = {"distance": DISTANCE_BUCKETS, "element": ELEMENT_BUCKETS}[cfg.buckets]

    def attach(self, base: nn.Module) -> None:
        self.ctx = install_relative_bias(base, self.num_buckets)
        self.betas = self.ctx.betas  # registered here: trained at method_lr, saved with the method

    def featurize(self, sample):
        return {"buckets": element_buckets(sample.graph, self.cfg.buckets)}

    def collate(self, items, rng: np.random.Generator | None):
        el = _round_up(max(it.features["buckets"].shape[0] for it in items) + 1, EL_MULTIPLE)  # + the none element
        none = self.num_buckets - 1
        bucket = np.full((_round_up(len(items), NDOC_MULTIPLE), el, el), none, dtype=np.uint8)
        local, doc, prefix, offset = [], [], [], 0
        for d, it in enumerate(items):
            if it.offset != offset:
                raise ValueError("relative bias expects the samples packed in order from position 0")
            b = it.features["buckets"]
            bucket[d, :b.shape[0], :b.shape[0]] = b
            n, n_prompt = it.length or len(it.prompt.ids), len(it.prompt.ids)
            lq = np.full(n, el - 1, dtype=np.int64)
            lq[:n_prompt] = np.where(it.prompt.elements >= 0, it.prompt.elements, el - 1)
            local.append(lq)
            doc.append(np.full(n, d, dtype=np.int64))
            prefix.append(np.arange(n) < n_prompt)
            offset += n
        return {"graph_bias_bucket": torch.from_numpy(bucket), "graph_bias_local": torch.from_numpy(np.concatenate(local)),
                "graph_bias_doc": torch.from_numpy(np.concatenate(doc)),
                "graph_bias_prefix": torch.from_numpy(np.concatenate(prefix))}

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_bias_bucket" not in graph:  # decode steps: no bias
            self.ctx.clear()
            return embeddings
        bucket = graph["graph_bias_bucket"].long()
        pad = embeddings.numel() // embeddings.shape[-1] - graph["graph_bias_local"].shape[0]
        local = F.pad(graph["graph_bias_local"], (0, pad), value=bucket.shape[-1] - 1)
        doc = F.pad(graph["graph_bias_doc"], (0, pad), value=-1)
        # packed training batch: embeddings [T, D] -> FlexAttention mask; a prefill is [1, L, D] (KV-cache SDPA)
        block_mask = packed_block_mask(doc, F.pad(graph["graph_bias_prefix"], (0, pad))) if embeddings.dim() == 2 else None
        self.ctx.set(bucket, doc, local, block_mask)
        return embeddings
