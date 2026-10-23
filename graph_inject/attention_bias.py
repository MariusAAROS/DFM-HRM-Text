"""Relative graph attention bias (phase 2): score(i, j) += beta_l[h, bucket(c(i), c(j))] in every attention layer.

c(i) is token i's graph element (the token -> element map of abs-pe). The bucket of a pair of elements is their
distance in the Levi graph (nodes and edges are both vertices, an edge is adjacent to its endpoints), as in
GTLM / Graphormer's spatial encoding; pairs where either side has no element get the "none" bucket, whose bias is
fixed at 0, so template and response tokens attend as in the text model. beta is zero-initialized: step 0 is
exactly the text model.

models/ is not edited. install_relative_bias() switches every models.layers.Attention to GraphBiasAttention by
`__class__` (weights kept). While the shared GraphBiasContext is inactive (every other method, decode steps) the
layer is the parent class. While it is active:

  training (no cache)   FlexAttention with the document + PrefixLM mask of the packed batch, instead of the FA2
                        varlen kernel, which takes no bias. The bias is gathered per layer into
                        [H, NDOC, EL, EL] (doc, local element of q, local element of k) once per forward and grad
                        mode: indexing a [H, nbuckets] table inside score_mod directly makes the backward ~17x
                        slower (atomic adds on a few hundred entries), the expanded table only ~2.5x.
  prefill (KV cache)    the same bias added to the SDPA float mask.

Probe on the RTX PRO 6000 (T=8192, 12 x 128, bf16): flex with beta = 0 matches varlen to 0.004; fwd+bwd per call
2.9 ms (varlen) vs 7.3 ms (flex + expanded bias).
"""
import numpy as np
import torch
import torch._dynamo
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from models.layers import Attention, apply_rotary_pos_emb
from graph_inject.structure import PromptGraph

MAX_DISTANCE = 8
# distance buckets: 0..MAX_DISTANCE (0 = same element), > MAX_DISTANCE, unreachable, none
DISTANCE_BUCKETS = MAX_DISTANCE + 4
# element buckets (identity control): same element, other element, none
ELEMENT_BUCKETS = 3

torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)  # one kernel per padded shape
_flex_attention = torch.compile(flex_attention, dynamic=False)


def element_buckets(graph: PromptGraph, mode: str) -> np.ndarray:
    """[E, E] uint8 bucket of every pair of graph elements (E = nnodes + nedges)."""
    n, e = graph.nnodes, len(graph.edges)
    if mode == "element":
        out = np.ones((n + e, n + e), dtype=np.uint8)
        np.fill_diagonal(out, 0)
        return out
    if mode != "distance":
        raise ValueError(f"unknown bucket mode {mode}")
    if n + e == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    ends = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    rows = np.concatenate([n + np.arange(e), n + np.arange(e)])
    levi = csr_matrix((np.ones(2 * e), (rows, ends.T.reshape(-1))), shape=(n + e, n + e))
    dist = shortest_path(levi, directed=False, unweighted=True)
    out = np.where(np.isinf(dist), MAX_DISTANCE + 2, np.minimum(dist, MAX_DISTANCE + 1))
    return out.astype(np.uint8)


class GraphBiasContext:
    """Per-forward graph tensors shared by every GraphBiasAttention layer (set by RelativeBiasMethod.inject)."""

    def __init__(self, betas: nn.ParameterList, num_buckets: int):
        self.betas = betas              # one [H, num_buckets - 1] table per layer; the "none" bucket is 0
        self.num_buckets = num_buckets
        self.active = False

    def set(self, bucket: Tensor, doc: Tensor, local: Tensor, block_mask: BlockMask | None):
        """bucket [NDOC, EL, EL]; per token: doc (-1 = padding) and local element (EL - 1 = none)."""
        self.bucket, self.doc, self.local, self.block_mask = bucket, doc.clamp_min(0), local, block_mask
        self._bias: dict[tuple[int, bool], Tensor] = {}
        self.active = True

    def clear(self):
        self.active = False
        self.bucket = self.doc = self.local = self.block_mask = None
        self._bias = {}

    def bias(self, layer: int) -> Tensor:
        """[H, NDOC, EL, EL] fp32 for this layer, gathered once per forward and grad mode."""
        key = (layer, torch.is_grad_enabled())
        if key not in self._bias:
            beta = self.betas[layer]
            table = torch.cat([beta, beta.new_zeros(beta.shape[0], 1)], dim=1)  # "none" bucket fixed at 0
            self._bias[key] = table[:, self.bucket]
        return self._bias[key]


class GraphBiasAttention(Attention):
    bias_ctx: GraphBiasContext
    bias_layer: int

    def forward(self, hidden_states: Tensor, cos_sin=None, cache=None, cache_lengths=None,
                stability_diagnostics=None, stability_scope=None, **seq_info) -> Tensor:
        if cache is not None or not self.bias_ctx.active:  # parent path; prefill bias in _attention_with_cache
            return super().forward(hidden_states, cos_sin, cache, cache_lengths, stability_diagnostics,
                                   stability_scope, **seq_info)
        # Attention.forward's projections and RoPE, then FlexAttention instead of flash_attn_varlen_prefixlm
        gqkv = self.gqkv_proj(hidden_states)
        gqkv = gqkv.unflatten(-1, (2 * self.num_heads + 2 * self.num_key_value_heads, self.head_dim))
        gate, query, key, value = gqkv.split((self.num_heads, self.num_heads, self.num_key_value_heads,
                                              self.num_key_value_heads), dim=-2)
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)

        ctx = self.bias_ctx
        bias, doc, local = ctx.bias(self.bias_layer), ctx.doc, ctx.local

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[h, doc[q_idx], local[q_idx], local[kv_idx]]

        q, k, v = (x.transpose(0, 1)[None] for x in (query, key, value))  # [T, H, D] -> [1, H, T, D]
        attn_output = _flex_attention(q, k, v, score_mod=score_mod, block_mask=ctx.block_mask)[0].transpose(0, 1)
        attn_output = (torch.sigmoid(gate) * attn_output).flatten(-2)
        return self.o_proj(attn_output)

    def _attention_with_cache(self, query, key, value, cache, cache_lengths, causal):
        ctx = self.bias_ctx
        if not ctx.active:
            return super()._attention_with_cache(query, key, value, cache, cache_lengths, causal)
        # Prefill of one prompt into an empty cache: Attention._attention_with_cache with the bias in the mask
        batch_size, seqlen = query.shape[:2]
        if batch_size != 1 or seqlen != ctx.local.shape[0]:
            raise ValueError("graph bias at inference: single-prompt prefill only")
        positions = cache_lengths[:, None] + torch.arange(seqlen, device=query.device, dtype=cache_lengths.dtype)
        cache.keys[0, positions[0]] = key[0]
        cache.values[0, positions[0]] = value[0]
        keys, values = cache.keys[:1, :seqlen], cache.values[:1, :seqlen]
        local = ctx.local
        attn_mask = ctx.bias(self.bias_layer)[:, 0][:, local[:, None], local[None, :]].to(query.dtype)  # [H, L, L]
        if causal:
            attn_mask = attn_mask.masked_fill(~torch.ones(seqlen, seqlen, dtype=torch.bool, device=query.device).tril(),
                                              torch.finfo(query.dtype).min)
        out = torch.nn.functional.scaled_dot_product_attention(query.transpose(1, 2), keys.transpose(1, 2),
                                                               values.transpose(1, 2), attn_mask=attn_mask[None])
        return out.transpose(1, 2)


def install_relative_bias(base: nn.Module, num_buckets: int) -> GraphBiasContext:
    """Switch every Attention of `base` to GraphBiasAttention; returns the shared context holding the betas."""
    layers = [m for m in base.modules() if isinstance(m, Attention)]
    if any(m.num_heads != m.num_key_value_heads for m in layers):
        raise NotImplementedError("grouped-query attention")
    device = next(base.parameters()).device
    betas = nn.ParameterList([nn.Parameter(torch.zeros(m.num_heads, num_buckets - 1, device=device)) for m in layers])
    ctx = GraphBiasContext(betas, num_buckets)
    for i, m in enumerate(layers):
        m.__class__ = GraphBiasAttention
        m.bias_ctx, m.bias_layer = ctx, i
    return ctx


def uninstall_relative_bias(base: nn.Module) -> None:
    """Back to plain Attention (tests that reuse one base model across methods)."""
    for m in base.modules():
        if isinstance(m, GraphBiasAttention):
            m.__class__ = Attention
            del m.bias_ctx, m.bias_layer


def packed_block_mask(doc: Tensor, prefix: Tensor) -> BlockMask:
    """Document + PrefixLM mask of a packed batch: same document, and the key is in the prompt or not after the
    query (the routing flash_attn_varlen_prefixlm implements). Padding tokens (doc = -1) attend nowhere."""
    t = doc.shape[0]

    def mask_mod(b, h, q_idx, kv_idx):
        return (doc[q_idx] == doc[kv_idx]) & (doc[q_idx] >= 0) & (prefix[kv_idx] | (kv_idx <= q_idx))

    return create_block_mask(mask_mod, None, None, t, t, device=doc.device)
