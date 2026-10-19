"""PrefixLM attention on PyTorch's built-in FlashAttention-2 varlen kernels (``torch.nn.attention.varlen``).

Used for SM120 (RTX PRO 6000 Blackwell / RTX 50xx): FA3 is SM90-only, and FA4 CuTe fails to compile its
SM120 varlen kernels (flash-attn-4 4.0.0b14 @ 4178915 and 4.0.0b19). PyTorch >= 2.11 cu13x wheels ship
FA2 varlen kernels for sm_120. Routing matches the ROCm backend: one bidirectional pass over prefix
tokens, then one causal pass of response queries over each document's prefix + response keys, using
FlashAttention's bottom-right causal alignment.
"""

from typing import Optional

import torch
from torch import Tensor

from models.flash_attention_prefixlm_common import PREFIXLM_ROUTING_KEYS, prefixlm_routing_from_tensors

__all__ = ["flash_attn_varlen_prefixlm"]


def _torch_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> Tensor:
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as exc:
        raise ImportError("accelerator_type=sm120 requires torch>=2.11 (torch.nn.attention.varlen.varlen_attn).") from exc

    return varlen_attn(
        q,
        k,
        v,
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        max_seqlen_q,
        max_seqlen_k,
        # (-1, 0) is causal with bottom-right alignment when seqlen_q < seqlen_k.
        window_size=(-1, 0) if causal else (-1, -1),
    )


def flash_attn_varlen_prefixlm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    prefix_lens: Tensor,
    causal_lens: Tensor,
    cu_seqlens: Tensor,
    total_seqlen: Tensor,
    numseqs: Tensor,
    max_seqlen_prefix: Tensor,
    max_seqlen_causal: Tensor,
    max_seqlen_all: Tensor,
    *,
    prefix_cu_seqlens: Optional[Tensor] = None,
    prefix_idx: Optional[Tensor] = None,
    causal_idx: Optional[Tensor] = None,
    active_key_idx: Optional[Tensor] = None,
    active_cu_seqlens_q: Optional[Tensor] = None,
    active_cu_seqlens_k: Optional[Tensor] = None,
) -> Tensor:
    routing_values = (
        prefix_cu_seqlens,
        prefix_idx,
        causal_idx,
        active_key_idx,
        active_cu_seqlens_q,
        active_cu_seqlens_k,
    )
    if all(value is None for value in routing_values):
        routing = prefixlm_routing_from_tensors(
            prefix_lens,
            causal_lens,
            cu_seqlens,
            total_seqlen,
            numseqs,
            max_seqlen_prefix,
            max_seqlen_causal,
            max_seqlen_all,
        )
    elif any(value is None for value in routing_values):
        raise ValueError(f"PrefixLM routing tensors must provide all of {PREFIXLM_ROUTING_KEYS}")
    else:
        routing = dict(zip(PREFIXLM_ROUTING_KEYS, routing_values, strict=True))  # type: ignore[arg-type]

    max_prefix = int(max_seqlen_prefix.item())

    out = torch.zeros_like(q)
    prefix_idx = routing["prefix_idx"]
    out[prefix_idx] = _torch_varlen(
        q[prefix_idx],
        k[prefix_idx],
        v[prefix_idx],
        cu_seqlens_q=routing["prefix_cu_seqlens"],
        cu_seqlens_k=routing["prefix_cu_seqlens"],
        max_seqlen_q=max_prefix,
        max_seqlen_k=max_prefix,
        causal=is_causal,
    )

    causal_idx = routing["causal_idx"]
    if causal_idx.numel() > 0:
        out[causal_idx] = _torch_varlen(
            q[causal_idx],
            k[routing["active_key_idx"]],
            v[routing["active_key_idx"]],
            cu_seqlens_q=routing["active_cu_seqlens_q"],
            cu_seqlens_k=routing["active_cu_seqlens_k"],
            # Upper bounds are accepted; reuse the packed-batch metadata instead of syncing reductions.
            max_seqlen_q=int(max_seqlen_causal.item()),
            max_seqlen_k=int(max_seqlen_all.item()),
            causal=True,
        )

    return out
