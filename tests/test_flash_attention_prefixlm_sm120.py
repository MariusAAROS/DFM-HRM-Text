import numpy as np
import pytest
import torch

import models.accelerator as accelerator
from models.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars


def test_sm120_is_a_cuda_accelerator_with_capability_12(monkeypatch) -> None:
    assert accelerator._expected_cuda_major("sm120") == 12

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _rank=0: (12, 0))
    assert accelerator.is_accelerator_available("sm120")
    assert not accelerator.is_accelerator_available("sm100")
    assert accelerator.torch_device_for_accelerator("sm120") == torch.device("cuda", 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _rank=0: (10, 0))
    assert not accelerator.is_accelerator_available("sm120")
    with pytest.raises(RuntimeError, match="capability 12.x"):
        accelerator.validate_accelerator_available("sm120")


def test_sm120_routes_prefixlm_attention_to_torch_varlen(monkeypatch) -> None:
    import models.flash_attention_prefixlm_dispatch as dispatch
    import models.flash_attention_prefixlm_torch_varlen as torch_varlen

    calls = []

    def fake_backend(q, *_args, **kwargs):
        calls.append(kwargs["prefix_idx"])
        return q

    monkeypatch.setattr(torch_varlen, "flash_attn_varlen_prefixlm", fake_backend)
    monkeypatch.setattr(dispatch, "get_accelerator_type", lambda: "sm120")
    q = torch.zeros(4, 1, 8)
    scalars = [torch.tensor(0)] * 5
    out = dispatch.flash_attn_varlen_prefixlm(q, q, q, False, q, q, q, *scalars, prefix_idx=q)
    assert out is q
    assert len(calls) == 1 and calls[0] is q


def test_sm120_prepares_prefixlm_batch_metadata(monkeypatch) -> None:
    from models.common import prepare_prefixlm_batch
    from models.flash_attention_prefixlm_common import PREFIXLM_PREPARED_KEYS

    tensors, scalars = compute_aux_seq_tensors_scalars(np.array([3, 4], dtype=np.int32), np.array([2, 0], dtype=np.int32), 16)
    batch = {k: torch.from_numpy(v) for k, v in tensors.items()} | {k: torch.tensor(v) for k, v in scalars.items()}
    monkeypatch.setattr("models.accelerator.get_accelerator_type", lambda: "sm120")
    prepared = prepare_prefixlm_batch(dict(batch))
    assert all(key in prepared for key in PREFIXLM_PREPARED_KEYS)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 12


# (prefix_len, causal_len) per packed sequence; covers zero-causal, long-prefix and many-short-docs batches.
PACKS = {
    "mixed": [(37, 5), (120, 0), (8, 64), (300, 1), (1, 1)],
    "long_prefix": [(900, 3), (2, 2)],
    "graphqa_like": [(150, 3)] * 20 + [(96, 9)] * 5,
}


def _packed_case(pack, padding: int, heads: int = 8, head_dim: int = 128, seed: int = 0):
    prefix = np.array([p for p, _ in pack], dtype=np.int32)
    causal = np.array([c for _, c in pack], dtype=np.int32)
    total = int(prefix.sum() + causal.sum()) + padding
    tensors, scalars = compute_aux_seq_tensors_scalars(prefix, causal, total)
    meta = [torch.from_numpy(tensors[k]).cuda() for k in ("prefix_lens", "causal_lens", "cu_seqlens")]
    meta += [torch.tensor(scalars[k]) for k in ("total_seqlen", "numseqs", "max_seqlen_prefix", "max_seqlen_causal", "max_seqlen_all")]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = [torch.randn(total, heads, head_dim, device="cuda", generator=gen) for _ in range(3)]
    grad_out = torch.randn(total, heads, head_dim, device="cuda", generator=gen)
    return qkv, grad_out, meta


def _run(fn, qkv, grad_out, meta, dtype, is_causal, **kwargs):
    q, k, v = [t.detach().to(dtype).requires_grad_(True) for t in qkv]
    out = fn(q, k, v, is_causal, *meta, **kwargs)
    out.backward(grad_out.to(dtype))
    return [t.float() for t in (out, q.grad, k.grad, v.grad)]


@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 GPU")
@pytest.mark.parametrize("precomputed_routing", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("pack_name", sorted(PACKS))
def test_sm120_prefixlm_matches_dense_reference(precomputed_routing: bool, is_causal: bool, pack_name: str) -> None:
    from models.flash_attention_prefixlm_common import prefixlm_routing_from_tensors
    from models.flash_attention_prefixlm_dense import flash_attn_varlen_prefixlm as dense
    from models.flash_attention_prefixlm_dispatch import flash_attn_varlen_prefixlm as dispatch

    previous = accelerator.get_accelerator_type()
    accelerator.set_accelerator_type("sm120")
    try:
        qkv, grad_out, meta = _packed_case(PACKS[pack_name], padding=7)
        routing = prefixlm_routing_from_tensors(*meta) if precomputed_routing else {}
        reference = _run(dense, qkv, grad_out, meta, torch.float32, is_causal)
        dense_bf16 = _run(dense, qkv, grad_out, meta, torch.bfloat16, is_causal)
        fast = _run(dispatch, qkv, grad_out, meta, torch.bfloat16, is_causal, **routing)
    finally:
        accelerator.set_accelerator_type(previous)

    total = int(meta[3].item())
    for name, got, base, ref in zip(("out", "dq", "dk", "dv"), fast, dense_bf16, reference):
        assert torch.isfinite(got).all(), name
        # Padding rows carry no tokens: outputs and gradients must be exactly zero there.
        assert got[total:].abs().max().item() == 0.0, name
        # FlashAttention-style tolerance: the bf16 kernel error is bounded by a small multiple of the
        # bf16 dense error against the fp32 reference.
        fast_err = (got - ref).abs().max().item()
        dense_err = (base - ref).abs().max().item()
        assert fast_err <= 2 * dense_err + 1e-3, f"{name}: sm120 err {fast_err:.3e} vs dense bf16 err {dense_err:.3e}"
