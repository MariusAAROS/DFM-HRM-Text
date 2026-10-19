"""HF HRM-Text -> native import (conversion/convert_from_hf.py): logit parity with transformers' HrmTextForCausalLM.

A random tiny HF model is saved, mapped onto the native model and both are run in fp32 on CPU over one
PrefixLM sequence: HF with token_type_ids (prompt bidirectional), native with the inference engine's
cache path (bidirectional prefill of the prompt, then causal one-token decodes).
"""
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

transformers = pytest.importorskip("transformers")
HrmTextConfig = getattr(transformers, "HrmTextConfig", None)
if HrmTextConfig is None:
    pytest.skip("transformers without hrm_text", allow_module_level=True)

from conversion.convert_from_hf import check_arch, load_hf_state, native_key  # noqa: E402
from models.accelerator import set_accelerator_type  # noqa: E402
from pretrain import load_model_class  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VOCAB, HIDDEN, HEADS, PER_STACK, H, L, SEQ = 512, 128, 2, 2, 2, 2, 64


def native_cfg(**arch_overrides) -> dict:
    overrides = [f"arch.n_layers={2 * PER_STACK}", f"arch.hidden_size={HIDDEN}", f"arch.num_heads={HEADS}",
                 f"arch.H_cycles={H}", f"arch.L_cycles={L}"] + [f"arch.{k}={v}" for k, v in arch_overrides.items()]
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="cfg_graphqa_hrm1b", overrides=overrides)
    return OmegaConf.to_container(cfg.arch, resolve=True) | {  # type: ignore[operator]
        "vocab_size": VOCAB, "max_seq_len": SEQ, "tokenizer_info": {}, "total_length": 0,
    } | OmegaConf.to_container(cfg.data, resolve=True)  # type: ignore[operator]


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    config = HrmTextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=512, num_hidden_layers=PER_STACK,
        num_attention_heads=HEADS, head_dim=HIDDEN // HEADS, H_cycles=H, L_cycles=L, L_bp_cycles=[0, L],
        max_position_embeddings=SEQ, initializer_range=HIDDEN ** -0.5, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    with torch.no_grad():
        for param in model.parameters():  # z_L_init is zero-initialised; make every tensor informative
            param.normal_(0.0, 0.05)
    path = tmp_path_factory.mktemp("hf_hrm")
    model.save_pretrained(path)
    return path


def test_native_key_mapping():
    assert native_key("model.embed_tokens.weight") == "embed_tokens.embedding_weight"
    assert native_key("model.z_L_init") == "model.zL_init"
    assert native_key("lm_head.weight") == "lm_head.weight"
    assert native_key("model.H_module.layers.3.attn.gqkv_proj.weight") == "model.H_level.core.layers.3.attn.gqkv_proj.weight"
    assert native_key("model.L_module.layers.0.mlp.down_proj.weight") == "model.L_level.core.layers.0.mlp.down_proj.weight"


def test_check_arch_rejects_mismatch(hf_dir):
    import json
    hf_cfg = json.loads((hf_dir / "config.json").read_text())
    check_arch(hf_cfg, native_cfg())
    with pytest.raises(ValueError, match="L_cycles"):
        check_arch(hf_cfg, native_cfg(L_cycles=3))
    with pytest.raises(ValueError, match="embedding_scale"):
        check_arch(hf_cfg, native_cfg(init_type="fixed_normal"))


def test_logit_parity_prefixlm(hf_dir):
    set_accelerator_type("cpu")
    # fp32 zL_init buffer (it is allocated in fwd_bwd_dtype); HF RMSNorm always runs in fp32, so fp32 is
    # also the precision at which the two implementations can agree exactly.
    cfg = native_cfg() | {"fwd_bwd_dtype": "float32"}
    native = load_model_class(cfg["head"])(load_model_class(cfg["name"])(cfg), cfg)
    native.load_state_dict({native_key(k): v for k, v in load_hf_state(hf_dir).items()}, strict=True)
    native.eval()
    hf = transformers.AutoModelForCausalLM.from_pretrained(hf_dir, attn_implementation="eager").eval()

    torch.manual_seed(1)
    prompt_len, answer_len = 11, 6
    tokens = torch.randint(0, VOCAB, (prompt_len + answer_len,))
    token_type_ids = (torch.arange(tokens.numel()) < prompt_len).long()
    with torch.no_grad():
        ref = hf(input_ids=tokens[None], token_type_ids=token_type_ids[None]).logits[0]

        cache = native.create_cache(max_batch_size=1, max_seq_len=SEQ, dtype=torch.float32, device="cpu")
        out = [native(carry=None, batch={"inputs": tokens[None, :prompt_len], "position_ids": torch.arange(prompt_len),
                                         "cache": cache, "cache_lengths": torch.zeros(1, dtype=torch.int32)})[-1][0]]
        for pos in range(prompt_len, tokens.numel()):
            out.append(native(carry=None, batch={"inputs": tokens[None, pos:pos + 1], "position_ids": torch.tensor([[pos]]),
                                                 "cache": cache, "cache_lengths": torch.tensor([pos], dtype=torch.int32)})[-1][0])
        got = torch.cat(out)

    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
