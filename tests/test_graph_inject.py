"""graph_inject: parsing, tokenization parity with data_io, alignment, and (GPU) method sanity on HRM-Text-1B.

CPU tests need data/graphs/graphqa and the tokenized train sets; GPU tests also need the converted
HRM-Text-1B dir and a free GPU (~25 GB).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from graph_inject.structure import NO_ELEMENT, parse_graphqa
from graph_inject.tokenize import PromptTokenizer, tokenizer_checkpoint

REPO = Path(__file__).resolve().parents[1]
GRAPHQA = REPO / "data/graphs/graphqa"
HF = "/work/dfm/marius-ortega/dfm-hrm-graphs/hf"
METHODS = ["text", "text-incident", "text-colors", "prompt-tuning", "graphtoken", "node-tokens", "abs-pe", "abs-id"]

pytestmark = pytest.mark.skipif(not (GRAPHQA / "hrm-text-grl").exists(), reason="GraphQA data not prepared")


def rows(split: str) -> list[dict]:
    with open(GRAPHQA / f"hrm-text-grl/standard/{split}.jsonl") as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_parse_matches_raw_counts(split):
    with open(GRAPHQA / f"baseline/standard/{split}.json") as f:
        raw = json.load(f)
    for row, ref in zip(rows(split), raw):
        g = parse_graphqa(row["instruction"])
        assert g.nnodes == int(ref["nnodes"]) and len(g.edges) == int(ref["nedges"])


@pytest.mark.parametrize("hf_name,tokenized", [("HRM-Text-1B", "graphqa_grl"), ("DFM-Mimir-v1.5", "graphqa_mimir"),
                                               ("DFM-Mimir", "graphqa_mimir")])
def test_token_ids_match_data_io(hf_name, tokenized):
    if not os.path.isdir(f"{HF}/{hf_name}"):
        pytest.skip(f"{hf_name} not converted")
    os.chdir(REPO)  # Mimir's tokenizer path is relative to the repo root
    tok = PromptTokenizer(tokenizer_checkpoint(f"{HF}/{hf_name}"))
    d = REPO / f"data/graphs/tokenized/{tokenized}/train.jsonl"
    tokens, arrays = np.load(d / "tokens.npy"), {k: np.load(d / f"{k}.npy") for k in ("inst_start", "inst_len",
                                                                                      "resp_start", "resp_len")}
    for i, r in enumerate(rows("train")):
        g = parse_graphqa(r["instruction"])
        enc = tok.encode_prompt(r["condition"], r["instruction"], g.char_elements(len(r["instruction"])))
        resp = tok.encode_response(r["condition"], r["instruction"], r["response"])
        s, n = arrays["inst_start"][i], arrays["inst_len"][i]
        assert enc.ids.tolist() == tokens[s:s + n].tolist()
        s, n = arrays["resp_start"][i], arrays["resp_len"][i]
        assert resp.tolist() == tokens[s:s + n].tolist()
        if i < 200:  # alignment: every token of node v decodes to digits of v (or the space before them)
            for t, el in zip(enc.ids, enc.elements):
                if NO_ELEMENT < el < g.nnodes:
                    text = tok.tokenizer.decode([int(t)]).strip()
                    assert text == "" or text in str(el), (text, el)


def test_soft_slots_before_content():
    tok = PromptTokenizer(tokenizer_checkpoint(f"{HF}/HRM-Text-1B"))
    r = rows("val")[0]
    enc = tok.encode_prompt(r["condition"], r["instruction"], None, num_soft_tokens=3)
    ids = tok.tokenizer.convert_ids_to_tokens(enc.ids.tolist())
    assert enc.soft_slots.tolist() == [2, 3, 4] and ids[5] == "####"


# ---------------------------------------------------------------- GPU: model + methods on HRM-Text-1B

@pytest.fixture(scope="module")
def setup():
    if not torch.cuda.is_available() or not os.path.isdir(f"{HF}/HRM-Text-1B"):
        pytest.skip("needs a GPU and the converted HRM-Text-1B")
    os.chdir(REPO)
    from models.accelerator import set_accelerator_type
    from simple_inference_engine import inference_load_checkpoint
    set_accelerator_type("sm120")
    ckpt = inference_load_checkpoint(f"{HF}/HRM-Text-1B", None, False, ckpt_tag="step_0")
    return ckpt


def build(ckpt, method_name: str, lora: bool):
    from graph_inject.lora import apply_lora, LoRALinear
    from graph_inject.methods import build_method
    from graph_inject.model import GraphInjectedLM, embedding_rms
    base = ckpt.model
    for module in list(base.modules()):  # undo LoRA from a previous build
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.base)
    base.requires_grad_(False).train()
    if lora:
        apply_lora(base, 16, 32)
    cfg = OmegaConf.load(REPO / f"graph_inject/configs/methods/{method_name}.yaml")
    method = build_method(cfg, 1536, embedding_rms(base)).cuda()
    method.setup(ckpt.tokenizer, base.embed_tokens)
    return GraphInjectedLM(base, method)


def batch_for(ckpt, model, n=6, split="val"):
    from graph_inject.data import encode_samples, load_samples, to_device_prefixlm, train_batch
    samples = encode_samples(load_samples(str(GRAPHQA / f"hrm-text-grl/standard/{split}.jsonl"))[:n], model.method,
                             PromptTokenizer(ckpt))
    batch, scalars = train_batch(samples, model.method, 8192, np.random.default_rng(0))
    return samples, to_device_prefixlm(batch, scalars, torch.device("cuda"))


@pytest.mark.parametrize("method_name", ["text", "abs-pe", "abs-id"])
def test_zero_init_is_identity(setup, method_name):
    """LoRA B = 0 and phi's output projections = 0: logits equal the pretrained model's."""
    ckpt = setup
    plain = build(ckpt, "text", lora=False)
    _, batch = batch_for(ckpt, plain)
    with torch.no_grad():
        ref = ckpt.model(None, {k: v for k, v in batch.items() if k not in ("labels",) and not k.startswith("graph_")},
                         bp_steps=5)[1]
    model = build(ckpt, method_name, lora=True)
    _, batch = batch_for(ckpt, model)
    with torch.no_grad():
        out = model(None, {k: v for k, v in batch.items() if k != "labels"}, bp_steps=5)[1]
    assert torch.equal(ref, out)


@pytest.mark.parametrize("method_name", METHODS)
def test_gradients_reach_trainable_modules(setup, method_name):
    ckpt = setup
    model = build(ckpt, method_name, lora=True)
    _, batch = batch_for(ckpt, model)
    _, loss, _ = model(None, batch, bp_steps=5)
    loss.backward()
    assert torch.isfinite(loss)
    zero_init = ("lora_A", "node_encoder", "edge_encoder")  # behind a zero-init matrix: no gradient at step 0
    for name, p in model.named_parameters():
        if p.requires_grad and not any(z in name for z in zero_init):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    # the method's graph module receives gradient through z_H (inject_x=False, bp_max_steps 5)
    method_grads = [p.grad for p in model.method.parameters() if p.grad is not None]
    assert method_grads or method_name.startswith("text")


@pytest.mark.parametrize("method_name", ["text", "graphtoken"])
def test_packed_logits_match_prefill(setup, method_name):
    """Training forward (packed varlen) and inference prefill (KV-cache SDPA) agree on the prompt's last token."""
    from graph_inject.generate import _prefill_batch
    import torch.utils._pytree as pytree
    ckpt = setup
    model = build(ckpt, method_name, lora=True).eval()
    samples, batch = batch_for(ckpt, model, n=3)
    with torch.no_grad():
        logits = model(None, {k: v for k, v in batch.items() if k != "labels"}, bp_steps=5)[1]
        cache = model.create_cache(max_batch_size=1, max_seq_len=2048, dtype=torch.bfloat16, device="cuda")
        offset = 0
        for s in samples:
            pre = model(None, _prefill_batch(s, model.method, pytree.tree_map(lambda x: x[0:1], cache), "cuda"))[1]
            last = offset + len(s.prompt.ids) - 1
            assert pre[0, -1].argmax() == logits[last].argmax()
            assert (pre[0, -1].float() - logits[last].float()).abs().max() < 0.5
            offset += len(s.prompt.ids) + len(s.response_ids) - 1


def test_generation_matches_inference_engine(setup):
    from graph_inject.generate import generate
    from simple_inference_engine import inference_generate
    ckpt = setup
    model = build(ckpt, "text", lora=False).eval()  # the engine infers its cache dtype from the first parameter
    samples, _ = batch_for(ckpt, model, n=24)
    ours = generate(model, ckpt.carry, model.method, samples, ckpt.stop_token_id(), max_generation=32, batch_size=8)
    ckpt_eval = type(ckpt)(model=ckpt.model.eval(), carry=ckpt.carry, tokenizer=ckpt.tokenizer,
                           tokenizer_info=ckpt.tokenizer_info)
    prompts = ((i, (s.sample.condition, s.sample.instruction)) for i, s in enumerate(samples))
    ref = {i: ids.tolist() for i, _, ids in inference_generate(ckpt_eval, prompts, max_tokens=2048, max_generation=32,
                                                               batch_size=8, return_tokens=True)}
    assert all(ours[i] == ref[i] for i in range(len(samples)))
