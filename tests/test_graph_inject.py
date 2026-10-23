"""graph_inject: parsing, tokenization parity with data_io, alignment, and (GPU) method sanity on HRM-Text-1B.

CPU tests need data/graphs/{graphqa,metaqa,kqapro} and the tokenized train sets; GPU tests also need the converted
HRM-Text-1B dir and a free GPU (~25 GB). Run with OMP_NUM_THREADS=1 (LapPE's eigensolver crawls on 192 threads).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from graph_inject.data import parse_graph
from graph_inject.structure import NO_ELEMENT, parse_graphqa
from graph_inject.tokenize import PromptTokenizer, tokenizer_checkpoint

REPO = Path(__file__).resolve().parents[1]
GRAPHQA = REPO / "data/graphs/graphqa"
DATA = REPO / "data/graphs"
HF = "/work/dfm/marius-ortega/dfm-hrm-graphs/hf"
METHODS = ["text", "text-incident", "text-colors", "prompt-tuning", "graphtoken", "node-tokens", "abs-pe", "abs-id",
           "rel-bias", "rel-bias-elem"]

pytestmark = pytest.mark.skipif(not (GRAPHQA / "hrm-text-grl").exists(), reason="GraphQA data not prepared")


def rows(split: str, dataset: str = "graphqa") -> list[dict]:
    with open(DATA / dataset / f"hrm-text-grl/standard/{split}.jsonl") as f:
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


# ---------------------------------------------------------------- knowledge graphs (MetaQA, KQA Pro)

@pytest.mark.parametrize("dataset", ["metaqa", "kqapro"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_kg_parse_matches_raw_counts(dataset, split):
    """Every row parses (parse_kg_facts also checks that each fact re-serializes to its line); one edge per fact
    plus one per qualifier; MetaQA questions name their topic entity."""
    raw_dir = "baseline-grl" if dataset == "metaqa" else "baseline"
    with open(DATA / dataset / f"{raw_dir}/standard/{split}.json") as f:
        raw = json.load(f)
    mentioned = 0
    for row, ref in zip(rows(split, dataset), raw, strict=True):
        g = parse_graph(row["instruction"], dataset)
        assert len(g.facts) == int(ref["nedges"])
        assert len(g.edges) == len(g.facts) + sum(len(f.qualifiers) for f in g.facts)
        assert len(g.node_names) == g.nnodes == len(set(g.node_names))
        mentioned += any(start >= g.question_start for start, _, _ in g.node_spans)
    if dataset == "metaqa":
        assert mentioned / len(raw) > 0.99


@pytest.mark.parametrize("hf_name,tokenized,dataset", [
    ("HRM-Text-1B", "metaqa_grl", "metaqa"), ("HRM-Text-1B", "kqapro_grl", "kqapro"),
    ("DFM-Mimir-v1.5", "metaqa_mimir", "metaqa"), ("DFM-Mimir-v1.5", "kqapro_mimir", "kqapro")])
def test_kg_token_ids_match_data_io(hf_name, tokenized, dataset):
    if not os.path.isdir(f"{HF}/{hf_name}"):
        pytest.skip(f"{hf_name} not converted")
    os.chdir(REPO)
    tok = PromptTokenizer(tokenizer_checkpoint(f"{HF}/{hf_name}"))
    d = REPO / f"data/graphs/tokenized/{tokenized}/train.jsonl"
    tokens, arrays = np.load(d / "tokens.npy"), {k: np.load(d / f"{k}.npy") for k in ("inst_start", "inst_len",
                                                                                      "resp_start", "resp_len")}
    for i, r in enumerate(rows("train", dataset)[:1000]):
        g = parse_graph(r["instruction"], dataset)
        enc = tok.encode_prompt(r["condition"], r["instruction"], g.char_elements(len(r["instruction"])))
        s, n = arrays["inst_start"][i], arrays["inst_len"][i]
        assert enc.ids.tolist() == tokens[s:s + n].tolist()
        s, n = arrays["resp_start"][i], arrays["resp_len"][i]
        assert tok.encode_response(r["condition"], r["instruction"], r["response"]).tolist() == tokens[s:s + n].tolist()
        if i < 100:  # alignment: a token of node v decodes to part of v's name (up to the separators it straddles)
            for t, el in zip(enc.ids, enc.elements):
                if NO_ELEMENT < el < g.nnodes:
                    text = tok.tokenizer.decode([int(t)]).strip(" ,()[];:'\"?.\n")
                    if "\ufffd" not in text:  # byte-level pieces of a multi-byte character decode to U+FFFD
                        assert text in g.node_names[el], (text, g.node_names[el])


@pytest.mark.parametrize("dataset", ["metaqa", "kqapro"])
def test_kg_text_encodings(dataset):
    from graph_inject.data import load_samples
    from graph_inject.methods.text import TextMethod
    samples = load_samples(str(DATA / dataset / "hrm-text-grl/standard/val.jsonl"), dataset)
    for enc in ("incident", "colors"):
        method = TextMethod(OmegaConf.create({"encoding": enc}), 1536, 1.0)
        for s in samples:
            text = method.rewrite_prompt(s)
            q = s.instruction[s.graph.question_start:]
            assert text.endswith(q) and text.startswith("#### Question\n")
            if enc == "incident" and s.graph.facts:
                assert all(f.subject in text and f.obj in text for f in s.graph.facts)


@pytest.mark.parametrize("dataset", ["metaqa", "kqapro"])
def test_val_subset_fixed_and_stratified(dataset):
    """The epoch-selection subset is the same for every run (fixed seed) and keeps each task's share (+-1)."""
    from collections import Counter
    from types import SimpleNamespace
    from graph_inject.data import load_samples, stratified_subset
    val = [SimpleNamespace(sample=s) for s in load_samples(str(DATA / dataset / "hrm-text-grl/standard/val.jsonl"), dataset)]
    sub = stratified_subset(val, 400)
    assert len(sub) == 400 and [x.sample.idx for x in sub] == [x.sample.idx for x in stratified_subset(val, 400)]
    full, part = Counter(x.sample.task for x in val), Counter(x.sample.task for x in sub)
    assert all(abs(part[t] - 400 * full[t] / len(val)) < 1 for t in full)


@pytest.mark.parametrize("dataset", ["metaqa", "kqapro"])
def test_kg_structural_features(dataset):
    """RWSE + LapPE on every val graph (incl. >= 100 nodes, where PyG would use ARPACK): finite, right shape."""
    from graph_inject.features import structural_features
    for r in rows("val", dataset):
        g = parse_graph(r["instruction"], dataset)
        x = structural_features(g, 16, 8)
        assert x.shape == (g.nnodes, 24) and np.isfinite(x).all()


def test_levi_buckets():
    """Path 0-1-2 plus isolated node 3; elements: nodes 0..3, edge (0,1) = 4, edge (1,2) = 5."""
    from graph_inject.attention_bias import MAX_DISTANCE, element_buckets
    from graph_inject.structure import PromptGraph
    g = PromptGraph(nnodes=4, edges=[(0, 1), (1, 2)], node_spans=[], edge_spans=[], description_span=(0, 0),
                    question_start=0)
    b = element_buckets(g, "distance")
    assert b[0, 0] == 0 and b[0, 4] == 1 and b[0, 1] == 2 and b[0, 5] == 3 and b[0, 2] == 4 and b[4, 5] == 2
    assert (b[3, :3] == MAX_DISTANCE + 2).all() and (b == b.T).all()
    e = element_buckets(g, "element")
    assert (np.diag(e) == 0).all() and e.sum() == 6 * 5


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
    from graph_inject.attention_bias import uninstall_relative_bias
    from graph_inject.lora import apply_lora, LoRALinear
    from graph_inject.methods import build_method
    from graph_inject.model import GraphInjectedLM, embedding_rms
    base = ckpt.model
    uninstall_relative_bias(base)  # undo rel-bias from a previous build
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


def batch_for(ckpt, model, n=6, split="val", dataset="graphqa", start=0):
    from graph_inject.data import encode_samples, load_samples, to_device_prefixlm, train_batch
    rows_ = load_samples(str(DATA / dataset / f"hrm-text-grl/standard/{split}.jsonl"), dataset, start + n)[start:]
    samples = encode_samples(rows_, model.method, PromptTokenizer(ckpt))
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


@pytest.mark.parametrize("dataset", ["graphqa", "metaqa", "kqapro"])
@pytest.mark.parametrize("method_name", METHODS)
def test_gradients_reach_trainable_modules(setup, method_name, dataset):
    ckpt = setup
    model = build(ckpt, method_name, lora=True)
    # KQA Pro val rows 13-15 include a prompt without facts (row 14): methods must handle empty graphs
    _, batch = batch_for(ckpt, model, dataset=dataset, **{"kqapro": {"n": 3, "start": 13}, "metaqa": {"n": 4}}.get(dataset, {}))
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


# ---------------------------------------------------------------- phase 2: relative attention bias

def _logits(model, batch):
    return model(None, {k: v for k, v in batch.items() if k != "labels"}, bp_steps=5)[1].float()


@pytest.mark.parametrize("dataset", ["graphqa", "metaqa"])
def test_rel_bias_zero_matches_text(setup, dataset):
    """beta = 0: FlexAttention (document + PrefixLM mask) gives the varlen kernel's logits."""
    ckpt = setup
    n = 6 if dataset == "graphqa" else 4
    text = build(ckpt, "text", lora=False)
    samples, batch = batch_for(ckpt, text, n=n, dataset=dataset)
    with torch.no_grad():
        ref = _logits(text, batch)
        model = build(ckpt, "rel-bias", lora=False)
        _, batch = batch_for(ckpt, model, n=n, dataset=dataset)
        out = _logits(model, batch)
        model.method.ctx.clear()
    real = sum(len(s.prompt.ids) + len(s.response_ids) - 1 for s in samples)
    agree = (ref[:real].argmax(-1) == out[:real].argmax(-1)).float().mean()
    assert agree > 0.98, agree
    last = np.cumsum([len(s.prompt.ids) + len(s.response_ids) - 1 for s in samples]) - 1
    assert (ref[last] - out[last]).abs().max() < 0.5


@pytest.mark.parametrize("method_name", ["rel-bias", "rel-bias-elem"])
def test_rel_bias_train_matches_prefill(setup, method_name):
    """Random beta: packed training logits (flex + bias) agree with KV-cache prefill (SDPA + bias) on every prompt
    position as closely as at beta = 0 (bf16 kernel noise), while beta itself moves the logits much more."""
    from graph_inject.generate import _prefill_batch
    import torch.utils._pytree as pytree
    ckpt = setup
    model = build(ckpt, method_name, lora=False).eval()
    samples, batch = batch_for(ckpt, model, n=3)

    def run():
        packed = _logits(model, batch)
        cache = model.create_cache(max_batch_size=1, max_seq_len=2048, dtype=torch.bfloat16, device="cuda")
        offset, pos, prefill = 0, [], []
        for s in samples:
            pre = model(None, _prefill_batch(s, model.method, pytree.tree_map(lambda x: x[0:1], cache), "cuda"))[1]
            prefill.append(pre[0].float())
            pos.append(torch.arange(offset, offset + len(s.prompt.ids)))
            offset += len(s.prompt.ids) + len(s.response_ids) - 1
        pos = torch.cat(pos)
        return packed[pos], torch.cat(prefill)

    torch.manual_seed(0)
    with torch.no_grad():
        packed0, prefill0 = run()
        for beta in model.method.betas:
            beta.normal_(0, 2.0)
        packed, prefill = run()
        model.method.ctx.clear()
    noise, gap = (prefill0 - packed0).abs().mean(), (prefill - packed).abs().mean()
    effect = (packed0 - packed).abs().mean()
    assert (prefill.argmax(-1) == packed.argmax(-1)).float().mean() > 0.98
    assert gap < 1.5 * noise and effect > 5 * gap, (noise, gap, effect)


def test_rel_bias_generation_zero_matches_text(setup):
    from graph_inject.generate import generate
    ckpt = setup
    text = build(ckpt, "text", lora=False).eval()
    samples, _ = batch_for(ckpt, text, n=8)
    ref = generate(text, ckpt.carry, text.method, samples, ckpt.stop_token_id(), max_generation=32, batch_size=4)
    model = build(ckpt, "rel-bias", lora=False).eval()
    samples, _ = batch_for(ckpt, model, n=8)
    ours = generate(model, ckpt.carry, model.method, samples, ckpt.stop_token_id(), max_generation=32, batch_size=4)
    assert ours == ref
