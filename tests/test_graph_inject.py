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
           "rel-bias", "rel-bias-elem", "graphtoken-faithful", "node-tokens-faithful"]

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


@pytest.mark.parametrize("dataset", ["graphqa", "metaqa", "kqapro"])
def test_token_occurrences(dataset):
    """Each graph token's occurrence is a span of its own element that overlaps the token; occurrences are -1 exactly
    off the graph; nodes mentioned several times (edge tuples, facts, the question) have several occurrences."""
    tok = PromptTokenizer(tokenizer_checkpoint(f"{HF}/HRM-Text-1B"))
    repeated = 0
    for r in rows("val", dataset)[:50]:
        g = parse_graph(r["instruction"], dataset)
        n = len(r["instruction"])
        enc = tok.encode_prompt(r["condition"], r["instruction"], g.char_elements(n), char_spans=g.char_spans(n))
        text, start = tok.render(r["condition"], r["instruction"])
        offsets = tok.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        spans = [(a, b, g.nnodes + e) for a, b, e in g.edge_spans] + list(g.node_spans)
        assert ((enc.occurrences == NO_ELEMENT) == (enc.elements == NO_ELEMENT)).all()
        for (ts, te), el, o in zip(offsets, enc.elements, enc.occurrences):
            if o >= 0:
                a, b, span_el = spans[o]
                assert span_el == el and ts - start < b and te - start > a
        node_occ = {}
        for el, o in zip(enc.elements, enc.occurrences):
            if NO_ELEMENT < el < g.nnodes:
                node_occ.setdefault(int(el), set()).add(int(o))
        repeated += any(len(v) > 1 for v in node_occ.values())
    assert repeated >= 40


def test_cl_owl_colors():
    """The official naming reaches exactly the six names of the paper's Figure 5, in hue order; the block carries the
    paper's instructions."""
    from graph_inject.methods.text import cl_owl_color, color_block
    names = [cl_owl_color(x) for x in np.linspace(0, 1, 1801)]
    assert set(names) == {"red", "crimson", "orange", "lime", "green", "teal"}
    assert names[0] == "red" and names[-1] == "teal" and cl_owl_color(0.5) == "lime"
    block = color_block(parse_graphqa(rows("val")[0]["instruction"]))
    assert "similarity metric among nodes" in block and "Color(i)=c" in block and "\n<<COLORS>>\nColor(0):" in block


@pytest.mark.parametrize("dataset", ["graphqa", "metaqa", "kqapro"])
def test_no_graph_text(dataset):
    """graph_text: false (the faithful arms) keeps the question and drops every edge / fact."""
    from graph_inject.data import load_samples
    from graph_inject.methods import build_method
    for name in ("graphtoken-faithful", "node-tokens-faithful"):
        cfg = OmegaConf.load(REPO / f"graph_inject/configs/methods/{name}.yaml")
        method = build_method(cfg, 1536, 1.0)
        for s in load_samples(str(DATA / dataset / "hrm-text-grl/standard/val.jsonl"), dataset, 100):
            text = method.rewrite_prompt(s)
            assert text.startswith("#### Question\nQ: ") and text.endswith(s.instruction[s.graph.question_start:])
            assert "The edges in G" not in text and "The facts are" not in text


def test_colors_fit_context():
    """KQA Pro text-colors prompts exceed the 4096-position RoPE table without truncation; the colour block is cut to
    the first nodes that fit, other prompts are untouched."""
    from graph_inject.data import load_samples
    from graph_inject.methods.text import PROMPT_TOKEN_BUDGET, TextMethod, color_block
    ckpt = tokenizer_checkpoint(f"{HF}/HRM-Text-1B")
    tok = PromptTokenizer(ckpt)
    method = TextMethod(OmegaConf.create({"encoding": "colors"}), 1536, 1.0)
    method.setup(ckpt.tokenizer, None)
    truncated = 0
    for s in load_samples(str(DATA / "kqapro/hrm-text-grl/standard/val.jsonl"), "kqapro"):
        text = method.rewrite_prompt(s)
        end = s.graph.description_span[1]
        full = s.instruction[:end] + color_block(s.graph) + s.instruction[end:]
        assert len(tok.encode_prompt(s.condition, text, None).ids) + 96 < 4096
        truncated += text != full
        assert text == full or len(ckpt.tokenizer(full, add_special_tokens=False)["input_ids"]) > PROMPT_TOKEN_BUDGET
    assert 0 < truncated < 20


V2 = GRAPHQA / "hrm-text-grl-v2/standard"


@pytest.mark.skipif(not V2.exists(), reason="run scripts/graph_inject/build_graphqa_v2.py")
def test_graphqa_v2_maxflow_and_dedup():
    """v2: MaximumFlow capacities in 1-10 are printed and the gold is the max flow of the printed graph; no val/test
    prompt is a train prompt; every other row is unchanged."""
    import networkx as nx
    from graph_inject.data import load_samples
    v2 = {k: [json.loads(line) for line in open(V2 / f"{k}.jsonl")] for k in ("train", "val", "test")}
    assert (len(v2["train"]), len(v2["val"]), len(v2["test"])) == (3080, 307, 324)
    old_train = {r["instruction"] for r in rows("train")}
    assert not any(r["instruction"] in old_train for k in ("val", "test") for r in v2[k])
    report = json.load(open(V2 / "dedup_report.json"))
    old_test = rows("test")
    for r, i in zip(v2["test"], report["test"]["kept_ids"]):
        assert r["task"] == old_test[i]["task"] and (r["task"] == "MaximumFlow" or r == old_test[i])
    for s in load_samples(str(V2 / "test.jsonl"), "graphqa_v2"):
        if s.task != "MaximumFlow":
            continue
        caps = [int(c) for c in __import__("re").findall(r"\(\d+, \d+, (\d+)\)", s.instruction)]
        assert len(caps) == len(s.graph.edges) and all(1 <= c <= 10 for c in caps)
        g = nx.Graph()
        g.add_nodes_from(range(s.graph.nnodes))
        g.add_edges_from((u, v, {"capacity": c}) for (u, v), c in zip(s.graph.edges, caps) if u != v)
        src, dst = s.graph.question_nodes()[:2]
        flow = nx.maximum_flow_value(g, src, dst) if nx.has_path(g, src, dst) else 0
        assert s.response.strip().rstrip(".") == str(int(flow))


def test_unsaturated_task_diagnostics():
    """edge_index / tri_hint / tri_cot touch only their task; the edge numbering ends at the gold edge count, the
    per-node triangle counts sum to 3x the gold, and the prompts still parse to the same graph."""
    from graph_inject.data import load_samples
    from graph_inject.generate import final_answer
    from graph_inject.methods import build_method
    samples = load_samples(str(GRAPHQA / "hrm-text-grl/standard/train.jsonl"))
    for name, task in (("text-edge-index", "EdgeCount"), ("text-tri-hint", "TriangleCounting"),
                       ("text-tri-cot", "TriangleCounting")):
        cfg = OmegaConf.load(REPO / f"graph_inject/configs/methods/{name}.yaml")
        cfg.pop("train", None)
        method = build_method(cfg, 1536, 1.0)
        for s in samples:
            prompt, response = method.rewrite_prompt(s), method.rewrite_response(s)
            if s.task != task:
                assert prompt == s.instruction and response == s.response
                continue
            assert parse_graphqa(prompt).edges == s.graph.edges
            gold = int(s.response.strip().rstrip("."))
            if name == "text-edge-index":
                assert f"e{gold}: (" in prompt and f"e{gold + 1}: (" not in prompt
            elif name == "text-tri-hint":
                counts = [int(c) for c in __import__("re").findall(r"node \d+: (\d+)", prompt.split("Triangles per node:")[1])]
                assert len(counts) == s.graph.nnodes and sum(counts) == 3 * gold
            else:
                assert response.endswith(f"Answer: {gold}.")
                assert final_answer(response, response).strip() == f"{gold}."
    assert final_answer("node 0: 1. Answer: 7.", "x Answer: 7.") == " 7." and final_answer(" 7.", " 7.") == " 7."


def test_graphtoken_query_readout_empty_graph():
    """graphtoken-faithful at a single-prompt prefill of a KQA Pro prompt without facts (0 nodes)."""
    from graph_inject.data import encode_samples, load_samples
    from graph_inject.methods import build_method
    from graph_inject.methods.base import CollateItem
    cfg = OmegaConf.load(REPO / "graph_inject/configs/methods/graphtoken-faithful.yaml")
    method = build_method(cfg, 1536, 1.0)
    rows_ = load_samples(str(DATA / "kqapro/hrm-text-grl/standard/val.jsonl"), "kqapro", 15)[13:]
    for s in encode_samples(rows_, method, PromptTokenizer(tokenizer_checkpoint(f"{HF}/HRM-Text-1B"))):
        graph = method.collate([CollateItem(s.sample, s.features, s.prompt, offset=0)], rng=None)
        out = method.inject(torch.zeros(1, len(s.prompt.ids), 1536), graph)
        assert torch.isfinite(out).all() and out[0, s.prompt.soft_slots].abs().sum() > 0
    assert any(s.graph.nnodes == 0 for s in rows_)


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
    frozen = OmegaConf.load(REPO / f"graph_inject/configs/methods/{method_name}.yaml").get("train", {}).get("regime") == "frozen"
    model = build(ckpt, method_name, lora=not frozen)
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


def test_rel_bias_same_occurrence_is_zero(setup):
    """GTLM's b_SPD(u, u) = 0: random beta moves every graph-token pair except those within one span occurrence;
    two mentions of one node (distance 0, other occurrence) do get the learned bucket-0 bias."""
    from graph_inject.generate import _prefill_batch
    import torch.utils._pytree as pytree
    ckpt = setup
    model = build(ckpt, "rel-bias", lora=False).eval()
    samples, _ = batch_for(ckpt, model, n=1)
    torch.manual_seed(0)
    with torch.no_grad():
        for beta in model.method.betas:
            beta.normal_(0, 2.0)
        cache = model.create_cache(max_batch_size=1, max_seq_len=2048, dtype=torch.bfloat16, device="cuda")
        model(None, _prefill_batch(samples[0], model.method, pytree.tree_map(lambda x: x[0:1], cache), "cuda"))
        # the prefill leaves the context set (decode steps clear it)
        b = model.method.ctx.token_bias(0).cpu()
        model.method.ctx.clear()
    el = torch.from_numpy(samples[0].prompt.elements)
    occ = torch.from_numpy(samples[0].prompt.occurrences)
    graph = (el[:, None] >= 0) & (el[None, :] >= 0)
    same_occ = (occ[:, None] == occ[None, :]) & graph
    same_el_other_occ = (el[:, None] == el[None, :]) & ~same_occ & graph
    assert same_occ.any() and same_el_other_occ.any()
    assert (b[:, same_occ] == 0).all() and (b[:, ~graph] == 0).all()
    assert (b[:, same_el_other_occ] != 0).all() and (b[:, graph & ~same_occ] != 0).float().mean() > 0.99


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
