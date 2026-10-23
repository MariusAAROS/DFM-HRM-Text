"""GraphQA / MetaQA / KQA Pro samples -> encoded prompts -> packed PrefixLM training batches (same layout as dataset_new.V1Dataset).

Unlike the data_io pipeline, tokenization happens here (once per split), because methods may rewrite the
prompt, insert soft-token slots, or need the token -> graph-element alignment. The ids are identical to
data_io's (tests/test_graph_inject.py checks all train rows against data/graphs/tokenized/).
"""
import json
from dataclasses import dataclass

import numpy as np
import torch

from graph_inject.methods.base import CollateItem, GraphMethod, GraphQASample
from graph_inject.kg_structure import parse_kg_facts
from graph_inject.structure import PromptGraph, parse_graphqa
from graph_inject.tokenize import EncodedPrompt, PromptTokenizer
from models.common import IGNORE_LABEL_ID, wrap_tensor, prepare_prefixlm_batch
from models.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars


@dataclass
class EncodedSample:
    sample: GraphQASample
    prompt: EncodedPrompt
    response_ids: np.ndarray   # incl. stop token(s), as supervised in training
    gold_ids: list[int]        # answer ids for tf_em
    features: dict[str, np.ndarray]

    @property
    def train_length(self) -> int:
        return len(self.prompt.ids) + len(self.response_ids) - 1


def parse_graph(text: str, dataset: str) -> PromptGraph:
    return parse_graphqa(text) if dataset == "graphqa" else parse_kg_facts(text, dataset)


def load_samples(path: str, dataset: str = "graphqa", limit: int | None = None) -> list[GraphQASample]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:limit]
    return [GraphQASample(idx=i, condition=r.get("condition", "direct"), instruction=r["instruction"],
                          response=r["response"], task=r.get("task", ""), graph=parse_graph(r["instruction"], dataset))
            for i, r in enumerate(rows)]


def stratified_subset(samples: list, n: int, seed: int = 0) -> list:
    """n samples drawn without replacement, per task in proportion to the task's share (largest remainders), in
    their original order. The seed is fixed, not the run's: every run and every epoch scores the same subset."""
    if n >= len(samples):
        return samples
    by_task: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_task.setdefault(s.sample.task, []).append(i)
    tasks = sorted(by_task)
    quota = {t: n * len(by_task[t]) / len(samples) for t in tasks}
    take = {t: int(quota[t]) for t in tasks}
    for t in sorted(tasks, key=lambda t: quota[t] - take[t], reverse=True)[:n - sum(take.values())]:
        take[t] += 1
    rng = np.random.default_rng(seed)
    keep = sorted(i for t in tasks for i in rng.permutation(by_task[t])[:take[t]])
    return [samples[i] for i in keep]


def encode_samples(samples: list[GraphQASample], method: GraphMethod, tokenizer: PromptTokenizer) -> list[EncodedSample]:
    out = []
    for s in samples:
        prompt = method.rewrite_prompt(s)
        char_elements = None
        if method.needs_elements:
            if prompt != s.instruction:
                raise ValueError("element alignment needs the original prompt")
            char_elements = s.graph.char_elements(len(prompt))
        out.append(EncodedSample(
            sample=s,
            prompt=tokenizer.encode_prompt(s.condition, prompt, char_elements, method.num_soft_tokens(s)),
            response_ids=tokenizer.encode_response(s.condition, prompt, s.response),
            gold_ids=tokenizer.gold_answer_ids(s.response),
            features=method.featurize(s)))
    return out


def pack_batches(samples: list[EncodedSample], max_tokens: int, rng: np.random.Generator) -> list[list[int]]:
    """Shuffle, then fill batches of at most max_tokens in order."""
    batches, current, used = [], [], 0
    for i in rng.permutation(len(samples)):
        n = samples[i].train_length
        if n > max_tokens:
            raise ValueError(f"sample {i} has {n} tokens > {max_tokens}")
        if used + n > max_tokens:
            batches.append(current)
            current, used = [], 0
        current.append(int(i))
        used += n
    if current:
        batches.append(current)
    return batches


def train_batch(samples: list[EncodedSample], method: GraphMethod, max_tokens: int,
                rng: np.random.Generator | None) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Packed batch exactly as dataset_new.V1Dataset._load_batch builds it (target_only), plus graph keys."""
    inputs, labels, position_ids, prefix_lens, causal_lens, items = [], [], [], [], [], []
    offset = 0
    for s in samples:
        inst, resp = s.prompt.ids, s.response_ids
        inputs += [inst, resp[:-1]]
        labels += [np.full(len(inst) - 1, IGNORE_LABEL_ID), resp]
        position_ids += [np.arange(len(inst)), np.arange(len(inst), len(inst) + len(resp) - 1)]
        prefix_lens.append(len(inst))
        causal_lens.append(len(resp) - 1)
        items.append(CollateItem(sample=s.sample, features=s.features, prompt=s.prompt, offset=offset,
                                 length=len(inst) + len(resp) - 1))
        offset += len(inst) + len(resp) - 1
    batch = {k: np.concatenate(v).astype(np.int32) for k, v in
             (("inputs", inputs), ("labels", labels), ("position_ids", position_ids))}
    pad = max_tokens - batch["inputs"].shape[0]
    for k, fill in (("inputs", 0), ("labels", IGNORE_LABEL_ID), ("position_ids", 0)):
        batch[k] = np.pad(batch[k], (0, pad), constant_values=fill)
    tensors, scalars = compute_aux_seq_tensors_scalars(np.array(prefix_lens, dtype=np.int32),
                                                       np.array(causal_lens, dtype=np.int32), max_tokens)
    out = {k: torch.from_numpy(v) for k, v in (batch | tensors).items()}
    return out | method.collate(items, rng), scalars


def to_device_prefixlm(batch: dict[str, torch.Tensor], scalars: dict[str, int], device: torch.device) -> dict:
    """Move to the GPU and add the PrefixLM routing tensors, as pretrain.py's train loop does."""
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return prepare_prefixlm_batch(batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in scalars.items()})
