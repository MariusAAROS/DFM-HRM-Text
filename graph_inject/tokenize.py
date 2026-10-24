"""Prompt/response tokenization identical to training (data_io) and inference (simple_inference_engine),
plus the graph element c(i) of every prompt token and the placeholder slots for soft tokens.

The token ids come from InferenceCheckpoint.tokenize_prompt, the function eval_graphqa.py uses. Character
offsets come from a second, fast-tokenizer pass over the rendered prompt string; the two id sequences are
asserted equal, so the alignment cannot silently drift from the ids the model sees.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from graph_inject.structure import NO_ELEMENT

PLACEHOLDER_ID = 0  # Embedding rows at soft-token slots are overwritten, so the id itself is irrelevant.


@dataclass
class EncodedPrompt:
    ids: np.ndarray        # [L] instruction ids incl. template tokens and soft-token placeholders
    elements: np.ndarray   # [L] graph element of each token (NO_ELEMENT for template / placeholder tokens)
    soft_slots: np.ndarray # [k] positions of the soft-token placeholders
    occurrences: np.ndarray | None = None  # [L] span occurrence of each token's element (-1: none), if requested


def tokenizer_checkpoint(hf_dir: str):
    """An InferenceCheckpoint with the tokenizer set up as inference_load_checkpoint does, but no model."""
    from simple_inference_engine import InferenceCheckpoint, V1DatasetMeta

    with open(os.path.join(hf_dir, "train_metadata.yaml")) as f:
        info = V1DatasetMeta(**yaml.safe_load(f)).tokenizer_info
    path = info["tokenizer_path"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path) if os.path.isfile(path) else AutoTokenizer.from_pretrained(path)
    if info.get("template_mode") == "jinja_chat_template":
        tokenizer.chat_template = Path(info["chat_template_path"]).read_text()
        tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token = "<pad>", "<bos>", "<turn|>"
    return InferenceCheckpoint(model=None, carry=None, tokenizer=tokenizer, tokenizer_info=info)  # pyright: ignore[reportArgumentType]


class PromptTokenizer:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.tokenizer = ckpt.tokenizer
        self.info = ckpt.tokenizer_info
        self.chat = self.info.get("template_mode") == "jinja_chat_template"

    def _user_messages(self, prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    def render(self, condition: str, prompt: str) -> tuple[str, int]:
        """Prompt string exactly as tokenized, and the character offset where `prompt` starts in it."""
        if self.chat:
            text = self.tokenizer.apply_chat_template(
                self._user_messages(prompt), tokenize=False, add_generation_prompt=True,
                enable_thinking=self.info.get("enable_thinking", False))
            start = text.index(prompt.strip())
            if not prompt.startswith(prompt.strip()):
                raise ValueError("prompt with leading whitespace: character spans would shift")
            return text, start
        prefix = self.info["boq"] + "".join(self.info["condition_mapping"][c] for c in condition.split(","))
        return prefix + prompt + self.info["eoq"], len(prefix)

    def encode_prompt(self, condition: str, prompt: str, char_elements: np.ndarray | None,
                      num_soft_tokens: int = 0, char_spans: np.ndarray | None = None) -> EncodedPrompt:
        ids = np.asarray(self.ckpt.tokenize_prompt(condition, prompt), dtype=np.int64)
        text, start = self.render(condition, prompt)
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if enc["input_ids"] != ids.tolist():
            raise ValueError("offset pass disagrees with tokenize_prompt; alignment would be wrong")

        elements = np.full(len(ids), NO_ELEMENT, dtype=np.int64)
        if char_elements is not None:
            full = np.full(len(text), NO_ELEMENT, dtype=np.int64)
            n = min(len(char_elements), len(text) - start)  # the chat template trims trailing whitespace
            full[start:start + n] = char_elements[:n]
            for i, (s, e) in enumerate(enc["offset_mapping"]):
                elements[i] = token_element(full[s:e])
        occurrences = None
        if char_spans is not None:
            spans = np.full(len(text), NO_ELEMENT, dtype=np.int64)
            spans[start:start + n] = char_spans[:n]
            occurrences = np.full(len(ids), NO_ELEMENT, dtype=np.int64)
            for i, (s, e) in enumerate(enc["offset_mapping"]):
                if elements[i] != NO_ELEMENT:  # the span covering most of the token's characters of its element
                    occurrences[i] = token_element(spans[s:e][full[s:e] == elements[i]])

        # Soft-token slots go right before the prompt content (after <bos>/<|turn>user or <|im_start|><cond>).
        insert_at = next(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > start)
        slots = np.arange(insert_at, insert_at + num_soft_tokens)
        ids = np.insert(ids, insert_at, np.full(num_soft_tokens, PLACEHOLDER_ID))
        elements = np.insert(elements, insert_at, np.full(num_soft_tokens, NO_ELEMENT))
        if occurrences is not None:
            occurrences = np.insert(occurrences, insert_at, np.full(num_soft_tokens, NO_ELEMENT))
        return EncodedPrompt(ids=ids, elements=elements, soft_slots=slots, occurrences=occurrences)

    def encode_response(self, condition: str, prompt: str, response: str) -> np.ndarray:
        """Response ids incl. stop token(s), as data_io supervises them."""
        if self.chat:
            msgs = self._user_messages(prompt)
            prompt_text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.info.get("enable_thinking", False))
            full = self.tokenizer.apply_chat_template(
                msgs + [{"role": "assistant", "content": response}], tokenize=False,
                enable_thinking=self.info.get("enable_thinking", False))
            if not full.startswith(prompt_text):
                raise ValueError("chat template renders the prompt differently with an assistant turn")
            text = full[len(prompt_text):]
            return np.asarray(self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
        ids = self.tokenizer(response, add_special_tokens=False)["input_ids"] + [self.ckpt.stop_token_id()]
        return np.asarray(ids, dtype=np.int64)

    def gold_answer_ids(self, response: str) -> list[int]:
        """Answer ids without the stop token, as eval_graphqa.py's tf_em compares them."""
        return self.tokenizer(response.strip() if self.chat else response, add_special_tokens=False)["input_ids"]


def token_element(chars: np.ndarray) -> int:
    """The graph element covering most of the token's characters; ties go to the node (nodes have the lowest ids).

    E.g. HRM's "Ġ" + "1" + "2" of " 12" in "(3, 12)": "Ġ" is the edge, "1" and "2" are node 12.
    """
    if chars.size == 0:
        return NO_ELEMENT
    vals, counts = np.unique(chars[chars != NO_ELEMENT], return_counts=True)
    if vals.size == 0:
        return NO_ELEMENT
    return int(vals[np.argmax(counts)])
