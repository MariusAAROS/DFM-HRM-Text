"""GraphInjectedLM: a pretrained LMHead(HRM) with a graph-injection method between embedding and HRM.

forward mirrors models/lm_head.py LMHead.forward (embed -> HRM -> lm_head -> CE loss summed and divided by
the number of supervised tokens; logits only at supervised positions), minus goldfish loss and distributed
reductions, plus one step: `method.inject` on the token embeddings. Batch keys starting with "graph_" belong to the method and are
not passed to the HRM (every other key reaches the attention kernel as a kwarg).
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_inject.methods.base import GraphMethod
from models.common import IGNORE_LABEL_ID

GRAPH_PREFIX = "graph_"


def split_graph_keys(batch: dict) -> tuple[dict, dict]:
    graph = {k: v for k, v in batch.items() if k.startswith(GRAPH_PREFIX)}
    rest = {k: v for k, v in batch.items() if not k.startswith(GRAPH_PREFIX)}
    return rest, graph


class GraphInjectedLM(nn.Module):
    def __init__(self, base: nn.Module, method: GraphMethod):
        super().__init__()
        self.base = base        # models.lm_head.LMHead
        self.method = method
        self.create_cache = base.create_cache

    def forward(self, carry, batch: dict[str, Tensor], **kwargs):
        batch, graph = split_graph_keys(batch)
        embeddings = self.method.inject(self.base.embed_tokens(batch["inputs"]), graph)
        seq_info = {k: v for k, v in batch.items() if k not in ("inputs", "labels")}
        new_carry, hidden = self.base.model(carry, embeddings, **seq_info, **kwargs)
        if "labels" not in batch:
            return new_carry, self.base.lm_head(hidden)

        # Logits only at supervised positions (the response): same loss, and a [T, 262k] fp32 tensor less for Mimir.
        labels = batch["labels"].long()
        valid = labels != IGNORE_LABEL_ID
        logits, labels = self.base.lm_head(hidden[valid]), labels[valid]
        loss = F.cross_entropy(logits.float(), labels, reduction="sum") / valid.sum()
        with torch.no_grad():
            metrics = {"loss": loss.detach(), "token_accuracy": (logits.argmax(-1) == labels).float().mean()}
        return new_carry, loss, metrics


def embedding_rms(base: nn.Module) -> float:
    """RMS of the scaled token embeddings (the scale soft tokens are normalized to)."""
    emb = base.embed_tokens
    with torch.no_grad():
        w = emb.embedding_weight.float()
        return float((w.pow(2).mean(-1).sqrt().mean() * emb.scale).item())
