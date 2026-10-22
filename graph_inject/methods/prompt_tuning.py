"""Static soft prompt (no graph information): the capacity control for the graph-conditioned soft prompts.

P-Tuning (Liu et al., 2022) with the MLP reparametrization, using peft's own PromptEncoder module:
P = MLP(Z), Z an embedding table of k virtual tokens. The same k vectors are written into the k
placeholder slots of every prompt.
"""
from peft import PromptEncoderConfig
from peft.tuners.p_tuning import PromptEncoder
import torch
from torch import Tensor

from graph_inject.methods.base import GraphMethod, ToEmbeddingScale, soft_positions, write_soft_tokens


class PromptTuningMethod(GraphMethod):
    def __init__(self, cfg, hidden_size: int, embed_rms: float):
        super().__init__(cfg, hidden_size, embed_rms)
        self.k = cfg.num_tokens
        self.encoder = PromptEncoder(PromptEncoderConfig(
            num_virtual_tokens=self.k, token_dim=hidden_size, encoder_hidden_size=cfg.encoder_hidden_size,
            encoder_reparameterization_type="MLP", num_transformer_submodules=1))
        self.scale = ToEmbeddingScale(embed_rms)

    def num_soft_tokens(self, sample) -> int:
        return self.k

    def collate(self, items, rng):
        return {"graph_soft_pos": soft_positions(items)}

    def inject(self, embeddings: Tensor, graph: dict[str, Tensor]) -> Tensor:
        if "graph_soft_pos" not in graph:
            return embeddings
        prompt = self.scale(self.encoder(torch.arange(self.k, device=embeddings.device)))
        n_prompts = graph["graph_soft_pos"].numel() // self.k
        return write_soft_tokens(embeddings, graph["graph_soft_pos"], prompt.repeat(n_prompts, 1))
