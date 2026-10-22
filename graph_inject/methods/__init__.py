"""Registry of graph-injection methods (config key `method.kind`)."""
from graph_inject.methods.absolute_pe import AbsolutePEMethod
from graph_inject.methods.base import GraphMethod
from graph_inject.methods.graph_token import GraphTokenMethod
from graph_inject.methods.node_tokens import NodeTokensMethod
from graph_inject.methods.prompt_tuning import PromptTuningMethod
from graph_inject.methods.text import TextMethod

METHODS: dict[str, type[GraphMethod]] = {
    "text": TextMethod,
    "prompt_tuning": PromptTuningMethod,
    "graph_token": GraphTokenMethod,
    "node_tokens": NodeTokensMethod,
    "absolute_pe": AbsolutePEMethod,
}


def build_method(cfg, hidden_size: int, embed_rms: float) -> GraphMethod:
    return METHODS[cfg.kind](cfg, hidden_size, embed_rms)
