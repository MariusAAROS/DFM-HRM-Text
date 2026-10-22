"""LoRA for the HRM-Text linear layers.

peft only wraps torch.nn.Linear (and a few HF layer types); HRM's projections are models.layers.LinearInit,
so the 30 lines below replace it. Initialization follows peft's default: A ~ kaiming_uniform(a=sqrt(5)),
B = 0, so a freshly wrapped model computes exactly the base function. The base weight stays frozen (and
usually bf16); A and B are fp32 and are cast to the activation dtype in forward.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.layers import LinearInit

DEFAULT_TARGETS = ("gqkv_proj", "o_proj", "gate_up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: LinearInit, rank: int, alpha: float):
        super().__init__()
        self.base = base
        out_features, in_features = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        delta = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + self.scaling * delta


def apply_lora(model: nn.Module, rank: int, alpha: float, targets=DEFAULT_TARGETS) -> int:
    """Wrap every LinearInit attribute named in `targets` (in place). Returns the number of wrapped layers."""
    wrapped = 0
    for module in list(model.modules()):
        for name in targets:
            layer = getattr(module, name, None)
            if isinstance(layer, LinearInit):
                setattr(module, name, LoRALinear(layer, rank, alpha))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(f"no LinearInit named {targets} found")
    return wrapped
