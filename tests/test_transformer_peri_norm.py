import torch
import torch.nn.functional as F
from torch import nn

from models.transformer import Transformer, TransformerBlock, TransformerConfig


class Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x, **_kwargs):
        return x * self.factor


CONFIG = dict(max_seq_len=8, n_layers=1, hidden_size=8, num_heads=1, expansion=1, init_type="lecun_normal",
              norm_eps=1e-6, pos_emb_type="none")


def rms(x):
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


def test_peri_norm_normalises_sublayer_inputs_and_outputs():
    block = TransformerBlock(TransformerConfig(**CONFIG, norm_type="peri"))
    block.attn, block.mlp = Scale(30.0), Scale(-7.0)
    x = torch.randn(5, 8) * 4

    h = x + rms(rms(x) * 30.0)
    expected = h + rms(rms(h) * -7.0)
    torch.testing.assert_close(block(x), expected)


def test_peri_norm_keeps_final_norm():
    model = Transformer(TransformerConfig(**CONFIG, norm_type="peri"))
    model.layers = nn.ModuleList([])
    x = torch.randn(5, 8) * 4
    torch.testing.assert_close(model(x), rms(x))
