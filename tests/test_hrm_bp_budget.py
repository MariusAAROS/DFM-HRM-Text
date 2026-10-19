import pytest
import torch
from torch import nn

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.baselines.trm_nocarry import TinyRecursiveModel
from models.common import resolve_bp_steps
from models.transformer import TransformerBlock


class RecordingBlock(TransformerBlock):
    """Stand-in Transformer block: counts applications that build an autograd graph."""

    def __init__(self, weight: float, dtype: torch.dtype) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(weight, dtype=dtype))
        self.inputs: list[torch.Tensor] = []
        self.grad_calls = 0

    def forward(self, hidden_states, **_kwargs):
        self.inputs.append(hidden_states.detach().clone())
        out = torch.tanh(hidden_states * self.weight)
        self.grad_calls += int(out.requires_grad)
        return out


BASE = {
    "max_seq_len": 8,
    "n_layers": 2,
    "hidden_size": 4,
    "num_heads": 1,
    "expansion": 1,
    "init_type": "lecun_normal",
    "norm_type": "pre",
    "norm_eps": 1e-6,
    "pos_emb_type": "none",
    "half_layers": True,
    "bp_min_steps": 2,
    "bp_max_steps": 5,
}


def make_hrm(dtype: torch.dtype = torch.float32, **overrides) -> HierarchicalReasoningModel:
    model = HierarchicalReasoningModel(BASE | {"H_cycles": 3, "L_cycles": 4, "fwd_bwd_dtype": str(dtype).removeprefix("torch.")} | overrides)
    model.H_level.core.layers = nn.ModuleList([RecordingBlock(0.7, dtype)])
    model.L_level.core.layers = nn.ModuleList([RecordingBlock(0.6, dtype)])
    model.train()
    return model


def make_trm(**overrides) -> TinyRecursiveModel:
    model = TinyRecursiveModel(BASE | {"H_cycles": 3, "L_cycles": 4, "H_bp_steps": 1, "L_bp_steps": 1} | overrides)
    model.L_level.core.layers = nn.ModuleList([RecordingBlock(0.6, torch.float32)])
    model.train()
    return model


def grad_blocks(model) -> int:
    blocks = {id(level.core.layers[0]): level.core.layers[0] for level in (getattr(model, "H_level", None), model.L_level) if level is not None}
    return sum(block.grad_calls for block in blocks.values())


def test_legacy_warmup_formula_preserved():
    for H in range(1, 9):
        for L in range(1, 13):
            model = HierarchicalReasoningModel(BASE | {"H_cycles": H, "L_cycles": L})
            for bp_steps in range(0, 8):
                legacy_H = min(H, bp_steps - 1)
                assert model._resolve_bp_steps(bp_steps) == (legacy_H, bp_steps - legacy_H)


def test_resolve_bp_steps_sentinel_full_and_clamp():
    for H, L in ((1, 1), (2, 3), (4, 6), (6, 12)):
        assert resolve_bp_steps(H, L, -1, -1) == (H, H * L)
        assert resolve_bp_steps(H, L, 0, 0, full_backprop=True) == (H, H * L)
        assert resolve_bp_steps(H, L, 99, 99) == (H, H * L)
        assert resolve_bp_steps(H, L, 1, 2) == (1, min(2, H * L))


def test_hrm_full_backprop_builds_graph_through_every_block():
    H, L = 3, 4
    for overrides in ({"full_backprop": True}, {"H_bp_steps": -1, "L_bp_steps": -1}):
        model = make_hrm(**overrides)
        model(None, torch.randn(8, 4, requires_grad=True), bp_steps=2)
        assert grad_blocks(model) == H * (L + 1)


def test_hrm_full_backprop_disables_warmup():
    model = make_hrm(full_backprop=True, bp_warmup_ratio=0.2)
    assert model.bp_warmup_ratio == 0.0


def test_hrm_truncated_budget_counts():
    model = make_hrm()
    H_bp, L_bp = model._resolve_bp_steps(5)
    model(None, torch.randn(8, 4, requires_grad=True), bp_steps=5)
    assert (H_bp, L_bp) == (3, 2)
    assert grad_blocks(model) == H_bp + L_bp


def test_hrm_explicit_budget_ignores_schedule():
    model = make_hrm(H_bp_steps=1, L_bp_steps=4)
    model(None, torch.randn(8, 4, requires_grad=True), bp_steps=2)
    assert grad_blocks(model) == 5


@pytest.mark.parametrize("overrides", [
    {"H_bp_steps": -1},
    {"L_bp_steps": 3},
    {"full_backprop": True, "H_bp_steps": 1, "L_bp_steps": 1},
])
def test_hrm_ambiguous_budgets_rejected(overrides):
    with pytest.raises(ValueError):
        HierarchicalReasoningModel(BASE | {"H_cycles": 2, "L_cycles": 3} | overrides)


def test_hrm_full_backprop_gradient_is_exact_truncated_is_not():
    def grad_error(model) -> float:
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        _, out = model(None, x, bp_steps=2)
        # Under truncation the input may not reach the graph at all; its autograd gradient is then zero.
        (autograd,) = torch.autograd.grad(out.sum(), x, allow_unused=True, materialize_grads=True)
        numeric = torch.zeros_like(x)
        eps = 1e-6
        with torch.no_grad():
            for idx in range(x.numel()):
                delta = torch.zeros_like(x).view(-1)
                delta[idx] = eps
                delta = delta.view_as(x)
                plus = model(None, x + delta, bp_steps=2)[1].sum()
                minus = model(None, x - delta, bp_steps=2)[1].sum()
                numeric.view(-1)[idx] = (plus - minus) / (2 * eps)
        return float((autograd - numeric).abs().max())

    assert grad_error(make_hrm(torch.float64, full_backprop=True)) < 1e-6
    assert grad_error(make_hrm(torch.float64)) > 1e-4


def test_hrm_inject_x_feeds_embedding_to_every_l_step():
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    plain = make_hrm()
    injected = make_hrm(inject_x=True)
    injected.load_state_dict(plain.state_dict())
    with torch.no_grad():
        plain(None, x, bp_steps=2)
        injected(None, x, bp_steps=2)

    plain_l = plain.L_level.core.layers[0].inputs
    injected_l = injected.L_level.core.layers[0].inputs
    # First L step: z_L = zL_init, z_H = x, so the injected input is zL_init + 2x.
    torch.testing.assert_close(plain_l[0], plain.zL_init + x)
    torch.testing.assert_close(injected_l[0], injected.zL_init + 2 * x)


def test_trm_budgets():
    H, L = 3, 4
    full = make_trm(full_backprop=True)
    assert (full.H_bp_steps, full.L_bp_steps) == (H, H * L)
    full(None, torch.randn(8, 4, requires_grad=True))
    assert grad_blocks(full) == H * (L + 1)

    every = make_trm(H_bp_steps=-1, L_bp_steps=-1)
    assert (every.H_bp_steps, every.L_bp_steps) == (H, H * L)

    trunc = make_trm()
    trunc(None, torch.randn(8, 4, requires_grad=True))
    assert grad_blocks(trunc) == 2


def test_auto_module_lr_counts_follow_fixed_budgets():
    from types import SimpleNamespace

    from models.module_learning_rates import configured_module_rates

    def rates(**arch):
        config = SimpleNamespace(lr=1.0, lr_auto=True, lr_embeddings=None, lr_head=None, lr_h=None, lr_l=None,
                                 arch={"name": "baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel",
                                       "H_cycles": 3, "L_cycles": 4} | arch)
        return configured_module_rates(config, bp_steps=5)

    assert rates() == pytest.approx(dict(embeddings=1.0, head=1.0, h=1 / 3, l=1 / 2))
    assert rates(full_backprop=True) == pytest.approx(dict(embeddings=1.0, head=1.0, h=1 / 3, l=1 / 12))
    assert rates(H_bp_steps=1, L_bp_steps=-1) == pytest.approx(dict(embeddings=1.0, head=1.0, h=1.0, l=1 / 12))


def test_logged_bp_steps_counts_blocks_with_gradient():
    from types import SimpleNamespace

    state = SimpleNamespace(step=0, total_steps=100)
    assert make_hrm(H_cycles=3, L_cycles=6, full_backprop=True).compute_train_extra_args(state) == {"bp_steps": 3 * (6 + 1)}
    assert make_hrm(H_cycles=3, L_cycles=6, H_bp_steps=1, L_bp_steps=4).compute_train_extra_args(state) == {"bp_steps": 5}
    assert make_hrm(H_cycles=3, L_cycles=6, bp_warmup_ratio=0.0).compute_train_extra_args(state) == {"bp_steps": 5}
