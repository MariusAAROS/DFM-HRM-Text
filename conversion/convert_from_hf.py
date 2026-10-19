"""Import a Hugging Face HRM-Text checkpoint (e.g. sapientinc/HRM-Text-1B) into this stack.

Inverse of conversion/convert_to_hf.py (vllm_packed layout). Writes a checkpoint directory that both
consumers of native checkpoints read:

    <out_dir>/all_config.yaml, train_metadata.yaml   composed training config + dataset metadata
    <out_dir>/unsharded_step_0.pt                    {"model": native state dict, "optim": {"state": {}}}
    <out_dir>/checkpoint_state_step_0.json           carry_policy=none
    <out_dir>/hf_config.json                         the source config.json

- Fine-tune: `pretrain.py ... init_weights_path=<out_dir>/unsharded_step_0.pt` (weights only; fresh
  optimizer, step and data position).
- Zero-shot eval: `scripts/graphs/eval_graphqa.py --ckpt_path <out_dir> --ckpt_tag step_0`.

The architecture comes from the training config (--config-name + overrides), so a fine-tune and its
zero-shot baseline share one source of truth; the script refuses to write when that config disagrees with
the HF config.json (layers, widths, cycles, embedding scale, ...).

Usage:
    .venv/bin/python -m conversion.convert_from_hf --hf_model sapientinc/HRM-Text-1B \\
        --out_dir /work/dfm/marius-ortega/dfm-hrm-graphs/hf/HRM-Text-1B \\
        --config-name cfg_graphqa_hrm1b data.path=data/graphs/sampled/graphqa_grl_e20
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_new import V1Dataset  # noqa: E402
from pretrain import PretrainConfig, load_model_class  # noqa: E402
from conversion.convert_to_hf import _compute_l_bp_cycles, per_stack_layers  # noqa: E402
from models.transformer import TransformerConfig  # noqa: E402

HRM_ARCH = "baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel"


def resolve_hf_dir(hf_model: str, revision: str | None) -> Path:
    if Path(hf_model).is_dir():
        return Path(hf_model)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(hf_model, revision=revision, allow_patterns=["*.json", "*.safetensors"]))


def load_hf_state(hf_dir: Path) -> dict[str, torch.Tensor]:
    index = hf_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for name in files:
        state.update(load_file(hf_dir / name))
    return state


def native_key(key: str) -> str:
    if key == "model.embed_tokens.weight":
        return "embed_tokens.embedding_weight"
    if key == "model.z_L_init":
        return "model.zL_init"
    key = re.sub(r"^model\.H_module\.layers\.", "model.H_level.core.layers.", key)
    return re.sub(r"^model\.L_module\.layers\.", "model.L_level.core.layers.", key)


def check_arch(hf_cfg: dict, cfg: dict) -> None:
    """Fail unless the native config builds the same network as the HF config.json."""
    if hf_cfg.get("model_type") != "hrm_text":
        raise ValueError(f"Unsupported model_type {hf_cfg.get('model_type')!r}; expected 'hrm_text'")
    if cfg["name"] != HRM_ARCH:
        raise ValueError(f"arch.name must be {HRM_ARCH} to load HRM-Text weights, got {cfg['name']}")
    hidden = cfg["hidden_size"]
    transformer = TransformerConfig(**cfg)
    # Older config.json files store the per-stack depth in num_hidden_layers; transformers >= 5 saves it as
    # num_layers_per_stack and rewrites num_hidden_layers to the KV-cache slot count.
    # They also omit num_key_value_heads (= num_attention_heads) and move rope_theta into rope_parameters.
    hf_cfg = hf_cfg | {
        "num_layers_per_stack": hf_cfg.get("num_layers_per_stack") or hf_cfg.get("num_hidden_layers"),
        "num_key_value_heads": hf_cfg.get("num_key_value_heads") or hf_cfg.get("num_attention_heads"),
        "rope_theta": hf_cfg.get("rope_theta") or hf_cfg.get("rope_parameters", {}).get("rope_theta"),
    }
    expected = {
        "num_layers_per_stack": per_stack_layers(cfg),
        "hidden_size": hidden,
        "num_attention_heads": cfg["num_heads"],
        "num_key_value_heads": cfg["num_heads"],
        "head_dim": hidden // cfg["num_heads"],
        "intermediate_size": transformer.intermediate_size,
        "H_cycles": cfg["H_cycles"],
        "L_cycles": cfg["L_cycles"],
        "rms_norm_eps": cfg["norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "embedding_scale": 1.0 / transformer.init_config.in_std,
        "vocab_size": cfg["vocab_size"],
    }
    mismatches = []
    for key, value in expected.items():
        got = hf_cfg.get(key)
        same = math.isclose(got, value, rel_tol=1e-6) if isinstance(value, float) and got is not None else got == value
        if not same:
            mismatches.append(f"  {key}: HF={got!r} native={value!r}")
    for key in ("L_sliding_window", "H_sliding_window"):
        if hf_cfg.get(key) is not None:
            mismatches.append(f"  {key}: HF={hf_cfg[key]!r} (sliding windows are not supported here)")
    if hf_cfg.get("rope_parameters", {}).get("rope_type", "default") != "default":
        mismatches.append(f"  rope_parameters: HF={hf_cfg['rope_parameters']!r} (only default RoPE is supported)")
    if cfg["norm_type"] != "pre" or cfg.get("pos_emb_type", "rope") != "rope" or cfg.get("H_override"):
        mismatches.append("  norm_type/pos_emb_type/H_override: HRM-Text uses pre-norm, RoPE, symmetric H/L")
    if mismatches:
        raise ValueError("Training config does not match the HF checkpoint:\n" + "\n".join(mismatches))
    # Gradient budget: informational, it does not change the weights.
    if "L_bp_cycles" in hf_cfg and list(hf_cfg["L_bp_cycles"]) != _compute_l_bp_cycles(cfg):
        print(f"[import] note: HF L_bp_cycles={hf_cfg['L_bp_cycles']} vs config {_compute_l_bp_cycles(cfg)} "
              "(bp_max_steps / H_bp_steps / L_bp_steps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf_model", required=True, help="HF repo id or local directory.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--config-name", dest="config_name", default="cfg_graphqa_hrm1b")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data.path=...")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        hydra_cfg = compose(config_name=args.config_name, overrides=args.overrides)
    config = PretrainConfig(**OmegaConf.to_container(hydra_cfg, resolve=True))  # type: ignore[arg-type]
    metadata = V1Dataset._load_metadata(SimpleNamespace(config=SimpleNamespace(dataset_path=config.data.path)))  # type: ignore[arg-type]
    model_cfg = config.arch.model_dump() | metadata.model_dump() | config.data.model_dump()

    hf_dir = resolve_hf_dir(args.hf_model, args.revision)
    hf_cfg = json.loads((hf_dir / "config.json").read_text())
    check_arch(hf_cfg, model_cfg)

    with torch.device("meta"):
        model = load_model_class(config.arch.head)(load_model_class(config.arch.name)(model_cfg), model_cfg)
    expected = model.state_dict()

    state = {native_key(k): v.contiguous() for k, v in load_hf_state(hf_dir).items()}
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    shapes = [f"  {k}: HF {tuple(state[k].shape)} vs native {tuple(expected[k].shape)}"
              for k in sorted(set(state) & set(expected)) if state[k].shape != expected[k].shape]
    if missing or unexpected or shapes:
        raise RuntimeError("HF tensors do not map onto the native model:\n"
                           + "\n".join([f"  missing {k}" for k in missing] + [f"  unexpected {k}" for k in unexpected] + shapes))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "optim": {"state": {}}}, args.out_dir / "unsharded_step_0.pt")
    (args.out_dir / "checkpoint_state_step_0.json").write_text(json.dumps(
        {"step": 0, "carry_policy": "none", "source": args.hf_model, "revision": hf_dir.name}, indent=2) + "\n")
    (args.out_dir / "hf_config.json").write_text((hf_dir / "config.json").read_text())
    with open(args.out_dir / "all_config.yaml", "wt") as f:
        yaml.dump(config.model_dump(), f)
    with open(args.out_dir / "train_metadata.yaml", "wt") as f:
        yaml.dump(metadata.model_dump(), f)
    n_params = sum(v.numel() for v in state.values())
    print(f"[import] {args.hf_model} ({hf_dir.name}) -> {args.out_dir}: {len(state)} tensors, {n_params / 1e9:.3f}B params")
    print(f"[import] fine-tune with init_weights_path={os.path.abspath(args.out_dir / 'unsharded_step_0.pt')}")


if __name__ == "__main__":
    main()
