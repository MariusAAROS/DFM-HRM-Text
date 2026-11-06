"""Merge a graph_inject LoRA (trainable_ep<N>.pt) into a Mimir HF model dir, for vLLM eval and VeRL (wiki:
mimir-graph-benchmark, E1 / E2 -> E3).

W_hf += (alpha / rank) * B @ A for every LoRA pair, in fp32, stored back in the base dtype (bf16). Native and hub
layouts both pack the projections (gqkv_proj, gate_up_proj), so a LoRA on a native module maps to one hub tensor:
    <H|L>_level.core.layers.<i>.<attn|mlp>.<proj>.lora_[AB]  ->  model.<H|L>_module.layers.<i>.<attn|mlp>.<proj>.weight
Every LoRA pair must map to an existing tensor of the right shape, and only LoRA parameters may be in the file (a
method with its own modules cannot be merged). The out dir symlinks every other file of --base (use an hf_view dir, so
the tokenizer flag is right) and gets a new model.safetensors.

    .venv/bin/python -m graph_bench.merge_lora_hf --base $HF/DFM-Mimir-v1.5-hfview \\
        --lora $RUN/trainable_ep3.pt --config $RUN/config.yaml --out $HF/mimir15-e2-ep3
"""
import argparse
import re
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

LORA = re.compile(r"(?:^|\.)([HL])_level\.core\.layers\.(\d+)\.(attn|mlp)\.(\w+)\.lora_([AB])$")


def merge(base_dir: Path, lora_path: Path, scaling: float, out: Path) -> int:
    weights = load_file(str(base_dir / "model.safetensors"))
    lora = torch.load(lora_path, map_location="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in lora.items():
        m = LORA.search(name)
        if m is None:
            raise ValueError(f"not a LoRA parameter of a layer projection: {name}")
        level, layer, block, proj, ab = m.groups()
        pairs.setdefault(f"model.{level}_module.layers.{layer}.{block}.{proj}.weight", {})[ab] = t
    for key, p in pairs.items():
        if key not in weights or set(p) != {"A", "B"}:
            raise ValueError(f"{key}: no such hub tensor or incomplete pair {sorted(p)}")
        w = weights[key]
        delta = scaling * (p["B"].float() @ p["A"].float())
        if delta.shape != w.shape:
            raise ValueError(f"{key}: delta {tuple(delta.shape)} vs weight {tuple(w.shape)}")
        weights[key] = (w.float() + delta).to(w.dtype)
    out.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        dst = out / f.name
        if f.name not in ("model.safetensors", "MERGED_FROM") and not dst.exists():  # a merged base has its own
            dst.symlink_to(f.resolve())
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "MERGED_FROM").write_text(f"base {base_dir.resolve()}\nlora {lora_path.resolve()}\nscaling {scaling}\n")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="The run's config.yaml (lora.rank, lora.alpha).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("regime") != "lora":
        raise SystemExit(f"regime {cfg.get('regime')}: only LoRA runs can be merged")
    n = merge(args.base, args.lora, cfg["lora"]["alpha"] / cfg["lora"]["rank"], args.out)
    print(f"{args.out}: merged {n} LoRA pairs")


if __name__ == "__main__":
    main()
