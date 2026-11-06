"""Merge a VeRL GRPO LoRA checkpoint into a Mimir HF (hub-layout) dir, for vLLM eval (wiki: mimir-graph-benchmark, E3/E4).

A VeRL FSDP checkpoint (<ckpt>/global_step_N/actor/model_world_size_1_rank_0.pt) holds the whole peft model in fp32,
in transformers' unpacked layout: base_model.model.model.<H|L>_module.layers.<i>.{self_attn,mlp}.<proj>.{base_layer,
lora_A.default,lora_B.default}.weight. The hub file packs the projections the way transformers' hrm_text conversion
mapping chunks them (dim 0, in this order):
    attn.gqkv_proj = [self_attn.gate_proj, q_proj, k_proj, v_proj]    mlp.gate_up_proj = [mlp.gate_proj, up_proj]
    attn.o_proj = self_attn.o_proj                                     mlp.down_proj = mlp.down_proj
Merged weight = base_layer + (lora_alpha / r) * B @ A (fp32, lora_train_meta.json), stored in the hub dtype (bf16).

Checks: the base_layer tensors are frozen copies of --base, so each packed concat of them must equal the --base tensor
exactly (this pins the packing order and the base); every other checkpoint tensor (embeddings, norms, z inits) must
equal its --base tensor too; every LoRA pair must be complete. The out dir symlinks every other file of --base (an
hf_view or merged dir, so the tokenizer flag is right) and gets a new model.safetensors.

    python -m graph_bench.rl.merge_verl_lora --ckpt $ROOT/e3_grpo_e2/ckpt/global_step_200/actor \\
        --base $HF/mimir15-sft-cot-ep3 --out $HF/mimir15-e3-grpo-step200
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PREFIX = "base_model.model."
LAYER = re.compile(r"^model\.([HL])_module\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.(base_layer|lora_A\.default|lora_B\.default)\.weight$")
PACK = {  # (block, hub proj) -> transformers projections, in chunk order
    ("self_attn", "gqkv_proj"): ["gate_proj", "q_proj", "k_proj", "v_proj"],
    ("self_attn", "o_proj"): ["o_proj"],
    ("mlp", "gate_up_proj"): ["gate_proj", "up_proj"],
    ("mlp", "down_proj"): ["down_proj"],
}
HUB_BLOCK = {"self_attn": "attn", "mlp": "mlp"}


def merge(ckpt: Path, base_dir: Path, out: Path) -> dict:
    meta = json.loads((ckpt / "lora_train_meta.json").read_text())
    scaling = meta["lora_alpha"] / meta["r"]
    sd = torch.load(ckpt / "model_world_size_1_rank_0.pt", mmap=True, weights_only=False, map_location="cpu")
    hub = load_file(str(base_dir / "model.safetensors"))

    parts: dict[tuple, dict[str, torch.Tensor]] = {}  # (level, layer, block, proj) -> {base_layer, A, B}
    other = 0
    for name, t in sd.items():
        if not name.startswith(PREFIX):
            raise ValueError(f"unexpected key {name}")
        key = name[len(PREFIX):]
        m = LAYER.match(key)
        if m is None:
            if "lora_" in key or "base_layer" in key:
                raise ValueError(f"LoRA tensor outside a layer projection: {key}")
            if key not in hub:
                raise ValueError(f"{key}: not in {base_dir}/model.safetensors")
            if not torch.equal(t.to(hub[key].dtype), hub[key]):
                raise ValueError(f"{key}: differs from the base (wrong --base?)")
            other += 1
            continue
        level, layer, block, proj, kind = m.groups()
        parts.setdefault((level, layer, block, proj), {})[kind.split(".")[0]] = t

    merged_tensors, pairs = 0, 0
    for (level, layer, block, hub_proj), projs in [((lv, ly, b, hp), ps) for (lv, ly) in
                                                   sorted({(k[0], k[1]) for k in parts}) for (b, hp), ps in PACK.items()]:
        hub_key = f"model.{level}_module.layers.{layer}.{HUB_BLOCK[block]}.{hub_proj}.weight"
        base_chunks, new_chunks = [], []
        for proj in projs:
            p = parts.get((level, layer, block, proj))
            if p is None or "base_layer" not in p:
                raise ValueError(f"{level}{layer}.{block}.{proj}: no base_layer in the checkpoint")
            w = p["base_layer"].float()
            base_chunks.append(w)
            if "lora_A" in p or "lora_B" in p:
                if set(p) != {"base_layer", "lora_A", "lora_B"}:
                    raise ValueError(f"{level}{layer}.{block}.{proj}: incomplete LoRA pair {sorted(p)}")
                w = w + scaling * (p["lora_B"].float() @ p["lora_A"].float())
                pairs += 1
            new_chunks.append(w)
        ref = hub[hub_key]
        if not torch.equal(torch.cat(base_chunks).to(ref.dtype), ref):
            raise ValueError(f"{hub_key}: packed base_layer {projs} differs from the base tensor (packing or --base)")
        hub[hub_key] = torch.cat(new_chunks).to(ref.dtype)
        merged_tensors += 1
    n_proj = sum(len(v) for v in PACK.values())
    if pairs == 0 or len(parts) != merged_tensors // len(PACK) * n_proj:
        raise ValueError(f"{len(parts)} projections in the checkpoint, {merged_tensors} hub tensors rebuilt, {pairs} pairs")

    out.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        dst = out / f.name
        if f.name not in ("model.safetensors", "MERGED_FROM") and not dst.exists():
            dst.symlink_to(f.resolve())
    save_file(hub, str(out / "model.safetensors"), metadata={"format": "pt"})
    info = {"base": str(base_dir.resolve()), "ckpt": str(ckpt.resolve()), "scaling": scaling, "lora_pairs": pairs,
            "hub_tensors_rebuilt": merged_tensors, "other_tensors_checked": other}
    (out / "MERGED_FROM").write_text("".join(f"{k} {v}\n" for k, v in info.items()))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="<ckpt>/global_step_N/actor")
    ap.add_argument("--base", type=Path, required=True, help="The HF dir GRPO started from (MODEL_PATH).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(merge(args.ckpt, args.base, args.out)))


if __name__ == "__main__":
    main()
