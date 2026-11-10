"""LoRA SFT of a Hugging Face baseline on the graph_bench SFT mixture (wiki: mimir-graph-benchmark, E1 / E2 for the
baselines). The Mimir arms train with graph_inject (native HRM); this is the same recipe for transformers models:

  - data: the same rows (data/graphs/bench/sft/train_{answer,cot}.jsonl: {instruction, response}).
  - prompt: the user turn is the row's instruction, rendered with the model's chat template (enable_thinking=False),
    then tokenized without special tokens: exactly the token ids graph_bench/eval/eval_vllm.py --style instruct feeds
    vLLM. Target: the response, then the template's end-of-turn string (<|im_end|>, <|endoftext|>, ...) on which vLLM
    stops. Loss on the target tokens only.
  - LoRA rank 16 / alpha 32 on every linear layer of the blocks (lm_head excluded), adapters in fp32, base bf16.
  - AdamW (0.9, 0.999), lr 2e-4, weight decay 0, linear warmup over 20 steps then constant, grad clip 1.0.
  - steps of <= 8,192 tokens: rows shuffled per epoch (numpy rng [seed, epoch]) and filled greedily in order, as
    graph_inject.data.pack_batches; loss = summed token CE / target tokens of the step. Rows run one at a time (no
    padding, no cross-row attention) with gradient accumulation; activation checkpointing on.
  - 3 epochs; the adapter is saved after every epoch (<out>/adapter_ep<N>/).

--merge writes <out>/merged_ep<N>/: the base checkpoint's own safetensors with W += (alpha/rank) B @ A added to each
adapted tensor (fp32, stored back in bf16) and every other file symlinked, so vLLM loads it exactly as the base
(Qwen3.5's checkpoint is the vision-language class; the text model trained here maps onto its language_model keys).

    /home/ucloud/miniforge3/bin/python -m graph_bench.sft_hf --model Qwen/Qwen2.5-7B-Instruct \\
        --train data/graphs/bench/sft/train_answer.jsonl --out $ROOT/sft_baselines/qwen25_7b/answer --merge
"""
import argparse
import glob
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

IGNORE = -100


def resolve(model: str) -> Path:
    """A local dir, or the cached hub snapshot of a repo id (offline; no download)."""
    if Path(model).is_dir():
        return Path(model)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snaps = [Path(p) for p in glob.glob(str(hub / f"models--{model.replace('/', '--')}/snapshots/*"))
             if (Path(p) / "config.json").exists()]
    if len(snaps) != 1:
        raise SystemExit(f"{model}: {len(snaps)} cached snapshots with a config.json")
    return snaps[0]


def end_of_turn(tok, chat_kwargs: dict) -> str:
    """The string the template puts after an assistant reply, up to the stop token: '<|im_end|>' for Qwen."""
    user = [{"role": "user", "content": "Q?"}]
    prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True, **chat_kwargs)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": "R"}], tokenize=False, **chat_kwargs)
    if not full.startswith(prompt + "R"):
        raise SystemExit(f"chat template: the reply does not follow the generation prompt\n{prompt!r}\n{full!r}")
    end = full[len(prompt) + 1:].rstrip("\n")
    if not end:
        raise SystemExit("chat template puts nothing after the reply")
    return end


def encode(rows: list[dict], tok, chat_kwargs: dict, end: str) -> list[tuple[list[int], list[int]]]:
    out = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False,
                                       add_generation_prompt=True, **chat_kwargs)
        prompt = tok(text, add_special_tokens=False)["input_ids"]  # as eval_vllm (token ids, no added <bos>)
        target = tok(r["response"] + end, add_special_tokens=False)["input_ids"]
        out.append((prompt, target))
    return out


def pack_batches(lengths: list[int], max_tokens: int, rng: np.random.Generator) -> list[list[int]]:
    """graph_inject.data.pack_batches: shuffle, then fill steps of at most max_tokens in order."""
    batches, current, used = [], [], 0
    for i in rng.permutation(len(lengths)):
        n = lengths[i]
        if used + n > max_tokens and current:
            batches.append(current)
            current, used = [], 0
        current.append(int(i))
        used += n
    if current:
        batches.append(current)
    return batches


def lora_targets(model) -> list[str]:
    """Every nn.Linear inside the decoder layers (attention, linear attention, MLP); not lm_head."""
    names = {n.split(".")[-1] for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and re.search(r"\.layers\.\d+\.", n)}
    return sorted(names)


def train(args, path: Path, out: Path):
    import wandb
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(path)
    chat_kwargs = {"enable_thinking": False}
    end = end_of_turn(tok, chat_kwargs)
    rows = [json.loads(line) for line in open(args.train, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    data = encode(rows, tok, chat_kwargs, end)
    lengths = [len(p) + len(t) for p, t in data]
    too_long = [i for i, n in enumerate(lengths) if n > args.max_seq_len]
    print(f"encoded {len(data)} rows in {time.time() - t0:.0f}s, end-of-turn {end!r}, max length {max(lengths)}, "
          f"{sum(lengths) / 1e6:.1f}M tokens/epoch, {len(too_long)} over {args.max_seq_len} (dropped)", flush=True)
    keep = [i for i in range(len(data)) if lengths[i] <= args.max_seq_len]

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    targets = lora_targets(model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                             target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA r{args.rank} on {targets}: {sum(p.numel() for p in trainable) / 1e6:.1f}M trainable", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    run = wandb.init(project=args.wandb_project, group=args.wandb_group, job_type="train", name=args.run_name,
                     config=vars(args) | {"model_path": str(path), "lora_targets": targets, "end_of_turn": end,
                                          "rows": len(keep), "dropped_too_long": len(too_long)},
                     mode=args.wandb_mode)
    step, autocast = 0, torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng([args.seed, epoch])
        sub = [lengths[i] for i in keep]
        t_epoch, tokens_seen = time.time(), 0
        for batch in pack_batches(sub, args.tokens_per_step, rng):
            idx = [keep[j] for j in batch]
            n_target = sum(len(data[i][1]) for i in idx)
            loss_sum, correct = 0.0, 0
            for i in idx:
                prompt, target = data[i]
                ids = torch.tensor([prompt + target[:-1]], device=device)
                labels = torch.tensor(target, device=device)
                with autocast:  # logits only for the positions that predict a target token
                    logits = model(input_ids=ids, logits_to_keep=len(target)).logits[0].float()
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                (loss / n_target).backward()
                loss_sum += loss.item()
                correct += (logits.argmax(-1) == labels).sum().item()
                tokens_seen += ids.shape[1]
            step += 1
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, step / args.warmup_steps) if args.warmup_steps else args.lr
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % args.log_every == 0 or step == 1:
                log = {"train/loss": loss_sum / n_target, "train/token_accuracy": correct / n_target,
                       "train/grad_norm": grad_norm.item(), "train/epoch": epoch,
                       "train/tokens_per_s": tokens_seen / (time.time() - t_epoch),
                       "train/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0}
                run.log(log, step=step)
                print(f"epoch {epoch} step {step} " + " ".join(f"{k[6:]}={v:.4g}" for k, v in log.items()), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        print(f"epoch {epoch} done in {time.time() - t_epoch:.0f}s ({step} steps)", flush=True)
        model.save_pretrained(out / f"adapter_ep{epoch}")
        if args.max_steps and step >= args.max_steps:
            break
    run.finish()
    return epoch


def merge(base: Path, adapter: Path, out: Path) -> int:
    """W += (alpha/rank) B @ A on the base checkpoint's own tensors (suffix match on layers.<i>.<...>.weight)."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in lora.items():
        m = re.search(r"\.(layers\.\d+\..+)\.lora_([AB])\.weight$", name)
        if m is None:
            raise ValueError(f"unexpected adapter tensor {name}")
        pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = t
    shards = sorted(base.glob("*.safetensors"))
    keys = {}
    for s in shards:
        with safe_open(str(s), "pt") as f:
            for k in f.keys():
                keys[k] = s
    target = {}
    for suffix in pairs:
        # skip the vision tower and Qwen3.5's multi-token-prediction head (mtp.layers.*, same suffixes; LoRA is
        # applied to the language model only)
        hits = [k for k in keys if k.endswith("." + suffix) and "visual" not in k and not k.startswith("mtp.")]
        if len(hits) != 1:
            raise ValueError(f"{suffix}: {len(hits)} matching base tensors {hits[:3]}")
        target[hits[0]] = suffix
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for s in shards:
        tensors = load_file(str(s))
        for k in tensors:
            if k in target:
                p = pairs[target[k]]
                delta = scaling * (p["B"].float() @ p["A"].float())
                if delta.shape != tensors[k].shape:
                    raise ValueError(f"{k}: delta {tuple(delta.shape)} vs {tuple(tensors[k].shape)}")
                tensors[k] = (tensors[k].float() + delta).to(tensors[k].dtype)
                done += 1
        save_file(tensors, str(out / s.name), metadata={"format": "pt"})
        del tensors
    if done != len(pairs):
        raise ValueError(f"merged {done} of {len(pairs)} LoRA pairs")
    for f in base.iterdir():
        if not f.name.endswith(".safetensors") and not (out / f.name).exists():
            (out / f.name).symlink_to(f.resolve())
    (out / "MERGED_FROM").write_text(f"base {base.resolve()}\nlora {adapter.resolve()}\nscaling {scaling}\n")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Repo id (cached snapshot) or local dir.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tokens-per-step", type=int, default=8192)
    ap.add_argument("--max-seq-len", type=int, default=8192, help="Longer rows are dropped (counted in the log).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="First N rows (smoke tests).")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--merge", action="store_true", help="Merge the last epoch's adapter into <out>/merged_ep<N>.")
    ap.add_argument("--merge-only", type=int, default=None, help="Skip training; merge adapter_ep<N>.")
    ap.add_argument("--wandb-project", default="Mimir-Graph-RLVR")
    ap.add_argument("--wandb-group", default="mimir-bench-sft-baselines")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    path = resolve(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    epoch = args.merge_only or train(args, path, args.out)
    if args.merge or args.merge_only:
        n = merge(path, args.out / f"adapter_ep{epoch}", args.out / f"merged_ep{epoch}")
        print(f"{args.out}/merged_ep{epoch}: merged {n} LoRA pairs", flush=True)


if __name__ == "__main__":
    main()
