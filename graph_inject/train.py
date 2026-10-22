"""Fine-tune a pretrained HRM-Text model on GraphQA with one graph-injection method, then evaluate.

    .venv/bin/python -m graph_inject.train model=hrm1b method=graph_token [key=value ...]

Configs: graph_inject/configs/base.yaml <- configs/models/<model>.yaml <- configs/methods/<method>.yaml <- CLI.
Single GPU, no FSDP, no torch.compile. The base model is loaded from the converted HF dir with
simple_inference_engine.inference_load_checkpoint (bf16). Regimes:

  lora    base frozen in bf16, LoRA (fp32) on every H/L projection + the method's modules     (default)
  frozen  base frozen, only the method's modules train (original GraphToken setting)
  full    every weight trains: fp32 params under bf16 autocast (trainer check against pretrain.py)

Every epoch: greedy generation on val (eval_graphqa metrics). The best epoch by val tf_em (ties: earliest,
as scripts/graphs/run_graphqa_hrm1b.sh) is evaluated on test. Outputs in <out_root>/<model>/<method>_s<seed>/:
config.yaml, eval_val_ep<N>.json, eval_test.json, best_trainable.pt (trainable params only).
"""
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from graph_inject.data import EncodedSample, encode_samples, load_samples, pack_batches, to_device_prefixlm, train_batch
from graph_inject.generate import generate, score
from graph_inject.lora import apply_lora
from graph_inject.methods import build_method
from graph_inject.model import GraphInjectedLM, embedding_rms
from graph_inject.tokenize import PromptTokenizer
from models.accelerator import set_accelerator_type
from models.adam_atan2 import AdamATan2
from simple_inference_engine import inference_load_checkpoint

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def load_config(argv: list[str]):
    cli = OmegaConf.from_dotlist(argv)
    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "base.yaml"))
    cfg.model = OmegaConf.load(os.path.join(CONFIG_DIR, "models", f"{cli.pop('model')}.yaml"))
    method_cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "methods", f"{cli.pop('method')}.yaml"))
    cfg = OmegaConf.merge(cfg, method_cfg.pop("train", {}), {"method": method_cfg}, cli)
    return cfg


def build_model(cfg, device):
    ckpt = inference_load_checkpoint(cfg.model.hf_dir, None, False, ckpt_tag="step_0")
    base = ckpt.model.train()
    base.requires_grad_(cfg.regime == "full")
    if cfg.regime == "full":
        base.float()
    elif cfg.regime == "lora":
        apply_lora(base, cfg.lora.rank, cfg.lora.alpha)
    elif cfg.regime != "frozen":
        raise ValueError(f"unknown regime {cfg.regime}")
    method = build_method(cfg.method, base.embed_tokens.embedding_weight.shape[1], embedding_rms(base)).to(device)
    method.setup(ckpt.tokenizer, base.embed_tokens)
    model = GraphInjectedLM(base, method)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError(f"nothing to train: regime={cfg.regime} with method {cfg.method.name}")
    return ckpt, model


def build_optimizer(cfg, model: GraphInjectedLM):
    method_params = [p for p in model.method.parameters() if p.requires_grad]
    base_params = [p for p in model.base.parameters() if p.requires_grad]
    groups = [g for g in ({"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                          {"params": method_params, "lr": cfg.method_lr, "weight_decay": 0.0}) if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=tuple(cfg.betas))
    if cfg.optimizer == "adam_atan2":
        return AdamATan2(groups, betas=tuple(cfg.betas))
    raise ValueError(f"unknown optimizer {cfg.optimizer}")


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def evaluate(cfg, ckpt, model, samples: list[EncodedSample], autocast) -> dict:
    stop_id = ckpt.stop_token_id()
    model.eval()
    with autocast():
        gens = generate(model, ckpt.carry, model.method, samples, stop_id,
                        max_generation=cfg.eval.max_generation, batch_size=cfg.eval.batch_size)
    model.train()
    return score(samples, gens, lambda ids: ckpt.tokenizer.decode(ids), stop_id)


def log_eval(prefix: str, report: dict, step: int):
    keys = ("greedy_acc", "greedy_acc_answerable", "greedy_em", "tf_em", "f1")
    wandb.log({**{f"{prefix}/{k}": report[k] for k in keys},
               **{f"{prefix}/task/{t}": a for t, a in report["task_accuracy"].items()}}, step=step)
    print(f"[{prefix}] " + "  ".join(f"{k}={report[k]:.4f}" for k in keys), flush=True)


def main(argv: list[str]):
    cfg = load_config(argv)
    run_name = f"{cfg.model.name}-{cfg.method.name}-s{cfg.seed}"
    out_dir = os.path.join(cfg.out_root, cfg.model.name, f"{cfg.method.name}_s{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg), flush=True)

    set_accelerator_type(cfg.accelerator_type)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    ckpt, model = build_model(cfg, device)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cfg.regime == "full" else nullcontext
    bp_steps = model.base.compute_train_extra_args(None)["bp_steps"]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_method = sum(p.numel() for p in model.method.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,} (method {n_method:,}), bp_steps={bp_steps}", flush=True)

    tokenizer = PromptTokenizer(ckpt)
    t0 = time.time()
    splits = {k: encode_samples(load_samples(cfg.data[k]), model.method, tokenizer) for k in ("train", "val", "test")}
    if cfg.eval.limit:
        splits["val"], splits["test"] = splits["val"][:cfg.eval.limit], splits["test"][:cfg.eval.limit]
    print(f"encoded splits in {time.time() - t0:.0f}s: " + ", ".join(
        f"{k}={len(v)} (max prompt {max(len(s.prompt.ids) for s in v)})" for k, v in splits.items()), flush=True)

    optimizer = build_optimizer(cfg, model)
    wandb.init(project=cfg.wandb.project, group=cfg.wandb.group, name=run_name, job_type="train",
               tags=[cfg.model.name, cfg.method.name, cfg.regime], mode=cfg.wandb.mode,
               config=OmegaConf.to_container(cfg, resolve=True) | {"trainable_params": n_trainable,
                                                                    "method_params": n_method})

    step, best = 0, None
    for epoch in range(1, cfg.epochs + 1):
        rng = np.random.default_rng([cfg.seed, epoch])
        t_epoch = time.time()
        for indices in pack_batches(splits["train"], cfg.tokens_per_step, rng):
            batch, scalars = train_batch([splits["train"][i] for i in indices], model.method, cfg.tokens_per_step, rng)
            batch = to_device_prefixlm(batch, scalars, device)
            step += 1
            warmup = min(1.0, step / cfg.warmup_steps) if cfg.warmup_steps else 1.0
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * warmup
            with autocast():
                _, loss, metrics = model(carry=ckpt.carry, batch=batch, bp_steps=bp_steps)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                       cfg.grad_clip if cfg.grad_clip else math.inf)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % cfg.log_every == 0 or step == 1:
                log = {"train/loss": metrics["loss"].item(), "train/token_accuracy": metrics["token_accuracy"].item(),
                       "train/grad_norm": grad_norm.item(), "train/epoch": epoch,
                       "train/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                wandb.log(log, step=step)
                print(f"epoch {epoch} step {step} " + " ".join(f"{k[6:]}={v:.4g}" for k, v in log.items()), flush=True)
            if cfg.max_steps and step >= cfg.max_steps:
                break
        epoch_time = time.time() - t_epoch
        print(f"epoch {epoch} done in {epoch_time:.0f}s ({step / max(epoch, 1):.0f} steps/epoch)", flush=True)

        report = evaluate(cfg, ckpt, model, splits["val"], autocast)
        report |= {"epoch": epoch, "step": step, "epoch_time_s": epoch_time}
        with open(os.path.join(out_dir, f"eval_val_ep{epoch}.json"), "w") as f:
            json.dump(report, f, indent=1)
        log_eval("val", report, step)
        if best is None or report["tf_em"] > best["tf_em"]:
            best = {"epoch": epoch, "tf_em": report["tf_em"], "state": trainable_state(model)}
            torch.save(best["state"], os.path.join(out_dir, "best_trainable.pt"))
        if cfg.max_steps and step >= cfg.max_steps:
            break

    model.load_state_dict(best["state"], strict=False)
    report = evaluate(cfg, ckpt, model, splits["test"], autocast) | {"best_epoch": best["epoch"], "step": step}
    with open(os.path.join(out_dir, "eval_test.json"), "w") as f:
        json.dump(report, f, indent=1)
    log_eval("test", report, step)
    wandb.summary.update({"best_epoch": best["epoch"], "test_greedy_acc": report["greedy_acc"],
                          "test_greedy_acc_answerable": report["greedy_acc_answerable"]})
    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
