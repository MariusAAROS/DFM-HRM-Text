"""Make a loadable HF model dir for vLLM / VeRL whose tokenizer tokenizes like data_io and the native engine.

The DFM-Mimir checkpoints (v1, v1.5, HRM-Mimir-v1) set "fix_mistral_regex": true in tokenizer_config.json. transformers
then prepends a Mistral-style regex pre-tokenizer that tokenizer.json does not have, so "#### Question" becomes
"####", " ", "Question" instead of "####", " Question": every prompt differs from the ids the model was fine-tuned
and evaluated on natively (0/600 parity prompts identical; 600/600 with the flag off).

The view symlinks every file of the snapshot and writes tokenizer_config.json with the flag set to false:
    .venv/bin/python -m graph_bench.eval.hf_view \\
        ~/.cache/huggingface/hub/models--danish-foundation-models--DFM-Mimir-v1.5/snapshots/<rev> \\
        /work/dfm/marius-ortega/dfm-hrm-graphs/hf/DFM-Mimir-v1.5-hfview
"""
import argparse
import json
from pathlib import Path


def make_view(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        dst = out / f.name
        if f.name == "tokenizer_config.json":
            cfg = json.loads(f.read_text())
            cfg["fix_mistral_regex"] = False
            dst.unlink(missing_ok=True)
            dst.write_text(json.dumps(cfg, indent=2) + "\n")
        elif not dst.exists():
            dst.symlink_to(f.resolve())
    (out / "HF_VIEW_SOURCE").write_text(f"{src.resolve()}\nfix_mistral_regex forced to false (graph_bench/eval/hf_view.py)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="HF snapshot dir.")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    make_view(args.src.expanduser(), args.out)
    print(f"{args.out}: view of {args.src}")


if __name__ == "__main__":
    main()
