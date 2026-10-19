#!/usr/bin/env bash
# Build the repo-local .venv for SM120 (RTX PRO 6000 Blackwell) graph experiments.
#
#   bash scripts/graphs/setup_venv.sh            # creates/updates .venv
#   VENV=/path/to/venv bash scripts/graphs/setup_venv.sh
#
# SM120 attention runs on PyTorch's built-in FA2 varlen kernels (accelerator_type=sm120,
# models/flash_attention_prefixlm_torch_varlen.py), so no FlashAttention package is installed:
# FA3 is SM90-only and FA4 CuTe fails to compile SM120 varlen kernels (4.0.0b14 @ 4178915, 4.0.0b19).
# torch 2.11.0+cu130 matches the shared `hrm` conda env. vLLM / lm-eval / euroeval extras are
# intentionally skipped (not needed for graph experiments).
set -euo pipefail

cd "$(dirname "$0")/../.."
VENV=${VENV:-.venv}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}

[[ -x "$VENV/bin/python" ]] || uv venv --python 3.13 --prompt hrm-graphs "$VENV"
export VIRTUAL_ENV="$PWD/$VENV"

uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
# Base [project].dependencies; torch stays pinned to the CUDA build above.
uv pip install -r pyproject.toml "torch==$TORCH_VERSION" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
# Tests, analysis scripts, and evaluation/benchmarks.py imports (math-verify, base lm-eval without the
# vLLM extra) used by prepare_benchmark_data.py for GSM8k/MATH/MMLU prompts.
uv pip install pytest pandas matplotlib seaborn math-verify lm-eval "torch==$TORCH_VERSION" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match

"$VENV/bin/python" - <<'PY'
import torch
from torch.nn.attention.varlen import varlen_attn  # noqa: F401  (sm120 PrefixLM backend)
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
