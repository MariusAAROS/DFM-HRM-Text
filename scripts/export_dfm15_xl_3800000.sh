#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="/home/ucloud/miniforge3/envs/hrm/bin:$PATH"
python scripts/schedule_dfm15_xl_evaluation.py --finalize
exec python "$@"
