#!/bin/bash
# =====================================================================
# Fine-tune the pretrained HRM-Text-1B on GraphQA (config/cfg_graphqa_hrm1b.yaml) and evaluate it the way
# Graph-Representation-Learning-for-LLM did: val at every epoch, test on the epoch with the best val
# teacher-forced exact match (GRL's val/exact_match, here eval/tf_em), plus the zero-shot (step 0) model.
# Greedy-generation accuracy (eval/greedy_acc, the headline metric), greedy_em and per-task accuracy are reported too.
# Reports from before 2026-10-20 have accuracy / tf_exact_match (= greedy_acc / greedy_em) and no tf_em.
#
# W&B: project HRM-Graph, group graphqa-hrm1b (train and eval runs, told apart by job_type).
#
# Usage (from repo root):
#   bash scripts/graphs/run_graphqa_hrm1b.sh
#   RUN=graphqa_hrm1b_s1 OVERRIDES="seed=1" bash scripts/graphs/run_graphqa_hrm1b.sh
#   LOG_SPLITS=test ...   # log only the test report (best val epoch) to W&B; val JSONs stay on disk
#   Other HRM-Text checkpoints: HF_MODEL, HF_DIR, CONFIG, GROUP, TAG, ROOT (see run_graphqa_mimir_seeds.sh).
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

ROOT="${ROOT:-/work/dfm/marius-ortega/dfm-hrm-graphs/graphqa_hrm1b}"
HF_MODEL="${HF_MODEL:-sapientinc/HRM-Text-1B}"
HF_DIR="${HF_DIR:-/work/dfm/marius-ortega/dfm-hrm-graphs/hf/HRM-Text-1B}"  # = init_weights_path's directory in the config
EVAL_DIR="${EVAL_DIR:-data/graphs/graphqa/hrm-text-grl/standard}"
RUN="${RUN:-graphqa_hrm1b_s0}"
OVERRIDES="${OVERRIDES:-}"
LOG_SPLITS="${LOG_SPLITS:-val test}"  # eval reports sent to W&B
CONFIG="${CONFIG:-cfg_graphqa_hrm1b}"
GROUP="${GROUP:-graphqa-hrm1b}"
TAG="${TAG:-hrm1b-finetune}"
PY="${PY:-$PWD/.venv/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"  # never "disabled": all_config.yaml (needed by eval) is only written with a wandb run

# From the composed config: `epochs` may be inherited (cfg_graphqa_mimir sets none of its own).
EPOCHS="$("$PY" -c "
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir='$PWD/config', version_base=None):
    print(compose(config_name='$CONFIG').epochs)")"
[[ $EPOCHS =~ ^[0-9]+$ ]] || { echo "could not read epochs from $CONFIG: '$EPOCHS'" >&2; exit 1; }
CKPT="$ROOT/ckpts/$RUN"
mkdir -p "$ROOT/logs" "$CKPT"
log() { echo "[$(date '+%F %T')] $*"; }

# 1. Import the HF weights once (also the zero-shot eval checkpoint).
if [[ ! -f "$HF_DIR/unsharded_step_0.pt" ]]; then
  log "[import] $HF_MODEL -> $HF_DIR"
  "$PY" -m conversion.convert_from_hf --hf_model "$HF_MODEL" --out_dir "$HF_DIR" --config-name "$CONFIG"
fi

# 2. Train.
if [[ -d "$CKPT/fsdp2_epoch_$EPOCHS" ]]; then
  log "[skip] $RUN already trained"
else
  log "[train] $RUN ($OVERRIDES)"
  # shellcheck disable=SC2086
  torchrun --standalone --nnodes=1 --nproc_per_node=1 pretrain.py --config-name "$CONFIG" \
    init_weights_path="$HF_DIR/unsharded_step_0.pt" $OVERRIDES \
    run_name="$RUN" checkpoint_path="$CKPT" > "$ROOT/logs/train_$RUN.log" 2>&1
fi

eval_one() {  # $1 = ckpt dir, $2 = ckpt arg (--ckpt_epoch N | --ckpt_tag step_0), $3 = split, $4 = report path
  [[ -f "$4" ]] && return 0
  log "[eval] $(basename "$1") $2 $3"
  # shellcheck disable=SC2086
  "$PY" scripts/graphs/eval_graphqa.py --ckpt_path "$1" $2 --data "$EVAL_DIR/$3.jsonl" \
    --max_tokens 1536 --batch_size 32 --max_generation 96 --no_wandb --out "$4" \
    > "$ROOT/logs/eval_$(basename "$4" .json).log" 2>&1
}

# 3. Zero-shot baseline: the imported model before fine-tuning.
mkdir -p "$ROOT/ckpts/zeroshot"
for split in val test; do
  eval_one "$HF_DIR" "--ckpt_tag step_0" "$split" "$ROOT/ckpts/zeroshot/eval_grl_${split}_ep0.json"
done

# 4. Val at every epoch, then test on the best one (ties: earliest).
for e in $(seq 1 "$EPOCHS"); do
  eval_one "$CKPT" "--ckpt_epoch $e" val "$CKPT/eval_grl_val_ep$e.json"
done
BEST="$("$PY" - "$CKPT" "$EPOCHS" <<'PY'
import json, sys
ckpt, epochs = sys.argv[1], int(sys.argv[2])
def tf(r):  # pre-2026-10-20 reports: tf_exact_match (= greedy_em) was the selection metric
    return r["tf_em"] if "tf_em" in r else r["tf_exact_match"]
scores = {e: tf(json.load(open(f"{ckpt}/eval_grl_val_ep{e}.json"))) for e in range(1, epochs + 1)}
print(max(scores, key=lambda e: (scores[e], -e)))
PY
)"
log "[eval] best val epoch: $BEST"
eval_one "$CKPT" "--ckpt_epoch $BEST" test "$CKPT/eval_grl_test_ep$BEST.json"

# 5. One W&B eval run per report.
for run in zeroshot "$RUN"; do
  "$PY" scripts/graphs/log_graphqa_evals_to_wandb.py --root "$ROOT" --project HRM-Graph --group "$GROUP" \
    --tag "$TAG" --run "$run" --splits $LOG_SPLITS >> "$ROOT/logs/wandb_eval.log" 2>&1 \
    || log "[eval] wandb logging FAILED: $run (see $ROOT/logs/wandb_eval.log)"
done
"$PY" - "$ROOT" "$RUN" "$EPOCHS" "$BEST" <<'PY'
import json, sys
root, run, epochs, best = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
load = lambda p: json.load(open(p))
fmt = lambda r: f"{r.get('tf_em', float('nan')):6.3f} {r.get('greedy_em', r.get('tf_exact_match')):6.3f} {r.get('greedy_acc', r.get('accuracy')):6.3f}"
print(f"{'checkpoint':16s} {'split':5s} {'tf_em':>6s} {'g_em':>6s} {'g_acc':>6s}")
for split in ("val", "test"):
    r = load(f"{root}/ckpts/zeroshot/eval_grl_{split}_ep0.json")
    print(f"{'zero-shot':16s} {split:5s} {fmt(r)}")
for e in range(1, epochs + 1):
    r = load(f"{root}/ckpts/{run}/eval_grl_val_ep{e}.json")
    print(f"{run + ' ep' + str(e):16s} {'val':5s} {fmt(r)}")
r = load(f"{root}/ckpts/{run}/eval_grl_test_ep{best}.json")
print(f"{run + ' ep' + str(best):16s} {'test':5s} {fmt(r)}")
print("test per task:", {t: round(a, 3) for t, a in r["task_accuracy"].items()})
PY
