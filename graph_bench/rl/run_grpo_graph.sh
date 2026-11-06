#!/usr/bin/env bash
# GRPO on the graph sets with VeRL + vLLM (arms E3 / E3b / E4 of wiki: mimir-graph-benchmark).
# Adapted from /work/dfm/marius-ortega/jacob-hrm-text-rlvr/verl_hrm_text/run_grpo_hrm_text.sh.
#
# Smoke test:
#   TRAIN_FILE=... TEST_FILE=... TOTAL_TRAINING_STEPS=50 EXPERIMENT_NAME=smoke bash graph_bench/rl/run_grpo_graph.sh
# Detach real runs: setsid nohup bash graph_bench/rl/run_grpo_graph.sh > <log> 2>&1 < /dev/null &
#
# Environment: /home/ucloud/miniforge3 (vllm 0.27.1, verl 0.9.0, transformers 5.20.2), not the repo .venv.
# MODEL_PATH must be an HF-format checkpoint (conversion/convert_to_hf.py for native SFT outputs; merge LoRA first).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
JACOB=${JACOB:-/work/dfm/marius-ortega/jacob-hrm-text-rlvr}
PY=${PY:-/home/ucloud/miniforge3/bin/python3}
export PYTHONPATH="$REPO:$JACOB${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# W&B: default host (api.wandb.ai); do not set WANDB_BASE_URL. WANDB_RUN_ID + WANDB_RESUME=must continue a run.

MODEL_PATH=${MODEL_PATH:-danish-foundation-models/DFM-Mimir-v1.5}
TRAIN_FILE=${TRAIN_FILE:-$REPO/data/graphs/bench/verl/train.parquet}
TEST_FILE=${TEST_FILE:-$REPO/data/graphs/bench/verl/val.parquet}

# none: no thought prefill (policy SFT'd on plain-response traces, E3/E4); prompt: prefill scored as prompt (E3b).
export GRAPH_RL_PREFILL=${GRAPH_RL_PREFILL:-none}
export GRAPH_REWARD=${GRAPH_REWARD:-acc}
export GRAPH_RL_FUSE_SYNC=${GRAPH_RL_FUSE_SYNC:-1}  # 0 for the Bifrost vLLM plugin (separate weight names)

PROJECT_NAME=${PROJECT_NAME:-Mimir-Graph-RLVR}  # RL runs have their own W&B project (SFT / eval stay in HRM-Graph)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_graph}
# KL to the starting model (the actor with its LoRA disabled): REF_KL=True logs actor/kl_loss, KL_COEF weights it in the
# loss (0 = log only, no penalty). Costs one more forward per step (~86 s for E3-sized batches). Default on, except for
# e3_grpo_e2, which started without it: its resumes keep one config (decided 2026-11-06).
REF_KL=${REF_KL:-$([ "$EXPERIMENT_NAME" = e3_grpo_e2 ] && echo False || echo True)}
KL_COEF=${KL_COEF:-0.0}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
LR=${LR:-1e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-16}
PPO_MICRO_BATCH_PER_GPU=${PPO_MICRO_BATCH_PER_GPU:-4}
LOGPROB_MICRO_BATCH_PER_GPU=${LOGPROB_MICRO_BATCH_PER_GPU:-4}
CLIP_HIGH=${CLIP_HIGH:-0.28}                 # DAPO clip-higher
FILTER_GROUPS=${FILTER_GROUPS:-True}         # DAPO dynamic sampling: drop all-correct / all-wrong groups
GPU_MEM=${GPU_MEM:-0.4}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}         # Jacob's working value; try False (CUDA graphs) in the gate
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
EXTRA_VERL_ARGS=${EXTRA_VERL_ARGS:-}
# Dumps (one JSONL per step: prompt, response, score, gold, step): ROLLOUT_DATA_DIR gets the trained batch (the groups
# kept by filter_groups), VAL_DATA_DIR every validation pass. Unset: nothing is stored.
DUMP_ARGS="${ROLLOUT_DATA_DIR:+trainer.rollout_data_dir=$ROLLOUT_DATA_DIR} ${VAL_DATA_DIR:+trainer.validation_data_dir=$VAL_DATA_DIR}"
# Host-RAM caps: the container has 128 GB (cgroup, no swap; an OOM restarts it and kills every job). Ray would
# otherwise size its object store (on /dev/shm, counted in the cgroup) from system memory and see all 188 CPUs.
RAY_OBJ_STORE=${RAY_OBJ_STORE:-8000000000}
RAY_CPUS=${RAY_CPUS:-16}
# Ray helper processes can start slowly here: the raylet timed out waiting for its runtime-env agent (2026-11-03).
export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-300000}
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-300}

echo "[grpo_graph] model=$MODEL_PATH prefill=$GRAPH_RL_PREFILL reward=$GRAPH_REWARD ref_kl=$REF_KL kl_coef=$KL_COEF train=$TRAIN_FILE"
echo "[grpo_graph] container memory at start: $(( $(cat /sys/fs/cgroup/memory.current) / 1000000000 )) GB of $(( $(cat /sys/fs/cgroup/memory.max) / 1000000000 )) GB"

cd "$REPO"
"$PY" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable=${FILTER_GROUPS} \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=4 \
  data.train_files=${TRAIN_FILE} \
  data.val_files=${TEST_FILE} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.prompt_key=prompt \
  data.truncation=error \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.trust_remote_code=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.external_lib=graph_bench.rl.verl_ext \
  actor_rollout_ref.model.lora_rank=${LORA_RANK} \
  actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU} \
  actor_rollout_ref.actor.use_kl_loss=${REF_KL} \
  actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
  actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_PER_GPU} \
  reward_model.enable=False \
  reward.custom_reward_function.path=$REPO/graph_bench/rl/reward.py \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  ray_kwargs.ray_init.num_cpus=${RAY_CPUS} \
  +ray_kwargs.ray_init.object_store_memory=${RAY_OBJ_STORE} \
  +ray_kwargs.ray_init.include_dashboard=False \
  ${DUMP_ARGS} \
  ${EXTRA_VERL_ARGS}
