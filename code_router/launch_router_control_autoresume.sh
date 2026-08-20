#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${VLA_DATA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONFIG="${CONFIG:-examples/router_robocasa_so101_active.yaml}"
RUN_NAME="${RUN_NAME:-policycache_float16_b24_ga1_workers8_6gpu_5000_20260426_195123}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs/pi05_skill_router_lora}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/runs/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/cache/pi05_policy_ready_float16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
STEPS_PER_CHANNEL="${STEPS_PER_CHANNEL:-500}"
ROUTER_STEPS_PER_CHANNEL="${ROUTER_STEPS_PER_CHANNEL:-500}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-320}"
SAVE_EVERY="${SAVE_EVERY:-50}"
ROUTER_METRICS_EVERY="${ROUTER_METRICS_EVERY:-20}"
ROUTER_RENDEZVOUS_TIMEOUT_SECONDS="${ROUTER_RENDEZVOUS_TIMEOUT_SECONDS:-1800}"
ROUTER_RENDEZVOUS_WARN_SECONDS="${ROUTER_RENDEZVOUS_WARN_SECONDS:-120}"
RANK_STARTUP_STAGGER_SECONDS="${RANK_STARTUP_STAGGER_SECONDS:-60}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-45}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/logs}"
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/router_control_autoresume_${RUN_NAME}.log}"

mkdir -p "${LOG_DIR}"

attempt=1
shopt -s nullglob
for existing_log in "${LOG_DIR}/router_control_autoresume_${RUN_NAME}_attempt"*.log; do
  existing_name="${existing_log##*_attempt}"
  existing_number="${existing_name%.log}"
  if [[ "${existing_number}" =~ ^[0-9]+$ && $((10#${existing_number})) -ge "${attempt}" ]]; then
    attempt=$((10#${existing_number} + 1))
  fi
done
shopt -u nullglob

while true; do
  attempt_log="${LOG_DIR}/router_control_autoresume_${RUN_NAME}_attempt$(printf '%04d' "${attempt}").log"
  {
    echo "[router-autoresume] attempt=${attempt} started_at=$(date -u --iso-8601=seconds)"
    echo "[router-autoresume] run_dir=${RUN_DIR}"
    echo "[router-autoresume] attempt_log=${attempt_log}"
    if [[ -f "${RUN_DIR}/router_control_trainer_state.pt" ]]; then
      python - "${RUN_DIR}/router_control_trainer_state.pt" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
state = torch.load(path, map_location="cpu")
steps = state.get("channel_steps") or []
print(
    "[router-autoresume] checkpoint="
    + str(
        {
            "step": int(state.get("step", 0)),
            "channel_steps_sum": int(sum(int(v) for v in steps)),
            "num_channels": len(steps),
            "channel_cursor": int(state.get("channel_cursor", 0)),
        }
    ),
    flush=True,
)
PY
    else
      echo "[router-autoresume] checkpoint=missing"
    fi
  } | tee -a "${MASTER_LOG}"

  set +e
  torchrun --standalone --nproc-per-node "${NPROC_PER_NODE}" train_router_lora_sharded.py \
    --config "${CONFIG}" \
    --run-name "${RUN_NAME}" \
    --steps-per-channel "${STEPS_PER_CHANNEL}" \
    --router-steps-per-channel "${ROUTER_STEPS_PER_CHANNEL}" \
    --batch-size "${BATCH_SIZE}" \
    --router-batch-size "${ROUTER_BATCH_SIZE}" \
    --gradient-accumulation-steps 1 \
    --num-workers 8 \
    --save-every "${SAVE_EVERY}" \
    --prefetch-batches 8 \
    --router-num-workers 0 \
    --router-prefetch-batches 0 \
    --router-prefetch-workers 1 \
    --router-metrics-every "${ROUTER_METRICS_EVERY}" \
    --router-rendezvous-timeout-seconds "${ROUTER_RENDEZVOUS_TIMEOUT_SECONDS}" \
    --router-rendezvous-warn-seconds "${ROUTER_RENDEZVOUS_WARN_SECONDS}" \
    --torch-num-threads 1 \
    --torch-num-interop-threads 1 \
    --rank-startup-stagger-seconds "${RANK_STARTUP_STAGGER_SECONDS}" \
    --adapter-completion-poll-seconds 10 \
    --cache-root "${CACHE_ROOT}" \
    --cache-image-storage-dtype float16 \
    --require-policy-cache \
    --confirm-long-run \
    --skip-adapter-phase \
    --router-impl lora_control \
    --router-control-adapter router_control \
    --router-control-checkpoint "${RUN_DIR}" \
    > "${attempt_log}" 2>&1
  exit_code=$?
  set -e

  echo "[router-autoresume] attempt=${attempt} exit_code=${exit_code} finished_at=$(date -u --iso-8601=seconds)" | tee -a "${MASTER_LOG}"
  if [[ "${exit_code}" -eq 0 ]]; then
    echo "[router-autoresume] completed_at=$(date -u --iso-8601=seconds)" | tee -a "${MASTER_LOG}"
    exit 0
  fi

  if [[ "${MAX_ATTEMPTS}" -gt 0 && "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "[router-autoresume] reached_max_attempts=${MAX_ATTEMPTS}" | tee -a "${MASTER_LOG}" >&2
    exit "${exit_code}"
  fi

  attempt=$((attempt + 1))
  echo "[router-autoresume] retrying_after_seconds=${RETRY_SLEEP_SECONDS}" | tee -a "${MASTER_LOG}"
  sleep "${RETRY_SLEEP_SECONDS}"
done
