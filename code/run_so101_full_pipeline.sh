#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${VLA_DATA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CODE_DIR="${SCRIPT_DIR}"
SKILL_ID="${SKILL_ID:-so101_pick_cube}"
GPUS="${GPUS:-0 1}"
DTYPE="${DTYPE:-bfloat16}"
STEPS="${STEPS:-240000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-2000}"
EVAL_SUBSET_WINDOWS="${EVAL_SUBSET_WINDOWS:-1024}"
LOG_EVERY="${LOG_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
DECAY_STEPS="${DECAY_STEPS:-240000}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
FULL_EVAL_LOG_EVERY_BATCHES="${FULL_EVAL_LOG_EVERY_BATCHES:-500}"
RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-1}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_PREFIX_BS8="${RUN_PREFIX_BS8:-so101_full_${RUN_TS}_bs8}"
RUN_PREFIX_BS4="${RUN_PREFIX_BS4:-so101_full_${RUN_TS}_bs4}"
PIPELINE_LOG="${PIPELINE_LOG:-}"

cd "${CODE_DIR}"
export PYTHONUNBUFFERED=1

echo "[pipeline] start skill=${SKILL_ID} run_ts=${RUN_TS} gpus=${GPUS}"
echo "[pipeline] register raw skill begin"
python register_lerobot_skill.py --skill-id "${SKILL_ID}" --video-backend pyav --overwrite

echo "[pipeline] prepare begin"
python prepare_skill_dataset.py --skill-id "${SKILL_ID}"

run_train() {
  local batch_size="$1"
  local run_name_prefix="$2"
  local extra_args=()
  if [[ "${RESUME_IF_EXISTS}" == "1" ]]; then
    extra_args+=(--resume-if-exists)
  fi
  echo "[pipeline] train begin batch_size=${batch_size} run_name_prefix=${run_name_prefix} gpus=${GPUS}"
  python train_skill_parallel.py \
    --skill-ids "${SKILL_ID}" \
    --groups A B C \
    --gpus ${GPUS} \
    --log-dir "${DATA_ROOT}/logs" \
    --run-name-prefix "${run_name_prefix}" \
    --steps "${STEPS}" \
    --batch-size "${batch_size}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every-steps "${SAVE_EVERY_STEPS}" \
    --eval-subset-windows "${EVAL_SUBSET_WINDOWS}" \
    --log-every "${LOG_EVERY}" \
    --num-workers "${NUM_WORKERS}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --decay-steps "${DECAY_STEPS}" \
    --grad-clip-norm "${GRAD_CLIP_NORM}" \
    --dtype "${DTYPE}" \
    --full-eval-log-every-batches "${FULL_EVAL_LOG_EVERY_BATCHES}" \
    "${extra_args[@]}" \
    --overwrite
}

if run_train 8 "${RUN_PREFIX_BS8}"; then
  echo "[pipeline] train finished with batch_size=8"
  exit 0
else
  status=$?
fi
if [[ -n "${PIPELINE_LOG}" ]] && grep -Eiq "out of memory|cuda out of memory|oom" "${PIPELINE_LOG}"; then
  echo "[pipeline] batch_size=8 failed with OOM, retrying batch_size=4"
  run_train 4 "${RUN_PREFIX_BS4}"
  echo "[pipeline] train finished with batch_size=4"
  exit 0
fi

echo "[pipeline] batch_size=8 failed without OOM, exit_status=${status}"
exit "${status}"
