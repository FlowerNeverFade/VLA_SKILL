#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${VLA_DATA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONFIG="examples/router_robocasa_so101_active.yaml"
CACHE_ROOT="${VLA_POLICY_CACHE_ROOT:-${DATA_ROOT}/cache/pi05_policy_ready_float16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
STEPS_PER_CHANNEL="${STEPS_PER_CHANNEL:-500}"
RUN_STEM="policycache_float16_maxvram_${NPROC_PER_NODE}gpu_${STEPS_PER_CHANNEL}steps_ga1_workers8_$(date -u +%Y%m%d_%H%M%S)"
TRAIN_BATCH_CANDIDATES=(24)
CACHE_SPLITS=(train)
CACHE_PARALLEL_CHANNELS=8
CACHE_NUM_WORKERS=8

echo "[full-float16] started_at=$(date -u --iso-8601=seconds)"
echo "[full-float16] config=${CONFIG}"
echo "[full-float16] cache_root=${CACHE_ROOT}"
echo "[full-float16] run_stem=${RUN_STEM}"
echo "[full-float16] nproc_per_node=${NPROC_PER_NODE}"
echo "[full-float16] steps_per_channel=${STEPS_PER_CHANNEL}"
echo "[full-float16] cache_splits=${CACHE_SPLITS[*]}"
echo "[full-float16] cache_parallel_channels=${CACHE_PARALLEL_CHANNELS}"
echo "[full-float16] cache_num_workers=${CACHE_NUM_WORKERS}"
echo "[full-float16] train_batch_candidates=${TRAIN_BATCH_CANDIDATES[*]}"
df -h "${DATA_ROOT}"

python build_router_policy_cache.py \
  --config "${CONFIG}" \
  --cache-root "${CACHE_ROOT}" \
  --splits "${CACHE_SPLITS[@]}" \
  --batch-size 16 \
  --shard-size 256 \
  --num-workers "${CACHE_NUM_WORKERS}" \
  --parallel-channels "${CACHE_PARALLEL_CHANNELS}" \
  --image-storage-dtype float16

echo "[full-float16] cache_done_at=$(date -u --iso-8601=seconds)"
du -sh "${CACHE_ROOT}" || true
df -h "${DATA_ROOT}"

PROBE_CONFIG="${CACHE_ROOT}/probe_first4_channels.yaml"
python - "${CONFIG}" "${PROBE_CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
payload = yaml.safe_load(src.read_text(encoding="utf-8"))
payload["channels"] = payload["channels"][:4]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"[full-float16] wrote_probe_config={dst}")
PY

SELECTED_BATCH=""
for batch_size in "${TRAIN_BATCH_CANDIDATES[@]}"; do
  echo "[full-float16] probing_batch_size=${batch_size} started_at=$(date -u --iso-8601=seconds)"
  if torchrun --standalone --nproc-per-node "${NPROC_PER_NODE}" train_router_lora_sharded.py \
    --config "${PROBE_CONFIG}" \
    --run-name "${RUN_STEM}_probe_b${batch_size}" \
    --steps-per-channel 1 \
    --batch-size "${batch_size}" \
    --gradient-accumulation-steps 1 \
    --num-workers 4 \
    --prefetch-batches 2 \
    --cache-root "${CACHE_ROOT}" \
    --cache-image-storage-dtype float16 \
    --require-policy-cache \
    --router-impl lora_control \
    --skip-router-phase; then
    SELECTED_BATCH="${batch_size}"
    echo "[full-float16] selected_batch_size=${SELECTED_BATCH}"
    break
  fi
  echo "[full-float16] probing_batch_size=${batch_size} failed_at=$(date -u --iso-8601=seconds)"
done

if [[ -z "${SELECTED_BATCH}" ]]; then
  echo "[full-float16] no batch size candidate succeeded" >&2
  exit 1
fi

FULL_RUN_NAME="${RUN_STEM}_b${SELECTED_BATCH}"
TRAIN_BATCHES=()
TAKE=0
for batch_size in "${TRAIN_BATCH_CANDIDATES[@]}"; do
  if [[ "${batch_size}" == "${SELECTED_BATCH}" ]]; then
    TAKE=1
  fi
  if [[ "${TAKE}" == "1" ]]; then
    TRAIN_BATCHES+=("${batch_size}")
  fi
done

for batch_size in "${TRAIN_BATCHES[@]}"; do
  echo "[full-float16] full_train_batch_size=${batch_size} run_name=${FULL_RUN_NAME} started_at=$(date -u --iso-8601=seconds)"
  if torchrun --standalone --nproc-per-node "${NPROC_PER_NODE}" train_router_lora_sharded.py \
    --config "${CONFIG}" \
    --run-name "${FULL_RUN_NAME}" \
    --steps-per-channel "${STEPS_PER_CHANNEL}" \
    --batch-size "${batch_size}" \
    --gradient-accumulation-steps 1 \
    --num-workers 8 \
    --prefetch-batches 8 \
    --cache-root "${CACHE_ROOT}" \
    --cache-image-storage-dtype float16 \
    --require-policy-cache \
    --router-impl lora_control \
    --confirm-long-run; then
    echo "[full-float16] full_train_succeeded_batch_size=${batch_size}"
    echo "[full-float16] finished_at=$(date -u --iso-8601=seconds)"
    exit 0
  fi
  echo "[full-float16] full_train_batch_size=${batch_size} failed_at=$(date -u --iso-8601=seconds); retrying lower batch if available"
done

echo "[full-float16] full training failed for all batch candidates" >&2
exit 1
