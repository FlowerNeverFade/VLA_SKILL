# PI05 Multi-Skill LoRA Router Prototype

This is an independent prototype for multi-skill PI05 LoRA routing.

V1 design:

- one task/skill maps to one LoRA channel
- every channel uses LoRA target group `C`
- the default router is a reserved `router_control` LoRA adapter plus a minimal `LayerNorm + Linear` head
- router input is `pooled_router_control_context + explicit state_embedding`
- PI05 base stays frozen
- router CE loss updates only `router_control` LoRA and the linear router head
- policy loss updates only the ground-truth channel adapter

Default output root (under `VLA_DATA_ROOT`):

```bash
${VLA_DATA_ROOT}/outputs/pi05_skill_router_lora
```

Example config:

```bash
examples/router_experiment.yaml
```

RoboCasa + SO101 active-task workflow:

```bash
python register_robocasa_active_tasks.py --overwrite
python prepare_router_skills.py
python generate_router_config.py --steps-per-channel 10
python train_router_lora_sharded.py \
  --config examples/router_robocasa_so101_active.yaml \
  --steps-per-channel 10 \
  --router-impl lora_control
```

DDP multi-GPU training is opt-in through `torchrun`. `train.batch_size` is
interpreted as the per-device batch size, so four GPUs with `batch_size: 8`
produce an effective global batch of 32:

```bash
torchrun --nproc_per_node 4 train_router_lora.py \
  --config examples/router_robocasa_so101_active.yaml \
  --steps-per-channel 10
```

Skill-sharded multi-worker training assigns different skills to different GPUs
for the adapter phase, then trains one synchronized LoRA-control router:

```bash
torchrun --nproc-per-node 6 train_router_lora_sharded.py \
  --config examples/router_robocasa_so101_active.yaml \
  --steps-per-channel 10 \
  --gradient-accumulation-steps 6 \
  --router-impl lora_control
```

Policy-ready tensor cache removes online LeRobot video decoding and PI05 CPU
preprocessing from training. Build it before a long run:

```bash
python build_router_policy_cache.py \
  --config examples/router_robocasa_so101_active.yaml \
  --cache-root "${VLA_DATA_ROOT}/cache/pi05_policy_ready" \
  --splits train val \
  --batch-size 16 \
  --shard-size 256 \
  --num-workers 8 \
  --image-storage-dtype float16
```

Then require the cache during skill-sharded training so missing or stale cache
does not silently fall back to online video decoding:

```bash
torchrun --standalone --nproc-per-node 4 train_router_lora_sharded.py \
  --config examples/router_robocasa_so101_active.yaml \
  --run-name policycache_float16_b16_ga2_workers8 \
  --steps-per-channel 20000 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --num-workers 8 \
  --prefetch-batches 8 \
  --cache-root "${VLA_DATA_ROOT}/cache/pi05_policy_ready" \
  --cache-image-storage-dtype float16 \
  --require-policy-cache \
  --router-impl lora_control \
  --confirm-long-run
```

LoRA-control router artifacts are saved as `router_control/`,
`router_control_head.pt`, and `router_control_meta.json`. Legacy `router.pt`
files are only used with `--router-impl hard_top1`.

For the full run, use:

```bash
python generate_router_config.py --steps-per-channel 20000
torchrun --standalone --nproc-per-node 6 train_router_lora_sharded.py \
  --config examples/router_robocasa_so101_active.yaml \
  --steps-per-channel 20000 \
  --router-impl lora_control \
  --confirm-long-run
```

The RoboCasa registration scans data parquet files and only registers task
indices that actually have frames and episodes.

Run tests:

```bash
python -m pytest -q
```

Train entrypoint:

```bash
python train_router_lora_sharded.py --config examples/router_experiment.yaml --router-impl lora_control
```

The default config is a real PI05 experiment template. It is not launched by
tests and should be adjusted before a long run.
