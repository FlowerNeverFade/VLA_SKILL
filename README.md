# VLA Skill LoRA and Multi-Skill Router

This repository contains the algorithm code used to package robot tasks as
PI0.5/VLA skills. A skill is a task-specific LoRA adapter with its own data
schema, normalization statistics, processor bundle, and router metadata. The
router combines many skill adapters and can train them with DDP or skill-sharded
workers.

The repository contains code only. Robot datasets, PI0.5 checkpoints, tokenizer
files, generated adapters, policy caches, videos, and experiment logs are
intentionally excluded from Git.

## Layout

| Path | Purpose |
| --- | --- |
| `code/` | Single-skill PI0.5 LoRA training, LeRobot task conversion, evaluation, and inference helpers. |
| `code_router/` | Multi-skill routing, RoboCasa registration, policy-ready caching, DDP, and skill-sharded training. |
| `code/tests/` | Lightweight data, LoRA-target, conversion, and evaluation tests. |
| `code_router/tests/` | Router, cache, distributed, schedule, and checkpoint tests. |
| `pi05_policy_server.py` | Minimal HTTP runtime for one base or skill-LoRA policy. |

## Runtime files

Use a separate runtime directory for large files. The default layout is:

```text
<runtime>/
  model/pi05_base/          # PI0.5 base checkpoint
  model/google/...          # optional local PaliGemma tokenizer
  datasets/                 # LeRobot/RoboCasa datasets
  skill/                    # generated skill.yaml, splits.json, stats.json
  outputs/                  # adapters and evaluation results
  cache/                    # policy-ready tensor cache
```

Set `VLA_DATA_ROOT=<runtime>` before running commands. Individual locations can
be overridden with `VLA_BASE_MODEL_PATH`, `VLA_SKILL_DIR`, `VLA_OUTPUT_ROOT`,
`VLA_ROUTER_OUTPUT_ROOT`, `VLA_POLICY_CACHE_ROOT`, and `PI05_TOKENIZER_PATH`.

## Install

The two directories are separate projects because both contain a local
`vla_skill` package. Install only the project you are using in a given virtual
environment:

```bash
python -m pip install -e code
# or, in a separate environment:
python -m pip install -e code_router
```

The `lerobot` package must be a version compatible with the PI0.5 policy API;
CUDA-enabled PyTorch should be installed according to the target machine.

## Single-skill workflow

Convert one LeRobot task into a skill directory, validate it, then train a
task-specific adapter:

```bash
cd code
python convert_lerobot_task_to_skill.py \
  --dataset-dir "$VLA_DATA_ROOT/datasets/my_dataset" \
  --repo-id owner/my_dataset \
  --skill-root "$VLA_DATA_ROOT/skill" \
  --skill-id my_task
python prepare_skill_dataset.py --skill-root "$VLA_DATA_ROOT/skill" --skill-id my_task
python train_skill_lora.py --skill-id my_task --group C
```

`train_skill_parallel.py` and `eval_base_sharded.py` provide multi-GPU
training/evaluation helpers. The conversion scripts also support raw LeRobot
task metadata without materializing a second copy of the source dataset.

After an adapter has been trained, the minimal HTTP runtime can be started from
the repository root:

```bash
python pi05_policy_server.py \
  --skill-root "$VLA_DATA_ROOT/skill" \
  --skill-id my_task \
  --base-model-path "$VLA_DATA_ROOT/model/pi05_base" \
  --adapter-dir "$VLA_DATA_ROOT/outputs/pi05_skill_lora/my_task/C/run/best"
```

## Multi-skill router workflow

Register and prepare active RoboCasa tasks, generate a channel configuration,
build the optional policy-ready cache, and launch the sharded trainer:

```bash
cd code_router
python register_robocasa_active_tasks.py \
  --dataset-dir "$VLA_DATA_ROOT/datasets/RoboCasa/datasets_hf/robocasa_target_atomic" \
  --skill-root "$VLA_DATA_ROOT/skill" --overwrite
python prepare_router_skills.py --skill-root "$VLA_DATA_ROOT/skill"
python generate_router_config.py --steps-per-channel 10
python build_router_policy_cache.py \
  --config examples/router_robocasa_so101_active.yaml \
  --cache-root "$VLA_DATA_ROOT/cache/pi05_policy_ready" \
  --splits train val --image-storage-dtype float16
torchrun --standalone --nproc-per-node 4 train_router_lora_sharded.py \
  --config examples/router_robocasa_so101_active.yaml \
  --steps-per-channel 1000 --router-impl lora_control \
  --require-policy-cache --confirm-long-run
```

The adapter phase assigns channels to workers. The router-control phase then
synchronizes a reserved LoRA adapter and a small classification head. Checkpoint
state includes per-channel progress so interrupted long runs can resume.

## Tests

```bash
python -m pytest -q code/tests
python -m pytest -q code_router/tests
python -m compileall -q code code_router
```

Tests use synthetic materialized episodes and do not download data or load a
GPU checkpoint.

## Scope and upstream components

This code integrates with LeRobot, PEFT, Transformers, and the PI0.5 policy
implementation. Their models, datasets, and licenses remain upstream and are
not redistributed here. No model weights or private experiment credentials are
part of this repository.
