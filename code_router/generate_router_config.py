#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH, DEFAULT_SKILL_ROOT
from vla_skill_router.constants import DEFAULT_ROUTER_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an active RoboCasa + SO101 router-LoRA YAML config.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_SKILL_ROOT / "robocasa_active_tasks.json",
    )
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROUTER_OUTPUT_ROOT)
    parser.add_argument("--so101-skill-id", type=str, default="so101_pick_cube")
    parser.add_argument("--steps-per-channel", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path(__file__).resolve().parent / "examples" / "router_robocasa_so101_active.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    prepare_summary_path = args.manifest_path.with_name("robocasa_prepare_summary.json")
    if prepare_summary_path.is_file():
        prepare_summary = json.loads(prepare_summary_path.read_text(encoding="utf-8"))
        robocasa_items = [
            {"skill_id": item["skill_id"], "task_index": item.get("task_index"), "task_name": item.get("task_name")}
            for item in prepare_summary.get("prepared_skills", [])
        ]
    else:
        robocasa_items = manifest["active_tasks"]
    channels = []
    for task in robocasa_items:
        channels.append(
            {
                "channel_id": task["skill_id"],
                "skill_id": task["skill_id"],
                "lora_group": "C",
            }
        )
    channels.append({"channel_id": args.so101_skill_id, "skill_id": args.so101_skill_id, "lora_group": "C"})

    payload = {
        "base_model_path": str(args.base_model_path),
        "skill_root": str(args.skill_root),
        "output_root": str(args.output_root),
        "channels": channels,
        "router": {
            "type": "lora_control",
            "feature_hook": "pi05_paligemma_with_expert",
            "state_embed_dim": 64,
            "hidden_dim": 256,
            "use_previous_skill": False,
        },
        "train": {
            "steps_per_channel": args.steps_per_channel,
            "batch_size": args.batch_size,
            "eval_every": 5000,
            "save_every": 5000,
            "log_every": 10,
            "num_workers": 2,
            "lr": 2.5e-5,
            "router_lr": 1.0e-4,
            "router_ce_weight": 1.0,
            "device": args.device,
            "dtype": args.dtype,
        },
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"output_config": str(args.output_config), "channels": len(channels)}, indent=2))


if __name__ == "__main__":
    main()
