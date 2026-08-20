#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vla_skill.dataset import build_window_dataset, load_skill_spec, load_stats
from vla_skill.pi05 import build_processors
from vla_skill_router.constants import DEFAULT_ROUTER_CONTROL_ADAPTER, ROUTER_IMPL_HARD_TOP1, ROUTER_IMPL_LORA_CONTROL
from vla_skill_router.action import crop_action
from vla_skill_router.config import load_experiment_config
from vla_skill_router.features import PI05PrefixFeatureExtractor
from vla_skill_router.real_runtime import (
    activate_router_control_adapter,
    build_obs_batch,
    build_hard_top1_router_from_config,
    build_lora_control_router_from_config,
    load_first_skill_policy,
    load_or_initialize_channels,
    load_or_initialize_router_control_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard top-1 router inference and activate the selected LoRA.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--obs-json", type=Path)
    parser.add_argument("--input-skill-id", type=str)
    parser.add_argument("--sample-skill-id", type=str)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--router-impl", choices=(ROUTER_IMPL_LORA_CONTROL, ROUTER_IMPL_HARD_TOP1), default=ROUTER_IMPL_LORA_CONTROL)
    parser.add_argument("--router-control-adapter", default=DEFAULT_ROUTER_CONTROL_ADAPTER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    policy = load_first_skill_policy(cfg)
    policy, _ = load_or_initialize_channels(
        cfg,
        policy,
        adapter_root=args.checkpoint_dir / "channels",
        is_trainable=False,
    )
    if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
        policy = load_or_initialize_router_control_adapter(
            cfg,
            policy,
            adapter_name=args.router_control_adapter,
            adapter_dir=args.checkpoint_dir / args.router_control_adapter,
            is_trainable=False,
            remove_existing_adapters=False,
        )

    if args.obs_json:
        if not args.input_skill_id:
            raise SystemExit("--obs-json requires --input-skill-id for v1 preprocessing.")
        input_skill_spec, raw_batch = build_obs_batch(cfg, obs_json=args.obs_json, skill_id=args.input_skill_id)
    else:
        if not args.sample_skill_id or args.sample_index is None:
            raise SystemExit("Dataset sample mode requires --sample-skill-id and --sample-index.")
        input_skill_spec = load_skill_spec(cfg.skill_root, args.sample_skill_id)
        dataset = build_window_dataset(input_skill_spec, split=args.split)
        sample = dataset[args.sample_index]
        raw_batch = {key: value for key, value in sample.items() if key != "action"}

    preprocessor, postprocessor = build_processors(
        input_skill_spec,
        load_stats(input_skill_spec),
        device=cfg.train.device,
        tokenizer_name_or_path=cfg.train.tokenizer_name_or_path,
    )
    proc_batch = preprocessor(raw_batch)
    feature_extractor = PI05PrefixFeatureExtractor(
        policy,
        disable_adapters=args.router_impl != ROUTER_IMPL_LORA_CONTROL,
    )
    if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
        activate_router_control_adapter(policy, adapter_name=args.router_control_adapter, trainable=False)
    with torch.no_grad():
        context = feature_extractor(proc_batch)
    if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
        router = build_lora_control_router_from_config(cfg, context_dim=context.shape[-1]).to(context.device)
        router.load_state_dict(
            torch.load(args.checkpoint_dir / f"{args.router_control_adapter}_head.pt", map_location=context.device)
        )
    else:
        router = build_hard_top1_router_from_config(cfg, context_dim=context.shape[-1]).to(context.device)
        router.load_state_dict(torch.load(args.checkpoint_dir / "router.pt", map_location=context.device))
    router.eval()

    with torch.no_grad():
        selected, probs = router.select(context, proc_batch["observation.state"].to(context.device))
        channel_index = int(selected[0].item())
        channel = cfg.channels[channel_index]
        if hasattr(policy, "set_adapter"):
            policy.set_adapter(channel.channel_id)
        pred = policy.predict_action_chunk(proc_batch)
        selected_skill_spec = load_skill_spec(cfg.skill_root, channel.skill_id)
        pred = crop_action(pred, selected_skill_spec.action_dim)
        pred = postprocessor(pred)

    print(
        json.dumps(
            {
                "selected_channel": channel.channel_id,
                "selected_skill_id": channel.skill_id,
                "router_probs": probs[0].detach().cpu().tolist(),
                "predicted_action": pred.squeeze(0).detach().cpu().tolist(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
