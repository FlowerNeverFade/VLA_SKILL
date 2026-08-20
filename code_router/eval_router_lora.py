#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vla_skill.dataset import load_skill_spec, load_stats
from vla_skill.pi05 import build_processors
from vla_skill_router.constants import DEFAULT_ROUTER_CONTROL_ADAPTER, ROUTER_IMPL_HARD_TOP1, ROUTER_IMPL_LORA_CONTROL
from vla_skill_router.config import load_experiment_config
from vla_skill_router.features import PI05PrefixFeatureExtractor
from vla_skill_router.real_runtime import (
    activate_channel_adapter,
    activate_router_control_adapter,
    build_dataset_for_channel,
    build_hard_top1_router_from_config,
    build_lora_control_router_from_config,
    load_first_skill_policy,
    load_or_initialize_channels,
    load_or_initialize_router_control_adapter,
    pi05_masked_policy_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate router accuracy and GT-channel policy loss.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--router-impl", choices=(ROUTER_IMPL_LORA_CONTROL, ROUTER_IMPL_HARD_TOP1), default=ROUTER_IMPL_LORA_CONTROL)
    parser.add_argument("--router-control-adapter", default=DEFAULT_ROUTER_CONTROL_ADAPTER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    policy = load_first_skill_policy(cfg)
    adapter_root = args.checkpoint_dir / "channels" if args.checkpoint_dir else None
    policy, _ = load_or_initialize_channels(cfg, policy, adapter_root=adapter_root, is_trainable=False)
    if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
        if args.checkpoint_dir is None:
            raise SystemExit("--router-impl lora_control requires --checkpoint-dir.")
        policy = load_or_initialize_router_control_adapter(
            cfg,
            policy,
            adapter_name=args.router_control_adapter,
            adapter_dir=args.checkpoint_dir / args.router_control_adapter,
            is_trainable=False,
            remove_existing_adapters=False,
        )
    feature_extractor = PI05PrefixFeatureExtractor(
        policy,
        disable_adapters=args.router_impl != ROUTER_IMPL_LORA_CONTROL,
    )

    datasets = [build_dataset_for_channel(cfg, index, split=args.split) for index in range(len(cfg.channels))]
    loaders = [
        DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
        for dataset in datasets
    ]
    preprocessors = []
    for channel in cfg.channels:
        skill_spec = load_skill_spec(cfg.skill_root, channel.skill_id)
        preprocessors.append(
            build_processors(
                skill_spec,
                load_stats(skill_spec),
                device=cfg.train.device,
                tokenizer_name_or_path=cfg.train.tokenizer_name_or_path,
            )[0]
        )

    first_raw = next(iter(loaders[0]))
    first_proc = preprocessors[0](first_raw)
    first_proc["channel_index"] = first_raw["channel_index"]
    first_proc["action_dim"] = first_raw["action_dim"]
    if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
        activate_router_control_adapter(policy, adapter_name=args.router_control_adapter, trainable=False)
    with torch.no_grad():
        first_context = feature_extractor(first_proc)
    if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
        router = build_lora_control_router_from_config(cfg, context_dim=first_context.shape[-1]).to(first_context.device)
        router.load_state_dict(
            torch.load(args.checkpoint_dir / f"{args.router_control_adapter}_head.pt", map_location=first_context.device)
        )
    else:
        router = build_hard_top1_router_from_config(cfg, context_dim=first_context.shape[-1]).to(first_context.device)
    if args.checkpoint_dir and args.router_impl == ROUTER_IMPL_HARD_TOP1:
        router.load_state_dict(torch.load(args.checkpoint_dir / "router.pt", map_location=first_context.device))
    router.eval()

    total = 0
    correct = 0
    policy_loss_sum = 0.0
    batches = 0
    with torch.no_grad():
        for channel_index, loader in enumerate(loaders):
            for batch_index, raw_batch in enumerate(loader):
                if batch_index >= args.max_batches:
                    break
                proc_batch = preprocessors[channel_index](raw_batch)
                proc_batch["channel_index"] = raw_batch["channel_index"]
                proc_batch["action_dim"] = raw_batch["action_dim"]
                if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
                    activate_router_control_adapter(policy, adapter_name=args.router_control_adapter, trainable=False)
                context = feature_extractor(proc_batch)
                logits = router(context, proc_batch["observation.state"].to(context.device)).logits
                labels = raw_batch["channel_index"].to(context.device)
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
                activate_channel_adapter(policy, cfg.channels[channel_index].channel_id)
                policy_loss_sum += float(pi05_masked_policy_loss(policy, proc_batch).item())
                batches += 1

    print(
        json.dumps(
            {
                "split": args.split,
                "router_accuracy": correct / max(1, total),
                "mean_policy_loss": policy_loss_sum / max(1, batches),
                "num_examples": total,
                "num_batches": batches,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
