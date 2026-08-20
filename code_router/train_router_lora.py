#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vla_skill.dataset import load_skill_spec, load_stats
from vla_skill.io_utils import ensure_dir, json_ready, timestamp_run_name, utc_now_iso, write_json, write_yaml
from vla_skill.pi05 import build_processors
from vla_skill_router.config import ExperimentConfig, load_experiment_config
from vla_skill_router.distributed import (
    DistributedInfo,
    barrier,
    broadcast_object,
    cleanup_distributed,
    effective_global_batch_size,
    init_distributed,
    reduce_mean_scalar,
)
from vla_skill_router.features import PI05PrefixFeatureExtractor
from vla_skill_router.real_runtime import (
    build_dataset_for_channel,
    build_router_from_config,
    iter_adapter_parameters,
    load_first_skill_policy,
    load_or_initialize_channels,
    pi05_masked_policy_loss,
)


class CyclingLoader:
    def __init__(self, loader: DataLoader, sampler: DistributedSampler | None = None):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.iterator = None

    def next(self) -> dict[str, Any]:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def reset(self) -> None:
        self.epoch = 0
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.iterator = None


class PolicyLossWrapper(nn.Module):
    def __init__(self, policy: Any):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict[str, Any], channel_id: str) -> torch.Tensor:
        self.policy.train()
        _activate_adapter_without_grad_toggle(self.policy, channel_id)
        return pi05_masked_policy_loss(self.policy, batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-skill PI05 LoRA channels plus a hard top-1 router.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--steps-per-channel", type=int)
    parser.add_argument("--confirm-long-run", action="store_true")
    return parser.parse_args()


def _unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _adapter_disable_context(policy):
    disable_adapter = getattr(policy, "disable_adapter", None)
    if disable_adapter is None:
        return nullcontext()
    return disable_adapter()


def _activate_adapter_without_grad_toggle(policy, channel_id: str) -> None:
    """Switch the active PEFT adapter without changing DDP-visible trainability."""
    changed = False
    if hasattr(policy, "active_adapter"):
        policy.active_adapter = channel_id
        changed = True
    base_model = getattr(policy, "base_model", None)
    if base_model is not None and hasattr(base_model, "active_adapter"):
        base_model.active_adapter = channel_id
        changed = True
    for module in policy.modules() if hasattr(policy, "modules") else ():
        if hasattr(module, "_active_adapter"):
            module._active_adapter = [channel_id]
            changed = True
    if not changed and hasattr(policy, "set_adapter"):
        policy.set_adapter(channel_id)


def _set_lora_parameters_trainable(policy) -> None:
    for name, param in policy.named_parameters():
        param.requires_grad = "lora_" in name or "adapter" in name


def _make_loader(dataset, cfg: ExperimentConfig, dist_info: DistributedInfo) -> CyclingLoader:
    sampler = None
    if dist_info.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.device.startswith("cuda"),
        persistent_workers=cfg.train.num_workers > 0,
        drop_last=False,
    )
    return CyclingLoader(loader, sampler=sampler)


def _prepare_batch(raw_batch, preprocessor):
    proc_batch = preprocessor(raw_batch)
    for key in ("channel_index", "action_dim"):
        proc_batch[key] = raw_batch[key]
    return proc_batch


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _save_single_adapter(policy, path: Path, channel_id: str) -> None:
    ensure_dir(path)
    try:
        policy.save_pretrained(str(path), selected_adapters=[channel_id])
    except TypeError:
        _activate_adapter_without_grad_toggle(policy, channel_id)
        policy.save_pretrained(str(path))


def _save_checkpoint(
    *,
    run_dir: Path,
    cfg: ExperimentConfig,
    policy,
    router,
    policy_optimizer: torch.optim.Optimizer,
    router_optimizer: torch.optim.Optimizer,
    step: int,
    channel_steps: list[int],
    dist_info: DistributedInfo,
    final: bool = False,
) -> None:
    if not dist_info.is_rank0:
        return
    ensure_dir(run_dir)
    unwrapped_router = _unwrap(router)
    write_json(
        run_dir / "router_meta.json",
        {
            "step": step,
            "channel_steps": channel_steps,
            "context_dim": unwrapped_router.context_dim,
            "channel_ids": cfg.channel_ids,
            "skill_ids": cfg.skill_ids,
            "distributed": {
                "world_size": dist_info.world_size,
                "backend": dist_info.backend,
                "per_device_batch_size": cfg.train.batch_size,
                "effective_global_batch_size": effective_global_batch_size(cfg.train.batch_size, dist_info.world_size),
            },
            "final": final,
            "saved_at": utc_now_iso(),
        },
    )
    torch.save(unwrapped_router.state_dict(), run_dir / "router.pt")
    torch.save(
        {
            "step": step,
            "channel_steps": channel_steps,
            "policy_optimizer": policy_optimizer.state_dict(),
            "router_optimizer": router_optimizer.state_dict(),
            "rng_state": _rng_state(),
            "distributed": {
                "world_size": dist_info.world_size,
                "backend": dist_info.backend,
            },
        },
        run_dir / "trainer_state.pt",
    )
    channels_dir = ensure_dir(run_dir / "channels")
    for channel in cfg.channels:
        _save_single_adapter(policy, channels_dir / channel.channel_id, channel.channel_id)


def _next_channel(
    *,
    global_step: int,
    channel_cursor: int,
    channel_steps: list[int],
    steps_per_channel: int | None,
) -> tuple[int | None, int]:
    if steps_per_channel is None:
        channel_index = (global_step - 1) % len(channel_steps)
        return channel_index, channel_index + 1

    for offset in range(len(channel_steps)):
        channel_index = (channel_cursor + offset) % len(channel_steps)
        if channel_steps[channel_index] < steps_per_channel:
            return channel_index, (channel_index + 1) % len(channel_steps)
    return None, channel_cursor


def _resolve_steps_per_channel(args: argparse.Namespace, cfg: ExperimentConfig) -> int | None:
    return args.steps_per_channel if args.steps_per_channel is not None else cfg.train.steps_per_channel


def _validate_long_run(args: argparse.Namespace, cfg: ExperimentConfig, steps_per_channel: int | None) -> None:
    if steps_per_channel is None:
        return
    total_updates = steps_per_channel * len(cfg.channels)
    if steps_per_channel >= 20_000 and not args.confirm_long_run:
        raise SystemExit(
            f"Refusing long run without --confirm-long-run: "
            f"{len(cfg.channels)} channels * {steps_per_channel} steps = {total_updates} updates."
        )


def _resolved_config_payload(
    *,
    config_path: Path,
    cfg: ExperimentConfig,
    steps_per_channel: int | None,
    total_updates: int,
    dist_info: DistributedInfo,
) -> dict[str, Any]:
    return json_ready(
        {
            "config_path": str(config_path),
            "base_model_path": str(cfg.base_model_path),
            "skill_root": str(cfg.skill_root),
            "output_root": str(cfg.output_root),
            "channels": [channel.__dict__ for channel in cfg.channels],
            "router": cfg.router.__dict__,
            "train": cfg.train.__dict__,
            "steps_per_channel": steps_per_channel,
            "total_micro_updates": total_updates,
            "distributed": {
                "enabled": dist_info.is_distributed,
                "backend": dist_info.backend,
                "rank": dist_info.rank,
                "local_rank": dist_info.local_rank,
                "world_size": dist_info.world_size,
                "per_device_batch_size": cfg.train.batch_size,
                "effective_global_batch_size": effective_global_batch_size(cfg.train.batch_size, dist_info.world_size),
            },
        }
    )


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    steps_per_channel = _resolve_steps_per_channel(args, cfg)
    _validate_long_run(args, cfg, steps_per_channel)

    dist_info = init_distributed(
        backend=cfg.train.distributed_backend,
        device_hint=cfg.train.device,
    )
    if dist_info.is_distributed and cfg.train.device.startswith("cuda"):
        from dataclasses import replace

        cfg = replace(cfg, train=replace(cfg.train, device=f"cuda:{dist_info.local_rank}"))

    run_name = args.run_name or cfg.run_name
    if dist_info.is_distributed:
        if dist_info.is_rank0:
            run_name = run_name or timestamp_run_name("router_lora")
        run_name = broadcast_object(run_name, dist_info)
    else:
        run_name = run_name or timestamp_run_name("router_lora")
    run_dir = cfg.output_root / "runs" / run_name
    total_updates = steps_per_channel * len(cfg.channels) if steps_per_channel is not None else cfg.train.steps

    try:
        if dist_info.is_rank0:
            ensure_dir(run_dir)
            write_yaml(
                run_dir / "resolved_config.yaml",
                _resolved_config_payload(
                    config_path=args.config,
                    cfg=cfg,
                    steps_per_channel=steps_per_channel,
                    total_updates=total_updates,
                    dist_info=dist_info,
                ),
            )
        barrier(dist_info)

        torch.manual_seed(cfg.train.seed + dist_info.rank)
        random.seed(cfg.train.seed + dist_info.rank)
        np.random.seed(cfg.train.seed + dist_info.rank)

        policy = load_first_skill_policy(cfg)
        policy, loaded_adapters = load_or_initialize_channels(cfg, policy)
        _set_lora_parameters_trainable(policy)

        datasets = [build_dataset_for_channel(cfg, index, split="train") for index in range(len(cfg.channels))]
        loaders = [_make_loader(dataset, cfg, dist_info) for dataset in datasets]
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

        feature_extractor = PI05PrefixFeatureExtractor(policy)
        first_raw = loaders[0].next()
        first_proc = _prepare_batch(first_raw, preprocessors[0])
        with torch.no_grad(), _adapter_disable_context(policy):
            first_context = feature_extractor(first_proc)
        loaders[0].reset()
        router = build_router_from_config(cfg, context_dim=first_context.shape[-1]).to(first_context.device)

        policy_loss_model: nn.Module = PolicyLossWrapper(policy)
        router_model: nn.Module = router
        if dist_info.is_distributed:
            device_ids = [dist_info.local_rank] if cfg.train.device.startswith("cuda") and torch.cuda.is_available() else None
            policy_loss_model = DistributedDataParallel(
                policy_loss_model,
                device_ids=device_ids,
                find_unused_parameters=True,
            )
            router_model = DistributedDataParallel(router_model, device_ids=device_ids)

        policy_optimizer = torch.optim.AdamW(list(iter_adapter_parameters(policy)), lr=cfg.train.lr)
        router_optimizer = torch.optim.AdamW(_unwrap(router_model).parameters(), lr=cfg.train.router_lr)

        metrics_path = run_dir / "metrics.jsonl"
        if dist_info.is_rank0:
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "loaded_adapters": {key: str(value) if value else None for key, value in loaded_adapters.items()},
                        "distributed": {
                            "world_size": dist_info.world_size,
                            "backend": dist_info.backend,
                            "per_device_batch_size": cfg.train.batch_size,
                            "effective_global_batch_size": effective_global_batch_size(
                                cfg.train.batch_size,
                                dist_info.world_size,
                            ),
                        },
                        "steps_per_channel": steps_per_channel,
                        "total_micro_updates": total_updates,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        channel_steps = [0 for _ in cfg.channels]
        channel_cursor = 0
        for global_step in range(1, total_updates + 1):
            channel_index, channel_cursor = _next_channel(
                global_step=global_step,
                channel_cursor=channel_cursor,
                channel_steps=channel_steps,
                steps_per_channel=steps_per_channel,
            )
            if channel_index is None:
                break

            channel = cfg.channels[channel_index]
            raw_batch = loaders[channel_index].next()
            proc_batch = _prepare_batch(raw_batch, preprocessors[channel_index])
            labels = proc_batch["channel_index"]
            if labels.ndim == 0:
                labels = labels[None]
            labels = labels.to(device=first_context.device, dtype=torch.long)

            policy_optimizer.zero_grad(set_to_none=True)
            router_optimizer.zero_grad(set_to_none=True)
            policy_loss_model.train()
            router_model.train()

            with torch.no_grad(), _adapter_disable_context(policy):
                pooled_context = feature_extractor(proc_batch)
            router_out = router_model(
                pooled_context.detach(),
                proc_batch["observation.state"].to(device=pooled_context.device),
            )
            router_ce_loss = F.cross_entropy(router_out.logits, labels)
            policy_loss = policy_loss_model(proc_batch, channel.channel_id)
            loss = policy_loss + cfg.train.router_ce_weight * router_ce_loss
            loss.backward()
            policy_optimizer.step()
            router_optimizer.step()
            channel_steps[channel_index] += 1

            with torch.no_grad():
                pred = torch.argmax(router_out.logits, dim=-1)
                router_accuracy = (pred == labels).float().mean().item()
            device = pooled_context.device
            record = {
                "kind": "train",
                "step": global_step,
                "channel_step": channel_steps[channel_index],
                "channel_id": channel.channel_id,
                "skill_id": channel.skill_id,
                "loss": reduce_mean_scalar(float(loss.detach().cpu().item()), device=device, info=dist_info),
                "policy_loss": reduce_mean_scalar(float(policy_loss.detach().cpu().item()), device=device, info=dist_info),
                "router_ce_loss": reduce_mean_scalar(
                    float(router_ce_loss.detach().cpu().item()),
                    device=device,
                    info=dist_info,
                ),
                "router_accuracy": reduce_mean_scalar(router_accuracy, device=device, info=dist_info),
                "timestamp": utc_now_iso(),
            }
            if dist_info.is_rank0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
                if global_step == 1 or global_step % cfg.train.log_every == 0 or global_step == total_updates:
                    print(json.dumps(record, ensure_ascii=False), flush=True)

            if dist_info.is_rank0 and (global_step % cfg.train.save_every == 0 or global_step == total_updates):
                _save_checkpoint(
                    run_dir=run_dir,
                    cfg=cfg,
                    policy=policy,
                    router=router_model,
                    policy_optimizer=policy_optimizer,
                    router_optimizer=router_optimizer,
                    step=global_step,
                    channel_steps=channel_steps,
                    dist_info=dist_info,
                    final=global_step == total_updates,
                )

        if dist_info.is_rank0:
            write_json(
                run_dir / "summary.json",
                {
                    "run_dir": str(run_dir),
                    "completed_at": utc_now_iso(),
                    "channel_steps": channel_steps,
                    "total_micro_updates": sum(channel_steps),
                    "distributed": {
                        "world_size": dist_info.world_size,
                        "backend": dist_info.backend,
                    },
                },
            )
        barrier(dist_info)
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
