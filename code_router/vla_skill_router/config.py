from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_LORA_GROUP,
    DEFAULT_ROUTER_CONTROL_ADAPTER,
    DEFAULT_ROUTER_CE_WEIGHT,
    DEFAULT_ROUTER_FEATURE_HOOK,
    DEFAULT_ROUTER_HIDDEN_DIM,
    DEFAULT_ROUTER_OUTPUT_ROOT,
    DEFAULT_ROUTER_STATE_EMBED_DIM,
    DEFAULT_ROUTER_TYPE,
    DEFAULT_SEED,
    DEFAULT_SKILL_ROOT,
    SUPPORTED_ROUTER_TYPES,
)


@dataclass(frozen=True)
class ChannelConfig:
    channel_id: str
    skill_id: str
    lora_group: str = DEFAULT_LORA_GROUP
    init_adapter_dir: Path | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChannelConfig":
        init_adapter_dir = payload.get("init_adapter_dir")
        return cls(
            channel_id=str(payload["channel_id"]),
            skill_id=str(payload["skill_id"]),
            lora_group=str(payload.get("lora_group", DEFAULT_LORA_GROUP)).upper(),
            init_adapter_dir=Path(init_adapter_dir) if init_adapter_dir else None,
        )


@dataclass(frozen=True)
class RouterConfig:
    type: str = DEFAULT_ROUTER_TYPE
    feature_hook: str = DEFAULT_ROUTER_FEATURE_HOOK
    state_embed_dim: int = DEFAULT_ROUTER_STATE_EMBED_DIM
    hidden_dim: int = DEFAULT_ROUTER_HIDDEN_DIM
    use_previous_skill: bool = False
    previous_skill_embed_dim: int | None = None
    context_dim: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RouterConfig":
        payload = payload or {}
        previous_skill_embed_dim = payload.get("previous_skill_embed_dim")
        context_dim = payload.get("context_dim")
        return cls(
            type=str(payload.get("type", DEFAULT_ROUTER_TYPE)),
            feature_hook=str(payload.get("feature_hook", DEFAULT_ROUTER_FEATURE_HOOK)),
            state_embed_dim=int(payload.get("state_embed_dim", DEFAULT_ROUTER_STATE_EMBED_DIM)),
            hidden_dim=int(payload.get("hidden_dim", DEFAULT_ROUTER_HIDDEN_DIM)),
            use_previous_skill=bool(payload.get("use_previous_skill", False)),
            previous_skill_embed_dim=int(previous_skill_embed_dim) if previous_skill_embed_dim is not None else None,
            context_dim=int(context_dim) if context_dim is not None else None,
        )


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    steps_per_channel: int | None = None
    batch_size: int = 4
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    num_workers: int = 2
    seed: int = DEFAULT_SEED
    lr: float = 2.5e-5
    router_lr: float = 1.0e-4
    router_ce_weight: float = DEFAULT_ROUTER_CE_WEIGHT
    device: str = "cuda"
    dtype: str = "bfloat16"
    tokenizer_name_or_path: str | None = None
    distributed_backend: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainConfig":
        payload = payload or {}
        steps_per_channel = payload.get("steps_per_channel")
        return cls(
            steps=int(payload.get("steps", 1000)),
            steps_per_channel=int(steps_per_channel) if steps_per_channel is not None else None,
            batch_size=int(payload.get("batch_size", 4)),
            eval_every=int(payload.get("eval_every", 100)),
            save_every=int(payload.get("save_every", payload.get("save_every_steps", 500))),
            log_every=int(payload.get("log_every", 10)),
            num_workers=int(payload.get("num_workers", 2)),
            seed=int(payload.get("seed", DEFAULT_SEED)),
            lr=float(payload.get("lr", payload.get("learning_rate", 2.5e-5))),
            router_lr=float(payload.get("router_lr", 1.0e-4)),
            router_ce_weight=float(payload.get("router_ce_weight", DEFAULT_ROUTER_CE_WEIGHT)),
            device=str(payload.get("device", "cuda")),
            dtype=str(payload.get("dtype", "bfloat16")),
            tokenizer_name_or_path=payload.get("tokenizer_name_or_path"),
            distributed_backend=(
                str(payload["distributed_backend"]) if payload.get("distributed_backend") is not None else None
            ),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    channels: tuple[ChannelConfig, ...]
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH
    skill_root: Path = DEFAULT_SKILL_ROOT
    output_root: Path = DEFAULT_ROUTER_OUTPUT_ROOT
    router: RouterConfig = field(default_factory=RouterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        channels_payload = payload.get("channels")
        if not isinstance(channels_payload, list) or not channels_payload:
            raise ValueError("Router experiment config must define a non-empty `channels` list.")
        cfg = cls(
            channels=tuple(ChannelConfig.from_dict(item) for item in channels_payload),
            base_model_path=Path(payload.get("base_model_path", DEFAULT_BASE_MODEL_PATH)),
            skill_root=Path(payload.get("skill_root", DEFAULT_SKILL_ROOT)),
            output_root=Path(payload.get("output_root", DEFAULT_ROUTER_OUTPUT_ROOT)),
            router=RouterConfig.from_dict(payload.get("router")),
            train=TrainConfig.from_dict(payload.get("train")),
            run_name=payload.get("run_name"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        channel_ids = [item.channel_id for item in self.channels]
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError(f"Duplicate channel_id values are not allowed: {channel_ids}.")
        skill_ids = [item.skill_id for item in self.channels]
        if len(set(skill_ids)) != len(skill_ids):
            raise ValueError(f"Each skill_id may appear in only one channel for v1: {skill_ids}.")
        invalid_groups = [item.lora_group for item in self.channels if item.lora_group != DEFAULT_LORA_GROUP]
        if invalid_groups:
            raise ValueError(
                f"v1 router channels must use LoRA group {DEFAULT_LORA_GROUP}; got {invalid_groups}."
            )
        if DEFAULT_ROUTER_CONTROL_ADAPTER in channel_ids:
            raise ValueError(f"`{DEFAULT_ROUTER_CONTROL_ADAPTER}` is reserved for the router-control LoRA adapter.")
        if self.router.type not in SUPPORTED_ROUTER_TYPES:
            raise ValueError(f"Unsupported router.type={self.router.type!r}; expected one of {SUPPORTED_ROUTER_TYPES!r}.")
        if self.router.state_embed_dim <= 0:
            raise ValueError("router.state_embed_dim must be positive.")
        if self.router.hidden_dim <= 0:
            raise ValueError("router.hidden_dim must be positive.")
        if self.train.steps <= 0:
            raise ValueError("train.steps must be positive.")
        if self.train.steps_per_channel is not None and self.train.steps_per_channel <= 0:
            raise ValueError("train.steps_per_channel must be positive when set.")
        if self.train.batch_size <= 0:
            raise ValueError("train.batch_size must be positive.")

    @property
    def channel_id_to_index(self) -> dict[str, int]:
        return {item.channel_id: index for index, item in enumerate(self.channels)}

    @property
    def skill_id_to_index(self) -> dict[str, int]:
        return {item.skill_id: index for index, item in enumerate(self.channels)}

    @property
    def channel_ids(self) -> list[str]:
        return [item.channel_id for item in self.channels]

    @property
    def skill_ids(self) -> list[str]:
        return [item.skill_id for item in self.channels]


def load_experiment_config(path: Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object.")
    return ExperimentConfig.from_dict(payload)
