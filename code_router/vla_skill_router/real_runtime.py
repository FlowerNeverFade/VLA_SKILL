from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft import get_peft_model

from vla_skill.dataset import build_single_observation_batch, build_window_dataset, load_obs_json, load_skill_spec, load_stats
from vla_skill.lora import build_lora_config
from vla_skill.pi05 import build_processors, load_base_policy, make_skill_policy_config
from vla_skill.schema import SkillSpec

from .action import crop_action
from .adapters import resolve_adapter_dir, set_only_named_adapter_trainable
from .config import ExperimentConfig
from .constants import DEFAULT_ROUTER_CONTROL_ADAPTER, ROUTER_IMPL_LORA_CONTROL
from .features import PI05PrefixFeatureExtractor
from .policy_cache import DEFAULT_IMAGE_STORAGE_DTYPE, PolicyCacheError, PolicyCacheIterableDataset
from .router_model import HardTop1Router, LoRAControlRouter


def build_router_from_config(cfg: ExperimentConfig, *, context_dim: int):
    if cfg.router.type == ROUTER_IMPL_LORA_CONTROL:
        return build_lora_control_router_from_config(cfg, context_dim=context_dim)
    return build_hard_top1_router_from_config(cfg, context_dim=context_dim)


def build_hard_top1_router_from_config(cfg: ExperimentConfig, *, context_dim: int) -> HardTop1Router:
    return HardTop1Router(
        context_dim=context_dim,
        num_channels=len(cfg.channels),
        state_embed_dim=cfg.router.state_embed_dim,
        hidden_dim=cfg.router.hidden_dim,
        use_previous_skill=cfg.router.use_previous_skill,
        previous_skill_embed_dim=cfg.router.previous_skill_embed_dim,
    )


def build_lora_control_router_from_config(cfg: ExperimentConfig, *, context_dim: int) -> LoRAControlRouter:
    return LoRAControlRouter(
        context_dim=context_dim,
        num_channels=len(cfg.channels),
        state_embed_dim=cfg.router.state_embed_dim,
    )


def make_router_shell_skill_spec(cfg: ExperimentConfig) -> SkillSpec:
    skill_specs = [load_skill_spec(cfg.skill_root, channel.skill_id) for channel in cfg.channels]
    return SkillSpec(
        skill_id="router_shell",
        display_name="Router Shell",
        state_dim=max(spec.state_dim for spec in skill_specs),
        action_dim=max(spec.action_dim for spec in skill_specs),
        camera_names=skill_specs[0].camera_names,
        chunk_size=max(spec.chunk_size for spec in skill_specs),
        n_action_steps=max(spec.n_action_steps for spec in skill_specs),
    )


def load_router_shell_policy(cfg: ExperimentConfig):
    skill_spec = make_router_shell_skill_spec(cfg)
    config = make_skill_policy_config(
        skill_spec,
        base_model_path=cfg.base_model_path,
        device=cfg.train.device,
        dtype=cfg.train.dtype,
    )
    return load_base_policy(config, base_model_path=cfg.base_model_path, strict=True)


def load_first_skill_policy(cfg: ExperimentConfig):
    return load_router_shell_policy(cfg)


def _resolve_channel_adapter_dir(cfg: ExperimentConfig, channel, *, adapter_root: Path | None = None) -> Path | None:
    if adapter_root is not None:
        candidate = Path(adapter_root) / channel.channel_id
        if candidate.exists():
            return candidate
    return resolve_adapter_dir(
        skill_output_dir=cfg.output_root / channel.skill_id,
        explicit_adapter_dir=channel.init_adapter_dir,
    ).adapter_dir


def load_or_initialize_channels(
    cfg: ExperimentConfig,
    policy,
    *,
    adapter_root: Path | None = None,
    is_trainable: bool = True,
) -> tuple[Any, dict[str, Path | None]]:
    loaded: dict[str, Path | None] = {}
    peft_model = policy
    if not isinstance(policy, PeftModel):
        for param in policy.parameters():
            param.requires_grad_(False)
        first = cfg.channels[0]
        first_adapter_dir = _resolve_channel_adapter_dir(cfg, first, adapter_root=adapter_root)
        if first_adapter_dir is not None:
            peft_model = PeftModel.from_pretrained(
                policy,
                str(first_adapter_dir),
                adapter_name=first.channel_id,
                is_trainable=is_trainable,
            )
            loaded[first.channel_id] = first_adapter_dir
        else:
            peft_model = get_peft_model(
                policy,
                build_lora_config(
                    "C",
                    base_model_name_or_path=str(cfg.base_model_path),
                    inference_mode=False,
                ),
                adapter_name=first.channel_id,
            )
            loaded[first.channel_id] = None
    for index, channel in enumerate(cfg.channels):
        adapter_name = channel.channel_id
        if index == 0:
            continue
        adapter_dir = _resolve_channel_adapter_dir(cfg, channel, adapter_root=adapter_root)
        if adapter_dir is not None:
            peft_model.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=is_trainable)
            loaded[adapter_name] = adapter_dir
        else:
            peft_model.add_adapter(
                adapter_name,
                peft_config=build_lora_config(
                    "C",
                    base_model_name_or_path=str(cfg.base_model_path),
                    inference_mode=False,
                ),
            )
            loaded[adapter_name] = None
    if not is_trainable:
        for param in peft_model.parameters():
            param.requires_grad_(False)
    return peft_model, loaded


def load_or_initialize_router_control_adapter(
    cfg: ExperimentConfig,
    policy,
    *,
    adapter_name: str = DEFAULT_ROUTER_CONTROL_ADAPTER,
    adapter_dir: Path | None = None,
    is_trainable: bool = True,
    remove_existing_adapters: bool = True,
):
    if adapter_name in cfg.channel_ids:
        raise ValueError(f"`{adapter_name}` is reserved for router control and cannot also be a skill channel.")
    peft_model = policy
    if not isinstance(policy, PeftModel):
        for param in policy.parameters():
            param.requires_grad_(False)
        if adapter_dir is not None and Path(adapter_dir).exists():
            peft_model = PeftModel.from_pretrained(
                policy,
                str(adapter_dir),
                adapter_name=adapter_name,
                is_trainable=is_trainable,
            )
        else:
            peft_model = get_peft_model(
                policy,
                build_lora_config(
                    "C",
                    base_model_name_or_path=str(cfg.base_model_path),
                    inference_mode=not is_trainable,
                ),
                adapter_name=adapter_name,
            )
    else:
        if remove_existing_adapters:
            for existing_adapter in list(policy.peft_config):
                if existing_adapter != adapter_name:
                    policy.delete_adapter(existing_adapter)
        if adapter_name not in policy.peft_config:
            if adapter_dir is not None and Path(adapter_dir).exists():
                policy.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=is_trainable)
            else:
                policy.add_adapter(
                    adapter_name,
                    peft_config=build_lora_config(
                        "C",
                        base_model_name_or_path=str(cfg.base_model_path),
                        inference_mode=not is_trainable,
                    ),
                )
        peft_model = policy
    activate_router_control_adapter(peft_model, adapter_name=adapter_name, trainable=is_trainable)
    return peft_model


def load_or_initialize_single_channel(
    cfg: ExperimentConfig,
    policy,
    channel_index: int,
) -> tuple[Any, Path | None]:
    channel = cfg.channels[channel_index]
    adapter_name = channel.channel_id
    loaded_adapter_dir: Path | None = None
    peft_model = policy

    if not isinstance(policy, PeftModel):
        for param in policy.parameters():
            param.requires_grad_(False)
        resolved = resolve_adapter_dir(
            skill_output_dir=cfg.output_root / channel.skill_id,
            explicit_adapter_dir=channel.init_adapter_dir,
        )
        if resolved.adapter_dir is not None:
            peft_model = PeftModel.from_pretrained(
                policy,
                str(resolved.adapter_dir),
                adapter_name=adapter_name,
                is_trainable=True,
            )
            loaded_adapter_dir = resolved.adapter_dir
        else:
            peft_model = get_peft_model(
                policy,
                build_lora_config(
                    "C",
                    base_model_name_or_path=str(cfg.base_model_path),
                    inference_mode=False,
                ),
                adapter_name=adapter_name,
            )
    else:
        for existing_adapter in list(policy.peft_config):
            if existing_adapter != adapter_name:
                policy.delete_adapter(existing_adapter)
        if adapter_name not in policy.peft_config:
            resolved = resolve_adapter_dir(
                skill_output_dir=cfg.output_root / channel.skill_id,
                explicit_adapter_dir=channel.init_adapter_dir,
            )
            if resolved.adapter_dir is not None:
                policy.load_adapter(str(resolved.adapter_dir), adapter_name=adapter_name, is_trainable=True)
                loaded_adapter_dir = resolved.adapter_dir
            else:
                policy.add_adapter(
                    adapter_name,
                    peft_config=build_lora_config(
                        "C",
                        base_model_name_or_path=str(cfg.base_model_path),
                        inference_mode=False,
                    ),
                )
        peft_model = policy

    activate_channel_adapter(peft_model, adapter_name)
    return peft_model, loaded_adapter_dir


def activate_channel_adapter(policy, channel_id: str) -> None:
    if hasattr(policy, "set_adapter"):
        policy.set_adapter(channel_id)
    set_only_named_adapter_trainable(policy, channel_id)


def activate_router_control_adapter(
    policy,
    *,
    adapter_name: str = DEFAULT_ROUTER_CONTROL_ADAPTER,
    trainable: bool = False,
) -> None:
    if hasattr(policy, "set_adapter"):
        policy.set_adapter(adapter_name)
    if trainable:
        set_only_named_adapter_trainable(policy, adapter_name)
    else:
        for param in policy.parameters():
            param.requires_grad_(False)


def iter_adapter_parameters(policy):
    for name, param in policy.named_parameters():
        if "lora_" in name or "adapter" in name:
            yield param


def iter_trainable_parameters(module):
    for param in module.parameters():
        if param.requires_grad:
            yield param


def _resolve_pi05_policy(policy):
    candidates = []
    base_model = getattr(policy, "base_model", None)
    if base_model is not None:
        candidates.append(getattr(base_model, "model", None))
    candidates.append(getattr(policy, "model", None))
    candidates.append(policy)
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "_preprocess_images") and hasattr(candidate, "prepare_action"):
            return candidate
    raise TypeError("Could not resolve the underlying PI05Policy from the policy wrapper.")


def pi05_masked_policy_loss(policy, batch: dict[str, Any]) -> torch.Tensor:
    from lerobot.policies.pi05 import modeling_pi05 as pi05_modeling

    pi05_policy = _resolve_pi05_policy(policy)
    images, img_masks = pi05_policy._preprocess_images(batch)
    tokens = batch[pi05_modeling.OBS_LANGUAGE_TOKENS]
    masks = batch[pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK]
    actions = pi05_policy.prepare_action(batch)
    losses = pi05_policy.model.forward(images, img_masks, tokens, masks, actions)
    action_dims = batch["action_dim"].to(device=losses.device, dtype=torch.long)
    dim_ids = torch.arange(losses.shape[-1], device=losses.device)[None, None, :]
    mask = dim_ids < action_dims[:, None, None]
    return (losses * mask).sum() / (mask.sum().clamp_min(1) * losses.shape[1])


def build_dataset_for_channel(
    cfg: ExperimentConfig,
    channel_index: int,
    split: str,
    *,
    cache_root: Path | None = None,
    require_policy_cache: bool = False,
    cache_rank: int = 0,
    cache_world_size: int = 1,
    cache_seed: int | None = None,
    cache_image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
):
    from .data import ChannelMeta, SkillChannelDataset

    channel = cfg.channels[channel_index]
    skill_spec = load_skill_spec(cfg.skill_root, channel.skill_id)
    if cache_root is not None and skill_spec.source.type == "lerobot":
        try:
            return PolicyCacheIterableDataset(
                cache_root=cache_root,
                skill_spec=skill_spec,
                split=split,
                tokenizer_name_or_path=cfg.train.tokenizer_name_or_path,
                image_storage_dtype=cache_image_storage_dtype,
                seed=cfg.train.seed + channel_index if cache_seed is None else cache_seed,
                rank=cache_rank,
                world_size=cache_world_size,
                shuffle=split == "train",
            )
        except (FileNotFoundError, PolicyCacheError):
            if require_policy_cache:
                raise
    elif require_policy_cache and skill_spec.source.type == "lerobot":
        raise FileNotFoundError(
            f"Policy cache is required but no cache_root was provided for skill={channel.skill_id} split={split}."
        )
    dataset = build_window_dataset(skill_spec, split=split)
    return SkillChannelDataset(
        dataset,
        ChannelMeta(
            channel_id=channel.channel_id,
            skill_id=channel.skill_id,
            channel_index=channel_index,
            action_dim=skill_spec.action_dim,
        ),
    )


def build_obs_batch(cfg: ExperimentConfig, *, obs_json: Path, skill_id: str):
    payload = load_obs_json(obs_json)
    skill_spec = load_skill_spec(cfg.skill_root, skill_id)
    image_paths = payload.get("image_paths") or payload.get("images")
    if not isinstance(image_paths, dict):
        raise ValueError("obs.json must contain `image_paths` or `images`.")
    return skill_spec, build_single_observation_batch(
        skill_spec,
        task=str(payload.get("task", skill_spec.display_name)),
        state=payload["state"],
        image_paths=image_paths,
    )


@torch.no_grad()
def route_and_predict(
    *,
    cfg: ExperimentConfig,
    policy,
    router: HardTop1Router,
    raw_batch: dict[str, Any],
    channel_specs_by_index: dict[int, Any],
) -> dict[str, Any]:
    feature_extractor = PI05PrefixFeatureExtractor(policy)
    preprocessor, postprocessor = None, None
    # The caller should pass a preprocessed batch for real inference if using this low-level helper.
    pooled_context = feature_extractor(raw_batch)
    selected, probs = router.select(
        pooled_context,
        raw_batch["observation.state"].to(device=pooled_context.device),
    )
    channel_index = int(selected[0].item())
    channel = cfg.channels[channel_index]
    if hasattr(policy, "set_adapter"):
        policy.set_adapter(channel.channel_id)
    pred = policy.predict_action_chunk(raw_batch)
    skill_spec = channel_specs_by_index[channel_index]
    pred = crop_action(pred, skill_spec.action_dim)
    return {
        "selected_channel": channel.channel_id,
        "selected_skill_id": channel.skill_id,
        "router_probs": probs[0].detach().cpu().tolist(),
        "predicted_action": pred.detach().cpu().squeeze(0).tolist(),
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
    }
