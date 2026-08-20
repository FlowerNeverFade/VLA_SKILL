from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from peft import PeftModel

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .constants import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_LOCAL_TOKENIZER_PATH,
    DEFAULT_TOKENIZER_REPO,
    IMAGE_FIELDS,
    IMAGE_RESOLUTION,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
)
from .schema import SkillSpec
from .stats import stats_to_torch


def _load_base_config_dict(base_model_path: Path) -> dict[str, Any]:
    with (base_model_path / "config.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.pop("type", None)
    return payload


def _as_policy_feature(feature: PolicyFeature | dict[str, Any]) -> PolicyFeature:
    if isinstance(feature, PolicyFeature):
        return feature
    feature_type = feature["type"]
    if isinstance(feature_type, str):
        feature_type = FeatureType(feature_type)
    return PolicyFeature(type=feature_type, shape=tuple(feature["shape"]))


def _normalize_config_features(config: Any) -> Any:
    if config.input_features:
        config.input_features = {name: _as_policy_feature(feature) for name, feature in config.input_features.items()}
    if config.output_features:
        config.output_features = {
            name: _as_policy_feature(feature) for name, feature in config.output_features.items()
        }
    return config


def make_skill_policy_config(
    skill_spec: SkillSpec,
    *,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
    train_expert_only: bool = True,
    gradient_checkpointing: bool = False,
    compile_model: bool = False,
) -> Any:
    payload = _load_base_config_dict(base_model_path)
    payload["device"] = device
    payload["dtype"] = dtype
    payload["push_to_hub"] = False
    payload["repo_id"] = None
    payload["private"] = None
    payload["chunk_size"] = skill_spec.chunk_size
    payload["n_action_steps"] = skill_spec.n_action_steps
    payload["max_state_dim"] = MAX_STATE_DIM
    payload["max_action_dim"] = MAX_ACTION_DIM
    payload["gradient_checkpointing"] = gradient_checkpointing
    payload["compile_model"] = compile_model
    payload["train_expert_only"] = train_expert_only
    payload["input_features"] = {
        image_field: {"type": "VISUAL", "shape": [3, *IMAGE_RESOLUTION]} for image_field in IMAGE_FIELDS
    }
    payload["input_features"]["observation.state"] = {"type": "STATE", "shape": [skill_spec.state_dim]}
    payload["output_features"] = {
        "action": {"type": "ACTION", "shape": [skill_spec.action_dim]},
    }
    config = make_policy_config("pi05", **payload)
    config.pretrained_path = Path(base_model_path)
    return _normalize_config_features(config)


def make_generic_inference_config(
    *,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Any:
    payload = _load_base_config_dict(base_model_path)
    payload["device"] = device
    payload["dtype"] = dtype
    payload["push_to_hub"] = False
    payload["repo_id"] = None
    payload["private"] = None
    config = make_policy_config("pi05", **payload)
    config.pretrained_path = Path(base_model_path)
    return _normalize_config_features(config)


def load_base_policy(
    config: Any,
    *,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    strict: bool = True,
) -> PI05Policy:
    return PI05Policy.from_pretrained(
        str(base_model_path),
        config=config,
        local_files_only=True,
        strict=strict,
    )


def load_skill_peft_policy(
    skill_spec: SkillSpec,
    adapter_dir: Path,
    *,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> PeftModel:
    config = make_skill_policy_config(skill_spec, base_model_path=base_model_path, device=device, dtype=dtype)
    policy = load_base_policy(config, base_model_path=base_model_path, strict=True)
    peft_model = PeftModel.from_pretrained(policy, str(adapter_dir), is_trainable=False)
    peft_model.eval()
    return peft_model


def load_skill_base_policy(
    skill_spec: SkillSpec,
    *,
    base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> PI05Policy:
    config = make_skill_policy_config(skill_spec, base_model_path=base_model_path, device=device, dtype=dtype)
    policy = load_base_policy(config, base_model_path=base_model_path, strict=True)
    policy.eval()
    return policy


def resolve_tokenizer_name_or_path(tokenizer_name_or_path: str | Path | None = None) -> str:
    if tokenizer_name_or_path is not None:
        return str(tokenizer_name_or_path)

    env_override = os.environ.get("PI05_TOKENIZER_PATH")
    if env_override:
        return env_override

    if DEFAULT_LOCAL_TOKENIZER_PATH.is_dir():
        return str(DEFAULT_LOCAL_TOKENIZER_PATH)

    allow_patterns = ["tokenizer*", "*.model", "special_tokens_map.json", "config.json"]
    try:
        return snapshot_download(
            DEFAULT_TOKENIZER_REPO,
            allow_patterns=allow_patterns,
            local_files_only=True,
            max_workers=2,
        )
    except Exception:
        return DEFAULT_TOKENIZER_REPO


def make_pi05_processors(
    config: Any,
    dataset_stats: dict[str, dict[str, Any]],
    *,
    tokenizer_name_or_path: str | Path | None = None,
):
    tokenizer_name_or_path = resolve_tokenizer_name_or_path(tokenizer_name_or_path)
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name_or_path,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def build_processors(
    skill_spec: SkillSpec,
    stats_payload: dict[str, Any],
    *,
    device: str = "cuda",
    tokenizer_name_or_path: str | Path | None = None,
):
    config = make_skill_policy_config(
        skill_spec,
        device=device,
        dtype="bfloat16" if device.startswith("cuda") else "float32",
        train_expert_only=True,
        gradient_checkpointing=False,
        compile_model=False,
    )
    config.device = device
    preprocessor, postprocessor = make_pi05_processors(
        config,
        dataset_stats=stats_to_torch(stats_payload),
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    return preprocessor, postprocessor
