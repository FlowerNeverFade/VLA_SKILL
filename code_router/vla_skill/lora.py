from __future__ import annotations

import re
from typing import Iterable

import torch.nn as nn
from peft import LoraConfig

from .constants import EXPECTED_LORA_TARGET_COUNTS, LORA_DEFAULTS

_ATTN_A = r".*\.gemma_expert\.model\.layers\.\d+\.self_attn\.(?:q|v)_proj"
_ATTN_B = r".*\.gemma_expert\.model\.layers\.\d+\.self_attn\.(?:q|k|v|o)_proj"
_MLP = r".*\.gemma_expert\.model\.layers\.\d+\.mlp\.(?:gate|up|down)_proj"
_PROJ = r"model\.(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)"


def canonical_group(group: str) -> str:
    value = group.upper()
    if value not in EXPECTED_LORA_TARGET_COUNTS:
        raise ValueError(f"Unsupported LoRA group `{group}`.")
    return value


def get_lora_group_regex(group: str) -> str:
    group = canonical_group(group)
    if group == "A":
        return rf"^(?:{_ATTN_A})$"
    if group == "B":
        return rf"^(?:{_ATTN_B})$"
    if group == "C":
        return rf"^(?:{_ATTN_B}|{_MLP}|{_PROJ})$"
    return rf"^(?:{_PROJ})$"


def build_lora_config(
    group: str,
    *,
    base_model_name_or_path: str,
    r: int = LORA_DEFAULTS["r"],
    lora_alpha: int = LORA_DEFAULTS["lora_alpha"],
    lora_dropout: float = LORA_DEFAULTS["lora_dropout"],
    bias: str = LORA_DEFAULTS["bias"],
    inference_mode: bool = False,
) -> LoraConfig:
    return LoraConfig(
        base_model_name_or_path=base_model_name_or_path,
        inference_mode=inference_mode,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=get_lora_group_regex(group),
        modules_to_save=[],
    )


def collect_target_module_names(policy: nn.Module, group: str) -> list[str]:
    pattern = re.compile(get_lora_group_regex(group))
    return sorted(
        name
        for name, module in policy.named_modules()
        if name and pattern.match(name) and isinstance(module, nn.Linear)
    )


def validate_lora_group_targets(policy: nn.Module, group: str) -> list[str]:
    group = canonical_group(group)
    matched = collect_target_module_names(policy, group)
    expected = EXPECTED_LORA_TARGET_COUNTS[group]
    if len(matched) != expected:
        raise ValueError(
            f"LoRA group {group} matched {len(matched)} modules, expected {expected}. "
            f"Sample matches: {matched[:5]}"
        )
    return matched


def iter_group_names(groups: Iterable[str] | None) -> list[str]:
    if groups is None:
        return list(EXPECTED_LORA_TARGET_COUNTS.keys())
    return [canonical_group(group) for group in groups]
