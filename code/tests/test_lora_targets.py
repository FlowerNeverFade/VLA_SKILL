from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH, EXPECTED_LORA_TARGET_COUNTS
from vla_skill.lora import get_lora_group_regex


def test_lora_regex_counts_match_pi05_base() -> None:
    module_names = set()
    with safe_open(str(DEFAULT_BASE_MODEL_PATH / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith(".weight"):
                module_names.add(f"model.{key[:-7]}")

    counts = {}
    for group, expected in EXPECTED_LORA_TARGET_COUNTS.items():
        import re

        pattern = re.compile(get_lora_group_regex(group))
        counts[group] = len([name for name in module_names if pattern.match(name)])
        assert counts[group] == expected

    assert counts == EXPECTED_LORA_TARGET_COUNTS
