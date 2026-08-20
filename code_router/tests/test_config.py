from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vla_skill_router.config import load_experiment_config
from vla_skill_router.constants import DEFAULT_ROUTER_OUTPUT_ROOT


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_experiment_config_maps_channels(tmp_path: Path) -> None:
    path = tmp_path / "router.yaml"
    _write_yaml(
        path,
        {
            "channels": [
                {"channel_id": "pick", "skill_id": "pick_cube", "lora_group": "C"},
                {"channel_id": "stove", "skill_id": "turn_stove", "lora_group": "C"},
            ],
            "router": {"state_embed_dim": 16, "hidden_dim": 32},
            "train": {"steps": 3, "batch_size": 2, "router_ce_weight": 0.7},
        },
    )

    cfg = load_experiment_config(path)

    assert cfg.output_root == DEFAULT_ROUTER_OUTPUT_ROOT
    assert cfg.channel_id_to_index == {"pick": 0, "stove": 1}
    assert cfg.skill_id_to_index == {"pick_cube": 0, "turn_stove": 1}
    assert cfg.router.state_embed_dim == 16
    assert cfg.train.router_ce_weight == pytest.approx(0.7)


def test_load_experiment_config_rejects_duplicate_channels(tmp_path: Path) -> None:
    path = tmp_path / "router.yaml"
    _write_yaml(
        path,
        {
            "channels": [
                {"channel_id": "pick", "skill_id": "pick_cube"},
                {"channel_id": "pick", "skill_id": "turn_stove"},
            ],
        },
    )

    with pytest.raises(ValueError, match="Duplicate channel_id"):
        load_experiment_config(path)


def test_load_experiment_config_rejects_non_c_group(tmp_path: Path) -> None:
    path = tmp_path / "router.yaml"
    _write_yaml(path, {"channels": [{"channel_id": "pick", "skill_id": "pick_cube", "lora_group": "A"}]})

    with pytest.raises(ValueError, match="LoRA group C"):
        load_experiment_config(path)


def test_router_control_adapter_name_is_reserved(tmp_path: Path) -> None:
    path = tmp_path / "router.yaml"
    _write_yaml(path, {"channels": [{"channel_id": "router_control", "skill_id": "pick_cube"}]})

    with pytest.raises(ValueError, match="reserved"):
        load_experiment_config(path)


def test_load_experiment_config_accepts_lora_control_router_type(tmp_path: Path) -> None:
    path = tmp_path / "router.yaml"
    _write_yaml(
        path,
        {
            "channels": [{"channel_id": "pick", "skill_id": "pick_cube"}],
            "router": {"type": "lora_control"},
        },
    )

    cfg = load_experiment_config(path)

    assert cfg.router.type == "lora_control"
