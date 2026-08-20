from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vla_skill_router.action import crop_action, masked_action_mse, pad_state
from vla_skill_router.adapters import resolve_adapter_dir


def test_pad_state_pads_to_max_dim() -> None:
    state = torch.ones(2, 3)
    padded = pad_state(state, max_state_dim=5)

    assert tuple(padded.shape) == (2, 5)
    assert torch.allclose(padded[:, :3], state)
    assert torch.allclose(padded[:, 3:], torch.zeros(2, 2))


def test_masked_action_mse_ignores_invalid_dims() -> None:
    pred = torch.tensor([[[1.0, 2.0, 100.0]], [[1.0, 1.0, 1.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    loss = masked_action_mse(pred, target, torch.tensor([2, 3]))

    assert loss == pytest.approx((1.0 + 4.0 + 1.0 + 1.0 + 1.0) / 5.0)


def test_crop_action_uses_selected_skill_dim() -> None:
    action = torch.zeros(2, 50, 8)

    assert tuple(crop_action(action, 6).shape) == (2, 50, 6)


def test_resolve_adapter_dir_falls_back_from_stale_json_to_best_symlink(tmp_path: Path) -> None:
    skill_output_dir = tmp_path / "pick_cube"
    real_adapter = skill_output_dir / "C" / "run_001" / "best"
    real_adapter.mkdir(parents=True)
    (skill_output_dir / "best").symlink_to(Path("C") / "run_001" / "best")
    stale = tmp_path / "missing" / "adapter"
    (skill_output_dir / "best_adapter.json").write_text(
        json.dumps({"adapter_dir": str(stale)}),
        encoding="utf-8",
    )

    resolved = resolve_adapter_dir(skill_output_dir=skill_output_dir)

    assert resolved.adapter_dir == skill_output_dir / "best"
    assert resolved.adapter_dir.exists()
    assert resolved.stale_path == stale
    assert resolved.source == "best_symlink"
