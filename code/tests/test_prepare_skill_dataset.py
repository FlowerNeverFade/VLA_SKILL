from __future__ import annotations

from pathlib import Path

import pytest

from vla_skill.dataset import SkillDataError, prepare_skill_directory

from .helpers import make_toy_skill


def test_prepare_skill_directory_writes_splits_and_train_only_stats(tmp_path: Path) -> None:
    skill_dir = make_toy_skill(
        tmp_path,
        skill_id="pick_mug",
        display_name="Pick Mug",
        task="pick the mug and place it on the tray",
        num_episodes=10,
        num_frames=4,
        seed=3,
    )

    summary = prepare_skill_directory(skill_dir)

    assert summary["train_episodes"] == 9
    assert summary["val_episodes"] == 1
    assert skill_dir.joinpath("splits.json").is_file()
    assert skill_dir.joinpath("stats.json").is_file()

    stats = __import__("json").load(skill_dir.joinpath("stats.json").open("r", encoding="utf-8"))
    assert stats["observation.state"]["count"] == [9 * 4]
    assert stats["action"]["count"] == [9 * 4]


def test_prepare_skill_directory_raises_with_missing_camera_frame(tmp_path: Path) -> None:
    skill_dir = make_toy_skill(
        tmp_path,
        skill_id="broken_skill",
        display_name="Broken Skill",
        task="do something",
        num_episodes=2,
        num_frames=4,
        seed=4,
    )
    broken_frame = skill_dir / "episodes" / "ep_000" / "images" / "base_0_rgb" / "000003.jpg"
    broken_frame.unlink()

    with pytest.raises(SkillDataError) as exc_info:
        prepare_skill_directory(skill_dir)

    message = str(exc_info.value)
    assert "broken_skill" in message
    assert "ep_000" in message
    assert "base_0_rgb" in message
