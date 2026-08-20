from __future__ import annotations

from pathlib import Path

from vla_skill import dataset as dataset_module
from vla_skill.dataset import build_lerobot_dataset, clear_lerobot_dataset_cache
from vla_skill.schema import SkillSpec, SourceSpec


class FakeLeRobotDataset:
    constructed: list[dict] = []
    getitem_calls = 0

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.meta = type("Meta", (), {"fps": 20.0})()
        self.hf_dataset = []
        FakeLeRobotDataset.constructed.append(kwargs)

    def __getitem__(self, index: int):
        FakeLeRobotDataset.getitem_calls += 1
        return {"index": index}


def _make_lerobot_skill(dataset_dir: Path, *, task_index: int) -> SkillSpec:
    return SkillSpec(
        skill_id=f"skill_{task_index}",
        display_name=f"Skill {task_index}",
        state_dim=2,
        action_dim=3,
        source=SourceSpec(
            type="lerobot",
            dataset_dir=dataset_dir,
            repo_id="fake/robocasa",
            task_index=task_index,
            video_backend="pyav",
            camera_mapping={
                "base_0_rgb": "observation.images.robot0_agentview_left",
                "left_wrist_0_rgb": "observation.images.robot0_eye_in_hand",
                "right_wrist_0_rgb": "observation.images.robot0_agentview_right",
            },
        ),
    )


def test_lerobot_dataset_is_cached_by_source(monkeypatch, tmp_path: Path) -> None:
    clear_lerobot_dataset_cache()
    FakeLeRobotDataset.constructed = []
    FakeLeRobotDataset.getitem_calls = 0
    monkeypatch.setattr(dataset_module, "LeRobotDataset", FakeLeRobotDataset)
    dataset_dir = tmp_path / "robocasa"
    dataset_dir.mkdir()

    dataset_a, backend_a, _ = build_lerobot_dataset(_make_lerobot_skill(dataset_dir, task_index=0))
    dataset_b, backend_b, _ = build_lerobot_dataset(_make_lerobot_skill(dataset_dir, task_index=2))

    assert dataset_a is dataset_b
    assert backend_a == backend_b == "pyav"
    assert len(FakeLeRobotDataset.constructed) == 1
    assert FakeLeRobotDataset.getitem_calls == 1
    clear_lerobot_dataset_cache()
