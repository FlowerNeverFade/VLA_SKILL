from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

import vla_skill.dataset as dataset_module
from vla_skill.dataset import build_window_dataset, load_skill_spec, prepare_skill_directory


class FakeLeRobotDataset:
    def __init__(self, *, repo_id: str, root: Path, download_videos: bool, video_backend: str, return_uint8: bool):
        self.repo_id = repo_id
        self.root = root
        self.download_videos = download_videos
        self.video_backend = video_backend
        self.return_uint8 = return_uint8
        self.meta = type("Meta", (), {"fps": 30.0})()
        self.hf_dataset = []
        self._rows = []
        for episode_index in range(3):
            for frame_index in range(4):
                state = np.full((6,), episode_index * 10 + frame_index, dtype=np.float32)
                action = np.full((6,), episode_index * 100 + frame_index, dtype=np.float32)
                self.hf_dataset.append(
                    {
                        "task_index": 0,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "observation.state": state,
                        "action": action,
                    }
                )
                image = torch.full((3, 64, 64), fill_value=episode_index * 20 + frame_index, dtype=torch.uint8)
                self._rows.append(
                    {
                        "observation.images.front": image.clone(),
                        "observation.images.wrist": image.clone(),
                        "observation.images.overhead": image.clone(),
                        "task": "pick cube",
                    }
                )

    def __getitem__(self, index: int) -> dict:
        return {**self.hf_dataset[index], **self._rows[index]}


def test_prepare_and_sample_raw_lerobot_skill(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dataset_module, "LeRobotDataset", FakeLeRobotDataset)

    dataset_dir = tmp_path / "dataset"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "tasks.json").write_text(json.dumps({"0": "pick cube"}), encoding="utf-8")

    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "pick_cube_raw"
    skill_dir.mkdir(parents=True)
    skill_yaml = {
        "skill_id": "pick_cube_raw",
        "display_name": "pick cube",
        "state_dim": 6,
        "action_dim": 6,
        "camera_names": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "chunk_size": 2,
        "n_action_steps": 2,
        "router": {
            "description": "pick cube",
            "aliases": ["pick cube"],
            "keywords": ["cube"],
            "regexes": [],
            "priority": 0,
        },
        "source": {
            "type": "lerobot",
            "dataset_dir": str(dataset_dir),
            "repo_id": "fake/repo",
            "task_index": 0,
            "video_backend": "pyav",
            "camera_mapping": {
                "base_0_rgb": "observation.images.overhead",
                "left_wrist_0_rgb": "observation.images.wrist",
                "right_wrist_0_rgb": "observation.images.front",
            },
        },
    }
    (skill_dir / "skill.yaml").write_text(yaml.safe_dump(skill_yaml, sort_keys=False), encoding="utf-8")

    summary = prepare_skill_directory(skill_dir)

    assert summary["source_type"] == "lerobot"
    assert summary["train_episodes"] == 2
    assert summary["val_episodes"] == 1
    stats = json.loads((skill_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["observation.state"]["count"] == [8]
    assert stats["action"]["count"] == [8]

    dataset = build_window_dataset(load_skill_spec(skill_root, "pick_cube_raw"), split="val")
    sample = dataset[0]
    assert tuple(sample["observation.state"].shape) == (6,)
    assert tuple(sample["action"].shape) == (2, 6)
    assert tuple(sample["observation.images.base_0_rgb"].shape) == (3, 224, 224)
    assert tuple(sample["observation.images.left_wrist_0_rgb"].shape) == (3, 224, 224)
    assert tuple(sample["observation.images.right_wrist_0_rgb"].shape) == (3, 224, 224)
