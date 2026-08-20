from __future__ import annotations

from pathlib import Path

import torch

import convert_lerobot_task_to_skill as converter


class FakeLeRobotDataset:
    def __init__(self, *, video_backend: str, fail_torchcodec: bool = True):
        self.video_backend = video_backend
        self.fail_torchcodec = fail_torchcodec
        self.meta = type("Meta", (), {"fps": 30.0})()
        self.hf_dataset = []
        self._items = {}
        task = "pick up the object and place it in the target location"
        for episode_index in (0, 1):
            for frame_index in range(52):
                global_index = len(self.hf_dataset)
                row = {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "task_index": 0,
                    "observation.state": [float(episode_index), float(frame_index)] * 3,
                    "action": [float(frame_index), float(episode_index)] * 3,
                }
                self.hf_dataset.append(row)
                image = torch.full((3, 16, 16), fill_value=episode_index * 40 + frame_index, dtype=torch.uint8)
                self._items[global_index] = {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "task_index": 0,
                    "task": task,
                    "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
                    "action": torch.tensor(row["action"], dtype=torch.float32),
                    "observation.images.front": image.clone(),
                    "observation.images.overhead": image.clone(),
                    "observation.images.wrist": image.clone(),
                }

    def __getitem__(self, index: int):
        if self.video_backend == "torchcodec" and self.fail_torchcodec:
            raise RuntimeError("torchcodec unavailable")
        return self._items[index]


def test_resolve_video_backend_falls_back_to_pyav(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)

    def fake_ctor(**kwargs):
        return FakeLeRobotDataset(video_backend=kwargs["video_backend"], fail_torchcodec=True)

    monkeypatch.setattr(converter, "LeRobotDataset", fake_ctor)

    backend, reason = converter.resolve_video_backend(dataset_dir, "fake/repo", "auto")

    assert backend == "pyav"
    assert reason is not None
    assert "torchcodec unavailable" in reason


def test_convert_task_exports_skill_from_fake_lerobot(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta" / "tasks.json").write_text(
        '{"0": "pick up the object and place it in the target location"}',
        encoding="utf-8",
    )

    def fake_ctor(**kwargs):
        return FakeLeRobotDataset(video_backend=kwargs["video_backend"], fail_torchcodec=True)

    monkeypatch.setattr(converter, "LeRobotDataset", fake_ctor)

    summary = converter.convert_task(
        dataset_dir=dataset_dir,
        repo_id="fake/repo",
        skill_root=tmp_path / "skills",
        task_index=0,
        skill_id="so101_pick_cube_smoke",
        max_episodes=2,
        overwrite=False,
        jpeg_quality=90,
        video_backend="auto",
    )

    skill_dir = tmp_path / "skills" / "so101_pick_cube_smoke"
    assert summary["episodes"] == 2
    assert summary["resolved_video_backend"] == "pyav"
    assert (skill_dir / "skill.yaml").is_file()
    assert (skill_dir / "splits.json").is_file()
    assert (skill_dir / "stats.json").is_file()

    episode_dir = skill_dir / "episodes" / "episode_000000"
    assert (episode_dir / "state.npy").is_file()
    assert (episode_dir / "action.npy").is_file()
    assert len(tuple((episode_dir / "images" / "base_0_rgb").glob("*.jpg"))) == 52
    assert len(tuple((episode_dir / "images" / "left_wrist_0_rgb").glob("*.jpg"))) == 52
    assert len(tuple((episode_dir / "images" / "right_wrist_0_rgb").glob("*.jpg"))) == 52

    meta = __import__("json").load((episode_dir / "meta.json").open("r", encoding="utf-8"))
    assert meta["camera_mapping"] == converter.CAMERA_SOURCE_BY_TARGET
    assert meta["source_video_backend"] == "pyav"
