from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vla_skill.constants import DEFAULT_SKILL_CAMERA_NAMES


def make_toy_skill(
    skill_root: Path,
    *,
    skill_id: str,
    display_name: str,
    task: str,
    aliases: list[str] | None = None,
    keywords: list[str] | None = None,
    regexes: list[str] | None = None,
    priority: int = 0,
    num_episodes: int = 10,
    num_frames: int = 4,
    state_dim: int = 8,
    action_dim: int = 7,
    chunk_size: int = 2,
    camera_names: tuple[str, ...] = DEFAULT_SKILL_CAMERA_NAMES,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    skill_dir = skill_root / skill_id
    episodes_dir = skill_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    skill_yaml = {
        "skill_id": skill_id,
        "display_name": display_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_names": list(camera_names),
        "chunk_size": chunk_size,
        "n_action_steps": chunk_size,
        "router": {
            "description": task,
            "aliases": aliases or [display_name],
            "keywords": keywords or [],
            "regexes": regexes or [],
            "priority": priority,
        },
    }
    (skill_dir / "skill.yaml").write_text(yaml.safe_dump(skill_yaml, sort_keys=False), encoding="utf-8")

    for episode_index in range(num_episodes):
        episode_id = f"ep_{episode_index:03d}"
        episode_dir = episodes_dir / episode_id
        (episode_dir / "images").mkdir(parents=True, exist_ok=True)

        state = rng.normal(loc=0.0, scale=1.0, size=(num_frames, state_dim)).astype(np.float32)
        action = rng.normal(loc=0.0, scale=0.25, size=(num_frames, action_dim)).astype(np.float32)
        np.save(episode_dir / "state.npy", state)
        np.save(episode_dir / "action.npy", action)

        meta = {
            "episode_id": episode_id,
            "task": task,
            "num_frames": num_frames,
            "fps": 5.0,
        }
        (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        for camera_name in camera_names:
            camera_dir = episode_dir / "images" / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in range(num_frames):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[:, :, 0] = (episode_index * 13 + frame_index * 19) % 255
                image[:, :, 1] = (frame_index * 37 + len(camera_name) * 11) % 255
                image[:, :, 2] = (episode_index * 7 + len(skill_id) * 23) % 255
                Image.fromarray(image).save(camera_dir / f"{frame_index:06d}.jpg", quality=90)

    return skill_dir
