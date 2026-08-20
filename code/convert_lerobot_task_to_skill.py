#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_skill.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_N_ACTION_STEPS,
    DEFAULT_SKILL_ROOT,
    DATA_ROOT,
    IMAGE_RESOLUTION,
)
from vla_skill.dataset import prepare_skill_directory
from vla_skill.io_utils import ensure_dir, load_json, remove_if_exists, write_json, write_yaml

DEFAULT_DATASET_DIR = DATA_ROOT / "datasets" / "gpudad_so101_pick_cube_chunked"
DEFAULT_REPO_ID = "gpudad/so101_pick_cube_chunked"
DEFAULT_SKILL_ID = "so101_pick_cube"
DEFAULT_TASK_INDEX = 0

CAMERA_SOURCE_BY_TARGET = {
    "base_0_rgb": "observation.images.overhead",
    "left_wrist_0_rgb": "observation.images.wrist",
    "right_wrist_0_rgb": "observation.images.front",
}
TASK_ALIAS_OVERRIDES = ("pick cube", "pick up cube")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "to",
    "of",
    "in",
    "on",
    "from",
    "with",
    "up",
    "it",
    "place",
    "pick",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one local LeRobot v3 task into a PI05 skill directory.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--task-index", type=int, default=DEFAULT_TASK_INDEX)
    parser.add_argument("--skill-id", type=str, default=DEFAULT_SKILL_ID)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--video-backend", choices=("auto", "torchcodec", "pyav"), default="auto")
    return parser.parse_args()


def infer_keywords(task_name: str) -> list[str]:
    keywords = []
    for token in re.findall(r"[a-z0-9]+", task_name.lower()):
        if token in STOPWORDS or len(token) <= 2:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:8]


def load_task_name(dataset_dir: Path, task_index: int) -> str:
    tasks_path = dataset_dir / "meta" / "tasks.json"
    payload = load_json(tasks_path)
    try:
        return str(payload[str(task_index)])
    except KeyError as exc:
        raise ValueError(f"Task index {task_index} not found in {tasks_path}.") from exc


def _dataset_kwargs(dataset_dir: Path, repo_id: str, backend: str) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "root": dataset_dir,
        "download_videos": False,
        "video_backend": backend,
        "return_uint8": True,
    }


def resolve_video_backend(dataset_dir: Path, repo_id: str, requested_backend: str) -> tuple[str, str | None]:
    if requested_backend != "auto":
        dataset = LeRobotDataset(**_dataset_kwargs(dataset_dir, repo_id, requested_backend))
        _ = dataset[0]
        return requested_backend, None

    try:
        dataset = LeRobotDataset(**_dataset_kwargs(dataset_dir, repo_id, "torchcodec"))
        _ = dataset[0]
        return "torchcodec", None
    except Exception as exc:  # noqa: BLE001
        fallback_reason = f"{type(exc).__name__}: {exc}"

    dataset = LeRobotDataset(**_dataset_kwargs(dataset_dir, repo_id, "pyav"))
    _ = dataset[0]
    return "pyav", fallback_reason


def build_dataset(dataset_dir: Path, repo_id: str, video_backend: str) -> LeRobotDataset:
    return LeRobotDataset(**_dataset_kwargs(dataset_dir, repo_id, video_backend))


def collect_episode_frame_indices(
    hf_dataset: Any,
    *,
    task_index: int,
    max_episodes: int | None,
) -> list[tuple[int, list[int]]]:
    episodes: OrderedDict[int, list[int]] = OrderedDict()
    for global_index, row in enumerate(hf_dataset):
        if int(row["task_index"]) != task_index:
            continue
        episode_index = int(row["episode_index"])
        if episode_index not in episodes:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            episodes[episode_index] = []
        episodes[episode_index].append(global_index)
    return list(episodes.items())


def _image_to_pil(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        image = image_value.convert("RGB")
    elif torch.is_tensor(image_value):
        tensor = image_value.detach().cpu()
        if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
            raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
        if tensor.dtype == torch.uint8:
            array = tensor.permute(1, 2, 0).contiguous().numpy()
        else:
            array = (
                tensor.clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(dtype=torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .numpy()
            )
        image = Image.fromarray(array).convert("RGB")
    elif isinstance(image_value, np.ndarray):
        array = np.asarray(image_value)
        if array.ndim != 3:
            raise ValueError(f"Unsupported ndarray image shape: {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = np.rint(array * 255.0).astype(np.uint8)
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        image = Image.fromarray(array).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_value)!r}")

    if image.size != IMAGE_RESOLUTION[::-1]:
        image = image.resize(IMAGE_RESOLUTION[::-1], Image.Resampling.BILINEAR)
    return image


def _write_jpeg(image_value: Any, path: Path, *, quality: int) -> None:
    ensure_dir(path.parent)
    _image_to_pil(image_value).save(path, format="JPEG", quality=quality)


def _episode_complete(episode_dir: Path, *, num_frames: int, state_dim: int, action_dim: int) -> bool:
    try:
        if not (episode_dir / "meta.json").is_file():
            return False
        state = np.load(episode_dir / "state.npy")
        action = np.load(episode_dir / "action.npy")
        if state.shape != (num_frames, state_dim):
            return False
        if action.shape != (num_frames, action_dim):
            return False
        for camera_name in CAMERA_SOURCE_BY_TARGET:
            camera_dir = episode_dir / "images" / camera_name
            if not camera_dir.is_dir():
                return False
            if len(tuple(camera_dir.glob("*.jpg"))) != num_frames:
                return False
    except Exception:  # noqa: BLE001
        return False
    return True


def convert_task(
    *,
    dataset_dir: Path,
    repo_id: str,
    skill_root: Path,
    task_index: int,
    skill_id: str,
    max_episodes: int | None,
    overwrite: bool,
    jpeg_quality: int,
    video_backend: str,
) -> dict[str, Any]:
    task_name = load_task_name(dataset_dir, task_index)
    resolved_backend, fallback_reason = resolve_video_backend(dataset_dir, repo_id, video_backend)
    dataset = build_dataset(dataset_dir, repo_id, resolved_backend)
    episode_items = collect_episode_frame_indices(dataset.hf_dataset, task_index=task_index, max_episodes=max_episodes)
    if not episode_items:
        raise ValueError(f"Task index {task_index} produced zero episodes from {dataset_dir}.")
    print(
        f"[convert] task_index={task_index} task_name={task_name!r} "
        f"episodes={len(episode_items)} backend={resolved_backend}",
        flush=True,
    )
    if fallback_reason:
        print(f"[convert] backend fallback reason: {fallback_reason}", flush=True)

    first_row = dataset.hf_dataset[episode_items[0][1][0]]
    state_dim = int(np.asarray(first_row["observation.state"]).shape[0])
    action_dim = int(np.asarray(first_row["action"]).shape[0])

    skill_dir = skill_root / skill_id
    if skill_dir.exists() and overwrite:
        remove_if_exists(skill_dir)
    ensure_dir(skill_dir)
    episodes_dir = ensure_dir(skill_dir / "episodes")

    write_yaml(
        skill_dir / "skill.yaml",
        {
            "skill_id": skill_id,
            "display_name": task_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "camera_names": list(CAMERA_SOURCE_BY_TARGET.keys()),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "n_action_steps": DEFAULT_N_ACTION_STEPS,
            "router": {
                "description": task_name,
                "aliases": [task_name, *TASK_ALIAS_OVERRIDES],
                "keywords": infer_keywords(task_name),
                "regexes": [],
                "priority": 0,
            },
        },
    )

    exported_episodes = 0
    skipped_episodes = 0
    total_frames = 0
    for export_index, (episode_index, frame_indices) in enumerate(episode_items):
        episode_id = f"episode_{episode_index:06d}"
        episode_dir = episodes_dir / episode_id
        num_frames = len(frame_indices)
        if episode_dir.exists() and _episode_complete(
            episode_dir,
            num_frames=num_frames,
            state_dim=state_dim,
            action_dim=action_dim,
        ):
            skipped_episodes += 1
            total_frames += num_frames
            if skipped_episodes == 1 or skipped_episodes % 10 == 0 or skipped_episodes == len(episode_items):
                print(
                    f"[convert] skipped episode {episode_id} ({export_index + 1}/{len(episode_items)})",
                    flush=True,
                )
            continue
        if episode_dir.exists():
            remove_if_exists(episode_dir)
        ensure_dir(episode_dir)

        camera_dirs = {
            camera_name: ensure_dir(episode_dir / "images" / camera_name)
            for camera_name in CAMERA_SOURCE_BY_TARGET
        }
        states = []
        actions = []
        for local_frame, global_index in enumerate(frame_indices):
            item = dataset[global_index]
            frame_name = f"{local_frame:06d}.jpg"
            for camera_name, source_key in CAMERA_SOURCE_BY_TARGET.items():
                _write_jpeg(item[source_key], camera_dirs[camera_name] / frame_name, quality=jpeg_quality)
            states.append(np.asarray(item["observation.state"], dtype=np.float32))
            actions.append(np.asarray(item["action"], dtype=np.float32))

        np.save(episode_dir / "state.npy", np.stack(states, axis=0))
        np.save(episode_dir / "action.npy", np.stack(actions, axis=0))
        write_json(
            episode_dir / "meta.json",
            {
                "episode_id": episode_id,
                "task": task_name,
                "num_frames": num_frames,
                "fps": float(dataset.meta.fps),
                "source_repo_id": repo_id,
                "source_dataset_dir": str(dataset_dir),
                "source_task_index": task_index,
                "source_episode_index": int(episode_index),
                "source_video_backend": resolved_backend,
                "camera_mapping": dict(CAMERA_SOURCE_BY_TARGET),
                "export_index": export_index,
            },
        )
        exported_episodes += 1
        total_frames += num_frames
        if exported_episodes == 1 or exported_episodes % 10 == 0 or exported_episodes == len(episode_items):
            print(
                f"[convert] exported episode {episode_id} ({export_index + 1}/{len(episode_items)}) "
                f"frames={num_frames}",
                flush=True,
            )

    prepare_summary = prepare_skill_directory(skill_dir)
    summary = {
        "skill_id": skill_id,
        "skill_dir": str(skill_dir),
        "task_index": task_index,
        "task_name": task_name,
        "episodes": len(episode_items),
        "exported_episodes": exported_episodes,
        "skipped_episodes": skipped_episodes,
        "total_frames": total_frames,
        "resolved_video_backend": resolved_backend,
        "fallback_reason": fallback_reason,
        "prepare": prepare_summary,
    }
    write_json(skill_dir / "conversion_summary.json", summary)
    print(f"[convert] done summary={summary}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    summary = convert_task(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        skill_root=args.skill_root,
        task_index=args.task_index,
        skill_id=args.skill_id,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        jpeg_quality=args.jpeg_quality,
        video_backend=args.video_backend,
    )
    print(summary)


if __name__ == "__main__":
    main()
