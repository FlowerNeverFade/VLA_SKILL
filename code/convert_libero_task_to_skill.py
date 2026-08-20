#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from vla_skill.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_N_ACTION_STEPS,
    DEFAULT_SKILL_CAMERA_NAMES,
    DEFAULT_SKILL_ROOT,
    DATA_ROOT,
    IMAGE_RESOLUTION,
)
from vla_skill.dataset import prepare_skill_directory
from vla_skill.io_utils import ensure_dir, remove_if_exists, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one LIBERO task from LeRobot parquet into a PI05 skill dir.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATA_ROOT / "datasets" / "HuggingFaceVLA_libero",
    )
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--skill-id", type=str)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", value)


def infer_keywords(task_name: str) -> list[str]:
    stopwords = {
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
        "both",
        "it",
        "place",
        "put",
        "pick",
    }
    keywords = []
    for token in re.findall(r"[a-z0-9]+", task_name.lower()):
        if token in stopwords or len(token) <= 2:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:8]


def load_task_name(dataset_root: Path, task_index: int) -> str:
    table = pq.read_table(dataset_root / "meta" / "tasks.parquet")
    for row in table.to_pylist():
        if int(row["task_index"]) == task_index:
            return str(row["__index_level_0__"])
    raise ValueError(f"Task index {task_index} not found in {dataset_root / 'meta' / 'tasks.parquet'}.")


def load_info_json(dataset_root: Path) -> dict[str, Any]:
    import json

    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _decode_image(image_payload: dict[str, Any]) -> Image.Image:
    image = Image.open(BytesIO(image_payload["bytes"])).convert("RGB")
    if image.size != IMAGE_RESOLUTION[::-1]:
        image = image.resize(IMAGE_RESOLUTION[::-1], Image.Resampling.BILINEAR)
    return image


def _write_jpeg(image_payload: dict[str, Any], path: Path, *, quality: int) -> None:
    ensure_dir(path.parent)
    image = _decode_image(image_payload)
    image.save(path, format="JPEG", quality=quality)


def convert_task(
    *,
    dataset_root: Path,
    skill_root: Path,
    task_index: int,
    skill_id: str | None,
    max_episodes: int | None,
    overwrite: bool,
    jpeg_quality: int,
) -> dict[str, Any]:
    task_name = load_task_name(dataset_root, task_index)
    skill_id = skill_id or f"libero_{task_index:02d}_{slugify(task_name)}"
    skill_dir = skill_root / skill_id

    if skill_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Skill directory already exists: {skill_dir}")
        remove_if_exists(skill_dir)
    ensure_dir(skill_dir)

    info = load_info_json(dataset_root)
    fps = float(info["fps"])
    data_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    needed_columns = [
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "frame_index",
        "episode_index",
        "task_index",
    ]
    selected_episode_ids: set[int] | None = None

    for parquet_path in data_files:
        table = pq.read_table(parquet_path, columns=needed_columns)
        for row in table.to_pylist():
            if int(row["task_index"]) != task_index:
                continue
            episode_index = int(row["episode_index"])
            if selected_episode_ids is not None and episode_index not in selected_episode_ids:
                continue
            rows_by_episode[episode_index].append(row)
            if max_episodes is not None and selected_episode_ids is None and len(rows_by_episode) >= max_episodes:
                selected_episode_ids = set(sorted(rows_by_episode)[:max_episodes])

    if not rows_by_episode:
        raise ValueError(f"Task index {task_index} produced zero episodes from {dataset_root}.")

    episode_items = sorted(rows_by_episode.items())
    if max_episodes is not None:
        episode_items = episode_items[:max_episodes]

    episodes_dir = ensure_dir(skill_dir / "episodes")
    total_frames = 0
    for export_index, (episode_index, rows) in enumerate(episode_items):
        rows.sort(key=lambda item: int(item["frame_index"]))
        episode_id = f"episode_{episode_index:06d}"
        episode_dir = ensure_dir(episodes_dir / episode_id)
        camera_dirs = {
            camera_name: ensure_dir(episode_dir / "images" / camera_name)
            for camera_name in DEFAULT_SKILL_CAMERA_NAMES
        }

        states = []
        actions = []
        for local_frame, row in enumerate(rows):
            frame_name = f"{local_frame:06d}.jpg"
            _write_jpeg(row["observation.images.image"], camera_dirs["base_0_rgb"] / frame_name, quality=jpeg_quality)
            _write_jpeg(
                row["observation.images.image2"],
                camera_dirs["left_wrist_0_rgb"] / frame_name,
                quality=jpeg_quality,
            )
            _write_jpeg(
                row["observation.images.image2"],
                camera_dirs["right_wrist_0_rgb"] / frame_name,
                quality=jpeg_quality,
            )
            states.append(np.asarray(row["observation.state"], dtype=np.float32))
            actions.append(np.asarray(row["action"], dtype=np.float32))

        state_arr = np.stack(states, axis=0)
        action_arr = np.stack(actions, axis=0)
        np.save(episode_dir / "state.npy", state_arr)
        np.save(episode_dir / "action.npy", action_arr)
        write_json(
            episode_dir / "meta.json",
            {
                "episode_id": episode_id,
                "task": task_name,
                "num_frames": int(len(rows)),
                "fps": fps,
                "source_dataset": str(dataset_root),
                "source_task_index": task_index,
                "source_episode_index": episode_index,
                "camera_mapping": {
                    "base_0_rgb": "observation.images.image",
                    "left_wrist_0_rgb": "observation.images.image2",
                    "right_wrist_0_rgb": "observation.images.image2",
                },
                "export_index": export_index,
            },
        )
        total_frames += len(rows)

    first_row = rows_by_episode[episode_items[0][0]][0]
    state_dim = int(np.asarray(first_row["observation.state"]).shape[0])
    action_dim = int(np.asarray(first_row["action"]).shape[0])
    write_yaml(
        skill_dir / "skill.yaml",
        {
            "skill_id": skill_id,
            "display_name": task_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "camera_names": list(DEFAULT_SKILL_CAMERA_NAMES),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "n_action_steps": DEFAULT_N_ACTION_STEPS,
            "router": {
                "description": task_name,
                "aliases": [task_name],
                "keywords": infer_keywords(task_name),
                "regexes": [],
                "priority": 0,
            },
        },
    )

    prepare_summary = prepare_skill_directory(skill_dir)
    summary = {
        "skill_id": skill_id,
        "skill_dir": str(skill_dir),
        "task_index": task_index,
        "task_name": task_name,
        "episodes": len(episode_items),
        "total_frames": total_frames,
        "prepare": prepare_summary,
    }
    write_json(skill_dir / "conversion_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = convert_task(
        dataset_root=args.dataset_root,
        skill_root=args.skill_root,
        task_index=args.task_index,
        skill_id=args.skill_id,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
        jpeg_quality=args.jpeg_quality,
    )
    print(summary)


if __name__ == "__main__":
    main()
