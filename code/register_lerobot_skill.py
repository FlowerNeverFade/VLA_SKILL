#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_skill.constants import DATA_ROOT, DEFAULT_CHUNK_SIZE, DEFAULT_N_ACTION_STEPS, DEFAULT_SKILL_ROOT
from vla_skill.io_utils import ensure_dir, load_json, remove_if_exists, write_yaml

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
    parser = argparse.ArgumentParser(description="Register a raw LeRobot task as a PI05 skill without materializing JPEGs.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--task-index", type=int, default=DEFAULT_TASK_INDEX)
    parser.add_argument("--skill-id", type=str, default=DEFAULT_SKILL_ID)
    parser.add_argument("--video-backend", choices=("auto", "torchcodec", "pyav"), default="pyav")
    parser.add_argument("--overwrite", action="store_true")
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


def build_dataset(dataset_dir: Path, repo_id: str, video_backend: str) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=repo_id,
        root=dataset_dir,
        download_videos=False,
        video_backend=video_backend,
        return_uint8=True,
    )


def find_first_matching_row(dataset: LeRobotDataset, task_index: int) -> dict:
    for row in dataset.hf_dataset:
        if int(row["task_index"]) == task_index:
            return row
    raise ValueError(f"Task index {task_index} produced zero rows in dataset {dataset.root}.")


def main() -> None:
    args = parse_args()
    task_name = load_task_name(args.dataset_dir, args.task_index)
    dataset = build_dataset(args.dataset_dir, args.repo_id, args.video_backend)
    first_row = find_first_matching_row(dataset, args.task_index)
    state_dim = int(np.asarray(first_row["observation.state"]).shape[0])
    action_dim = int(np.asarray(first_row["action"]).shape[0])

    skill_dir = args.skill_root / args.skill_id
    if skill_dir.exists() and args.overwrite:
        remove_if_exists(skill_dir)
    ensure_dir(skill_dir)

    write_yaml(
        skill_dir / "skill.yaml",
        {
            "skill_id": args.skill_id,
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
            "source": {
                "type": "lerobot",
                "dataset_dir": str(args.dataset_dir),
                "repo_id": args.repo_id,
                "task_index": args.task_index,
                "video_backend": args.video_backend,
                "camera_mapping": dict(CAMERA_SOURCE_BY_TARGET),
            },
        },
    )

    payload = {
        "skill_id": args.skill_id,
        "skill_dir": str(skill_dir),
        "task_index": args.task_index,
        "task_name": task_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "source_type": "lerobot",
        "video_backend": args.video_backend,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
