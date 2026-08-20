from __future__ import annotations

import json
import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vla_skill.constants import DEFAULT_CHUNK_SIZE, DEFAULT_N_ACTION_STEPS
from vla_skill.dataset import LEROBOT_RECORDS_FILENAME, split_episode_ids
from vla_skill.io_utils import ensure_dir, remove_if_exists, write_json, write_yaml
from vla_skill.stats import compute_vector_stats

_DATA_ROOT = Path(os.environ.get("VLA_DATA_ROOT", os.environ.get("VLA_SKILL_ROOT", Path(__file__).resolve().parents[2])))
DEFAULT_ROBOCASA_DATASET_DIR = Path(
    os.environ.get(
        "ROBOCASA_DATASET_DIR",
        _DATA_ROOT / "datasets" / "RoboCasa" / "datasets_hf" / "robocasa_target_atomic",
    )
)
DEFAULT_ROBOCASA_REPO_ID = "robocasa/robocasa_target_atomic"
DEFAULT_ROBOCASA_SKILL_PREFIX = "robocasa_target_atomic"
ROBOCASA_CAMERA_MAPPING = {
    "base_0_rgb": "observation.images.robot0_agentview_left",
    "left_wrist_0_rgb": "observation.images.robot0_eye_in_hand",
    "right_wrist_0_rgb": "observation.images.robot0_agentview_right",
}
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
    "by",
    "for",
    "into",
    "onto",
}


@dataclass(frozen=True)
class ActiveTask:
    task_index: int
    task_name: str
    frames: int
    episodes: int
    skill_id: str


def slugify_task_name(task_name: str, *, max_tokens: int = 8) -> str:
    tokens = [token for token in re.findall(r"[a-z0-9]+", task_name.lower()) if token not in STOPWORDS]
    slug = "_".join(tokens[:max_tokens])
    return slug or "task"


def infer_keywords(task_name: str) -> list[str]:
    keywords = []
    for token in re.findall(r"[a-z0-9]+", task_name.lower()):
        if token in STOPWORDS or len(token) <= 2:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:8]


def load_tasks_parquet(dataset_dir: Path) -> dict[int, str]:
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"RoboCasa tasks.parquet not found: {tasks_path}")
    tasks_df = pd.read_parquet(tasks_path)
    if "task_index" not in tasks_df.columns:
        raise ValueError(f"{tasks_path} must contain a `task_index` column.")
    return {int(row["task_index"]): str(index) for index, row in tasks_df.iterrows()}


def _data_parquet_paths(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))


def _load_info(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "meta" / "info.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scan_active_tasks(
    dataset_dir: Path = DEFAULT_ROBOCASA_DATASET_DIR,
    *,
    skill_prefix: str = DEFAULT_ROBOCASA_SKILL_PREFIX,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    task_names = load_tasks_parquet(dataset_dir)
    counts: dict[int, dict[str, int]] = {}
    for parquet_path in _data_parquet_paths(dataset_dir):
        df = pd.read_parquet(parquet_path, columns=["task_index", "episode_index"])
        grouped = df.groupby("task_index").agg(frames=("task_index", "size"), episodes=("episode_index", "nunique"))
        for task_index, row in grouped.iterrows():
            item = counts.setdefault(int(task_index), {"frames": 0, "episodes": 0})
            item["frames"] += int(row["frames"])
            item["episodes"] += int(row["episodes"])

    active_tasks: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    for task_index, task_name in sorted(task_names.items()):
        count = counts.get(task_index, {"frames": 0, "episodes": 0})
        skill_id = f"{skill_prefix}_{task_index:03d}_{slugify_task_name(task_name)}"
        payload = {
            "task_index": task_index,
            "task_name": task_name,
            "frames": int(count["frames"]),
            "episodes": int(count["episodes"]),
            "skill_id": skill_id,
        }
        if count["frames"] > 0 and count["episodes"] > 0:
            active_tasks.append(payload)
        else:
            skipped_tasks.append(payload)

    return {
        "dataset_dir": str(dataset_dir),
        "total_tasks_in_metadata": len(task_names),
        "active_task_count": len(active_tasks),
        "skipped_task_count": len(skipped_tasks),
        "active_tasks": active_tasks,
        "skipped_tasks": skipped_tasks,
    }


def write_active_task_manifest(
    *,
    dataset_dir: Path,
    output_path: Path,
    skill_prefix: str = DEFAULT_ROBOCASA_SKILL_PREFIX,
) -> dict[str, Any]:
    manifest = scan_active_tasks(dataset_dir, skill_prefix=skill_prefix)
    write_json(output_path, manifest)
    return manifest


def register_active_tasks_as_skills(
    *,
    dataset_dir: Path,
    repo_id: str,
    skill_root: Path,
    manifest_path: Path,
    overwrite: bool = False,
    video_backend: str = "pyav",
    skill_prefix: str = DEFAULT_ROBOCASA_SKILL_PREFIX,
) -> dict[str, Any]:
    manifest = write_active_task_manifest(dataset_dir=dataset_dir, output_path=manifest_path, skill_prefix=skill_prefix)
    info = _load_info(Path(dataset_dir))
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    registered = []
    for task in manifest["active_tasks"]:
        skill_dir = Path(skill_root) / task["skill_id"]
        if skill_dir.exists() and overwrite:
            remove_if_exists(skill_dir)
        ensure_dir(skill_dir)
        write_yaml(
            skill_dir / "skill.yaml",
            {
                "skill_id": task["skill_id"],
                "display_name": task["task_name"],
                "state_dim": state_dim,
                "action_dim": action_dim,
                "camera_names": list(ROBOCASA_CAMERA_MAPPING.keys()),
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "n_action_steps": DEFAULT_N_ACTION_STEPS,
                "router": {
                    "description": task["task_name"],
                    "aliases": [task["task_name"]],
                    "keywords": infer_keywords(task["task_name"]),
                    "regexes": [],
                    "priority": 0,
                },
                "source": {
                    "type": "lerobot",
                    "dataset_dir": str(dataset_dir),
                    "repo_id": repo_id,
                    "task_index": int(task["task_index"]),
                    "video_backend": video_backend,
                    "camera_mapping": dict(ROBOCASA_CAMERA_MAPPING),
                },
            },
        )
        registered.append({"skill_id": task["skill_id"], "skill_dir": str(skill_dir), "task_index": task["task_index"]})
    summary = {
        "dataset_dir": str(dataset_dir),
        "repo_id": repo_id,
        "skill_root": str(skill_root),
        "manifest_path": str(manifest_path),
        "registered_skill_count": len(registered),
        "registered_skills": registered,
        "skipped_task_count": manifest["skipped_task_count"],
    }
    write_json(Path(manifest_path).with_name("robocasa_registration_summary.json"), summary)
    return summary


def _build_episode_index(dataset_dir: Path) -> dict[int, dict[int, list[int]]]:
    task_episode_indices: dict[int, dict[int, list[int]]] = {}
    for parquet_path in _data_parquet_paths(dataset_dir):
        df = pd.read_parquet(parquet_path, columns=["task_index", "episode_index", "index"])
        for row in df.itertuples(index=False):
            task_index = int(row.task_index)
            episode_index = int(row.episode_index)
            global_index = int(row.index)
            task_episode_indices.setdefault(task_index, {}).setdefault(episode_index, []).append(global_index)
    for episodes in task_episode_indices.values():
        for frame_indices in episodes.values():
            frame_indices.sort()
    return task_episode_indices


def _collect_train_arrays(
    dataset_dir: Path,
    train_episode_ids_by_task: dict[int, set[str]],
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    states_by_task: dict[int, list[np.ndarray]] = {task_index: [] for task_index in train_episode_ids_by_task}
    actions_by_task: dict[int, list[np.ndarray]] = {task_index: [] for task_index in train_episode_ids_by_task}
    for parquet_path in _data_parquet_paths(dataset_dir):
        df = pd.read_parquet(
            parquet_path,
            columns=["task_index", "episode_index", "observation.state", "action"],
        )
        for task_index, train_episode_ids in train_episode_ids_by_task.items():
            if not train_episode_ids:
                continue
            episode_ints = {int(value.removeprefix("episode_")) for value in train_episode_ids}
            subset = df[(df["task_index"].astype(int) == task_index) & (df["episode_index"].astype(int).isin(episode_ints))]
            if subset.empty:
                continue
            states_by_task[task_index].append(np.stack(subset["observation.state"].to_numpy()).astype(np.float32))
            actions_by_task[task_index].append(np.stack(subset["action"].to_numpy()).astype(np.float32))
    return states_by_task, actions_by_task


def prepare_active_robocasa_skills(
    *,
    dataset_dir: Path,
    skill_root: Path,
    manifest_path: Path,
    seed: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    info = _load_info(Path(dataset_dir))
    fps = float(info["fps"])
    task_episode_indices = _build_episode_index(Path(dataset_dir))

    task_splits: dict[int, dict[str, list[str]]] = {}
    train_episode_ids_by_task: dict[int, set[str]] = {}
    summaries = []
    prepared_tasks = []
    skipped_short_tasks = []
    for task in manifest["active_tasks"]:
        task_index = int(task["task_index"])
        episodes = task_episode_indices.get(task_index, {})
        eligible = {
            f"episode_{episode_index:06d}": frame_indices
            for episode_index, frame_indices in sorted(episodes.items())
            if len(frame_indices) >= chunk_size
        }
        if not eligible:
            skipped_short_tasks.append({**task, "reason": f"no episodes with >= {chunk_size} frames"})
            continue
        splits = split_episode_ids(list(eligible), seed=seed)
        task_splits[task_index] = splits
        train_episode_ids_by_task[task_index] = set(splits["train"])
        prepared_tasks.append((task, eligible, splits))

    states_by_task, actions_by_task = _collect_train_arrays(Path(dataset_dir), train_episode_ids_by_task)

    for task, eligible, splits in prepared_tasks:
        task_index = int(task["task_index"])
        skill_dir = Path(skill_root) / task["skill_id"]
        state_parts = states_by_task.get(task_index) or []
        action_parts = actions_by_task.get(task_index) or []
        if not state_parts or not action_parts:
            skipped_short_tasks.append({**task, "reason": "empty train split after global prepare"})
            continue
        states = np.concatenate(state_parts, axis=0)
        actions = np.concatenate(action_parts, axis=0)
        stats = {
            "observation.state": compute_vector_stats(states),
            "action": compute_vector_stats(actions),
        }
        write_json(skill_dir / "splits.json", splits)
        write_json(skill_dir / "stats.json", stats)

        records_payload = {
            "task_index": task_index,
            "task_name": task["task_name"],
            "resolved_video_backend": "pyav",
            "records": {
                episode_id: {
                    "source_episode_index": int(episode_id.removeprefix("episode_")),
                    "frame_indices": frame_indices,
                    "num_frames": len(frame_indices),
                    "fps": fps,
                }
                for episode_id, frame_indices in eligible.items()
            },
        }
        write_json(skill_dir / LEROBOT_RECORDS_FILENAME, records_payload)

        source_summary = {
            "skill_id": task["skill_id"],
            "source_type": "lerobot",
            "task_index": task_index,
            "task_name": task["task_name"],
            "episodes": len(eligible),
            "train_episodes": len(splits["train"]),
            "val_episodes": len(splits["val"]),
            "train_windows": sum(max(0, len(eligible[eid]) - chunk_size + 1) for eid in splits["train"]),
            "val_windows": sum(max(0, len(eligible[eid]) - chunk_size + 1) for eid in splits["val"]),
            "splits_path": str(skill_dir / "splits.json"),
            "stats_path": str(skill_dir / "stats.json"),
            "records_path": str(skill_dir / LEROBOT_RECORDS_FILENAME),
        }
        write_json(skill_dir / "source_summary.json", source_summary)
        summaries.append(source_summary)

    result = {
        "dataset_dir": str(dataset_dir),
        "skill_root": str(skill_root),
        "manifest_path": str(manifest_path),
        "prepared_skill_count": len(summaries),
        "skipped_short_task_count": len(skipped_short_tasks),
        "prepared_skills": summaries,
        "skipped_short_tasks": skipped_short_tasks,
    }
    write_json(Path(manifest_path).with_name("robocasa_prepare_summary.json"), result)
    return result
