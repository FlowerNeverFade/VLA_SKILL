from __future__ import annotations

from collections import OrderedDict
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, resize

from .constants import DEFAULT_SEED, IMAGE_FIELD_BY_CAMERA, IMAGE_RESOLUTION
from .io_utils import load_json, write_json
from .schema import SkillSpec
from .stats import compute_vector_stats


class SkillDataError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    episode_dir: Path
    meta: dict[str, Any]
    num_frames: int
    fps: float
    task: str
    state_shape: tuple[int, int]
    action_shape: tuple[int, int]
    image_paths: dict[str, tuple[Path, ...]]


@dataclass(frozen=True)
class LeRobotEpisodeRecord:
    episode_id: str
    source_episode_index: int
    frame_indices: tuple[int, ...]
    num_frames: int
    fps: float
    task: str


def discover_skill_dirs(skill_root: Path) -> list[Path]:
    return sorted(path for path in skill_root.iterdir() if path.is_dir() and (path / "skill.yaml").is_file())


def load_skill_spec(skill_root: Path, skill_id: str) -> SkillSpec:
    skill_dir = skill_root / skill_id
    if not skill_dir.is_dir():
        raise SkillDataError(f"Skill directory does not exist: {skill_dir}")
    return SkillSpec.load(skill_dir)


def _load_meta(path: Path, skill_id: str, episode_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise SkillDataError(f"skill `{skill_id}` episode `{episode_id}` is missing {path.name}.")
    data = load_json(path)
    if not isinstance(data, dict):
        raise SkillDataError(f"skill `{skill_id}` episode `{episode_id}` has a non-object {path.name}.")
    for field in ("episode_id", "task", "num_frames", "fps"):
        if field not in data:
            raise SkillDataError(f"skill `{skill_id}` episode `{episode_id}` meta.json is missing `{field}`.")
    return data


def _load_array(path: Path, skill_id: str, episode_id: str, name: str) -> np.ndarray:
    if not path.is_file():
        raise SkillDataError(f"skill `{skill_id}` episode `{episode_id}` is missing {path.name}.")
    array = np.load(path)
    if array.ndim != 2:
        raise SkillDataError(
            f"skill `{skill_id}` episode `{episode_id}` {name} must be 2D [T, D], got shape={array.shape}."
        )
    return np.asarray(array, dtype=np.float32)


def validate_episode_directory(skill_spec: SkillSpec, episode_dir: Path) -> EpisodeRecord:
    episode_id = episode_dir.name
    meta = _load_meta(episode_dir / "meta.json", skill_spec.skill_id, episode_id)
    state = _load_array(episode_dir / "state.npy", skill_spec.skill_id, episode_id, "state.npy")
    action = _load_array(episode_dir / "action.npy", skill_spec.skill_id, episode_id, "action.npy")

    if state.shape[1] != skill_spec.state_dim:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` episode `{episode_id}` state dim={state.shape[1]} "
            f"does not match skill.yaml state_dim={skill_spec.state_dim}."
        )
    if action.shape[1] != skill_spec.action_dim:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` episode `{episode_id}` action dim={action.shape[1]} "
            f"does not match skill.yaml action_dim={skill_spec.action_dim}."
        )

    num_frames = int(meta["num_frames"])
    if num_frames < skill_spec.chunk_size:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` episode `{episode_id}` num_frames={num_frames} "
            f"is smaller than chunk_size={skill_spec.chunk_size}."
        )
    if state.shape[0] != num_frames:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` episode `{episode_id}` state length={state.shape[0]} "
            f"does not match meta num_frames={num_frames}."
        )
    if action.shape[0] != num_frames:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` episode `{episode_id}` action length={action.shape[0]} "
            f"does not match meta num_frames={num_frames}."
        )

    image_paths: dict[str, tuple[Path, ...]] = {}
    for camera_name in skill_spec.camera_names:
        camera_dir = episode_dir / "images" / camera_name
        if not camera_dir.is_dir():
            raise SkillDataError(
                f"skill `{skill_spec.skill_id}` episode `{episode_id}` is missing image directory {camera_dir}."
            )
        frames = tuple(sorted(camera_dir.glob("*.jpg")))
        if len(frames) != num_frames:
            raise SkillDataError(
                f"skill `{skill_spec.skill_id}` episode `{episode_id}` camera `{camera_name}` has {len(frames)} "
                f"frames but meta num_frames={num_frames}."
            )
        image_paths[camera_name] = frames

    return EpisodeRecord(
        episode_id=episode_id,
        episode_dir=episode_dir,
        meta=meta,
        num_frames=num_frames,
        fps=float(meta["fps"]),
        task=str(meta["task"]),
        state_shape=tuple(state.shape),
        action_shape=tuple(action.shape),
        image_paths=image_paths,
    )


def collect_episode_records(skill_spec: SkillSpec) -> dict[str, EpisodeRecord]:
    if not skill_spec.episodes_dir.is_dir():
        raise SkillDataError(f"skill `{skill_spec.skill_id}` is missing episodes directory {skill_spec.episodes_dir}.")
    records = {
        episode_dir.name: validate_episode_directory(skill_spec, episode_dir)
        for episode_dir in sorted(skill_spec.episodes_dir.iterdir())
        if episode_dir.is_dir()
    }
    if not records:
        raise SkillDataError(f"skill `{skill_spec.skill_id}` has no episode directories under {skill_spec.episodes_dir}.")
    return records


def _load_lerobot_task_name(skill_spec: SkillSpec) -> str:
    source = skill_spec.source
    assert source.dataset_dir is not None
    tasks_path = source.dataset_dir / "meta" / "tasks.json"
    payload = load_json(tasks_path)
    try:
        return str(payload[str(source.task_index)])
    except KeyError as exc:
        raise SkillDataError(f"Task index {source.task_index} not found in {tasks_path}.") from exc


def _lerobot_dataset_kwargs(skill_spec: SkillSpec, video_backend: str) -> dict[str, Any]:
    source = skill_spec.source
    assert source.dataset_dir is not None
    return {
        "repo_id": source.repo_id,
        "root": source.dataset_dir,
        "download_videos": False,
        "video_backend": video_backend,
        "return_uint8": True,
    }


def resolve_lerobot_video_backend(skill_spec: SkillSpec) -> tuple[str, str | None]:
    requested_backend = skill_spec.source.video_backend
    if requested_backend != "auto":
        dataset = LeRobotDataset(**_lerobot_dataset_kwargs(skill_spec, requested_backend))
        _ = dataset[0]
        return requested_backend, None

    try:
        dataset = LeRobotDataset(**_lerobot_dataset_kwargs(skill_spec, "torchcodec"))
        _ = dataset[0]
        return "torchcodec", None
    except Exception as exc:  # noqa: BLE001
        fallback_reason = f"{type(exc).__name__}: {exc}"

    dataset = LeRobotDataset(**_lerobot_dataset_kwargs(skill_spec, "pyav"))
    _ = dataset[0]
    return "pyav", fallback_reason


def build_lerobot_dataset(skill_spec: SkillSpec) -> tuple[LeRobotDataset, str, str | None]:
    if skill_spec.source.type != "lerobot":
        raise SkillDataError(f"skill `{skill_spec.skill_id}` is not configured for raw LeRobot loading.")
    resolved_backend, fallback_reason = resolve_lerobot_video_backend(skill_spec)
    dataset = LeRobotDataset(**_lerobot_dataset_kwargs(skill_spec, resolved_backend))
    return dataset, resolved_backend, fallback_reason


def collect_lerobot_episode_records(
    skill_spec: SkillSpec,
    dataset: LeRobotDataset | None = None,
) -> tuple[dict[str, LeRobotEpisodeRecord], str, str | None]:
    if dataset is None:
        dataset, resolved_backend, fallback_reason = build_lerobot_dataset(skill_spec)
    else:
        resolved_backend = skill_spec.source.video_backend
        fallback_reason = None

    task_name = _load_lerobot_task_name(skill_spec)
    source = skill_spec.source
    episodes: OrderedDict[int, list[int]] = OrderedDict()
    for global_index, row in enumerate(dataset.hf_dataset):
        if int(row["task_index"]) != source.task_index:
            continue
        episode_index = int(row["episode_index"])
        episodes.setdefault(episode_index, []).append(global_index)

    if not episodes:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` source task_index={source.task_index} produced zero episodes "
            f"from {source.dataset_dir}."
        )

    records: dict[str, LeRobotEpisodeRecord] = {}
    for episode_index, frame_indices in episodes.items():
        episode_id = f"episode_{episode_index:06d}"
        num_frames = len(frame_indices)
        if num_frames < skill_spec.chunk_size:
            raise SkillDataError(
                f"skill `{skill_spec.skill_id}` source episode `{episode_id}` has num_frames={num_frames} "
                f"smaller than chunk_size={skill_spec.chunk_size}."
            )
        records[episode_id] = LeRobotEpisodeRecord(
            episode_id=episode_id,
            source_episode_index=episode_index,
            frame_indices=tuple(frame_indices),
            num_frames=num_frames,
            fps=float(dataset.meta.fps),
            task=task_name,
        )
    return records, resolved_backend, fallback_reason


def split_episode_ids(episode_ids: list[str], seed: int = DEFAULT_SEED, val_ratio: float = 0.1) -> dict[str, list[str]]:
    shuffled = list(sorted(episode_ids))
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return {"train": shuffled, "val": []}

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    return {"train": train_ids, "val": val_ids}


def compute_skill_stats(skill_spec: SkillSpec, episode_ids: list[str]) -> dict[str, Any]:
    states = []
    actions = []
    for episode_id in episode_ids:
        episode_dir = skill_spec.episodes_dir / episode_id
        states.append(np.asarray(np.load(episode_dir / "state.npy"), dtype=np.float32))
        actions.append(np.asarray(np.load(episode_dir / "action.npy"), dtype=np.float32))
    if not states or not actions:
        raise SkillDataError(f"skill `{skill_spec.skill_id}` has an empty train split; cannot compute stats.")
    state_matrix = np.concatenate(states, axis=0)
    action_matrix = np.concatenate(actions, axis=0)
    return {
        "observation.state": compute_vector_stats(state_matrix),
        "action": compute_vector_stats(action_matrix),
    }


def compute_lerobot_stats(
    skill_spec: SkillSpec,
    episode_records: dict[str, LeRobotEpisodeRecord],
    episode_ids: list[str],
    hf_dataset: Any,
) -> dict[str, Any]:
    states = []
    actions = []
    total_episodes = len(episode_ids)
    for index, episode_id in enumerate(episode_ids, start=1):
        record = episode_records[episode_id]
        state_rows = []
        action_rows = []
        for global_index in record.frame_indices:
            row = hf_dataset[global_index]
            state_rows.append(np.asarray(row["observation.state"], dtype=np.float32))
            action_rows.append(np.asarray(row["action"], dtype=np.float32))
        states.append(np.stack(state_rows, axis=0))
        actions.append(np.stack(action_rows, axis=0))
        if index == 1 or index % 500 == 0 or index == total_episodes:
            print(
                f"[prepare] stats progress skill={skill_spec.skill_id} episodes={index}/{total_episodes}",
                flush=True,
            )
    if not states or not actions:
        raise SkillDataError(f"skill `{skill_spec.skill_id}` has an empty train split; cannot compute stats.")
    state_matrix = np.concatenate(states, axis=0)
    action_matrix = np.concatenate(actions, axis=0)
    return {
        "observation.state": compute_vector_stats(state_matrix),
        "action": compute_vector_stats(action_matrix),
    }


def prepare_skill_directory(skill_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    skill_spec = SkillSpec.load(skill_dir)
    if skill_spec.source.type == "lerobot":
        print(f"[prepare] raw lerobot skill={skill_spec.skill_id} seed={seed}", flush=True)
        dataset, resolved_backend, fallback_reason = build_lerobot_dataset(skill_spec)
        records, _, _ = collect_lerobot_episode_records(skill_spec, dataset=dataset)
        print(
            f"[prepare] collected episodes skill={skill_spec.skill_id} count={len(records)} backend={resolved_backend}",
            flush=True,
        )
        splits = split_episode_ids(list(records.keys()), seed=seed)
        stats = compute_lerobot_stats(skill_spec, records, splits["train"], dataset.hf_dataset)
        write_json(skill_spec.splits_path, splits)
        write_json(skill_spec.stats_path, stats)
        summary = {
            "skill_id": skill_spec.skill_id,
            "source_type": "lerobot",
            "resolved_video_backend": resolved_backend,
            "episodes": len(records),
            "train_episodes": len(splits["train"]),
            "val_episodes": len(splits["val"]),
            "train_windows": sum(max(0, records[eid].num_frames - skill_spec.chunk_size + 1) for eid in splits["train"]),
            "val_windows": sum(max(0, records[eid].num_frames - skill_spec.chunk_size + 1) for eid in splits["val"]),
            "splits_path": str(skill_spec.splits_path),
            "stats_path": str(skill_spec.stats_path),
        }
        if fallback_reason:
            summary["fallback_reason"] = fallback_reason
        write_json(skill_dir / "source_summary.json", summary)
        return summary
    records = collect_episode_records(skill_spec)
    splits = split_episode_ids(list(records.keys()), seed=seed)
    stats = compute_skill_stats(skill_spec, splits["train"])
    write_json(skill_spec.splits_path, splits)
    write_json(skill_spec.stats_path, stats)
    summary = {
        "skill_id": skill_spec.skill_id,
        "episodes": len(records),
        "train_episodes": len(splits["train"]),
        "val_episodes": len(splits["val"]),
        "train_windows": sum(max(0, records[eid].num_frames - skill_spec.chunk_size + 1) for eid in splits["train"]),
        "val_windows": sum(max(0, records[eid].num_frames - skill_spec.chunk_size + 1) for eid in splits["val"]),
        "splits_path": str(skill_spec.splits_path),
        "stats_path": str(skill_spec.stats_path),
    }
    return summary


def load_splits(skill_spec: SkillSpec) -> dict[str, list[str]]:
    if not skill_spec.splits_path.is_file():
        raise SkillDataError(f"skill `{skill_spec.skill_id}` is missing {skill_spec.splits_path}. Run prepare first.")
    data = load_json(skill_spec.splits_path)
    if not isinstance(data, dict):
        raise SkillDataError(f"{skill_spec.splits_path} must contain a JSON object.")
    return {
        "train": [str(item) for item in data.get("train", [])],
        "val": [str(item) for item in data.get("val", [])],
    }


def load_stats(skill_spec: SkillSpec) -> dict[str, Any]:
    if not skill_spec.stats_path.is_file():
        raise SkillDataError(f"skill `{skill_spec.skill_id}` is missing {skill_spec.stats_path}. Run prepare first.")
    return load_json(skill_spec.stats_path)


def _image_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).to(dtype=torch.float32) / 255.0
    return tensor


def _image_value_to_tensor(image_value: Any) -> torch.Tensor:
    if isinstance(image_value, Image.Image):
        tensor = pil_to_tensor(image_value.convert("RGB"))
    elif torch.is_tensor(image_value):
        tensor = image_value.detach().cpu()
        if tensor.ndim != 3:
            raise SkillDataError(f"Expected image tensor with 3 dims, got shape={tuple(tensor.shape)}.")
        if tensor.shape[0] not in (1, 3):
            if tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).contiguous()
            else:
                raise SkillDataError(f"Unsupported image tensor shape: {tuple(tensor.shape)}.")
    elif isinstance(image_value, np.ndarray):
        array = np.asarray(image_value)
        if array.ndim != 3:
            raise SkillDataError(f"Unsupported image ndarray shape: {array.shape}.")
        tensor = torch.from_numpy(array)
        if tensor.shape[0] not in (1, 3):
            if tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).contiguous()
            else:
                raise SkillDataError(f"Unsupported image ndarray shape: {array.shape}.")
    else:
        raise SkillDataError(f"Unsupported image value type: {type(image_value)!r}.")

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor.to(dtype=torch.float32)
    if float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    if tuple(tensor.shape[-2:]) != IMAGE_RESOLUTION:
        tensor = resize(tensor, list(IMAGE_RESOLUTION), antialias=True)
    return tensor


class SkillWindowDataset(Dataset[dict[str, Any]]):
    def __init__(self, skill_spec: SkillSpec, split: str):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split `{split}`.")
        self.skill_spec = skill_spec
        self.split = split
        self.episode_records = collect_episode_records(skill_spec)
        self.splits = load_splits(skill_spec)
        self.episode_ids = list(self.splits[split])
        self._episode_numeric_index = {episode_id: idx for idx, episode_id in enumerate(sorted(self.episode_records))}
        self.windows: list[tuple[str, int]] = []
        self._array_cache: dict[str, dict[str, np.ndarray]] = {}

        for episode_id in self.episode_ids:
            if episode_id not in self.episode_records:
                raise SkillDataError(
                    f"skill `{skill_spec.skill_id}` split `{split}` references unknown episode `{episode_id}`."
                )
            num_frames = self.episode_records[episode_id].num_frames
            for start_t in range(0, num_frames - skill_spec.chunk_size + 1):
                self.windows.append((episode_id, start_t))

    def __len__(self) -> int:
        return len(self.windows)

    def _load_episode_arrays(self, episode_id: str) -> dict[str, np.ndarray]:
        cached = self._array_cache.get(episode_id)
        if cached is None:
            episode_dir = self.skill_spec.episodes_dir / episode_id
            cached = {
                "state": np.asarray(np.load(episode_dir / "state.npy"), dtype=np.float32),
                "action": np.asarray(np.load(episode_dir / "action.npy"), dtype=np.float32),
            }
            self._array_cache[episode_id] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, start_t = self.windows[index]
        record = self.episode_records[episode_id]
        arrays = self._load_episode_arrays(episode_id)

        sample: dict[str, Any] = {
            "task": record.task,
            "episode_index": torch.tensor(self._episode_numeric_index[episode_id], dtype=torch.int64),
            "frame_index": torch.tensor(start_t, dtype=torch.int64),
            "observation.state": torch.from_numpy(arrays["state"][start_t].copy()),
            "action": torch.from_numpy(arrays["action"][start_t : start_t + self.skill_spec.chunk_size].copy()),
        }
        for camera_name in self.skill_spec.camera_names:
            field_name = IMAGE_FIELD_BY_CAMERA[camera_name]
            sample[field_name] = _image_to_tensor(record.image_paths[camera_name][start_t])
        return sample


class LeRobotWindowDataset(Dataset[dict[str, Any]]):
    def __init__(self, skill_spec: SkillSpec, split: str):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split `{split}`.")
        if skill_spec.source.type != "lerobot":
            raise ValueError(f"skill `{skill_spec.skill_id}` is not configured for a lerobot source.")
        self.skill_spec = skill_spec
        self.split = split
        self.dataset, self.resolved_video_backend, self.fallback_reason = build_lerobot_dataset(skill_spec)
        self.episode_records, _, _ = collect_lerobot_episode_records(skill_spec, dataset=self.dataset)
        self.splits = load_splits(skill_spec)
        self.episode_ids = list(self.splits[split])
        self._episode_numeric_index = {episode_id: idx for idx, episode_id in enumerate(sorted(self.episode_records))}
        self.windows: list[tuple[str, int]] = []
        self._array_cache: dict[str, dict[str, np.ndarray]] = {}

        for episode_id in self.episode_ids:
            if episode_id not in self.episode_records:
                raise SkillDataError(
                    f"skill `{skill_spec.skill_id}` split `{split}` references unknown episode `{episode_id}`."
                )
            num_frames = self.episode_records[episode_id].num_frames
            for start_t in range(0, num_frames - skill_spec.chunk_size + 1):
                self.windows.append((episode_id, start_t))

    def __len__(self) -> int:
        return len(self.windows)

    def _load_episode_arrays(self, episode_id: str) -> dict[str, np.ndarray]:
        cached = self._array_cache.get(episode_id)
        if cached is None:
            record = self.episode_records[episode_id]
            state_rows = []
            action_rows = []
            for global_index in record.frame_indices:
                row = self.dataset.hf_dataset[global_index]
                state_rows.append(np.asarray(row["observation.state"], dtype=np.float32))
                action_rows.append(np.asarray(row["action"], dtype=np.float32))
            cached = {
                "state": np.stack(state_rows, axis=0),
                "action": np.stack(action_rows, axis=0),
            }
            self._array_cache[episode_id] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, start_t = self.windows[index]
        record = self.episode_records[episode_id]
        arrays = self._load_episode_arrays(episode_id)
        global_index = record.frame_indices[start_t]
        item = self.dataset[global_index]

        sample: dict[str, Any] = {
            "task": record.task,
            "episode_index": torch.tensor(self._episode_numeric_index[episode_id], dtype=torch.int64),
            "frame_index": torch.tensor(start_t, dtype=torch.int64),
            "observation.state": torch.from_numpy(arrays["state"][start_t].copy()),
            "action": torch.from_numpy(arrays["action"][start_t : start_t + self.skill_spec.chunk_size].copy()),
        }
        for camera_name in self.skill_spec.camera_names:
            source_key = self.skill_spec.source.camera_mapping[camera_name]
            field_name = IMAGE_FIELD_BY_CAMERA[camera_name]
            sample[field_name] = _image_value_to_tensor(item[source_key])
        return sample


def build_window_dataset(skill_spec: SkillSpec, split: str) -> Dataset[dict[str, Any]]:
    if skill_spec.source.type == "lerobot":
        return LeRobotWindowDataset(skill_spec, split=split)
    return SkillWindowDataset(skill_spec, split=split)


def load_obs_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SkillDataError(f"{path} must contain a JSON object.")
    return payload


def build_single_observation_batch(
    skill_spec: SkillSpec,
    task: str,
    state: list[float] | np.ndarray,
    image_paths: dict[str, str | Path],
) -> dict[str, Any]:
    state_arr = np.asarray(state, dtype=np.float32)
    if state_arr.ndim != 1 or state_arr.shape[0] != skill_spec.state_dim:
        raise SkillDataError(
            f"skill `{skill_spec.skill_id}` expects state_dim={skill_spec.state_dim}, got shape={state_arr.shape}."
        )
    batch: dict[str, Any] = {
        "task": str(task),
        "observation.state": torch.from_numpy(state_arr.copy()),
    }
    for camera_name in skill_spec.camera_names:
        field_name = IMAGE_FIELD_BY_CAMERA[camera_name]
        if camera_name not in image_paths:
            raise SkillDataError(f"obs.json is missing image path for `{camera_name}`.")
        batch[field_name] = _image_to_tensor(Path(image_paths[camera_name]))
    return batch
