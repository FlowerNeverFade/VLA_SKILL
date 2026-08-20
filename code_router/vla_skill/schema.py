from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_N_ACTION_STEPS,
    DEFAULT_SKILL_CAMERA_NAMES,
    IMAGE_FIELD_BY_CAMERA,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
)
from .io_utils import load_yaml


@dataclass(frozen=True)
class RouterSpec:
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    regexes: list[str] = field(default_factory=list)
    priority: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RouterSpec":
        payload = payload or {}
        return cls(
            description=str(payload.get("description", "")),
            aliases=[str(item) for item in payload.get("aliases", [])],
            keywords=[str(item) for item in payload.get("keywords", [])],
            regexes=[str(item) for item in payload.get("regexes", [])],
            priority=int(payload.get("priority", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "aliases": list(self.aliases),
            "keywords": list(self.keywords),
            "regexes": list(self.regexes),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class SourceSpec:
    type: str = "materialized"
    dataset_dir: Path | None = None
    repo_id: str = ""
    task_index: int = 0
    video_backend: str = "auto"
    camera_mapping: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SourceSpec":
        payload = payload or {}
        dataset_dir = payload.get("dataset_dir")
        return cls(
            type=str(payload.get("type", "materialized")),
            dataset_dir=Path(dataset_dir) if dataset_dir else None,
            repo_id=str(payload.get("repo_id", "")),
            task_index=int(payload.get("task_index", 0)),
            video_backend=str(payload.get("video_backend", "auto")),
            camera_mapping={str(key): str(value) for key, value in (payload.get("camera_mapping") or {}).items()},
        )

    def validate(self, skill_id: str, camera_names: tuple[str, ...]) -> None:
        if self.type not in {"materialized", "lerobot"}:
            raise ValueError(
                f"skill `{skill_id}` has unsupported source.type={self.type!r}; expected `materialized` or `lerobot`."
            )
        if self.type == "materialized":
            return
        if self.dataset_dir is None:
            raise ValueError(f"skill `{skill_id}` source.dataset_dir is required for lerobot sources.")
        if not self.dataset_dir.is_dir():
            raise ValueError(f"skill `{skill_id}` source.dataset_dir does not exist: {self.dataset_dir}.")
        if not self.repo_id:
            raise ValueError(f"skill `{skill_id}` source.repo_id is required for lerobot sources.")
        if self.task_index < 0:
            raise ValueError(f"skill `{skill_id}` has invalid source.task_index={self.task_index}.")
        if self.video_backend not in {"auto", "torchcodec", "pyav"}:
            raise ValueError(
                f"skill `{skill_id}` has invalid source.video_backend={self.video_backend!r}; "
                "expected `auto`, `torchcodec`, or `pyav`."
            )
        missing = [camera_name for camera_name in camera_names if camera_name not in self.camera_mapping]
        if missing:
            raise ValueError(f"skill `{skill_id}` source.camera_mapping is missing cameras: {missing}.")
        extras = [camera_name for camera_name in self.camera_mapping if camera_name not in camera_names]
        if extras:
            raise ValueError(f"skill `{skill_id}` source.camera_mapping contains unknown cameras: {extras}.")
        for camera_name, source_key in self.camera_mapping.items():
            if not source_key.startswith("observation.images."):
                raise ValueError(
                    f"skill `{skill_id}` source.camera_mapping[{camera_name!r}] must point to an image key, "
                    f"got {source_key!r}."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "repo_id": self.repo_id,
            "task_index": self.task_index,
            "video_backend": self.video_backend,
            "camera_mapping": dict(self.camera_mapping),
        }
        if self.dataset_dir is not None:
            payload["dataset_dir"] = str(self.dataset_dir)
        return payload


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    display_name: str
    state_dim: int
    action_dim: int
    camera_names: tuple[str, ...] = DEFAULT_SKILL_CAMERA_NAMES
    chunk_size: int = DEFAULT_CHUNK_SIZE
    n_action_steps: int = DEFAULT_N_ACTION_STEPS
    router: RouterSpec = field(default_factory=RouterSpec)
    source: SourceSpec = field(default_factory=SourceSpec)
    skill_dir: Path | None = None

    @classmethod
    def load(cls, skill_dir: Path) -> "SkillSpec":
        payload = load_yaml(skill_dir / "skill.yaml")
        if not isinstance(payload, dict):
            raise ValueError(f"{skill_dir / 'skill.yaml'} must contain a YAML object.")
        spec = cls(
            skill_id=str(payload.get("skill_id", "")),
            display_name=str(payload.get("display_name", "")),
            state_dim=int(payload.get("state_dim", 0)),
            action_dim=int(payload.get("action_dim", 0)),
            camera_names=tuple(str(item) for item in payload.get("camera_names", DEFAULT_SKILL_CAMERA_NAMES)),
            chunk_size=int(payload.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            n_action_steps=int(payload.get("n_action_steps", DEFAULT_N_ACTION_STEPS)),
            router=RouterSpec.from_dict(payload.get("router")),
            source=SourceSpec.from_dict(payload.get("source")),
            skill_dir=skill_dir,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.skill_id:
            raise ValueError("skill.yaml is missing `skill_id`.")
        if not self.display_name:
            raise ValueError(f"skill `{self.skill_id}` is missing `display_name`.")
        if not (1 <= self.state_dim <= MAX_STATE_DIM):
            raise ValueError(
                f"skill `{self.skill_id}` has invalid state_dim={self.state_dim}; expected 1..{MAX_STATE_DIM}."
            )
        if not (1 <= self.action_dim <= MAX_ACTION_DIM):
            raise ValueError(
                f"skill `{self.skill_id}` has invalid action_dim={self.action_dim}; expected 1..{MAX_ACTION_DIM}."
            )
        if not self.camera_names:
            raise ValueError(f"skill `{self.skill_id}` must declare at least one camera.")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError(f"skill `{self.skill_id}` camera_names contains duplicates: {self.camera_names}.")
        invalid_cameras = [name for name in self.camera_names if name not in IMAGE_FIELD_BY_CAMERA]
        if invalid_cameras:
            raise ValueError(
                f"skill `{self.skill_id}` camera_names contains unsupported cameras: {invalid_cameras}."
            )
        if self.chunk_size <= 0:
            raise ValueError(f"skill `{self.skill_id}` has invalid chunk_size={self.chunk_size}.")
        if self.n_action_steps <= 0:
            raise ValueError(f"skill `{self.skill_id}` has invalid n_action_steps={self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"skill `{self.skill_id}` has n_action_steps={self.n_action_steps} > chunk_size={self.chunk_size}."
            )
        self.source.validate(self.skill_id, self.camera_names)

    @property
    def episodes_dir(self) -> Path:
        if self.skill_dir is None:
            raise ValueError(f"skill `{self.skill_id}` does not have a local skill_dir.")
        return self.skill_dir / "episodes"

    @property
    def splits_path(self) -> Path:
        if self.skill_dir is None:
            raise ValueError(f"skill `{self.skill_id}` does not have a local skill_dir.")
        return self.skill_dir / "splits.json"

    @property
    def stats_path(self) -> Path:
        if self.skill_dir is None:
            raise ValueError(f"skill `{self.skill_id}` does not have a local skill_dir.")
        return self.skill_dir / "stats.json"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "skill_id": self.skill_id,
            "display_name": self.display_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "camera_names": list(self.camera_names),
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "router": self.router.to_dict(),
            "source": self.source.to_dict(),
        }
        if self.skill_dir is not None:
            payload["skill_dir"] = str(self.skill_dir)
        return payload
