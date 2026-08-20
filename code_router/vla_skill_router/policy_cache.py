from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import os
import os
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import torch
from torch.utils.data import IterableDataset, get_worker_info

from vla_skill.constants import IMAGE_FIELDS, IMAGE_RESOLUTION
from vla_skill.io_utils import ensure_dir, load_json, utc_now_iso, write_json
from vla_skill.pi05 import resolve_tokenizer_name_or_path
from vla_skill.schema import SkillSpec


POLICY_CACHE_VERSION = 1
POLICY_CACHE_FORMAT = "policy_ready_float32"
_DATA_ROOT = Path(os.environ.get("VLA_DATA_ROOT", os.environ.get("VLA_SKILL_ROOT", Path(__file__).resolve().parents[2])))
DEFAULT_POLICY_CACHE_ROOT = Path(
    os.environ.get("VLA_POLICY_CACHE_ROOT", _DATA_ROOT / "cache" / "pi05_policy_ready")
)
IMAGE_STORAGE_DTYPES = ("float32", "float16", "uint8")
DEFAULT_IMAGE_STORAGE_DTYPE = "float32"


class PolicyCacheError(ValueError):
    pass


def policy_cache_split_dir(cache_root: Path, skill_id: str, split: str) -> Path:
    return Path(cache_root) / skill_id / split


def policy_cache_manifest_path(cache_root: Path, skill_id: str, split: str) -> Path:
    return policy_cache_split_dir(cache_root, skill_id, split) / "manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(_sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _json_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_policy_cache_payload(
    skill_spec: SkillSpec,
    *,
    split: str,
    tokenizer_name_or_path: str | Path | None = None,
    image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
) -> dict[str, Any]:
    if image_storage_dtype not in IMAGE_STORAGE_DTYPES:
        raise PolicyCacheError(f"Unsupported image_storage_dtype={image_storage_dtype!r}.")
    tokenizer_path = Path(resolve_tokenizer_name_or_path(tokenizer_name_or_path))
    skill_yaml_path = skill_spec.skill_dir / "skill.yaml" if skill_spec.skill_dir is not None else None
    source_payload: dict[str, Any] = {"type": skill_spec.source.type}
    if skill_spec.source.dataset_dir is not None:
        source_payload["dataset_dir"] = str(skill_spec.source.dataset_dir)
    source_payload["repo_id"] = skill_spec.source.repo_id
    source_payload["task_index"] = skill_spec.source.task_index
    source_payload["camera_mapping"] = dict(skill_spec.source.camera_mapping)
    source_payload["video_backend"] = skill_spec.source.video_backend
    return {
        "version": POLICY_CACHE_VERSION,
        "format": POLICY_CACHE_FORMAT,
        "skill_id": skill_spec.skill_id,
        "split": split,
        "state_dim": skill_spec.state_dim,
        "action_dim": skill_spec.action_dim,
        "camera_names": list(skill_spec.camera_names),
        "chunk_size": skill_spec.chunk_size,
        "n_action_steps": skill_spec.n_action_steps,
        "image_fields": list(IMAGE_FIELDS),
        "image_resolution": list(IMAGE_RESOLUTION),
        "image_storage_dtype": image_storage_dtype,
        "source": source_payload,
        "skill_yaml_sha256": _sha256_file(skill_yaml_path) if skill_yaml_path is not None and skill_yaml_path.is_file() else None,
        "stats_sha256": _sha256_file(skill_spec.stats_path) if skill_spec.stats_path.is_file() else None,
        "splits_sha256": _sha256_file(skill_spec.splits_path) if skill_spec.splits_path.is_file() else None,
        "tokenizer_name_or_path": str(tokenizer_path),
        "tokenizer_sha256": _sha256_tree(tokenizer_path),
    }


def expected_policy_cache_fingerprint(
    skill_spec: SkillSpec,
    *,
    split: str,
    tokenizer_name_or_path: str | Path | None = None,
    image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
) -> tuple[str, dict[str, Any]]:
    payload = expected_policy_cache_payload(
        skill_spec,
        split=split,
        tokenizer_name_or_path=tokenizer_name_or_path,
        image_storage_dtype=image_storage_dtype,
    )
    return _json_digest(payload), payload


def _field_schema(fields: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape[1:]),
            "dtype": str(value.dtype).replace("torch.", ""),
        }
        for key, value in sorted(fields.items())
    }


def _pack_image_tensor(tensor: torch.Tensor, image_storage_dtype: str) -> torch.Tensor:
    if image_storage_dtype == "float32":
        return tensor.to(device="cpu", dtype=torch.float32).contiguous()
    if image_storage_dtype == "float16":
        return tensor.to(device="cpu", dtype=torch.float16).contiguous()
    if image_storage_dtype == "uint8":
        return tensor.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).contiguous()
    raise PolicyCacheError(f"Unsupported image_storage_dtype={image_storage_dtype!r}.")


def select_policy_cache_fields(
    proc_batch: dict[str, Any],
    *,
    image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
) -> dict[str, torch.Tensor]:
    if image_storage_dtype not in IMAGE_STORAGE_DTYPES:
        raise PolicyCacheError(f"Unsupported image_storage_dtype={image_storage_dtype!r}.")
    action = proc_batch.get("action")
    if not torch.is_tensor(action) or action.ndim == 0:
        raise PolicyCacheError("Processed batch must contain batched tensor field `action`.")
    batch_size = int(action.shape[0])
    fields: dict[str, torch.Tensor] = {}
    for key, value in proc_batch.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim == 0 or int(value.shape[0]) != batch_size:
            continue
        if key in IMAGE_FIELDS:
            tensor = _pack_image_tensor(value, image_storage_dtype)
        else:
            tensor = value.detach().to(device="cpu").contiguous()
        fields[str(key)] = tensor
    required = ["action", "observation.state", "channel_index", "action_dim"]
    required.extend(IMAGE_FIELDS)
    missing = [key for key in required if key not in fields]
    if missing:
        raise PolicyCacheError(f"Processed batch is missing cache-required tensor fields: {missing}")
    return fields


def _concat_field_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = sorted(batches[0])
    for batch in batches[1:]:
        if sorted(batch) != keys:
            raise PolicyCacheError("Cannot concatenate cache batches with different fields.")
    return {key: torch.cat([batch[key] for batch in batches], dim=0).contiguous() for key in keys}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def write_policy_cache_shard(
    split_dir: Path,
    *,
    shard_index: int,
    skill_id: str,
    split: str,
    fields: dict[str, torch.Tensor],
    sample_start: int,
) -> dict[str, Any]:
    sample_count = int(next(iter(fields.values())).shape[0])
    shard_name = f"shard-{shard_index:06d}.pt"
    meta_name = f"shard-{shard_index:06d}.json"
    shard_path = split_dir / shard_name
    meta_path = split_dir / meta_name
    payload = {
        "version": POLICY_CACHE_VERSION,
        "format": POLICY_CACHE_FORMAT,
        "skill_id": skill_id,
        "split": split,
        "sample_start": sample_start,
        "sample_count": sample_count,
        "fields": fields,
    }
    _atomic_torch_save(payload, shard_path)
    meta = {
        "path": shard_name,
        "meta_path": meta_name,
        "sample_start": sample_start,
        "sample_count": sample_count,
        "field_schema": _field_schema(fields),
        "written_at": utc_now_iso(),
    }
    write_json(meta_path, meta)
    return meta


def load_existing_shard_metas(split_dir: Path) -> list[dict[str, Any]]:
    metas = []
    for path in sorted(split_dir.glob("shard-*.json")):
        try:
            payload = load_json(path)
            shard_path = split_dir / str(payload["path"])
        except Exception:
            # A killed cache build can leave a half-written sidecar. Ignore it;
            # the corresponding shard will be rebuilt from the last complete meta.
            continue
        if shard_path.is_file():
            metas.append(payload)
    metas.sort(key=lambda item: int(item["sample_start"]))
    return metas


def write_policy_cache_manifest(
    split_dir: Path,
    *,
    skill_spec: SkillSpec,
    split: str,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    shards: list[dict[str, Any]],
    image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
) -> dict[str, Any]:
    sample_count = sum(int(item["sample_count"]) for item in shards)
    field_schema = shards[0]["field_schema"] if shards else {}
    manifest = {
        "version": POLICY_CACHE_VERSION,
        "format": POLICY_CACHE_FORMAT,
        "skill_id": skill_spec.skill_id,
        "split": split,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "image_storage_dtype": image_storage_dtype,
        "sample_count": sample_count,
        "shard_count": len(shards),
        "field_schema": field_schema,
        "shards": shards,
        "created_at": utc_now_iso(),
    }
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def load_policy_cache_manifest(
    cache_root: Path,
    skill_spec: SkillSpec,
    *,
    split: str,
    tokenizer_name_or_path: str | Path | None = None,
    image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
    require_fingerprint_match: bool = True,
) -> dict[str, Any]:
    path = policy_cache_manifest_path(cache_root, skill_spec.skill_id, split)
    if not path.is_file():
        raise FileNotFoundError(f"Missing policy cache manifest: {path}")
    manifest = load_json(path)
    if int(manifest.get("version", -1)) != POLICY_CACHE_VERSION:
        raise PolicyCacheError(f"{path} has unsupported cache version {manifest.get('version')!r}.")
    if manifest.get("format") != POLICY_CACHE_FORMAT:
        raise PolicyCacheError(f"{path} has unsupported cache format {manifest.get('format')!r}.")
    if manifest.get("skill_id") != skill_spec.skill_id or manifest.get("split") != split:
        raise PolicyCacheError(f"{path} does not match skill={skill_spec.skill_id} split={split}.")
    if require_fingerprint_match:
        expected, _ = expected_policy_cache_fingerprint(
            skill_spec,
            split=split,
            tokenizer_name_or_path=tokenizer_name_or_path,
            image_storage_dtype=image_storage_dtype,
        )
        if manifest.get("fingerprint") != expected:
            raise PolicyCacheError(
                f"Policy cache fingerprint mismatch for skill={skill_spec.skill_id} split={split}. "
                "Rebuild the cache."
            )
    return manifest


def load_policy_cache_shard(cache_split_dir: Path, shard: dict[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(cache_split_dir / str(shard["path"]), map_location="cpu")
    if payload.get("version") != POLICY_CACHE_VERSION or payload.get("format") != POLICY_CACHE_FORMAT:
        raise PolicyCacheError(f"Invalid policy cache shard: {cache_split_dir / str(shard['path'])}")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise PolicyCacheError(f"Policy cache shard has no fields: {cache_split_dir / str(shard['path'])}")
    return fields


class PolicyCacheIterableDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        cache_root: Path,
        skill_spec: SkillSpec,
        split: str,
        tokenizer_name_or_path: str | Path | None = None,
        image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.skill_spec = skill_spec
        self.split = split
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.shuffle = shuffle
        self.manifest = load_policy_cache_manifest(
            self.cache_root,
            skill_spec,
            split=split,
            tokenizer_name_or_path=tokenizer_name_or_path,
            image_storage_dtype=image_storage_dtype,
        )
        self.split_dir = policy_cache_split_dir(self.cache_root, skill_spec.skill_id, split)
        self.shards = list(self.manifest.get("shards") or [])
        self.sample_count = int(self.manifest.get("sample_count", 0))

    def __len__(self) -> int:
        return self.sample_count

    def _assigned_shards(self) -> list[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        total_workers = max(1, self.world_size * num_workers)
        global_worker = self.rank * num_workers + worker_id
        order = list(range(len(self.shards)))
        rng = random.Random(self.seed + 9973 * self.rank + 1009 * worker_id)
        if self.shuffle:
            rng.shuffle(order)
        return [self.shards[index] for pos, index in enumerate(order) if pos % total_workers == global_worker]

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + 104729 * (self.rank + 1) + 15485863 * (worker_id + 1))
        for shard in self._assigned_shards():
            fields = load_policy_cache_shard(self.split_dir, shard)
            count = int(shard["sample_count"])
            indices = list(range(count))
            if self.shuffle:
                rng.shuffle(indices)
            for offset in indices:
                yield {key: value[offset] for key, value in fields.items()}


class PolicyCacheMapDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        cache_root: Path,
        skill_spec: SkillSpec,
        split: str,
        tokenizer_name_or_path: str | Path | None = None,
        image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.skill_spec = skill_spec
        self.split = split
        self.manifest = load_policy_cache_manifest(
            self.cache_root,
            skill_spec,
            split=split,
            tokenizer_name_or_path=tokenizer_name_or_path,
            image_storage_dtype=image_storage_dtype,
        )
        self.split_dir = policy_cache_split_dir(self.cache_root, skill_spec.skill_id, split)
        self.shards = list(self.manifest["shards"])
        self.starts = [int(item["sample_start"]) for item in self.shards]
        self.ends = [int(item["sample_start"]) + int(item["sample_count"]) for item in self.shards]
        self._cached_shard_index: int | None = None
        self._cached_fields: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return int(self.manifest["sample_count"])

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._cached_shard_index != shard_index or self._cached_fields is None:
            self._cached_fields = load_policy_cache_shard(self.split_dir, self.shards[shard_index])
            self._cached_shard_index = shard_index
        return self._cached_fields

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        shard = self.shards[shard_index]
        offset = index - int(shard["sample_start"])
        fields = self._load_shard(shard_index)
        return {key: value[offset] for key, value in fields.items()}


def remove_policy_cache_split(cache_root: Path, skill_id: str, split: str) -> None:
    path = policy_cache_split_dir(cache_root, skill_id, split)
    if path.exists():
        shutil.rmtree(path)
