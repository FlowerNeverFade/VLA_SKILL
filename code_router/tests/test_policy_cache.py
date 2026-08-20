from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vla_skill.constants import IMAGE_FIELDS
from vla_skill.schema import SkillSpec
from vla_skill_router.config import ChannelConfig, ExperimentConfig, TrainConfig
from vla_skill_router.policy_cache import (
    PolicyCacheError,
    PolicyCacheIterableDataset,
    PolicyCacheMapDataset,
    expected_policy_cache_fingerprint,
    policy_cache_split_dir,
    select_policy_cache_fields,
    write_policy_cache_manifest,
    write_policy_cache_shard,
)
from vla_skill_router.real_runtime import build_dataset_for_channel


def _write_skill(tmp_path: Path) -> tuple[SkillSpec, Path]:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_text("tokenizer", encoding="utf-8")
    skill_dir = tmp_path / "skill_a"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        f"""
skill_id: skill_a
display_name: Skill A
state_dim: 4
action_dim: 3
camera_names:
- base_0_rgb
- left_wrist_0_rgb
- right_wrist_0_rgb
chunk_size: 2
n_action_steps: 2
source:
  type: lerobot
  dataset_dir: {dataset_dir}
  repo_id: fake/repo
  task_index: 0
  video_backend: pyav
  camera_mapping:
    base_0_rgb: observation.images.base
    left_wrist_0_rgb: observation.images.left
    right_wrist_0_rgb: observation.images.right
""",
        encoding="utf-8",
    )
    (skill_dir / "stats.json").write_text('{"observation.state": {}, "action": {}}', encoding="utf-8")
    (skill_dir / "splits.json").write_text('{"train": ["episode_000000"], "val": []}', encoding="utf-8")
    return SkillSpec.load(skill_dir), tokenizer_path


def _fields(count: int, channel_index: int = 0) -> dict[str, torch.Tensor]:
    payload = {
        "action": torch.arange(count * 2 * 3, dtype=torch.float32).reshape(count, 2, 3),
        "observation.state": torch.zeros(count, 4),
        "observation.language.tokens": torch.ones(count, 8, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(count, 8, dtype=torch.bool),
        "channel_index": torch.full((count,), channel_index, dtype=torch.long),
        "action_dim": torch.full((count,), 3, dtype=torch.long),
    }
    for image_field in IMAGE_FIELDS:
        payload[image_field] = torch.zeros(count, 3, 224, 224)
    return payload


def _write_cache(
    cache_root: Path,
    skill_spec: SkillSpec,
    tokenizer_path: Path,
    *,
    counts=(2, 3),
    image_storage_dtype: str = "float32",
) -> None:
    split = "train"
    split_dir = policy_cache_split_dir(cache_root, skill_spec.skill_id, split)
    split_dir.mkdir(parents=True)
    shards = []
    sample_start = 0
    for index, count in enumerate(counts):
        meta = write_policy_cache_shard(
            split_dir,
            shard_index=index,
            skill_id=skill_spec.skill_id,
            split=split,
            fields=_fields(count),
            sample_start=sample_start,
        )
        shards.append(meta)
        sample_start += count
    fingerprint, payload = expected_policy_cache_fingerprint(
        skill_spec,
        split=split,
        tokenizer_name_or_path=tokenizer_path,
        image_storage_dtype=image_storage_dtype,
    )
    write_policy_cache_manifest(
        split_dir,
        skill_spec=skill_spec,
        split=split,
        fingerprint=fingerprint,
        fingerprint_payload=payload,
        shards=shards,
        image_storage_dtype=image_storage_dtype,
    )


def test_policy_cache_map_dataset_loads_samples(tmp_path: Path) -> None:
    skill_spec, tokenizer_path = _write_skill(tmp_path)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, skill_spec, tokenizer_path)

    dataset = PolicyCacheMapDataset(
        cache_root=cache_root,
        skill_spec=skill_spec,
        split="train",
        tokenizer_name_or_path=tokenizer_path,
    )

    assert len(dataset) == 5
    sample = dataset[3]
    assert sample["action"].shape == (2, 3)
    assert sample["channel_index"].item() == 0


def test_policy_cache_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    skill_spec, tokenizer_path = _write_skill(tmp_path)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, skill_spec, tokenizer_path)
    tokenizer_path.write_text("changed", encoding="utf-8")

    with pytest.raises(PolicyCacheError):
        PolicyCacheMapDataset(
            cache_root=cache_root,
            skill_spec=skill_spec,
            split="train",
            tokenizer_name_or_path=tokenizer_path,
        )


def test_policy_cache_iterable_shards_partition_by_rank(tmp_path: Path) -> None:
    skill_spec, tokenizer_path = _write_skill(tmp_path)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, skill_spec, tokenizer_path, counts=(2, 2))

    rank0 = list(
        PolicyCacheIterableDataset(
            cache_root=cache_root,
            skill_spec=skill_spec,
            split="train",
            tokenizer_name_or_path=tokenizer_path,
            rank=0,
            world_size=2,
            shuffle=False,
        )
    )
    rank1 = list(
        PolicyCacheIterableDataset(
            cache_root=cache_root,
            skill_spec=skill_spec,
            split="train",
            tokenizer_name_or_path=tokenizer_path,
            rank=1,
            world_size=2,
            shuffle=False,
        )
    )

    assert len(rank0) == 2
    assert len(rank1) == 2
    assert rank0[0]["action"][0, 0].item() == 0
    assert rank1[0]["action"][0, 0].item() == 0


def test_require_policy_cache_missing_does_not_fallback_to_lerobot(tmp_path: Path) -> None:
    skill_spec, tokenizer_path = _write_skill(tmp_path)
    cfg = ExperimentConfig(
        channels=(ChannelConfig(channel_id="channel_a", skill_id=skill_spec.skill_id),),
        skill_root=tmp_path,
        train=TrainConfig(tokenizer_name_or_path=str(tokenizer_path)),
    )

    with pytest.raises(FileNotFoundError):
        build_dataset_for_channel(
            cfg,
            0,
            "train",
            cache_root=tmp_path / "missing_cache",
            require_policy_cache=True,
        )


def test_policy_cache_can_store_images_as_uint8() -> None:
    fields = _fields(2)
    for image_field in IMAGE_FIELDS:
        fields[image_field] = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).view(1, 3, 1, 1).repeat(2, 1, 224, 224)

    packed = select_policy_cache_fields(fields, image_storage_dtype="uint8")

    for image_field in IMAGE_FIELDS:
        assert packed[image_field].dtype == torch.uint8
        assert packed[image_field][0, :, 0, 0].tolist() == [0, 128, 255]
    assert packed["action"].dtype == torch.float32


def test_policy_cache_uint8_fingerprint_must_match_requested_dtype(tmp_path: Path) -> None:
    skill_spec, tokenizer_path = _write_skill(tmp_path)
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, skill_spec, tokenizer_path, image_storage_dtype="uint8")

    PolicyCacheMapDataset(
        cache_root=cache_root,
        skill_spec=skill_spec,
        split="train",
        tokenizer_name_or_path=tokenizer_path,
        image_storage_dtype="uint8",
    )
    with pytest.raises(PolicyCacheError):
        PolicyCacheMapDataset(
            cache_root=cache_root,
            skill_spec=skill_spec,
            split="train",
            tokenizer_name_or_path=tokenizer_path,
            image_storage_dtype="float32",
        )
