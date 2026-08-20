#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from train_router_lora import _prepare_batch
from vla_skill.constants import IMAGE_FIELDS
from vla_skill.dataset import load_skill_spec, load_stats
from vla_skill.io_utils import ensure_dir
from vla_skill.pi05 import build_processors
from vla_skill_router.config import load_experiment_config
from vla_skill_router.policy_cache import (
    DEFAULT_IMAGE_STORAGE_DTYPE,
    DEFAULT_POLICY_CACHE_ROOT,
    IMAGE_STORAGE_DTYPES,
    PolicyCacheError,
    expected_policy_cache_fingerprint,
    load_existing_shard_metas,
    load_policy_cache_manifest,
    policy_cache_split_dir,
    remove_policy_cache_split,
    select_policy_cache_fields,
    write_policy_cache_manifest,
    write_policy_cache_shard,
)
from vla_skill_router.real_runtime import build_dataset_for_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy-ready tensor cache for router LoRA training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_POLICY_CACHE_ROOT)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--channel-ids", nargs="*")
    parser.add_argument("--max-channels", type=int)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--parallel-channels", type=int, default=1)
    parser.add_argument(
        "--image-storage-dtype",
        choices=IMAGE_STORAGE_DTYPES,
        default=DEFAULT_IMAGE_STORAGE_DTYPE,
        help="On-disk dtype for cached image tensors. Use uint8 to fit full RoboCasa/SO101 cache in about 1.2 TB.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _build_channel_split_cache(
    *,
    config_path: Path,
    cache_root: Path,
    channel_index: int,
    split: str,
    batch_size: int,
    shard_size: int,
    num_workers: int,
    max_samples: int | None,
    image_storage_dtype: str,
    overwrite: bool,
) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    channel = cfg.channels[channel_index]
    skill_spec = load_skill_spec(cfg.skill_root, channel.skill_id)
    if skill_spec.source.type != "lerobot":
        return {
            "skill_id": channel.skill_id,
            "channel_id": channel.channel_id,
            "split": split,
            "status": "skipped_non_lerobot",
        }

    split_dir = policy_cache_split_dir(cache_root, skill_spec.skill_id, split)
    if overwrite:
        remove_policy_cache_split(cache_root, skill_spec.skill_id, split)
    ensure_dir(split_dir)

    expected_fingerprint, fingerprint_payload = expected_policy_cache_fingerprint(
        skill_spec,
        split=split,
        tokenizer_name_or_path=cfg.train.tokenizer_name_or_path,
        image_storage_dtype=image_storage_dtype,
    )
    try:
        manifest = load_policy_cache_manifest(
            cache_root,
            skill_spec,
            split=split,
            tokenizer_name_or_path=cfg.train.tokenizer_name_or_path,
            image_storage_dtype=image_storage_dtype,
        )
        return {
            "skill_id": channel.skill_id,
            "channel_id": channel.channel_id,
            "split": split,
            "status": "already_complete",
            "sample_count": int(manifest["sample_count"]),
            "shard_count": int(manifest["shard_count"]),
        }
    except FileNotFoundError:
        pass
    except PolicyCacheError:
        raise

    raw_dataset = build_dataset_for_channel(cfg, channel_index, split=split)
    total_samples = len(raw_dataset)
    if max_samples is not None:
        total_samples = min(total_samples, int(max_samples))
    if total_samples <= 0:
        manifest = write_policy_cache_manifest(
            split_dir,
            skill_spec=skill_spec,
            split=split,
            fingerprint=expected_fingerprint,
            fingerprint_payload=fingerprint_payload,
            shards=[],
            image_storage_dtype=image_storage_dtype,
        )
        return {
            "skill_id": channel.skill_id,
            "channel_id": channel.channel_id,
            "split": split,
            "status": "skipped_zero_samples",
            "sample_count": int(manifest["sample_count"]),
            "shard_count": int(manifest["shard_count"]),
        }

    existing_shards = load_existing_shard_metas(split_dir)
    for shard_meta in existing_shards:
        schema = shard_meta.get("field_schema") or {}
        for image_field in IMAGE_FIELDS:
            dtype = (schema.get(image_field) or {}).get("dtype")
            if dtype is not None and dtype != image_storage_dtype:
                raise ValueError(
                    f"Existing partial cache for skill={channel.skill_id} split={split} stores "
                    f"{image_field} as {dtype}, not requested {image_storage_dtype}. "
                    "Use --overwrite or a new --cache-root."
                )
    completed_samples = sum(int(item["sample_count"]) for item in existing_shards)
    if completed_samples > total_samples:
        raise ValueError(
            f"Existing policy cache for skill={channel.skill_id} split={split} has "
            f"{completed_samples} samples, expected at most {total_samples}."
        )
    shard_index = len(existing_shards)
    shards = list(existing_shards)
    if completed_samples < total_samples:
        subset = Subset(raw_dataset, range(completed_samples, total_samples))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        preprocessor = build_processors(
            skill_spec,
            load_stats(skill_spec),
            device="cpu",
            tokenizer_name_or_path=cfg.train.tokenizer_name_or_path,
        )[0]
        pending_batches: list[dict[str, torch.Tensor]] = []
        pending_samples = 0
        sample_start = completed_samples
        for batch_index, raw_batch in enumerate(loader, start=1):
            proc_batch = _prepare_batch(raw_batch, preprocessor)
            fields = select_policy_cache_fields(proc_batch, image_storage_dtype=image_storage_dtype)
            pending_batches.append(fields)
            pending_samples += int(next(iter(fields.values())).shape[0])
            if pending_samples >= shard_size:
                shard_fields = {
                    key: torch.cat([item[key] for item in pending_batches], dim=0).contiguous()
                    for key in sorted(pending_batches[0])
                }
                meta = write_policy_cache_shard(
                    split_dir,
                    shard_index=shard_index,
                    skill_id=skill_spec.skill_id,
                    split=split,
                    fields=shard_fields,
                    sample_start=sample_start,
                )
                shards.append(meta)
                print(
                    json.dumps(
                        {
                            "kind": "cache_shard",
                            "skill_id": skill_spec.skill_id,
                            "split": split,
                            "shard_index": shard_index,
                            "sample_start": sample_start,
                            "sample_count": meta["sample_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                sample_start += int(meta["sample_count"])
                shard_index += 1
                pending_batches = []
                pending_samples = 0
            if batch_index == 1 or batch_index % 200 == 0:
                print(
                    json.dumps(
                        {
                            "kind": "cache_progress",
                            "skill_id": skill_spec.skill_id,
                            "split": split,
                            "processed_samples": completed_samples
                            + sum(int(item["sample_count"]) for item in shards[len(existing_shards) :])
                            + pending_samples,
                            "total_samples": total_samples,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if pending_batches:
            shard_fields = {
                key: torch.cat([item[key] for item in pending_batches], dim=0).contiguous()
                for key in sorted(pending_batches[0])
            }
            meta = write_policy_cache_shard(
                split_dir,
                shard_index=shard_index,
                skill_id=skill_spec.skill_id,
                split=split,
                fields=shard_fields,
                sample_start=sample_start,
            )
            shards.append(meta)

    manifest = write_policy_cache_manifest(
        split_dir,
        skill_spec=skill_spec,
        split=split,
        fingerprint=expected_fingerprint,
        fingerprint_payload=fingerprint_payload,
        shards=shards,
        image_storage_dtype=image_storage_dtype,
    )
    return {
        "skill_id": channel.skill_id,
        "channel_id": channel.channel_id,
        "split": split,
        "status": "complete",
        "sample_count": int(manifest["sample_count"]),
        "shard_count": int(manifest["shard_count"]),
    }


def _task_from_tuple(payload: tuple[Any, ...]) -> dict[str, Any]:
    return _build_channel_split_cache(
        config_path=payload[0],
        cache_root=payload[1],
        channel_index=payload[2],
        split=payload[3],
        batch_size=payload[4],
        shard_size=payload[5],
        num_workers=payload[6],
        max_samples=payload[7],
        image_storage_dtype=payload[8],
        overwrite=payload[9],
    )


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    selected_indices = list(range(len(cfg.channels)))
    if args.channel_ids:
        wanted = set(args.channel_ids)
        selected_indices = [index for index, channel in enumerate(cfg.channels) if channel.channel_id in wanted]
        missing = sorted(wanted - {cfg.channels[index].channel_id for index in selected_indices})
        if missing:
            raise SystemExit(f"Unknown channel ids: {missing}")
    if args.max_channels is not None:
        selected_indices = selected_indices[: args.max_channels]
    tasks = [
        (
            args.config,
            args.cache_root,
            channel_index,
            split,
            args.batch_size,
            args.shard_size,
            args.num_workers,
            args.max_samples_per_split,
            args.image_storage_dtype,
            args.overwrite,
        )
        for channel_index in selected_indices
        for split in args.splits
    ]
    ensure_dir(args.cache_root)
    print(
        json.dumps(
            {
                "kind": "cache_build_start",
                "cache_root": str(args.cache_root),
                "tasks": len(tasks),
                "channels": len(selected_indices),
                "splits": args.splits,
                "batch_size": args.batch_size,
                "shard_size": args.shard_size,
                "num_workers": args.num_workers,
                "parallel_channels": args.parallel_channels,
                "image_storage_dtype": args.image_storage_dtype,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results: list[dict[str, Any]] = []
    if args.parallel_channels <= 1:
        for task in tasks:
            result = _task_from_tuple(task)
            results.append(result)
            print(json.dumps({"kind": "cache_task_done", **result}, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel_channels) as pool:
            futures = [pool.submit(_task_from_tuple, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"kind": "cache_task_done", **result}, ensure_ascii=False), flush=True)
    print(json.dumps({"kind": "cache_build_done", "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
