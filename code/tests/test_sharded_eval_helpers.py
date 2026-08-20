from __future__ import annotations

from pathlib import Path

import pytest

from vla_skill.training import build_batch_shard_subset, merge_base_eval_shards, merge_eval_totals


def test_build_batch_shard_subset_preserves_global_batch_boundaries() -> None:
    dataset = list(range(10))

    shard0, meta0 = build_batch_shard_subset(dataset, batch_size=4, shard_id=0, num_shards=2)
    shard1, meta1 = build_batch_shard_subset(dataset, batch_size=4, shard_id=1, num_shards=2)

    assert meta0["total_batches"] == 3
    assert meta0["window_start"] == 0
    assert meta0["window_end"] == 4
    assert meta0["shard_num_batches"] == 1
    assert list(shard0) == [0, 1, 2, 3]

    assert meta1["window_start"] == 4
    assert meta1["window_end"] == 10
    assert meta1["shard_num_batches"] == 2
    assert list(shard1) == [4, 5, 6, 7, 8, 9]


def test_build_batch_shard_subset_rejects_too_many_shards() -> None:
    dataset = list(range(10))

    with pytest.raises(ValueError):
        build_batch_shard_subset(dataset, batch_size=4, shard_id=0, num_shards=4)


def test_merge_eval_totals_combines_partial_sums() -> None:
    merged = merge_eval_totals(
        [
            {"loss_sum": 2.0, "action_mse_sum": 1.0, "num_val_batches": 2.0},
            {"loss_sum": 4.5, "action_mse_sum": 2.5, "num_val_batches": 3.0},
        ]
    )

    assert merged["num_val_batches"] == pytest.approx(5.0)
    assert merged["val_loss"] == pytest.approx(1.3)
    assert merged["action_mse"] == pytest.approx(0.7)


def test_merge_base_eval_shards_writes_summary(tmp_path: Path) -> None:
    shard_payloads = [
        {
            "skill_id": "pick_cube",
            "model_type": "base",
            "base_model_path": "/tmp/base",
            "shard_id": 0,
            "num_shards": 2,
            "device": "cuda:0",
            "shard_num_windows": 8,
            "shard_num_batches": 2,
            "loss_sum": 0.8,
            "action_mse_sum": 0.4,
            "num_val_batches": 2.0,
            "val_loss": 0.4,
            "action_mse": 0.2,
            "shard_output_path": "/tmp/shard0.json",
        },
        {
            "skill_id": "pick_cube",
            "model_type": "base",
            "base_model_path": "/tmp/base",
            "shard_id": 1,
            "num_shards": 2,
            "device": "cuda:1",
            "shard_num_windows": 6,
            "shard_num_batches": 2,
            "loss_sum": 1.2,
            "action_mse_sum": 0.8,
            "num_val_batches": 2.0,
            "val_loss": 0.6,
            "action_mse": 0.4,
            "shard_output_path": "/tmp/shard1.json",
        },
    ]

    summary = merge_base_eval_shards(shard_payloads, output_root=tmp_path, write_result=True)

    assert summary["evaluation_mode"] == "sharded"
    assert summary["num_shards"] == 2
    assert summary["val_loss"] == pytest.approx(0.5)
    assert summary["action_mse"] == pytest.approx(0.3)
    assert (tmp_path / "pick_cube" / "base_eval_summary.json").is_file()
