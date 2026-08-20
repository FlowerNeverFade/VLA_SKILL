from __future__ import annotations

from torch.utils.data import Dataset

from vla_skill.training import build_eval_monitor_dataset


class RangeDataset(Dataset[int]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def test_build_eval_monitor_dataset_is_deterministic() -> None:
    dataset = RangeDataset(20)

    subset_a = build_eval_monitor_dataset(dataset, subset_windows=5, seed=123)
    subset_b = build_eval_monitor_dataset(dataset, subset_windows=5, seed=123)

    assert list(subset_a.indices) == list(subset_b.indices)
    assert len(subset_a) == 5


def test_build_eval_monitor_dataset_returns_full_dataset_when_subset_is_not_smaller() -> None:
    dataset = RangeDataset(8)

    same_dataset = build_eval_monitor_dataset(dataset, subset_windows=8, seed=7)
    oversized_dataset = build_eval_monitor_dataset(dataset, subset_windows=99, seed=7)
    disabled_subset = build_eval_monitor_dataset(dataset, subset_windows=0, seed=7)

    assert same_dataset is dataset
    assert oversized_dataset is dataset
    assert disabled_subset is dataset
