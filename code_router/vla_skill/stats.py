from __future__ import annotations

from typing import Any

import numpy as np
import torch

STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _as_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array for statistics, got shape={arr.shape}.")
    return arr


def compute_vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    arr = _as_float_array(values)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def validate_stats_payload(payload: dict[str, Any]) -> None:
    for feature_key in ("observation.state", "action"):
        if feature_key not in payload:
            raise ValueError(f"stats.json is missing `{feature_key}`.")
        feature_stats = payload[feature_key]
        for stat_key in STAT_KEYS:
            if stat_key not in feature_stats:
                raise ValueError(f"stats.json `{feature_key}` is missing `{stat_key}`.")


def stats_to_torch(payload: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    validate_stats_payload(payload)
    converted: dict[str, dict[str, torch.Tensor]] = {}
    for feature_key, feature_stats in payload.items():
        converted[feature_key] = {}
        for stat_key, value in feature_stats.items():
            if stat_key == "count":
                converted[feature_key][stat_key] = torch.tensor(value, dtype=torch.int64)
            else:
                converted[feature_key][stat_key] = torch.tensor(value, dtype=torch.float32)
    return converted
