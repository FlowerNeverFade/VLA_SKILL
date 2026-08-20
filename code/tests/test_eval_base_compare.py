from __future__ import annotations

import pytest

from vla_skill.training import compare_eval_results


def test_compare_eval_results_reports_improvement_over_base() -> None:
    adapter_summary = {
        "skill_id": "pick_mug",
        "model_type": "adapter",
        "val_loss": 0.12,
        "action_mse": 0.08,
    }
    base_summary = {
        "skill_id": "pick_mug",
        "model_type": "base",
        "val_loss": 0.20,
        "action_mse": 0.10,
    }

    comparison = compare_eval_results(adapter_summary, base_summary)

    assert comparison["skill_id"] == "pick_mug"
    assert comparison["improvement_over_base"]["val_loss"] == pytest.approx(0.08)
    assert comparison["improvement_over_base"]["action_mse"] == pytest.approx(0.02)
    assert comparison["better_than_base"]["val_loss"] is True
    assert comparison["better_than_base"]["action_mse"] is True
