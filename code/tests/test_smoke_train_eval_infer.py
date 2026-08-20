from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH
from vla_skill.dataset import SkillWindowDataset, load_skill_spec, prepare_skill_directory
from vla_skill.registry import SkillAdapterRegistry
from vla_skill.training import TrainRunConfig, evaluate_base_policy, evaluate_saved_adapter, train_skill_lora

from .helpers import make_toy_skill


@pytest.mark.smoke
@pytest.mark.skipif(
    os.environ.get("RUN_PI05_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="Set RUN_PI05_SMOKE=1 on a CUDA machine to run the PI05 smoke test.",
)
def test_smoke_train_eval_and_infer_across_groups(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    output_root = tmp_path / "outputs"
    make_toy_skill(
        skill_root,
        skill_id="pick_mug",
        display_name="Pick Mug",
        task="pick the mug and place it on the tray",
        aliases=["pick mug"],
        keywords=["mug", "tray"],
        regexes=["pick .* mug"],
        num_episodes=10,
        num_frames=4,
        chunk_size=2,
        seed=7,
    )
    prepare_skill_directory(skill_root / "pick_mug")

    for group in ("A", "B", "C"):
        summary = train_skill_lora(
            TrainRunConfig(
                skill_id="pick_mug",
                group=group,
                skill_root=skill_root,
                output_root=output_root,
                base_model_path=DEFAULT_BASE_MODEL_PATH,
                run_name=f"smoke_{group.lower()}",
                steps=1,
                batch_size=1,
                eval_every=1,
                log_every=1,
                num_workers=0,
                device="cuda",
                dtype="bfloat16",
                overwrite=True,
            )
        )
        assert Path(summary["best_adapter_dir"]).is_dir()
        assert Path(summary["last_adapter_dir"]).is_dir()

        eval_summary = evaluate_saved_adapter(
            skill_id="pick_mug",
            skill_root=skill_root,
            output_root=output_root,
            base_model_path=DEFAULT_BASE_MODEL_PATH,
            group=group,
            run_name=f"smoke_{group.lower()}",
            batch_size=1,
            num_workers=0,
            device="cuda",
            dtype="bfloat16",
        )
        assert "val_loss" in eval_summary
        assert "action_mse" in eval_summary

    base_eval_summary = evaluate_base_policy(
        skill_id="pick_mug",
        skill_root=skill_root,
        output_root=output_root,
        base_model_path=DEFAULT_BASE_MODEL_PATH,
        batch_size=1,
        num_workers=0,
        device="cuda",
        dtype="bfloat16",
    )
    assert "val_loss" in base_eval_summary
    assert "action_mse" in base_eval_summary

    registry = SkillAdapterRegistry(
        skill_root=skill_root,
        output_root=output_root,
        base_model_path=DEFAULT_BASE_MODEL_PATH,
        device="cuda",
        dtype="bfloat16",
    )
    dataset = SkillWindowDataset(load_skill_spec(skill_root, "pick_mug"), split="val")
    sample = dataset[0]
    raw_batch = {key: value for key, value in sample.items() if key != "action"}
    skill_spec, model, preprocessor, postprocessor, _ = registry.activate(task=sample["task"])
    pred = model.predict_action_chunk(preprocessor(raw_batch))
    pred = pred[:, :, : skill_spec.action_dim]
    pred = postprocessor(pred)
    assert tuple(pred.shape) == tuple(sample["action"].shape)
