from __future__ import annotations

from pathlib import Path

from vla_skill.training import (
    OPTIMIZER_STATE_FILENAME,
    SCHEDULER_STATE_FILENAME,
    TRAINER_STATE_FILENAME,
    TrainRunConfig,
    is_resumable_checkpoint_dir,
    is_resumable_run_dir,
)


def test_train_run_config_uses_resume_dir_name() -> None:
    resume_dir = Path("/tmp/pi05_runs/pick_cube/A/run_123")
    cfg = TrainRunConfig(skill_id="pick_cube", group="A", resume_from_run_dir=resume_dir)

    assert cfg.resolve_run_name() == "run_123"
    assert cfg.run_dir == resume_dir


def test_is_resumable_run_dir_requires_training_state_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    checkpoint_dir = run_dir / "last"
    checkpoint_dir.mkdir(parents=True)

    assert not is_resumable_checkpoint_dir(checkpoint_dir)
    assert not is_resumable_run_dir(run_dir)

    for name in (
        TRAINER_STATE_FILENAME,
        OPTIMIZER_STATE_FILENAME,
        SCHEDULER_STATE_FILENAME,
        "adapter_config.json",
    ):
        (checkpoint_dir / name).write_text("{}", encoding="utf-8")

    assert is_resumable_checkpoint_dir(checkpoint_dir)
    assert is_resumable_run_dir(run_dir)
