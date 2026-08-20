from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from vla_skill.dataset import _load_lerobot_task_name
from vla_skill.schema import SkillSpec
from vla_skill_router.robocasa import (
    prepare_active_robocasa_skills,
    register_active_tasks_as_skills,
    scan_active_tasks,
)


def _make_fake_robocasa(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 20,
        "features": {
            "observation.state": {"shape": [2]},
            "action": {"shape": [3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    tasks = pd.DataFrame({"task_index": [0, 1, 2]}, index=["Open drawer.", "Unused task.", "Close door."])
    tasks.to_parquet(root / "meta" / "tasks.parquet")
    rows = []
    index = 0
    for task_index, episode_index in [(0, 0), (0, 1), (2, 2), (2, 3)]:
        for frame in range(3):
            rows.append(
                {
                    "index": index,
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "frame_index": frame,
                    "observation.state": [float(task_index), float(frame)],
                    "action": [float(task_index), float(frame), 1.0],
                }
            )
            index += 1
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


def test_scan_active_tasks_skips_zero_sample_tasks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "robocasa"
    _make_fake_robocasa(dataset_dir)

    manifest = scan_active_tasks(dataset_dir)

    assert manifest["active_task_count"] == 2
    assert manifest["skipped_task_count"] == 1
    assert [item["task_index"] for item in manifest["active_tasks"]] == [0, 2]
    assert manifest["skipped_tasks"][0]["task_index"] == 1


def test_register_and_global_prepare_active_robocasa_skills(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "robocasa"
    skill_root = tmp_path / "skills"
    manifest_path = tmp_path / "robocasa_active_tasks.json"
    _make_fake_robocasa(dataset_dir)

    register_summary = register_active_tasks_as_skills(
        dataset_dir=dataset_dir,
        repo_id="fake/robocasa",
        skill_root=skill_root,
        manifest_path=manifest_path,
        overwrite=True,
    )
    prepare_summary = prepare_active_robocasa_skills(
        dataset_dir=dataset_dir,
        skill_root=skill_root,
        manifest_path=manifest_path,
        seed=123,
        chunk_size=2,
    )

    assert register_summary["registered_skill_count"] == 2
    assert prepare_summary["prepared_skill_count"] == 2
    first_skill = Path(register_summary["registered_skills"][0]["skill_dir"])
    assert (first_skill / "skill.yaml").is_file()
    assert (first_skill / "splits.json").is_file()
    assert (first_skill / "stats.json").is_file()
    assert (first_skill / "lerobot_records.json").is_file()
    stats = json.loads((first_skill / "stats.json").read_text(encoding="utf-8"))
    assert stats["observation.state"]["count"] == [3]


def test_lerobot_task_name_reads_tasks_parquet(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "robocasa"
    _make_fake_robocasa(dataset_dir)
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    payload = {
        "skill_id": "fake_skill",
        "display_name": "fake",
        "state_dim": 2,
        "action_dim": 3,
        "source": {
            "type": "lerobot",
            "dataset_dir": str(dataset_dir),
            "repo_id": "fake/robocasa",
            "task_index": 2,
            "camera_mapping": {
                "base_0_rgb": "observation.images.robot0_agentview_left",
                "left_wrist_0_rgb": "observation.images.robot0_eye_in_hand",
                "right_wrist_0_rgb": "observation.images.robot0_agentview_right",
            },
        },
    }
    (skill_dir / "skill.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert _load_lerobot_task_name(SkillSpec.load(skill_dir)) == "Close door."
