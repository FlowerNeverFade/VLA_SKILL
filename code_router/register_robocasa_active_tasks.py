#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_skill.constants import DEFAULT_SKILL_ROOT
from vla_skill_router.robocasa import (
    DEFAULT_ROBOCASA_DATASET_DIR,
    DEFAULT_ROBOCASA_REPO_ID,
    DEFAULT_ROBOCASA_SKILL_PREFIX,
    register_active_tasks_as_skills,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register RoboCasa active task_indices as router skills.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_ROBOCASA_DATASET_DIR)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_ROBOCASA_REPO_ID)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_SKILL_ROOT / "robocasa_active_tasks.json",
    )
    parser.add_argument("--skill-prefix", type=str, default=DEFAULT_ROBOCASA_SKILL_PREFIX)
    parser.add_argument("--video-backend", choices=("auto", "torchcodec", "pyav"), default="pyav")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = register_active_tasks_as_skills(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        skill_root=args.skill_root,
        manifest_path=args.manifest_path,
        overwrite=args.overwrite,
        video_backend=args.video_backend,
        skill_prefix=args.skill_prefix,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
