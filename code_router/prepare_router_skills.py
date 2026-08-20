#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_skill.constants import DEFAULT_SEED, DEFAULT_SKILL_ROOT
from vla_skill_router.robocasa import DEFAULT_ROBOCASA_DATASET_DIR, prepare_active_robocasa_skills


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare all active RoboCasa router skills with one global scan.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_ROBOCASA_DATASET_DIR)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_SKILL_ROOT / "robocasa_active_tasks.json",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_active_robocasa_skills(
        dataset_dir=args.dataset_dir,
        skill_root=args.skill_root,
        manifest_path=args.manifest_path,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
