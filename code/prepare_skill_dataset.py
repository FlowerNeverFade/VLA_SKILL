#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_skill.constants import DEFAULT_SEED, DEFAULT_SKILL_ROOT
from vla_skill.dataset import discover_skill_dirs, prepare_skill_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate skill directories and generate splits.json / stats.json.")
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--skill-id", type=str, help="Prepare one skill only.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skill_id:
        targets = [args.skill_root / args.skill_id]
    else:
        targets = discover_skill_dirs(args.skill_root)
    if not targets:
        raise SystemExit(f"No skill directories found under {args.skill_root}.")

    summaries = [prepare_skill_directory(path, seed=args.seed) for path in targets]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
