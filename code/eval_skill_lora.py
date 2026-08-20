#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_skill.constants import DEFAULT_BASE_MODEL_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_SKILL_ROOT
from vla_skill.io_utils import write_json
from vla_skill.training import compare_eval_results, evaluate_base_policy, evaluate_saved_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PI05 skill adapter on the val split.")
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--group", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def _resolve_compare_output_path(args: argparse.Namespace, adapter_summary: dict[str, object]) -> Path:
    adapter_dir = Path(str(adapter_summary["adapter_dir"]))
    if adapter_dir.name == "best":
        return adapter_dir.parent / "compare_with_base.json"
    return adapter_dir / "compare_with_base.json"


def main() -> None:
    args = parse_args()
    if args.base_only and args.compare_base:
        raise SystemExit("--base-only and --compare-base cannot be used together.")

    if args.base_only:
        summary = evaluate_base_policy(
            skill_id=args.skill_id,
            skill_root=args.skill_root,
            output_root=args.output_root,
            base_model_path=args.base_model_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            dtype=args.dtype,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            write_result=not args.no_write,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    adapter_summary = evaluate_saved_adapter(
        skill_id=args.skill_id,
        skill_root=args.skill_root,
        output_root=args.output_root,
        base_model_path=args.base_model_path,
        adapter_dir=args.adapter_dir,
        group=args.group,
        run_name=args.run_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        write_result=not args.no_write,
    )
    if not args.compare_base:
        print(json.dumps(adapter_summary, indent=2, ensure_ascii=False))
        return

    base_summary = evaluate_base_policy(
        skill_id=args.skill_id,
        skill_root=args.skill_root,
        output_root=args.output_root,
        base_model_path=args.base_model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        write_result=not args.no_write,
    )
    comparison = compare_eval_results(adapter_summary, base_summary)
    if not args.no_write:
        compare_path = _resolve_compare_output_path(args, adapter_summary)
        write_json(compare_path, comparison)
        comparison["comparison_path"] = str(compare_path)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
