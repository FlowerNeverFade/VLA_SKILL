#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vla_skill.constants import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SKILL_ROOT,
    LORA_GROUPS,
)
from vla_skill.dataset import discover_skill_dirs
from vla_skill.io_utils import timestamp_run_name, write_json
from vla_skill.training import TrainRunConfig, is_resumable_run_dir, train_skill_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA groups across one or more skills.")
    parser.add_argument("--skill-ids", nargs="*", help="Default: all skills under skill-root.")
    parser.add_argument("--groups", nargs="*", default=list(LORA_GROUPS), choices=list(LORA_GROUPS))
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--run-name-prefix", type=str)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every-steps", type=int, default=2000)
    parser.add_argument("--eval-subset-windows", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-steps", type=int, default=30000)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--full-eval-log-every-batches", type=int, default=500)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-full-eval-at-end", action="store_true")
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_skill_comparison(output_root: Path, skill_id: str, rows: list[dict]) -> None:
    skill_dir = output_root / skill_id
    write_json(skill_dir / "comparison.json", rows)
    with (skill_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["skill_id", "group", "run_name", "best_val_loss", "best_action_mse", "best_step", "run_dir"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "skill_id": row["skill_id"],
                    "group": row["group"],
                    "run_name": row["run_name"],
                    "best_val_loss": row["best_val_loss"],
                    "best_action_mse": row["best_action_mse"],
                    "best_step": row["best_step"],
                    "run_dir": row["run_dir"],
                }
            )


def main() -> None:
    args = parse_args()
    skill_ids = args.skill_ids or [path.name for path in discover_skill_dirs(args.skill_root)]
    if not skill_ids:
        raise SystemExit(f"No skills found under {args.skill_root}.")

    global_rows = []
    for skill_id in skill_ids:
        skill_rows = []
        for group in args.groups:
            print(f"[grid] launching skill={skill_id} group={group}", flush=True)
            run_name = None
            if args.run_name_prefix:
                run_name = f"{args.run_name_prefix}_{group.lower()}"
            resume_from_run_dir = None
            if args.resume_if_exists and run_name is not None:
                candidate_run_dir = args.output_root / skill_id / group / run_name
                if is_resumable_run_dir(candidate_run_dir):
                    resume_from_run_dir = candidate_run_dir
                    print(f"[grid] resuming skill={skill_id} group={group} run_dir={candidate_run_dir}", flush=True)
            cfg = TrainRunConfig(
                skill_id=skill_id,
                group=group,
                skill_root=args.skill_root,
                output_root=args.output_root,
                base_model_path=args.base_model_path,
                run_name=run_name,
                steps=args.steps,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                save_every_steps=args.save_every_steps,
                eval_subset_windows=args.eval_subset_windows,
                log_every=args.log_every,
                num_workers=args.num_workers,
                seed=args.seed,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                decay_steps=args.decay_steps,
                grad_clip_norm=args.grad_clip_norm,
                device=args.device,
                dtype=args.dtype,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                full_eval_at_end=not args.no_full_eval_at_end,
                full_eval_log_every_batches=args.full_eval_log_every_batches,
                resume_from_run_dir=resume_from_run_dir,
                gradient_checkpointing=args.gradient_checkpointing,
                compile_model=args.compile_model,
                overwrite=args.overwrite if resume_from_run_dir is None else False,
            )
            summary = train_skill_lora(cfg)
            skill_rows.append(summary)
            global_rows.append(summary)
            print(
                f"[grid] completed skill={skill_id} group={group} "
                f"best_val_loss={summary['best_val_loss']:.6f} action_mse={summary['best_action_mse']:.6f}",
                flush=True,
            )
        _write_skill_comparison(args.output_root, skill_id, skill_rows)

    global_path = args.output_root / f"grid_summary_{timestamp_run_name('pi05_skill')}.json"
    write_json(global_path, global_rows)
    print(json.dumps({"summary_path": str(global_path), "runs": global_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
