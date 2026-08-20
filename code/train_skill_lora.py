#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_skill.constants import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SKILL_ROOT,
)
from vla_skill.training import TrainRunConfig, train_skill_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one PI05 LoRA adapter for one skill and one target group.")
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--group", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--run-name", type=str)
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
    parser.add_argument("--resume-from-run-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainRunConfig(
        skill_id=args.skill_id,
        group=args.group,
        skill_root=args.skill_root,
        output_root=args.output_root,
        base_model_path=args.base_model_path,
        run_name=args.run_name,
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
        resume_from_run_dir=args.resume_from_run_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        overwrite=args.overwrite,
    )
    summary = train_skill_lora(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
