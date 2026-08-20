#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from vla_skill.constants import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_SKILL_ROOT,
    LORA_GROUPS,
)
from vla_skill.dataset import discover_skill_dirs
from vla_skill.io_utils import timestamp_run_name, write_json
from vla_skill.training import is_resumable_run_dir

OOM_PATTERN = re.compile(r"out of memory|cuda out of memory|oom", re.IGNORECASE)


@dataclass
class RunningJob:
    skill_id: str
    group: str
    gpu: str
    batch_size: int
    run_name: str
    log_path: Path
    process: subprocess.Popen
    log_handle: TextIO
    retry_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA groups in parallel across one or more GPUs.")
    parser.add_argument("--skill-ids", nargs="*", help="Default: all skills under skill-root.")
    parser.add_argument("--groups", nargs="*", default=list(LORA_GROUPS), choices=list(LORA_GROUPS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="Physical GPU ids, e.g. `0 1`.")
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_OUTPUT_ROOT.parent / "logs")
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
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--full-eval-log-every-batches", type=int, default=500)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-full-eval-at-end", action="store_true")
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="How often to poll running jobs for completion.",
    )
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


def _build_command(
    args: argparse.Namespace,
    skill_id: str,
    group: str,
    batch_size: int,
    run_name: str,
    *,
    resume_from_run_dir: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "train_skill_lora.py",
        "--skill-id",
        skill_id,
        "--group",
        group,
        "--skill-root",
        str(args.skill_root),
        "--output-root",
        str(args.output_root),
        "--base-model-path",
        str(args.base_model_path),
        "--run-name",
        run_name,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(batch_size),
        "--eval-every",
        str(args.eval_every),
        "--save-every-steps",
        str(args.save_every_steps),
        "--eval-subset-windows",
        str(args.eval_subset_windows),
        "--log-every",
        str(args.log_every),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--decay-steps",
        str(args.decay_steps),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--device",
        "cuda:0",
        "--dtype",
        args.dtype,
        "--full-eval-log-every-batches",
        str(args.full_eval_log_every_batches),
    ]
    if args.tokenizer_name_or_path:
        cmd.extend(["--tokenizer-name-or-path", args.tokenizer_name_or_path])
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.compile_model:
        cmd.append("--compile-model")
    if args.no_full_eval_at_end:
        cmd.append("--no-full-eval-at-end")
    if resume_from_run_dir is not None:
        cmd.extend(["--resume-from-run-dir", str(resume_from_run_dir)])
    if args.overwrite and resume_from_run_dir is None:
        cmd.append("--overwrite")
    return cmd


def _log_path(log_dir: Path, skill_id: str, group: str, run_name: str, gpu: str, batch_size: int) -> Path:
    safe_run_name = run_name.replace("/", "_")
    return log_dir / f"{skill_id}_{group}_{safe_run_name}_gpu{gpu}_bs{batch_size}.log"


def _launch_job(
    args: argparse.Namespace,
    *,
    skill_id: str,
    group: str,
    gpu: str,
    batch_size: int,
    run_name: str,
    retry_count: int = 0,
) -> RunningJob:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(args.log_dir, skill_id, group, run_name, gpu, batch_size)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    candidate_run_dir = args.output_root / skill_id / group / run_name
    resume_from_run_dir = candidate_run_dir if args.resume_if_exists and is_resumable_run_dir(candidate_run_dir) else None
    process = subprocess.Popen(
        _build_command(
            args,
            skill_id,
            group,
            batch_size,
            run_name,
            resume_from_run_dir=resume_from_run_dir,
        ),
        cwd=str(Path(__file__).resolve().parent),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(
        f"[parallel] launched skill={skill_id} group={group} gpu={gpu} batch_size={batch_size} "
        f"run_name={run_name} pid={process.pid} log={log_path} "
        f"resume_from={resume_from_run_dir}",
        flush=True,
    )
    return RunningJob(
        skill_id=skill_id,
        group=group,
        gpu=str(gpu),
        batch_size=batch_size,
        run_name=run_name,
        log_path=log_path,
        process=process,
        log_handle=log_handle,
        retry_count=retry_count,
    )


def _read_log_tail(path: Path, max_chars: int = 6000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]


def _load_run_summary(output_root: Path, skill_id: str, group: str, run_name: str) -> dict:
    summary_path = output_root / skill_id / group / run_name / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json for {skill_id}/{group}/{run_name}: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _maybe_retry_oom(args: argparse.Namespace, job: RunningJob) -> RunningJob | None:
    if job.batch_size <= 4:
        return None
    tail = _read_log_tail(job.log_path)
    if not OOM_PATTERN.search(tail):
        return None
    retry_run_name = f"{job.run_name}_retrybs4"
    print(
        f"[parallel] OOM detected skill={job.skill_id} group={job.group} gpu={job.gpu}; "
        f"retrying with batch_size=4 run_name={retry_run_name}",
        flush=True,
    )
    return _launch_job(
        args,
        skill_id=job.skill_id,
        group=job.group,
        gpu=job.gpu,
        batch_size=4,
        run_name=retry_run_name,
        retry_count=job.retry_count + 1,
    )


def _schedule_skill(args: argparse.Namespace, skill_id: str, groups: list[str]) -> list[dict]:
    queue: deque[str] = deque(groups)
    available_gpus: deque[str] = deque(str(gpu) for gpu in args.gpus)
    running: dict[int, RunningJob] = {}
    skill_rows: list[dict] = []
    default_prefix = args.run_name_prefix or timestamp_run_name(f"{skill_id}_parallel")

    while queue or running:
        while queue and available_gpus:
            group = queue.popleft()
            gpu = available_gpus.popleft()
            run_name = f"{default_prefix}_{group.lower()}"
            job = _launch_job(
                args,
                skill_id=skill_id,
                group=group,
                gpu=gpu,
                batch_size=args.batch_size,
                run_name=run_name,
            )
            running[job.process.pid] = job

        if not running:
            break

        time.sleep(max(0.1, args.poll_seconds))
        completed_pids = [pid for pid, job in running.items() if job.process.poll() is not None]
        for pid in completed_pids:
            job = running.pop(pid)
            return_code = job.process.returncode
            job.log_handle.close()
            if return_code == 0:
                summary = _load_run_summary(args.output_root, job.skill_id, job.group, job.run_name)
                skill_rows.append(summary)
                available_gpus.append(job.gpu)
                print(
                    f"[parallel] completed skill={job.skill_id} group={job.group} gpu={job.gpu} "
                    f"best_val_loss={summary['best_val_loss']:.6f} action_mse={summary['best_action_mse']:.6f}",
                    flush=True,
                )
                continue

            retry_job = _maybe_retry_oom(args, job)
            if retry_job is not None:
                running[retry_job.process.pid] = retry_job
                continue

            available_gpus.append(job.gpu)
            tail = _read_log_tail(job.log_path)
            raise RuntimeError(
                f"Parallel training failed for skill={job.skill_id} group={job.group} gpu={job.gpu} "
                f"exit_code={return_code}. Log tail:\n{tail}"
            )

    rows_by_group = {row["group"]: row for row in skill_rows}
    ordered_rows = [rows_by_group[group] for group in groups if group in rows_by_group]
    _write_skill_comparison(args.output_root, skill_id, ordered_rows)
    return ordered_rows


def main() -> None:
    args = parse_args()
    skill_ids = args.skill_ids or [path.name for path in discover_skill_dirs(args.skill_root)]
    if not skill_ids:
        raise SystemExit(f"No skills found under {args.skill_root}.")
    if not args.gpus:
        raise SystemExit("At least one GPU id is required.")

    global_rows = []
    for skill_id in skill_ids:
        print(
            f"[parallel] scheduling skill={skill_id} groups={args.groups} gpus={args.gpus} "
            f"batch_size={args.batch_size}",
            flush=True,
        )
        skill_rows = _schedule_skill(args, skill_id, list(args.groups))
        global_rows.extend(skill_rows)

    global_path = args.output_root / f"grid_summary_{timestamp_run_name('pi05_skill_parallel')}.json"
    write_json(global_path, global_rows)
    print(json.dumps({"summary_path": str(global_path), "runs": global_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
