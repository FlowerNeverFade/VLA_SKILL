#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from vla_skill.constants import DATA_ROOT, DEFAULT_BASE_MODEL_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_SKILL_ROOT
from vla_skill.io_utils import ensure_dir, load_json, timestamp_run_name, write_json
from vla_skill.training import evaluate_base_policy_shard, merge_base_eval_shards

DEFAULT_LOG_ROOT = DATA_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sharded multi-GPU full-val evaluation for the PI05 base model.")
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2, help="Per-shard dataloader workers.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--log-every-batches", type=int, default=500)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--shard-output-path", type=Path)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return parser.parse_args()


def _write_base_vs_adapters(skill_dir: Path, base_summary: dict[str, object]) -> dict[str, str] | None:
    comparison_path = skill_dir / "comparison.json"
    if not comparison_path.is_file():
        return None

    comparison = load_json(comparison_path)
    if not isinstance(comparison, list) or not comparison:
        return None

    rows: list[dict[str, object]] = []
    for item in comparison:
        rows.append(
            {
                "skill_id": item["skill_id"],
                "group": item["group"],
                "run_name": item["run_name"],
                "adapter_val_loss": item["best_val_loss"],
                "adapter_action_mse": item["best_action_mse"],
                "base_val_loss": base_summary["val_loss"],
                "base_action_mse": base_summary["action_mse"],
                "improvement_over_base_val_loss": float(base_summary["val_loss"]) - float(item["best_val_loss"]),
                "improvement_over_base_action_mse": float(base_summary["action_mse"]) - float(item["best_action_mse"]),
                "better_than_base_val_loss": float(item["best_val_loss"]) < float(base_summary["val_loss"]),
                "better_than_base_action_mse": float(item["best_action_mse"]) < float(base_summary["action_mse"]),
            }
        )

    out_json = skill_dir / "base_vs_adapters.json"
    out_csv = skill_dir / "base_vs_adapters.csv"
    write_json(out_json, {"base": base_summary, "adapters": rows})
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "base_vs_adapters_json": str(out_json),
        "base_vs_adapters_csv": str(out_csv),
    }


def _worker_mode(args: argparse.Namespace) -> None:
    if args.shard_id is None or args.num_shards is None or args.device is None:
        raise SystemExit("--worker requires --shard-id, --num-shards, and --device.")
    payload = evaluate_base_policy_shard(
        skill_id=args.skill_id,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        skill_root=args.skill_root,
        output_root=args.output_root,
        base_model_path=args.base_model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        log_every_batches=args.log_every_batches,
        shard_output_path=args.shard_output_path,
        write_result=True,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _orchestrator_mode(args: argparse.Namespace) -> None:
    gpus = [str(gpu) for gpu in args.gpus]
    num_shards = len(gpus)
    if num_shards <= 0:
        raise SystemExit("--gpus must contain at least one GPU id.")

    run_name = args.run_name or timestamp_run_name(f"base_eval_{args.skill_id}_sharded")
    shard_root = args.output_root / args.skill_id / "base_eval_shards" / run_name
    ensure_dir(shard_root)
    ensure_dir(args.log_root)

    launch_payload = {
        "skill_id": args.skill_id,
        "run_name": run_name,
        "gpus": gpus,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "dtype": args.dtype,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "log_every_batches": args.log_every_batches,
        "shard_root": str(shard_root),
    }
    write_json(shard_root / "launch.json", launch_payload)

    script_path = Path(__file__).resolve()
    handles = []
    processes: list[tuple[subprocess.Popen[str], int, str, Path, Path]] = []

    for shard_id, gpu_id in enumerate(gpus):
        shard_output_path = shard_root / f"shard_{shard_id:02d}.json"
        shard_log_path = shard_root / f"shard_{shard_id:02d}.log"
        device = f"cuda:{gpu_id}"
        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--skill-id",
            args.skill_id,
            "--skill-root",
            str(args.skill_root),
            "--output-root",
            str(args.output_root),
            "--base-model-path",
            str(args.base_model_path),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--dtype",
            args.dtype,
            "--device",
            device,
            "--shard-id",
            str(shard_id),
            "--num-shards",
            str(num_shards),
            "--shard-output-path",
            str(shard_output_path),
            "--log-every-batches",
            str(args.log_every_batches),
        ]
        if args.tokenizer_name_or_path is not None:
            cmd.extend(["--tokenizer-name-or-path", args.tokenizer_name_or_path])

        handle = shard_log_path.open("w", encoding="utf-8")
        handles.append(handle)
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((proc, shard_id, device, shard_output_path, shard_log_path))
        print(f"[launch] shard={shard_id}/{num_shards} device={device} pid={proc.pid} log={shard_log_path}", flush=True)

    failed = False
    try:
        for proc, shard_id, device, shard_output_path, shard_log_path in processes:
            exit_code = proc.wait()
            print(
                f"[wait] shard={shard_id}/{num_shards} device={device} pid={proc.pid} "
                f"exit={exit_code} output={shard_output_path}",
                flush=True,
            )
            if exit_code != 0:
                print(f"[error] shard={shard_id} failed, see {shard_log_path}", flush=True)
                failed = True
    finally:
        for handle in handles:
            handle.close()

    if failed:
        raise SystemExit(1)

    shard_payloads = [load_json(shard_output_path) for _, _, _, shard_output_path, _ in processes]
    base_summary = merge_base_eval_shards(
        shard_payloads,
        output_root=args.output_root,
        write_result=True,
    )
    extra_outputs = _write_base_vs_adapters(args.output_root / args.skill_id, base_summary)
    result_payload = {
        "base_eval_summary": str(args.output_root / args.skill_id / "base_eval_summary.json"),
        "shard_root": str(shard_root),
    }
    if extra_outputs is not None:
        result_payload.update(extra_outputs)
    print(json.dumps(result_payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_mode(args)
        return
    _orchestrator_mode(args)


if __name__ == "__main__":
    main()
