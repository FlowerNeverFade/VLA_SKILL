#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for skill-sharded adapters to finish, stop the legacy in-memory trainer, then launch lora_control router training."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--steps-per-channel", type=int, required=True)
    parser.add_argument("--router-steps-per-channel", type=int)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-batches", type=int, default=8)
    parser.add_argument("--cache-image-storage-dtype", default="float16")
    parser.add_argument("--router-control-adapter", default="router_control")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--log-path", type=Path, required=True)
    return parser.parse_args()


def _read_progress(run_dir: Path) -> dict:
    path = run_dir / "adapter_progress.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _status_counts(progress: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in (progress.get("channels") or {}).values():
        status = str(item.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _adapter_phase_complete(progress: dict) -> bool:
    counts = _status_counts(progress)
    total = len(progress.get("channels") or {})
    return total > 0 and counts.get("completed", 0) == total


def _matching_legacy_pids(run_name: str) -> list[int]:
    current = os.getpid()
    result = subprocess.run(
        ["pgrep", "-f", f"train_router_lora_sharded.py .*--run-name {run_name}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != current:
            pids.append(pid)
    return sorted(set(pids))


def _terminate_legacy_run(run_name: str, *, timeout_seconds: float) -> None:
    pids = _matching_legacy_pids(run_name)
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _matching_legacy_pids(run_name):
            return
        time.sleep(2.0)
    for pid in _matching_legacy_pids(run_name):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _launch_lora_control_router(args: argparse.Namespace) -> int:
    command = [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(args.nproc_per_node),
        "train_router_lora_sharded.py",
        "--config",
        str(args.config),
        "--run-name",
        args.run_name,
        "--steps-per-channel",
        str(args.steps_per_channel),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        "1",
        "--num-workers",
        str(args.num_workers),
        "--prefetch-batches",
        str(args.prefetch_batches),
        "--cache-root",
        str(args.cache_root),
        "--cache-image-storage-dtype",
        args.cache_image_storage_dtype,
        "--require-policy-cache",
        "--confirm-long-run",
        "--skip-adapter-phase",
        "--router-impl",
        "lora_control",
        "--router-control-adapter",
        args.router_control_adapter,
    ]
    if args.router_steps_per_channel is not None:
        command.extend(["--router-steps-per-channel", str(args.router_steps_per_channel)])
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(index) for index in range(args.nproc_per_node)))
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = args.log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return int(process.pid)


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "kind": "handoff_start",
                "run_name": args.run_name,
                "run_dir": str(args.run_dir),
                "log_path": str(args.log_path),
                "poll_seconds": args.poll_seconds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    while True:
        progress = _read_progress(args.run_dir)
        counts = _status_counts(progress)
        print(json.dumps({"kind": "handoff_poll", "counts": counts}, ensure_ascii=False), flush=True)
        if counts.get("failed", 0):
            raise SystemExit(f"Adapter phase has failed channels: {counts}")
        if _adapter_phase_complete(progress):
            break
        time.sleep(args.poll_seconds)

    print(json.dumps({"kind": "handoff_adapters_complete"}, ensure_ascii=False), flush=True)
    _terminate_legacy_run(args.run_name, timeout_seconds=args.terminate_timeout_seconds)
    pid = _launch_lora_control_router(args)
    print(json.dumps({"kind": "handoff_router_control_launched", "pid": pid}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
