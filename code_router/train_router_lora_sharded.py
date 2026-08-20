#!/usr/bin/env python
from __future__ import annotations

import argparse
import queue
from dataclasses import dataclass, replace
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset

from train_router_lora import (
    CyclingLoader,
    _adapter_disable_context,
    _activate_adapter_without_grad_toggle,
    _make_loader,
    _next_channel,
    _prepare_batch,
    _resolve_steps_per_channel,
    _save_single_adapter,
    _set_lora_parameters_trainable,
    _unwrap,
    _validate_long_run,
)
from vla_skill.constants import IMAGE_FIELDS
from vla_skill.dataset import load_skill_spec, load_stats
from vla_skill.io_utils import ensure_dir, json_ready, timestamp_run_name, utc_now_iso, write_json, write_yaml
from vla_skill.pi05 import build_processors
from vla_skill_router.config import ExperimentConfig, load_experiment_config
from vla_skill_router.constants import (
    DEFAULT_ROUTER_CONTROL_ADAPTER,
    ROUTER_IMPL_HARD_TOP1,
    ROUTER_IMPL_LORA_CONTROL,
)
from vla_skill_router.distributed import (
    DistributedInfo,
    barrier,
    broadcast_object,
    cleanup_distributed,
    effective_global_batch_size,
    init_distributed,
    reduce_mean_scalar,
)
from vla_skill_router.features import PI05PrefixFeatureExtractor
from vla_skill_router.policy_cache import DEFAULT_IMAGE_STORAGE_DTYPE, IMAGE_STORAGE_DTYPES, PolicyCacheIterableDataset
from vla_skill_router.real_runtime import (
    activate_router_control_adapter,
    build_hard_top1_router_from_config,
    build_lora_control_router_from_config,
    build_dataset_for_channel,
    iter_adapter_parameters,
    iter_trainable_parameters,
    load_first_skill_policy,
    load_or_initialize_router_control_adapter,
    load_or_initialize_single_channel,
    pi05_masked_policy_loss,
)
from vla_skill_router.sharded import (
    claim_next_channel,
    ensure_adapter_progress,
    expected_micro_batches,
    mark_channel_completed,
    mark_channel_failed,
    mark_channel_progress,
    progress_summary,
    resolve_gradient_accumulation_steps,
    wait_for_adapter_progress_complete,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PI05 router-LoRA with skill-sharded adapter workers plus a synchronized router phase."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--steps-per-channel", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--router-steps-per-channel", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--router-batch-size",
        type=int,
        help="Override only the synchronized router phase per-device batch size.",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-every", type=int, help="Override train.save_every without changing optimization.")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--require-policy-cache", action="store_true")
    parser.add_argument(
        "--cache-image-storage-dtype",
        choices=IMAGE_STORAGE_DTYPES,
        default=DEFAULT_IMAGE_STORAGE_DTYPE,
        help="Expected on-disk image dtype for policy cache. Use uint8 with compact caches.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="Prepared mini-batches to keep ahead per worker. Use 0 to disable the preprocessing thread.",
    )
    parser.add_argument(
        "--router-prefetch-batches",
        type=int,
        help="Prepared scheduled router batches to keep ahead. Defaults to --prefetch-batches.",
    )
    parser.add_argument(
        "--router-prefetch-workers",
        type=int,
        default=4,
        help="Background threads used to prepare scheduled router batches.",
    )
    parser.add_argument(
        "--router-metrics-every",
        type=int,
        default=1,
        help="Write synchronized router metrics every N steps. Higher values avoid per-step GPU/CPU sync.",
    )
    parser.add_argument(
        "--router-rendezvous-timeout-seconds",
        type=float,
        default=0.0,
        help="CPU file rendezvous timeout before each router DDP step. Use 0 to wait indefinitely.",
    )
    parser.add_argument(
        "--router-rendezvous-warn-seconds",
        type=float,
        default=300.0,
        help="Print a router rendezvous wait diagnostic every N seconds. Use 0 to disable warnings.",
    )
    parser.add_argument(
        "--router-num-workers",
        type=int,
        default=0,
        help="DataLoader workers used only by the synchronized router phase. Default 0 avoids per-skill worker fan-out.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        help="Optional torch intra-op CPU thread limit for each rank. This does not change training math.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        help="Optional torch inter-op CPU thread limit for each rank. This does not change training math.",
    )
    parser.add_argument(
        "--rank-startup-stagger-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional per-local-rank startup delay before model loading. "
            "Rank i sleeps i * seconds, reducing CPU memory/IO spikes without changing training math."
        ),
    )
    parser.add_argument(
        "--adapter-completion-poll-seconds",
        type=float,
        default=10.0,
        help="File-poll interval used after skill-sharded adapter workers finish their local queues.",
    )
    parser.add_argument("--skip-adapter-phase", action="store_true")
    parser.add_argument("--skip-router-phase", action="store_true")
    parser.add_argument(
        "--router-impl",
        choices=(ROUTER_IMPL_LORA_CONTROL, ROUTER_IMPL_HARD_TOP1),
        default=ROUTER_IMPL_LORA_CONTROL,
        help="Router phase implementation. lora_control trains the router_control LoRA adapter plus a linear head.",
    )
    parser.add_argument("--router-control-adapter", default=DEFAULT_ROUTER_CONTROL_ADAPTER)
    parser.add_argument(
        "--router-control-checkpoint",
        type=Path,
        help="Optional directory containing router_control/, router_control_head.pt, and trainer state to resume/init.",
    )
    parser.add_argument("--confirm-long-run", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class ChannelResources:
    loader: CyclingLoader
    preprocessor: Any | None
    preprocessed: bool
    lock: threading.Lock


def _make_single_worker_loader(
    dataset,
    cfg: ExperimentConfig,
    *,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
) -> CyclingLoader:
    worker_count = cfg.train.num_workers if num_workers is None else num_workers
    keep_workers = worker_count > 0 if persistent_workers is None else persistent_workers and worker_count > 0
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=not isinstance(dataset, IterableDataset),
        num_workers=worker_count,
        pin_memory=cfg.train.device.startswith("cuda"),
        persistent_workers=keep_workers,
        drop_last=False,
    )
    return CyclingLoader(loader)


def _make_channel_loader(
    dataset,
    cfg: ExperimentConfig,
    dist_info: DistributedInfo | None,
    *,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
) -> CyclingLoader:
    worker_count = cfg.train.num_workers if num_workers is None else num_workers
    keep_workers = worker_count > 0 if persistent_workers is None else persistent_workers and worker_count > 0
    if isinstance(dataset, IterableDataset):
        loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=cfg.train.device.startswith("cuda"),
            persistent_workers=keep_workers,
            drop_last=False,
        )
        return CyclingLoader(loader)
    if dist_info is None:
        return _make_single_worker_loader(
            dataset,
            cfg,
            num_workers=worker_count,
            persistent_workers=keep_workers,
        )
    return _make_loader(dataset, cfg, dist_info)


class LazyChannelResources:
    def __init__(
        self,
        cfg: ExperimentConfig,
        dist_info: DistributedInfo | None = None,
        *,
        cache_root: Path | None = None,
        require_policy_cache: bool = False,
        cache_image_storage_dtype: str = DEFAULT_IMAGE_STORAGE_DTYPE,
        num_workers: int | None = None,
        persistent_workers: bool | None = None,
    ):
        self.cfg = cfg
        self.dist_info = dist_info
        self.cache_root = cache_root
        self.require_policy_cache = require_policy_cache
        self.cache_image_storage_dtype = cache_image_storage_dtype
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self._items: dict[int, ChannelResources] = {}
        self._lock = threading.Lock()

    def get(self, channel_index: int) -> ChannelResources:
        with self._lock:
            item = self._items.get(channel_index)
            if item is not None:
                return item
            dataset = build_dataset_for_channel(
                self.cfg,
                channel_index,
                split="train",
                cache_root=self.cache_root,
                require_policy_cache=self.require_policy_cache,
                cache_rank=0 if self.dist_info is None else self.dist_info.rank,
                cache_world_size=1 if self.dist_info is None else self.dist_info.world_size,
                cache_seed=self.cfg.train.seed + channel_index,
                cache_image_storage_dtype=self.cache_image_storage_dtype,
            )
            loader = _make_channel_loader(
                dataset,
                self.cfg,
                self.dist_info,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
            channel = self.cfg.channels[channel_index]
            preprocessed = isinstance(dataset, PolicyCacheIterableDataset)
            if preprocessed:
                preprocessor = None
            else:
                skill_spec = load_skill_spec(self.cfg.skill_root, channel.skill_id)
                preprocessor = build_processors(
                    skill_spec,
                    load_stats(skill_spec),
                    device=self.cfg.train.device,
                    tokenizer_name_or_path=self.cfg.train.tokenizer_name_or_path,
                )[0]
            item = ChannelResources(
                loader=loader,
                preprocessor=preprocessor,
                preprocessed=preprocessed,
                lock=threading.Lock(),
            )
            self._items[channel_index] = item
            return item


def _move_batch_to_device(batch: Any, device: str, *, non_blocking: bool = True, key: str | None = None) -> Any:
    if torch.is_tensor(batch):
        moved = batch.to(device=device, non_blocking=non_blocking)
        if key in IMAGE_FIELDS and moved.dtype == torch.uint8:
            moved = moved.to(dtype=torch.float32).div_(255.0)
        return moved
    if isinstance(batch, dict):
        return {
            item_key: _move_batch_to_device(value, device, non_blocking=non_blocking, key=item_key)
            for item_key, value in batch.items()
        }
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device, non_blocking=non_blocking, key=key) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device, non_blocking=non_blocking, key=key) for value in batch)
    return batch


def _scheduled_router_labels(logits: torch.Tensor, channel_index: int) -> torch.Tensor:
    return torch.full(
        (int(logits.shape[0]),),
        int(channel_index),
        device=logits.device,
        dtype=torch.long,
    )


def _router_rendezvous_dir(run_dir: Path, name: str) -> Path:
    return run_dir / f".{name}_router_step_rendezvous"


def _reset_router_rendezvous_dir(run_dir: Path, name: str, dist_info: DistributedInfo) -> Path | None:
    if not dist_info.is_distributed:
        return None
    path = _router_rendezvous_dir(run_dir, name)
    if dist_info.is_rank0:
        if path.exists():
            shutil.rmtree(path)
        ensure_dir(path)
    barrier(dist_info)
    return path


def _write_router_rendezvous_marker(path: Path, *, rank: int, step: int) -> None:
    # Use a step-specific directory. A single per-rank marker file can be
    # overwritten by a fast rank entering step N+1 while a slower rank is still
    # checking step N, which turns the rendezvous into a permanent wait.
    step_path = path / f"step-{int(step):012d}"
    ensure_dir(step_path)
    final_path = step_path / f"rank{rank}.txt"
    tmp_path = step_path / f".rank{rank}.{os.getpid()}.tmp"
    tmp_path.write_text(f"{int(step)}\n{os.getpid()}\n{utc_now_iso()}\n", encoding="utf-8")
    tmp_path.replace(final_path)


def _read_router_rendezvous_step(path: Path) -> int | None:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        return int(first_line)
    except (OSError, IndexError, ValueError):
        return None


def _read_router_rendezvous_marker(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"step": None, "pid": None, "timestamp": None}
    try:
        step = int(lines[0])
    except (IndexError, ValueError):
        step = None
    try:
        pid = int(lines[1])
    except (IndexError, ValueError):
        pid = None
    return {
        "step": step,
        "pid": pid,
        "timestamp": lines[2] if len(lines) > 2 else None,
    }


def _router_rendezvous_snapshot(path: Path, world_size: int, *, step: int) -> dict[str, dict[str, Any]]:
    step_path = path / f"step-{int(step):012d}"
    return {
        f"rank{rank}": _read_router_rendezvous_marker(step_path / f"rank{rank}.txt")
        for rank in range(world_size)
    }


def _router_step_cpu_rendezvous(
    rendezvous_dir: Path | None,
    *,
    step: int,
    dist_info: DistributedInfo,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.25,
    warn_seconds: float = 300.0,
) -> None:
    if rendezvous_dir is None or not dist_info.is_distributed:
        return
    _write_router_rendezvous_marker(rendezvous_dir, rank=dist_info.rank, step=step)
    deadline = None if timeout_seconds is None or timeout_seconds <= 0 else time.monotonic() + timeout_seconds
    next_warning = None if warn_seconds <= 0 else time.monotonic() + warn_seconds
    while True:
        snapshot = _router_rendezvous_snapshot(rendezvous_dir, dist_info.world_size, step=step)
        missing = [
            rank_name
            for rank_name, marker in snapshot.items()
            if marker.get("step") != int(step)
        ]
        if not missing:
            return
        now = time.monotonic()
        if next_warning is not None and now >= next_warning:
            print(
                json.dumps(
                    {
                        "kind": "router_rendezvous_wait",
                        "step": int(step),
                        "rank": dist_info.rank,
                        "waiting_for": missing,
                        "markers": snapshot,
                        "timestamp": utc_now_iso(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            next_warning = now + warn_seconds
        if deadline is not None and now > deadline:
            raise TimeoutError(
                f"Timed out waiting for router step {step} CPU rendezvous "
                f"({rendezvous_dir}, rank={dist_info.rank}, world_size={dist_info.world_size}, "
                f"missing={missing}, markers={snapshot})."
            )
        time.sleep(poll_seconds)


class PrefetchedBatchIterator:
    def __init__(
        self,
        loader: CyclingLoader,
        preprocessor,
        *,
        prefetch_batches: int,
        device: str,
        preprocessed: bool = False,
    ) -> None:
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive.")
        self.loader = loader
        self.preprocessor = preprocessor
        self.device = device
        self.preprocessed = preprocessed
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_batches)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="batch-prefetch", daemon=True)
        try:
            first_proc_batch = self._prepare_next_batch()
            self._queue.put((first_proc_batch, None), block=False)
        except BaseException as exc:  # noqa: BLE001
            self._queue.put(exc, block=False)
        self._thread.start()

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _worker(self) -> None:
        stream = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(self.device)
            if device.index is not None:
                torch.cuda.set_device(device)
            stream = torch.cuda.Stream(device=device)
        while not self._stop.is_set():
            try:
                if stream is None:
                    proc_batch = self._prepare_next_batch()
                    event = None
                else:
                    with torch.cuda.stream(stream):
                        proc_batch = self._prepare_next_batch()
                    event = torch.cuda.Event()
                    event.record(stream)
                self._put((proc_batch, event))
            except BaseException as exc:  # noqa: BLE001
                self._put(exc)
                return

    def _prepare_next_batch(self) -> dict[str, Any]:
        batch = self.loader.next()
        if self.preprocessed:
            return _move_batch_to_device(batch, self.device)
        if self.preprocessor is None:
            raise RuntimeError("Online preprocessing requires a preprocessor.")
        return _prepare_batch(batch, self.preprocessor)

    def next(self) -> dict[str, Any]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise RuntimeError("Prefetch worker failed.") from item
        proc_batch, event = item
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        return proc_batch

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class RouterScheduledBatchPrefetcher:
    def __init__(
        self,
        *,
        resources: LazyChannelResources,
        cfg: ExperimentConfig,
        start_step: int,
        total_steps: int,
        channel_cursor: int,
        channel_steps: list[int],
        steps_per_channel: int,
        prefetch_batches: int,
        prefetch_workers: int = 1,
        device: str,
    ) -> None:
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive.")
        if prefetch_workers <= 0:
            raise ValueError("prefetch_workers must be positive.")
        self.resources = resources
        self.cfg = cfg
        self.next_step = int(start_step)
        self.expected_step = int(start_step)
        self.total_steps = int(total_steps)
        self.channel_cursor = int(channel_cursor) % max(1, len(channel_steps))
        self.channel_steps = list(channel_steps)
        self.steps_per_channel = int(steps_per_channel)
        self.max_ready = int(prefetch_batches)
        self.device = device
        self.prefetch_workers = int(prefetch_workers)
        self._task_queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_batches)
        self._ready: dict[int, Any] = {}
        self._condition = threading.Condition()
        self._error: BaseException | None = None
        self._active_workers = self.prefetch_workers
        self._stop = threading.Event()
        self.wait_diagnostic_seconds = 300.0
        self._next_wait_diagnostic = time.monotonic() + self.wait_diagnostic_seconds
        self._producer = threading.Thread(target=self._producer_loop, name="router-batch-scheduler", daemon=True)
        self._workers = [
            threading.Thread(target=self._worker, args=(index,), name=f"router-batch-prefetch-{index}", daemon=True)
            for index in range(self.prefetch_workers)
        ]
        for worker in self._workers:
            worker.start()
        self._producer.start()

    def _advance_schedule(self) -> tuple[int, int] | None:
        if self.next_step > self.total_steps:
            return None
        channel_index, self.channel_cursor = _next_channel(
            global_step=self.next_step,
            channel_cursor=self.channel_cursor,
            channel_steps=self.channel_steps,
            steps_per_channel=self.steps_per_channel,
        )
        if channel_index is None:
            self.next_step = self.total_steps + 1
            return None
        router_step = self.next_step
        self.next_step += 1
        self.channel_steps[channel_index] += 1
        return router_step, channel_index

    def _producer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                scheduled = self._advance_schedule()
                if scheduled is None:
                    break
                while not self._stop.is_set():
                    try:
                        self._task_queue.put(scheduled, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            for _ in range(self.prefetch_workers):
                while True:
                    try:
                        self._task_queue.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        if self._stop.is_set():
                            break

    def _prepare_batch_for(self, router_step: int, channel_index: int):
        resource = self.resources.get(channel_index)
        with resource.lock:
            raw_batch = resource.loader.next()
        proc_batch = (
            _move_batch_to_device(raw_batch, self.device)
            if resource.preprocessed
            else _prepare_batch(raw_batch, resource.preprocessor)
        )
        return router_step, channel_index, proc_batch

    def _worker(self, worker_index: int) -> None:
        stream = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(self.device)
            if device.index is not None:
                torch.cuda.set_device(device)
            stream = torch.cuda.Stream(device=device)
        try:
            while not self._stop.is_set():
                task = self._task_queue.get()
                if task is None:
                    return
                router_step, channel_index = task
                if stream is None:
                    item = self._prepare_batch_for(router_step, channel_index)
                    event = None
                else:
                    with torch.cuda.stream(stream):
                        item = self._prepare_batch_for(router_step, channel_index)
                    event = torch.cuda.Event()
                    event.record(stream)
                step_key = int(router_step)
                with self._condition:
                    while (
                        len(self._ready) >= self.max_ready
                        and step_key != self.expected_step
                        and not self._stop.is_set()
                    ):
                        self._condition.wait(timeout=0.2)
                    if self._stop.is_set():
                        return
                    self._ready[step_key] = (item, event)
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                self._active_workers -= 1
                self._condition.notify_all()

    def next(self):
        if self.expected_step > self.total_steps:
            return None
        with self._condition:
            while self.expected_step not in self._ready:
                if self._error is not None:
                    raise RuntimeError("Router prefetch worker failed.") from self._error
                if self._active_workers <= 0:
                    return None
                now = time.monotonic()
                if now >= self._next_wait_diagnostic:
                    print(
                        json.dumps(
                            {
                                "kind": "router_prefetch_wait",
                                "expected_step": self.expected_step,
                                "ready_steps": sorted(self._ready),
                                "next_scheduled_step": self.next_step,
                                "active_workers": self._active_workers,
                                "task_queue_size": self._task_queue.qsize(),
                                "max_ready": self.max_ready,
                                "timestamp": utc_now_iso(),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    self._next_wait_diagnostic = now + self.wait_diagnostic_seconds
                self._condition.wait(timeout=0.2)
            proc_item, event = self._ready.pop(self.expected_step)
            self.expected_step += 1
            self._condition.notify_all()
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        return proc_item

    def close(self) -> None:
        self._stop.set()
        self._producer.join(timeout=2.0)
        for worker in self._workers:
            worker.join(timeout=2.0)


def _resolved_config_payload(
    *,
    config_path: Path,
    cfg: ExperimentConfig,
    steps_per_channel: int,
    router_steps_per_channel: int | None,
    gradient_accumulation_steps: int,
    prefetch_batches: int,
    router_prefetch_batches: int | None,
    router_prefetch_workers: int | None,
    router_num_workers: int | None,
    router_metrics_every: int | None,
    router_batch_size: int | None,
    router_rendezvous_timeout_seconds: float | None,
    router_rendezvous_warn_seconds: float | None,
    cache_root: Path | None,
    require_policy_cache: bool,
    cache_image_storage_dtype: str,
    router_impl: str,
    router_control_adapter: str,
    router_control_checkpoint: Path | None,
    torch_num_threads: int | None,
    torch_num_interop_threads: int | None,
    rank_startup_stagger_seconds: float,
    dist_info: DistributedInfo,
) -> dict[str, Any]:
    return json_ready(
        {
            "mode": "skill_sharded",
            "config_path": str(config_path),
            "base_model_path": str(cfg.base_model_path),
            "skill_root": str(cfg.skill_root),
            "output_root": str(cfg.output_root),
            "channels": [channel.__dict__ for channel in cfg.channels],
            "router": cfg.router.__dict__,
            "train": cfg.train.__dict__,
            "steps_per_channel": steps_per_channel,
            "router_steps_per_channel": router_steps_per_channel,
            "total_adapter_optimizer_steps": steps_per_channel * len(cfg.channels),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "prefetch_batches": prefetch_batches,
            "router_prefetch_batches": router_prefetch_batches,
            "router_prefetch_workers": router_prefetch_workers,
            "router_num_workers": router_num_workers,
            "router_metrics_every": router_metrics_every,
            "router_batch_size": router_batch_size,
            "router_rendezvous_timeout_seconds": router_rendezvous_timeout_seconds,
            "router_rendezvous_warn_seconds": router_rendezvous_warn_seconds,
            "cache_root": str(cache_root) if cache_root is not None else None,
            "require_policy_cache": require_policy_cache,
            "cache_image_storage_dtype": cache_image_storage_dtype,
            "router_impl": router_impl,
            "router_control_adapter": router_control_adapter,
            "router_control_checkpoint": str(router_control_checkpoint) if router_control_checkpoint else None,
            "torch_num_threads": torch_num_threads,
            "torch_num_interop_threads": torch_num_interop_threads,
            "rank_startup_stagger_seconds": rank_startup_stagger_seconds,
            "adapter_effective_global_batch_size": cfg.train.batch_size * gradient_accumulation_steps,
            "distributed": {
                "enabled": dist_info.is_distributed,
                "backend": dist_info.backend,
                "rank": dist_info.rank,
                "local_rank": dist_info.local_rank,
                "world_size": dist_info.world_size,
                "per_device_batch_size": cfg.train.batch_size,
                "router_per_device_batch_size": router_batch_size or cfg.train.batch_size,
                "router_effective_global_batch_size": effective_global_batch_size(
                    router_batch_size or cfg.train.batch_size,
                    dist_info.world_size,
                ),
            },
        }
    )


def _stagger_rank_startup(dist_info: DistributedInfo, seconds: float) -> None:
    if seconds <= 0 or not dist_info.is_distributed:
        return
    delay = float(seconds) * float(dist_info.local_rank)
    if delay <= 0:
        return
    print(
        json.dumps(
            {
                "kind": "rank_startup_stagger",
                "rank": dist_info.rank,
                "local_rank": dist_info.local_rank,
                "sleep_seconds": delay,
                "timestamp": utc_now_iso(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    time.sleep(delay)


def _save_adapter_atomic(policy, final_path: Path, channel_id: str, *, rank: int) -> None:
    ensure_dir(final_path.parent)
    tmp_path = final_path.parent / f".{final_path.name}.rank{rank}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    _save_single_adapter(policy, tmp_path, channel_id)
    if final_path.exists():
        shutil.rmtree(final_path)
    tmp_path.rename(final_path)


def _append_rank_metric(run_dir: Path, rank: int, record: dict[str, Any]) -> None:
    path = run_dir / f"adapter_metrics_rank{rank}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")


def _ddp_static_graph_kwargs() -> dict[str, Any]:
    """Return DDP kwargs that preserve gradients while avoiding per-step graph scans."""
    parameters = inspect.signature(DistributedDataParallel).parameters
    kwargs: dict[str, Any] = {"broadcast_buffers": False}
    if "static_graph" in parameters:
        kwargs["static_graph"] = True
    if "gradient_as_bucket_view" in parameters:
        kwargs["gradient_as_bucket_view"] = True
    if "skip_all_reduce_unused_params" in parameters:
        kwargs["skip_all_reduce_unused_params"] = True
    return kwargs


def _derive_channel_cursor(channel_steps: list[int], *, steps_per_channel: int) -> int:
    """Recover the next round-robin cursor from completed per-channel steps."""
    if not channel_steps:
        return 0
    replay_steps = [0 for _ in channel_steps]
    cursor = 0
    for step in range(1, sum(int(value) for value in channel_steps) + 1):
        channel_index, cursor = _next_channel(
            global_step=step,
            channel_cursor=cursor,
            channel_steps=replay_steps,
            steps_per_channel=steps_per_channel,
        )
        if channel_index is None:
            break
        replay_steps[channel_index] += 1
    return cursor % len(channel_steps)


def _train_claimed_adapter(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    progress_path: Path,
    claim,
    policy,
    steps_per_channel: int,
    gradient_accumulation_steps: int,
    prefetch_batches: int,
    cache_root: Path | None,
    require_policy_cache: bool,
    cache_image_storage_dtype: str,
    rank: int,
) -> Any:
    channel = cfg.channels[claim.channel_index]
    policy, loaded_adapter_dir = load_or_initialize_single_channel(cfg, policy, claim.channel_index)
    _set_lora_parameters_trainable(policy)
    _activate_adapter_without_grad_toggle(policy, channel.channel_id)
    policy.train()

    resources = LazyChannelResources(
        cfg,
        cache_root=cache_root,
        require_policy_cache=require_policy_cache,
        cache_image_storage_dtype=cache_image_storage_dtype,
    )
    resource = resources.get(claim.channel_index)
    loader = resource.loader
    optimizer = torch.optim.AdamW(list(iter_adapter_parameters(policy)), lr=cfg.train.lr)

    prefetcher = (
        PrefetchedBatchIterator(
            loader,
            resource.preprocessor,
            prefetch_batches=prefetch_batches,
            device=cfg.train.device,
            preprocessed=resource.preprocessed,
        )
        if prefetch_batches > 0
        else None
    )
    try:
        for optimizer_step in range(1, steps_per_channel + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(gradient_accumulation_steps):
                if prefetcher is None:
                    batch = loader.next()
                    if resource.preprocessed:
                        proc_batch = _move_batch_to_device(batch, cfg.train.device)
                    else:
                        proc_batch = _prepare_batch(batch, resource.preprocessor)
                else:
                    proc_batch = prefetcher.next()
                _activate_adapter_without_grad_toggle(policy, channel.channel_id)
                loss = pi05_masked_policy_loss(policy, proc_batch) / float(gradient_accumulation_steps)
                loss.backward()
                loss_accum += float(loss.detach().cpu().item())
            optimizer.step()

            if optimizer_step == 1 or optimizer_step % cfg.train.log_every == 0 or optimizer_step == steps_per_channel:
                mark_channel_progress(
                    progress_path,
                    channel_id=channel.channel_id,
                    steps=optimizer_step,
                    rank=rank,
                )
                _append_rank_metric(
                    run_dir,
                    rank,
                    {
                        "kind": "adapter_train",
                        "rank": rank,
                        "channel_id": channel.channel_id,
                        "skill_id": channel.skill_id,
                        "optimizer_step": optimizer_step,
                        "micro_batches": expected_micro_batches(optimizer_step, gradient_accumulation_steps),
                        "policy_loss": loss_accum,
                        "loaded_adapter_dir": str(loaded_adapter_dir) if loaded_adapter_dir else None,
                        "prefetch_batches": prefetch_batches,
                        "policy_cache": resource.preprocessed,
                        "timestamp": utc_now_iso(),
                    },
                )
    finally:
        if prefetcher is not None:
            prefetcher.close()

    adapter_path = run_dir / "channels" / channel.channel_id
    _save_adapter_atomic(policy, adapter_path, channel.channel_id, rank=rank)
    mark_channel_completed(
        progress_path,
        channel_id=channel.channel_id,
        steps=steps_per_channel,
        adapter_path=adapter_path,
        rank=rank,
    )
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return policy


def run_adapter_phase(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    progress_path: Path,
    dist_info: DistributedInfo,
    steps_per_channel: int,
    gradient_accumulation_steps: int,
    prefetch_batches: int,
    cache_root: Path | None,
    require_policy_cache: bool,
    cache_image_storage_dtype: str,
) -> Any | None:
    policy = None
    while True:
        claim = claim_next_channel(progress_path, rank=dist_info.rank, pid=os.getpid())
        if claim is None:
            break
        try:
            if policy is None:
                policy = load_first_skill_policy(cfg)
            policy = _train_claimed_adapter(
                cfg=cfg,
                run_dir=run_dir,
                progress_path=progress_path,
                claim=claim,
                policy=policy,
                steps_per_channel=steps_per_channel,
                gradient_accumulation_steps=gradient_accumulation_steps,
                prefetch_batches=prefetch_batches,
                cache_root=cache_root,
                require_policy_cache=require_policy_cache,
                cache_image_storage_dtype=cache_image_storage_dtype,
                rank=dist_info.rank,
            )
        except Exception as exc:  # noqa: BLE001
            mark_channel_failed(progress_path, channel_id=claim.channel_id, rank=dist_info.rank, error=repr(exc))
            raise
    return policy


def _save_router_checkpoint(
    *,
    run_dir: Path,
    cfg: ExperimentConfig,
    router,
    router_optimizer: torch.optim.Optimizer,
    router_step: int,
    channel_steps: list[int],
    channel_cursor: int,
    dist_info: DistributedInfo,
    final: bool = False,
) -> None:
    if not dist_info.is_rank0:
        return
    unwrapped_router = _unwrap(router)
    write_json(
        run_dir / "router_meta.json",
        {
            "step": router_step,
            "channel_steps": channel_steps,
            "channel_cursor": channel_cursor,
            "context_dim": unwrapped_router.context_dim,
            "channel_ids": cfg.channel_ids,
            "skill_ids": cfg.skill_ids,
            "distributed": {
                "world_size": dist_info.world_size,
                "backend": dist_info.backend,
                "per_device_batch_size": cfg.train.batch_size,
                "effective_global_batch_size": effective_global_batch_size(cfg.train.batch_size, dist_info.world_size),
            },
            "final": final,
            "saved_at": utc_now_iso(),
        },
    )
    torch.save(unwrapped_router.state_dict(), run_dir / "router.pt")
    torch.save(
        {
            "step": router_step,
            "channel_steps": channel_steps,
            "channel_cursor": channel_cursor,
            "router_optimizer": router_optimizer.state_dict(),
        },
        run_dir / "router_trainer_state.pt",
    )


class LoRAControlRouterWrapper(nn.Module):
    def __init__(self, *, policy, router_head: nn.Module, adapter_name: str):
        super().__init__()
        self.policy = policy
        self.router_head = router_head
        self.adapter_name = adapter_name
        self.feature_extractor = PI05PrefixFeatureExtractor(
            policy,
            disable_adapters=False,
            no_grad=False,
        )

    def forward(self, batch: dict[str, Any]):
        _activate_adapter_without_grad_toggle(self.policy, self.adapter_name)
        pooled_context = self.feature_extractor(batch)
        return self.router_head(pooled_context, batch["observation.state"].to(pooled_context.device))


def _router_control_head_path(run_dir: Path, adapter_name: str) -> Path:
    return run_dir / f"{adapter_name}_head.pt"


def _router_control_meta_path(run_dir: Path, adapter_name: str) -> Path:
    return run_dir / f"{adapter_name}_meta.json"


def _router_control_state_path(run_dir: Path, adapter_name: str) -> Path:
    return run_dir / f"{adapter_name}_trainer_state.pt"


def _find_peft_adapter_dir(adapter_root: Path, adapter_name: str) -> Path | None:
    for candidate in (adapter_root, adapter_root / adapter_name):
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def _resolve_router_control_checkpoint_paths(
    checkpoint_dir: Path | None,
    *,
    adapter_name: str,
) -> tuple[Path | None, Path | None, Path | None]:
    if checkpoint_dir is None:
        return None, None, None
    checkpoint_dir = Path(checkpoint_dir)
    adapter_root = checkpoint_dir / adapter_name
    if not adapter_root.exists() and (checkpoint_dir / "adapter_model.safetensors").exists():
        adapter_root = checkpoint_dir
    adapter_dir = _find_peft_adapter_dir(adapter_root, adapter_name) if adapter_root.exists() else None
    head_path = _router_control_head_path(checkpoint_dir, adapter_name)
    if not head_path.exists() and adapter_name == DEFAULT_ROUTER_CONTROL_ADAPTER:
        legacy_head = checkpoint_dir / "router_control_head.pt"
        if legacy_head.exists():
            head_path = legacy_head
    state_path = _router_control_state_path(checkpoint_dir, adapter_name)
    return (
        adapter_dir,
        head_path if head_path.exists() else None,
        state_path if state_path.exists() else None,
    )


def _save_lora_control_router_checkpoint(
    *,
    run_dir: Path,
    cfg: ExperimentConfig,
    wrapper,
    optimizer: torch.optim.Optimizer,
    router_step: int,
    channel_steps: list[int],
    channel_cursor: int,
    dist_info: DistributedInfo,
    adapter_name: str,
    final: bool = False,
) -> None:
    if not dist_info.is_rank0:
        return
    unwrapped = _unwrap(wrapper)
    adapter_path = run_dir / adapter_name
    _save_adapter_atomic(unwrapped.policy, adapter_path, adapter_name, rank=dist_info.rank)
    saved_adapter_dir = _find_peft_adapter_dir(adapter_path, adapter_name) or adapter_path
    torch.save(unwrapped.router_head.state_dict(), _router_control_head_path(run_dir, adapter_name))
    write_json(
        _router_control_meta_path(run_dir, adapter_name),
        {
            "impl": ROUTER_IMPL_LORA_CONTROL,
            "adapter_name": adapter_name,
            "adapter_path": str(saved_adapter_dir),
            "head_path": str(_router_control_head_path(run_dir, adapter_name)),
            "step": router_step,
            "channel_steps": channel_steps,
            "channel_cursor": channel_cursor,
            "context_dim": unwrapped.router_head.context_dim,
            "channel_ids": cfg.channel_ids,
            "skill_ids": cfg.skill_ids,
            "distributed": {
                "world_size": dist_info.world_size,
                "backend": dist_info.backend,
                "per_device_batch_size": cfg.train.batch_size,
                "effective_global_batch_size": effective_global_batch_size(cfg.train.batch_size, dist_info.world_size),
            },
            "final": final,
            "saved_at": utc_now_iso(),
        },
    )
    torch.save(
        {
            "step": router_step,
            "channel_steps": channel_steps,
            "channel_cursor": channel_cursor,
            "optimizer": optimizer.state_dict(),
            "impl": ROUTER_IMPL_LORA_CONTROL,
            "adapter_name": adapter_name,
        },
        _router_control_state_path(run_dir, adapter_name),
    )


def run_hard_top1_router_phase(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    dist_info: DistributedInfo,
    policy,
    router_steps_per_channel: int,
    cache_root: Path | None,
    require_policy_cache: bool,
    cache_image_storage_dtype: str,
    router_num_workers: int,
    router_prefetch_batches: int,
    router_prefetch_workers: int,
    router_metrics_every: int,
    router_rendezvous_timeout_seconds: float | None,
    router_rendezvous_warn_seconds: float,
) -> None:
    if policy is None:
        policy = load_first_skill_policy(cfg)

    resources = LazyChannelResources(
        cfg,
        # Router DDP is synchronized by gradients, not by data ownership. Some
        # small skills do not have enough cached samples to shard across every
        # rank, so each rank keeps a full cycling stream instead of an empty
        # per-rank shard.
        dist_info=None,
        cache_root=cache_root,
        require_policy_cache=require_policy_cache,
        cache_image_storage_dtype=cache_image_storage_dtype,
        num_workers=router_num_workers,
        persistent_workers=False,
    )
    feature_extractor = PI05PrefixFeatureExtractor(policy)
    first_resource = resources.get(0)
    first_raw = first_resource.loader.next()
    first_proc = (
        _move_batch_to_device(first_raw, cfg.train.device)
        if first_resource.preprocessed
        else _prepare_batch(first_raw, first_resource.preprocessor)
    )
    with torch.no_grad(), _adapter_disable_context(policy):
        first_context = feature_extractor(first_proc)
    first_resource.loader.reset()

    router = build_hard_top1_router_from_config(cfg, context_dim=first_context.shape[-1]).to(first_context.device)
    router_model: nn.Module = router
    if dist_info.is_distributed:
        device_ids = [dist_info.local_rank] if cfg.train.device.startswith("cuda") and torch.cuda.is_available() else None
        router_model = DistributedDataParallel(router_model, device_ids=device_ids)
    router_optimizer = torch.optim.AdamW(_unwrap(router_model).parameters(), lr=cfg.train.router_lr)
    metrics_path = run_dir / "router_metrics.jsonl"

    total_router_updates = router_steps_per_channel * len(cfg.channels)
    channel_steps = [0 for _ in cfg.channels]
    prefetcher = (
        RouterScheduledBatchPrefetcher(
            resources=resources,
            cfg=cfg,
            start_step=1,
            total_steps=total_router_updates,
            channel_cursor=0,
            channel_steps=channel_steps,
            steps_per_channel=router_steps_per_channel,
            prefetch_batches=router_prefetch_batches,
            prefetch_workers=router_prefetch_workers,
            device=cfg.train.device,
        )
        if router_prefetch_batches > 0
        else None
    )
    channel_cursor = 0
    rendezvous_dir = _reset_router_rendezvous_dir(run_dir, "router", dist_info)
    try:
        for router_step in range(1, total_router_updates + 1):
            if prefetcher is None:
                channel_index, channel_cursor = _next_channel(
                    global_step=router_step,
                    channel_cursor=channel_cursor,
                    channel_steps=channel_steps,
                    steps_per_channel=router_steps_per_channel,
                )
                if channel_index is None:
                    break
                resource = resources.get(channel_index)
                raw_batch = resource.loader.next()
                proc_batch = (
                    _move_batch_to_device(raw_batch, cfg.train.device)
                    if resource.preprocessed
                    else _prepare_batch(raw_batch, resource.preprocessor)
                )
            else:
                scheduled = prefetcher.next()
                if scheduled is None:
                    break
                router_step, channel_index, proc_batch = scheduled
                channel_cursor = (channel_index + 1) % len(channel_steps)

            channel = cfg.channels[channel_index]
            _router_step_cpu_rendezvous(
                rendezvous_dir,
                step=router_step,
                dist_info=dist_info,
                timeout_seconds=router_rendezvous_timeout_seconds,
                warn_seconds=router_rendezvous_warn_seconds,
            )
            router_optimizer.zero_grad(set_to_none=True)
            router_model.train()
            with torch.no_grad(), _adapter_disable_context(policy):
                pooled_context = feature_extractor(proc_batch)
            router_out = router_model(pooled_context.detach(), proc_batch["observation.state"].to(pooled_context.device))
            labels = _scheduled_router_labels(router_out.logits, channel_index)
            router_ce_loss = F.cross_entropy(router_out.logits, labels)
            router_ce_loss.backward()
            router_optimizer.step()
            channel_steps[channel_index] += 1

            should_save = router_step % cfg.train.save_every == 0 or router_step == total_router_updates
            should_record = (
                router_step == 1
                or router_step % router_metrics_every == 0
                or should_save
            )
            if should_record:
                with torch.no_grad():
                    pred = torch.argmax(router_out.logits, dim=-1)
                    router_accuracy = (pred == labels).float().mean().item()
                record = {
                    "kind": "router_train",
                    "step": router_step,
                    "channel_step": channel_steps[channel_index],
                    "channel_id": channel.channel_id,
                    "skill_id": channel.skill_id,
                    "router_ce_loss": reduce_mean_scalar(
                        float(router_ce_loss.detach().cpu().item()),
                        device=pooled_context.device,
                        info=dist_info,
                    ),
                    "router_accuracy": reduce_mean_scalar(router_accuracy, device=pooled_context.device, info=dist_info),
                    "timestamp": utc_now_iso(),
                }
                if dist_info.is_rank0:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
                    if router_step == 1 or router_step % cfg.train.log_every == 0 or router_step == total_router_updates:
                        print(json.dumps(record, ensure_ascii=False), flush=True)

            if should_save:
                _save_router_checkpoint(
                    run_dir=run_dir,
                    cfg=cfg,
                    router=router_model,
                    router_optimizer=router_optimizer,
                    router_step=router_step,
                    channel_steps=channel_steps,
                    channel_cursor=channel_cursor,
                    dist_info=dist_info,
                    final=router_step == total_router_updates,
                )
    finally:
        if prefetcher is not None:
            prefetcher.close()


def run_lora_control_router_phase(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    dist_info: DistributedInfo,
    policy,
    router_steps_per_channel: int,
    cache_root: Path | None,
    require_policy_cache: bool,
    cache_image_storage_dtype: str,
    adapter_name: str,
    checkpoint_dir: Path | None,
    router_num_workers: int,
    router_prefetch_batches: int,
    router_prefetch_workers: int,
    router_metrics_every: int,
    router_rendezvous_timeout_seconds: float | None,
    router_rendezvous_warn_seconds: float,
) -> None:
    if adapter_name in cfg.channel_ids:
        raise ValueError(f"`{adapter_name}` is reserved for router control and cannot also be a skill channel.")
    if checkpoint_dir is None and (run_dir / adapter_name).exists() and _router_control_head_path(run_dir, adapter_name).exists():
        checkpoint_dir = run_dir
    checkpoint_adapter_dir, checkpoint_head_path, checkpoint_state_path = _resolve_router_control_checkpoint_paths(
        checkpoint_dir,
        adapter_name=adapter_name,
    )
    if policy is None:
        policy = load_first_skill_policy(cfg)
    policy = load_or_initialize_router_control_adapter(
        cfg,
        policy,
        adapter_name=adapter_name,
        adapter_dir=checkpoint_adapter_dir,
        is_trainable=True,
        remove_existing_adapters=True,
    )
    activate_router_control_adapter(policy, adapter_name=adapter_name, trainable=True)

    resources = LazyChannelResources(
        cfg,
        # Router DDP is synchronized by gradients, not by data ownership. Some
        # small skills do not have enough cached samples to shard across every
        # rank, so each rank keeps a full cycling stream instead of an empty
        # per-rank shard.
        dist_info=None,
        cache_root=cache_root,
        require_policy_cache=require_policy_cache,
        cache_image_storage_dtype=cache_image_storage_dtype,
        num_workers=router_num_workers,
        persistent_workers=False,
    )
    first_resource = resources.get(0)
    first_raw = first_resource.loader.next()
    first_proc = (
        _move_batch_to_device(first_raw, cfg.train.device)
        if first_resource.preprocessed
        else _prepare_batch(first_raw, first_resource.preprocessor)
    )
    first_feature_extractor = PI05PrefixFeatureExtractor(policy, disable_adapters=False, no_grad=True)
    _activate_adapter_without_grad_toggle(policy, adapter_name)
    first_context = first_feature_extractor(first_proc)
    first_resource.loader.reset()

    router_head = build_lora_control_router_from_config(cfg, context_dim=first_context.shape[-1]).to(
        first_context.device
    )
    if checkpoint_head_path is not None:
        router_head.load_state_dict(torch.load(checkpoint_head_path, map_location=first_context.device))

    wrapper: nn.Module = LoRAControlRouterWrapper(policy=policy, router_head=router_head, adapter_name=adapter_name)
    if dist_info.is_distributed:
        device_ids = [dist_info.local_rank] if cfg.train.device.startswith("cuda") and torch.cuda.is_available() else None
        wrapper = DistributedDataParallel(wrapper, device_ids=device_ids, **_ddp_static_graph_kwargs())
    optimizer = torch.optim.AdamW(list(iter_trainable_parameters(_unwrap(wrapper))), lr=cfg.train.router_lr)

    total_router_updates = router_steps_per_channel * len(cfg.channels)
    channel_steps = [0 for _ in cfg.channels]
    start_step = 1
    channel_cursor = 0
    if checkpoint_state_path is not None:
        state = torch.load(checkpoint_state_path, map_location=first_context.device)
        saved_channel_steps = state.get("channel_steps")
        if isinstance(saved_channel_steps, list) and len(saved_channel_steps) == len(channel_steps):
            channel_steps = [int(value) for value in saved_channel_steps]
        if "channel_cursor" in state:
            channel_cursor = int(state["channel_cursor"]) % len(channel_steps)
        else:
            channel_cursor = _derive_channel_cursor(channel_steps, steps_per_channel=router_steps_per_channel)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0)) + 1

    metrics_path = run_dir / f"{adapter_name}_metrics.jsonl"
    rendezvous_dir = _reset_router_rendezvous_dir(run_dir, adapter_name, dist_info)
    prefetcher = (
        RouterScheduledBatchPrefetcher(
            resources=resources,
            cfg=cfg,
            start_step=start_step,
            total_steps=total_router_updates,
            channel_cursor=channel_cursor,
            channel_steps=channel_steps,
            steps_per_channel=router_steps_per_channel,
            prefetch_batches=router_prefetch_batches,
            prefetch_workers=router_prefetch_workers,
            device=cfg.train.device,
        )
        if router_prefetch_batches > 0 and start_step <= total_router_updates
        else None
    )
    try:
        for router_step in range(start_step, total_router_updates + 1):
            if prefetcher is None:
                channel_index, channel_cursor = _next_channel(
                    global_step=router_step,
                    channel_cursor=channel_cursor,
                    channel_steps=channel_steps,
                    steps_per_channel=router_steps_per_channel,
                )
                if channel_index is None:
                    break
                resource = resources.get(channel_index)
                raw_batch = resource.loader.next()
                proc_batch = (
                    _move_batch_to_device(raw_batch, cfg.train.device)
                    if resource.preprocessed
                    else _prepare_batch(raw_batch, resource.preprocessor)
                )
            else:
                scheduled = prefetcher.next()
                if scheduled is None:
                    break
                router_step, channel_index, proc_batch = scheduled
                channel_cursor = (channel_index + 1) % len(channel_steps)

            channel = cfg.channels[channel_index]
            _router_step_cpu_rendezvous(
                rendezvous_dir,
                step=router_step,
                dist_info=dist_info,
                timeout_seconds=router_rendezvous_timeout_seconds,
                warn_seconds=router_rendezvous_warn_seconds,
            )
            optimizer.zero_grad(set_to_none=True)
            wrapper.train()
            router_out = wrapper(proc_batch)
            labels = _scheduled_router_labels(router_out.logits, channel_index)
            router_ce_loss = F.cross_entropy(router_out.logits, labels)
            router_ce_loss.backward()
            optimizer.step()
            channel_steps[channel_index] += 1

            should_save = router_step % cfg.train.save_every == 0 or router_step == total_router_updates
            should_record = (
                router_step == 1
                or router_step % router_metrics_every == 0
                or should_save
            )
            if should_record:
                with torch.no_grad():
                    pred = torch.argmax(router_out.logits, dim=-1)
                    router_accuracy = (pred == labels).float().mean().item()
                record = {
                    "kind": "router_control_train",
                    "step": router_step,
                    "channel_step": channel_steps[channel_index],
                    "channel_id": channel.channel_id,
                    "skill_id": channel.skill_id,
                    "router_ce_loss": reduce_mean_scalar(
                        float(router_ce_loss.detach().cpu().item()),
                        device=router_out.logits.device,
                        info=dist_info,
                    ),
                    "router_accuracy": reduce_mean_scalar(router_accuracy, device=router_out.logits.device, info=dist_info),
                    "adapter_name": adapter_name,
                    "timestamp": utc_now_iso(),
                }
                if dist_info.is_rank0:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
                    if router_step == 1 or router_step % cfg.train.log_every == 0 or router_step == total_router_updates:
                        print(json.dumps(record, ensure_ascii=False), flush=True)

            if should_save:
                _save_lora_control_router_checkpoint(
                    run_dir=run_dir,
                    cfg=cfg,
                    wrapper=wrapper,
                    optimizer=optimizer,
                    router_step=router_step,
                    channel_steps=channel_steps,
                    channel_cursor=channel_cursor,
                    dist_info=dist_info,
                    adapter_name=adapter_name,
                    final=router_step == total_router_updates,
                )
    finally:
        if prefetcher is not None:
            prefetcher.close()

    if start_step > total_router_updates and dist_info.is_rank0:
        print(
            json.dumps(
                {
                    "kind": "router_control_train",
                    "status": "already_complete",
                    "adapter_name": adapter_name,
                    "total_router_updates": total_router_updates,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.torch_num_threads is not None:
        if args.torch_num_threads <= 0:
            raise SystemExit("--torch-num-threads must be positive.")
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        if args.torch_num_interop_threads <= 0:
            raise SystemExit("--torch-num-interop-threads must be positive.")
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    cfg = load_experiment_config(args.config)
    train_cfg = cfg.train
    if args.batch_size is not None:
        train_cfg = replace(train_cfg, batch_size=args.batch_size)
    if args.num_workers is not None:
        train_cfg = replace(train_cfg, num_workers=args.num_workers)
    if args.save_every is not None:
        if args.save_every <= 0:
            raise SystemExit("--save-every must be positive.")
        train_cfg = replace(train_cfg, save_every=args.save_every)
    cfg = replace(cfg, train=train_cfg)
    if args.router_batch_size is not None and args.router_batch_size <= 0:
        raise SystemExit("--router-batch-size must be positive.")
    if args.prefetch_batches < 0:
        raise SystemExit("--prefetch-batches must be non-negative.")
    router_prefetch_batches = args.router_prefetch_batches
    if router_prefetch_batches is None:
        router_prefetch_batches = args.prefetch_batches
    if router_prefetch_batches < 0:
        raise SystemExit("--router-prefetch-batches must be non-negative.")
    if args.router_prefetch_workers <= 0:
        raise SystemExit("--router-prefetch-workers must be positive.")
    if args.router_num_workers < 0:
        raise SystemExit("--router-num-workers must be non-negative.")
    if args.router_metrics_every <= 0:
        raise SystemExit("--router-metrics-every must be positive.")
    if args.router_rendezvous_timeout_seconds < 0:
        raise SystemExit("--router-rendezvous-timeout-seconds must be non-negative.")
    if args.router_rendezvous_warn_seconds < 0:
        raise SystemExit("--router-rendezvous-warn-seconds must be non-negative.")
    if args.rank_startup_stagger_seconds < 0:
        raise SystemExit("--rank-startup-stagger-seconds must be non-negative.")
    router_rendezvous_timeout_seconds = (
        None if args.router_rendezvous_timeout_seconds <= 0 else args.router_rendezvous_timeout_seconds
    )
    if args.require_policy_cache and args.cache_root is None:
        raise SystemExit("--require-policy-cache requires --cache-root.")
    if args.router_control_adapter in cfg.channel_ids:
        raise SystemExit(
            f"--router-control-adapter {args.router_control_adapter!r} conflicts with a skill channel id."
        )
    steps_per_channel = _resolve_steps_per_channel(args, cfg)
    if steps_per_channel is None:
        raise SystemExit("Sharded training requires --steps-per-channel or train.steps_per_channel.")
    _validate_long_run(args, cfg, steps_per_channel)

    dist_info = init_distributed(backend=cfg.train.distributed_backend, device_hint=cfg.train.device)
    if dist_info.is_distributed and cfg.train.device.startswith("cuda"):
        cfg = replace(cfg, train=replace(cfg.train, device=f"cuda:{dist_info.local_rank}"))
    router_cfg = cfg
    if args.router_batch_size is not None:
        router_cfg = replace(cfg, train=replace(cfg.train, batch_size=args.router_batch_size))
    gradient_accumulation_steps = resolve_gradient_accumulation_steps(
        args.gradient_accumulation_steps,
        dist_info.world_size,
    )
    router_steps_per_channel = args.router_steps_per_channel
    if router_steps_per_channel is None:
        router_steps_per_channel = steps_per_channel
    if router_steps_per_channel <= 0:
        raise SystemExit("--router-steps-per-channel must be positive.")

    run_name = args.run_name or cfg.run_name
    if dist_info.is_distributed:
        if dist_info.is_rank0:
            run_name = run_name or timestamp_run_name("router_lora_sharded")
        run_name = broadcast_object(run_name, dist_info)
    else:
        run_name = run_name or timestamp_run_name("router_lora_sharded")
    run_dir = cfg.output_root / "runs" / run_name
    progress_path = run_dir / "adapter_progress.json"

    try:
        if dist_info.is_rank0:
            ensure_dir(run_dir)
            write_yaml(
                run_dir / "resolved_config.yaml",
                _resolved_config_payload(
                    config_path=args.config,
                    cfg=cfg,
                    steps_per_channel=steps_per_channel,
                    router_steps_per_channel=None if args.skip_router_phase else router_steps_per_channel,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    prefetch_batches=args.prefetch_batches,
                    router_prefetch_batches=None if args.skip_router_phase else router_prefetch_batches,
                    router_prefetch_workers=None if args.skip_router_phase else args.router_prefetch_workers,
                    router_num_workers=None if args.skip_router_phase else args.router_num_workers,
                    router_metrics_every=None if args.skip_router_phase else args.router_metrics_every,
                    router_batch_size=None if args.skip_router_phase else args.router_batch_size,
                    router_rendezvous_timeout_seconds=None
                    if args.skip_router_phase
                    else router_rendezvous_timeout_seconds,
                    router_rendezvous_warn_seconds=None
                    if args.skip_router_phase
                    else args.router_rendezvous_warn_seconds,
                    cache_root=args.cache_root,
                    require_policy_cache=args.require_policy_cache,
                    cache_image_storage_dtype=args.cache_image_storage_dtype,
                    router_impl=args.router_impl,
                    router_control_adapter=args.router_control_adapter,
                    router_control_checkpoint=args.router_control_checkpoint,
                    torch_num_threads=args.torch_num_threads,
                    torch_num_interop_threads=args.torch_num_interop_threads,
                    rank_startup_stagger_seconds=args.rank_startup_stagger_seconds,
                    dist_info=dist_info,
                ),
            )
            if not args.skip_adapter_phase:
                ensure_adapter_progress(progress_path, cfg, steps_per_channel=steps_per_channel)
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "mode": "skill_sharded",
                        "world_size": dist_info.world_size,
                        "steps_per_channel": steps_per_channel,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "prefetch_batches": args.prefetch_batches,
                        "router_prefetch_batches": None if args.skip_router_phase else router_prefetch_batches,
                        "router_prefetch_workers": None if args.skip_router_phase else args.router_prefetch_workers,
                        "router_num_workers": None if args.skip_router_phase else args.router_num_workers,
                        "router_metrics_every": None if args.skip_router_phase else args.router_metrics_every,
                        "router_batch_size": None if args.skip_router_phase else args.router_batch_size,
                        "router_rendezvous_timeout_seconds": None
                        if args.skip_router_phase
                        else router_rendezvous_timeout_seconds,
                        "router_rendezvous_warn_seconds": None
                        if args.skip_router_phase
                        else args.router_rendezvous_warn_seconds,
                        "cache_root": str(args.cache_root) if args.cache_root is not None else None,
                        "require_policy_cache": args.require_policy_cache,
                        "cache_image_storage_dtype": args.cache_image_storage_dtype,
                        "router_impl": args.router_impl,
                        "router_control_adapter": args.router_control_adapter,
                        "router_control_checkpoint": str(args.router_control_checkpoint)
                        if args.router_control_checkpoint
                        else None,
                        "torch_num_threads": args.torch_num_threads,
                        "torch_num_interop_threads": args.torch_num_interop_threads,
                        "rank_startup_stagger_seconds": args.rank_startup_stagger_seconds,
                        "adapter_effective_global_batch_size": cfg.train.batch_size * gradient_accumulation_steps,
                        "router_effective_global_batch_size": None
                        if args.skip_router_phase
                        else effective_global_batch_size(router_cfg.train.batch_size, dist_info.world_size),
                        "router_steps_per_channel": None if args.skip_router_phase else router_steps_per_channel,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        barrier(dist_info)

        torch.manual_seed(cfg.train.seed + dist_info.rank)
        random.seed(cfg.train.seed + dist_info.rank)
        np.random.seed(cfg.train.seed + dist_info.rank)
        _stagger_rank_startup(dist_info, args.rank_startup_stagger_seconds)

        policy = None
        if not args.skip_adapter_phase:
            policy = run_adapter_phase(
                cfg=cfg,
                run_dir=run_dir,
                progress_path=progress_path,
                dist_info=dist_info,
                steps_per_channel=steps_per_channel,
                gradient_accumulation_steps=gradient_accumulation_steps,
                prefetch_batches=args.prefetch_batches,
                cache_root=args.cache_root,
                require_policy_cache=args.require_policy_cache,
                cache_image_storage_dtype=args.cache_image_storage_dtype,
            )
            adapter_counts = wait_for_adapter_progress_complete(
                progress_path,
                poll_seconds=args.adapter_completion_poll_seconds,
            )
            if dist_info.is_rank0:
                print(
                    json.dumps(
                        {
                            "kind": "adapter_phase_complete",
                            "adapter_progress": adapter_counts,
                            "timestamp": utc_now_iso(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if not args.skip_router_phase:
            if args.router_impl == ROUTER_IMPL_LORA_CONTROL:
                run_lora_control_router_phase(
                    cfg=router_cfg,
                    run_dir=run_dir,
                    dist_info=dist_info,
                    policy=policy,
                    router_steps_per_channel=router_steps_per_channel,
                    cache_root=args.cache_root,
                    require_policy_cache=args.require_policy_cache,
                    cache_image_storage_dtype=args.cache_image_storage_dtype,
                    adapter_name=args.router_control_adapter,
                    checkpoint_dir=args.router_control_checkpoint,
                    router_num_workers=args.router_num_workers,
                    router_prefetch_batches=router_prefetch_batches,
                    router_prefetch_workers=args.router_prefetch_workers,
                    router_metrics_every=args.router_metrics_every,
                    router_rendezvous_timeout_seconds=router_rendezvous_timeout_seconds,
                    router_rendezvous_warn_seconds=args.router_rendezvous_warn_seconds,
                )
            else:
                run_hard_top1_router_phase(
                    cfg=router_cfg,
                    run_dir=run_dir,
                    dist_info=dist_info,
                    policy=policy,
                    router_steps_per_channel=router_steps_per_channel,
                    cache_root=args.cache_root,
                    require_policy_cache=args.require_policy_cache,
                    cache_image_storage_dtype=args.cache_image_storage_dtype,
                    router_num_workers=args.router_num_workers,
                    router_prefetch_batches=router_prefetch_batches,
                    router_prefetch_workers=args.router_prefetch_workers,
                    router_metrics_every=args.router_metrics_every,
                    router_rendezvous_timeout_seconds=router_rendezvous_timeout_seconds,
                    router_rendezvous_warn_seconds=args.router_rendezvous_warn_seconds,
                )
        barrier(dist_info)

        if dist_info.is_rank0:
            adapter_counts = progress_summary(progress_path) if progress_path.is_file() else {}
            write_json(
                run_dir / "summary.json",
                {
                    "run_dir": str(run_dir),
                    "completed_at": utc_now_iso(),
                    "mode": "skill_sharded",
                    "adapter_progress": adapter_counts,
                    "router_completed": not args.skip_router_phase,
                    "steps_per_channel": steps_per_channel,
                    "router_steps_per_channel": None if args.skip_router_phase else router_steps_per_channel,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "prefetch_batches": args.prefetch_batches,
                    "router_prefetch_batches": None if args.skip_router_phase else router_prefetch_batches,
                    "router_prefetch_workers": None if args.skip_router_phase else args.router_prefetch_workers,
                    "router_num_workers": None if args.skip_router_phase else args.router_num_workers,
                    "cache_root": str(args.cache_root) if args.cache_root is not None else None,
                    "require_policy_cache": args.require_policy_cache,
                    "cache_image_storage_dtype": args.cache_image_storage_dtype,
                    "router_impl": args.router_impl,
                    "router_control_adapter": args.router_control_adapter,
                    "world_size": dist_info.world_size,
                },
            )
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
