from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _default_backend(device_hint: str) -> str:
    if device_hint.startswith("cuda") and torch.cuda.is_available():
        return "nccl"
    return "gloo"


def resolve_backend(backend: str | None = None, *, device_hint: str = "cuda") -> str:
    if backend:
        return backend
    return _default_backend(device_hint)


def read_distributed_info(*, backend: str | None = None, device_hint: str = "cuda") -> DistributedInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_backend = resolve_backend(backend, device_hint=device_hint)
    return DistributedInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=resolved_backend if world_size > 1 else "none",
    )


def init_distributed(*, backend: str | None = None, device_hint: str = "cuda") -> DistributedInfo:
    info = read_distributed_info(backend=backend, device_hint=device_hint)
    if not info.is_distributed:
        return info
    if info.backend == "nccl" and device_hint.startswith("cuda"):
        torch.cuda.set_device(info.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=info.backend)
    return info


def cleanup_distributed(info: DistributedInfo) -> None:
    if info.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def effective_global_batch_size(per_device_batch_size: int, world_size: int) -> int:
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    return per_device_batch_size * world_size


def reduce_mean_scalar(value: float, *, device: torch.device | str, info: DistributedInfo) -> float:
    if not info.is_distributed:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(info.world_size)
    return float(tensor.detach().cpu().item())


def broadcast_object(value, info: DistributedInfo, *, src: int = 0):
    if not info.is_distributed:
        return value
    payload = [value if info.rank == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def barrier(info: DistributedInfo) -> None:
    if info.is_distributed:
        dist.barrier()
