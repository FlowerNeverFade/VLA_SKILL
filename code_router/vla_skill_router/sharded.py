from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator

from vla_skill.io_utils import utc_now_iso

from .config import ExperimentConfig


PROGRESS_VERSION = 1


@dataclass(frozen=True)
class ChannelClaim:
    channel_index: int
    channel_id: str
    skill_id: str
    attempts: int


def resolve_gradient_accumulation_steps(value: int | None, world_size: int) -> int:
    if value is None:
        return max(1, world_size)
    if value <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    return value


def expected_micro_batches(optimizer_steps: int, gradient_accumulation_steps: int) -> int:
    if optimizer_steps < 0:
        raise ValueError("optimizer_steps must be non-negative.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    return optimizer_steps * gradient_accumulation_steps


def _initial_progress(cfg: ExperimentConfig, *, steps_per_channel: int) -> dict[str, Any]:
    return {
        "version": PROGRESS_VERSION,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "steps_per_channel": steps_per_channel,
        "channels": {
            channel.channel_id: {
                "channel_index": index,
                "channel_id": channel.channel_id,
                "skill_id": channel.skill_id,
                "status": "pending",
                "steps": 0,
                "attempts": 0,
                "owner_rank": None,
                "owner_pid": None,
                "adapter_path": None,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "updated_at": utc_now_iso(),
            }
            for index, channel in enumerate(cfg.channels)
        },
    }


def _read_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = utc_now_iso()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


@contextmanager
def locked_progress(path: Path) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        payload = _read_progress(path)
        try:
            yield payload
        finally:
            _write_progress(path, payload)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def ensure_adapter_progress(path: Path, cfg: ExperimentConfig, *, steps_per_channel: int) -> dict[str, Any]:
    with locked_progress(path) as progress:
        if not progress:
            progress.update(_initial_progress(cfg, steps_per_channel=steps_per_channel))
        else:
            progress["steps_per_channel"] = steps_per_channel
            channels = progress.get("channels") or {}
            for channel in cfg.channels:
                item = channels.get(channel.channel_id)
                if item is None:
                    continue
                if item.get("status") in {"running", "failed"}:
                    item.update(
                        {
                            "status": "pending",
                            "owner_rank": None,
                            "owner_pid": None,
                            "started_at": None,
                            "error": None,
                            "updated_at": utc_now_iso(),
                        }
                    )
        return progress


def claim_next_channel(path: Path, *, rank: int, pid: int | None = None) -> ChannelClaim | None:
    with locked_progress(path) as progress:
        channels = progress.get("channels") or {}
        for channel_id, item in sorted(channels.items(), key=lambda pair: int(pair[1]["channel_index"])):
            if item.get("status") != "pending":
                continue
            attempts = int(item.get("attempts", 0)) + 1
            item.update(
                {
                    "status": "running",
                    "attempts": attempts,
                    "owner_rank": rank,
                    "owner_pid": pid,
                    "started_at": utc_now_iso(),
                    "completed_at": None,
                    "error": None,
                    "updated_at": utc_now_iso(),
                }
            )
            return ChannelClaim(
                channel_index=int(item["channel_index"]),
                channel_id=str(channel_id),
                skill_id=str(item["skill_id"]),
                attempts=attempts,
            )
    return None


def mark_channel_completed(
    path: Path,
    *,
    channel_id: str,
    steps: int,
    adapter_path: Path,
    rank: int,
) -> None:
    with locked_progress(path) as progress:
        item = progress["channels"][channel_id]
        item.update(
            {
                "status": "completed",
                "steps": int(steps),
                "owner_rank": rank,
                "adapter_path": str(adapter_path),
                "completed_at": utc_now_iso(),
                "error": None,
                "updated_at": utc_now_iso(),
            }
        )


def mark_channel_progress(path: Path, *, channel_id: str, steps: int, rank: int) -> None:
    with locked_progress(path) as progress:
        item = progress["channels"][channel_id]
        if item.get("status") == "completed":
            return
        item.update(
            {
                "status": "running",
                "steps": int(steps),
                "owner_rank": rank,
                "updated_at": utc_now_iso(),
            }
        )


def mark_channel_failed(path: Path, *, channel_id: str, rank: int, error: str) -> None:
    with locked_progress(path) as progress:
        item = progress["channels"][channel_id]
        item.update(
            {
                "status": "failed",
                "owner_rank": rank,
                "error": error,
                "updated_at": utc_now_iso(),
            }
        )


def progress_summary(path: Path) -> dict[str, int]:
    progress = _read_progress(path)
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for item in (progress.get("channels") or {}).values():
        status = str(item.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def wait_for_adapter_progress_complete(
    path: Path,
    *,
    poll_seconds: float = 10.0,
    timeout_seconds: float | None = None,
) -> dict[str, int]:
    """Wait for all sharded adapter workers through the progress file.

    The adapter phase intentionally lets ranks train different skills
    asynchronously. A distributed barrier here can time out on the long tail
    because early ranks may wait while other ranks are still training. Polling
    the file keeps the sharded phase independent, then the caller can enter the
    synchronized router phase only after all adapters are complete.
    """
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative.")
    deadline = None if timeout_seconds is None else time.time() + timeout_seconds
    while True:
        counts = progress_summary(path)
        total = sum(counts.values())
        if counts.get("failed", 0):
            raise RuntimeError(f"Adapter phase has failed channels: {counts}")
        if total > 0 and counts.get("completed", 0) == total:
            return counts
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for adapter phase completion: {counts}")
        time.sleep(poll_seconds)
