from __future__ import annotations


def next_round_robin_channel(step_counts: list[int], target_steps: int, start_index: int = 0) -> int | None:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive.")
    if not step_counts:
        return None
    total = len(step_counts)
    for offset in range(total):
        index = (start_index + offset) % total
        if step_counts[index] < target_steps:
            return index
    return None


def all_channels_complete(step_counts: list[int], target_steps: int) -> bool:
    return all(count >= target_steps for count in step_counts)
