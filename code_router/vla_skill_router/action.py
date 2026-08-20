from __future__ import annotations

import torch


def pad_state(state: torch.Tensor, max_state_dim: int) -> torch.Tensor:
    if state.ndim != 2:
        raise ValueError(f"state must have shape [B, D], got {tuple(state.shape)}.")
    if state.shape[-1] > max_state_dim:
        raise ValueError(f"state dim {state.shape[-1]} exceeds max_state_dim={max_state_dim}.")
    if state.shape[-1] == max_state_dim:
        return state
    pad_width = max_state_dim - state.shape[-1]
    return torch.nn.functional.pad(state, (0, pad_width))


def masked_action_mse(pred: torch.Tensor, target: torch.Tensor, action_dims: torch.Tensor | int) -> torch.Tensor:
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(f"pred and target must be [B, T, D], got {tuple(pred.shape)} and {tuple(target.shape)}.")
    if pred.shape[:2] != target.shape[:2]:
        raise ValueError(f"pred and target batch/time dims differ: {tuple(pred.shape)} vs {tuple(target.shape)}.")
    if pred.shape[-1] < target.shape[-1]:
        raise ValueError(f"pred action dim {pred.shape[-1]} is smaller than target dim {target.shape[-1]}.")

    device = pred.device
    if isinstance(action_dims, int):
        dims = torch.full((pred.shape[0],), action_dims, dtype=torch.long, device=device)
    else:
        dims = action_dims.to(device=device, dtype=torch.long)
    if dims.ndim != 1 or dims.shape[0] != pred.shape[0]:
        raise ValueError(f"action_dims must be scalar int or [B], got shape={tuple(dims.shape)}.")
    if torch.any(dims <= 0):
        raise ValueError("action_dims must be positive.")
    if torch.any(dims > pred.shape[-1]):
        raise ValueError("action_dims cannot exceed pred action dimension.")

    max_target_dim = target.shape[-1]
    if max_target_dim < pred.shape[-1]:
        target = torch.nn.functional.pad(target, (0, pred.shape[-1] - max_target_dim))
    dim_ids = torch.arange(pred.shape[-1], device=device)[None, None, :]
    mask = dim_ids < dims[:, None, None]
    loss_sum = ((pred - target) ** 2 * mask).sum()
    denom = mask.sum().clamp_min(1) * pred.shape[1]
    return loss_sum / denom


def crop_action(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action_dim <= 0:
        raise ValueError("action_dim must be positive.")
    if action.shape[-1] < action_dim:
        raise ValueError(f"Cannot crop action dim {action_dim} from tensor with dim {action.shape[-1]}.")
    return action[..., :action_dim]
