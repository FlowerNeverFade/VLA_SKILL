from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .adapters import set_module_requires_grad


Batch = dict[str, Any]
PolicyLossFn = Callable[[Any, Batch], torch.Tensor]
ActivateChannelFn = Callable[[int], None]


@dataclass(frozen=True)
class TrainStepOutput:
    loss: float
    policy_loss: float
    router_ce_loss: float
    router_accuracy: float


class RouterLoraTrainer:
    def __init__(
        self,
        *,
        policy: Any,
        router: torch.nn.Module,
        feature_extractor: Callable[[Batch], torch.Tensor],
        policy_optimizer: torch.optim.Optimizer,
        router_optimizer: torch.optim.Optimizer,
        policy_loss_fn: PolicyLossFn,
        activate_channel: ActivateChannelFn,
        router_ce_weight: float = 1.0,
    ):
        self.policy = policy
        self.router = router
        self.feature_extractor = feature_extractor
        self.policy_optimizer = policy_optimizer
        self.router_optimizer = router_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.activate_channel = activate_channel
        self.router_ce_weight = router_ce_weight

    def train_step(self, batch: Batch) -> TrainStepOutput:
        labels = batch["channel_index"]
        if labels.ndim == 0:
            labels = labels[None]
        labels = labels.to(next(self.router.parameters()).device, dtype=torch.long)

        self.policy_optimizer.zero_grad(set_to_none=True)
        self.router_optimizer.zero_grad(set_to_none=True)

        set_module_requires_grad(self.router, True)
        with torch.no_grad():
            pooled_context = self.feature_extractor(batch)
        router_out = self.router(
            pooled_context.detach(),
            batch["observation.state"].to(device=pooled_context.device),
        )
        router_ce_loss = F.cross_entropy(router_out.logits, labels)

        unique_labels = torch.unique(labels.detach().cpu())
        if unique_labels.numel() != 1:
            raise ValueError("v1 training expects each policy batch to contain exactly one channel.")
        gt_channel = int(unique_labels.item())
        self.activate_channel(gt_channel)
        policy_loss = self.policy_loss_fn(self.policy, batch)

        loss = policy_loss + self.router_ce_weight * router_ce_loss
        loss.backward()
        self.policy_optimizer.step()
        self.router_optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(router_out.logits, dim=-1)
            accuracy = (pred == labels).float().mean().item()
        return TrainStepOutput(
            loss=float(loss.detach().cpu().item()),
            policy_loss=float(policy_loss.detach().cpu().item()),
            router_ce_loss=float(router_ce_loss.detach().cpu().item()),
            router_accuracy=float(accuracy),
        )


def default_policy_loss_fn(policy: Any, batch: Batch) -> torch.Tensor:
    output = policy(batch)
    if isinstance(output, tuple):
        return output[0]
    return output
