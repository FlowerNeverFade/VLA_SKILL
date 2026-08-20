from __future__ import annotations

import copy

import torch
from torch import nn

from vla_skill_router.action import masked_action_mse
from vla_skill_router.features import StaticFeatureExtractor
from vla_skill_router.router_model import HardTop1Router
from vla_skill_router.trainer import RouterLoraTrainer


class ToyPolicy(nn.Module):
    def __init__(self, context_dim: int, action_dim: int, num_channels: int):
        super().__init__()
        self.base = nn.Linear(context_dim, context_dim)
        self.adapters = nn.ModuleList(nn.Linear(context_dim, action_dim) for _ in range(num_channels))
        self.active_channel = 0
        for param in self.base.parameters():
            param.requires_grad = False

    def set_active_channel(self, channel: int) -> None:
        self.active_channel = channel
        for index, adapter in enumerate(self.adapters):
            for param in adapter.parameters():
                param.requires_grad = index == channel

    def forward(self, batch):
        with torch.no_grad():
            hidden = self.base(batch["router_context"])
        pred = self.adapters[self.active_channel](hidden).unsqueeze(1)
        return masked_action_mse(pred, batch["action"], batch["action_dim"])


def _clone_params(module: nn.Module):
    return [param.detach().clone() for param in module.parameters()]


def _any_changed(before, module: nn.Module) -> bool:
    return any(not torch.allclose(old, new.detach()) for old, new in zip(before, module.parameters(), strict=True))


def _make_batch(channel_index: int, batch_size: int = 8):
    context = torch.randn(batch_size, 4)
    state = torch.randn(batch_size, 3) + float(channel_index)
    target = torch.full((batch_size, 1, 2), float(channel_index))
    return {
        "router_context": context,
        "observation.state": state,
        "action": target,
        "channel_index": torch.full((batch_size,), channel_index, dtype=torch.long),
        "action_dim": torch.full((batch_size,), 2, dtype=torch.long),
    }


def test_train_step_keeps_base_frozen_and_updates_router_and_current_adapter_only() -> None:
    torch.manual_seed(0)
    policy = ToyPolicy(context_dim=4, action_dim=2, num_channels=2)
    router = HardTop1Router(context_dim=4, num_channels=2, max_state_dim=6, state_embed_dim=3, hidden_dim=16)
    trainer = RouterLoraTrainer(
        policy=policy,
        router=router,
        feature_extractor=StaticFeatureExtractor(),
        policy_optimizer=torch.optim.SGD(policy.adapters.parameters(), lr=0.1),
        router_optimizer=torch.optim.SGD(router.parameters(), lr=0.1),
        policy_loss_fn=lambda model, batch: model(batch),
        activate_channel=policy.set_active_channel,
        router_ce_weight=1.0,
    )
    base_before = _clone_params(policy.base)
    adapter0_before = _clone_params(policy.adapters[0])
    adapter1_before = _clone_params(policy.adapters[1])
    router_before = _clone_params(router)

    output = trainer.train_step(_make_batch(channel_index=1))

    assert output.loss > 0.0
    assert not _any_changed(base_before, policy.base)
    assert not _any_changed(adapter0_before, policy.adapters[0])
    assert _any_changed(adapter1_before, policy.adapters[1])
    assert _any_changed(router_before, router)


def test_toy_router_learns_above_random() -> None:
    torch.manual_seed(7)
    policy = ToyPolicy(context_dim=4, action_dim=2, num_channels=2)
    router = HardTop1Router(context_dim=4, num_channels=2, max_state_dim=6, state_embed_dim=4, hidden_dim=16)
    trainer = RouterLoraTrainer(
        policy=policy,
        router=router,
        feature_extractor=StaticFeatureExtractor(),
        policy_optimizer=torch.optim.SGD(policy.adapters.parameters(), lr=0.05),
        router_optimizer=torch.optim.Adam(router.parameters(), lr=0.05),
        policy_loss_fn=lambda model, batch: model(batch),
        activate_channel=policy.set_active_channel,
        router_ce_weight=1.0,
    )

    for step in range(60):
        trainer.train_step(_make_batch(channel_index=step % 2, batch_size=16))

    with torch.no_grad():
        batch0 = _make_batch(channel_index=0, batch_size=32)
        batch1 = _make_batch(channel_index=1, batch_size=32)
        logits0 = router(batch0["router_context"], batch0["observation.state"]).logits
        logits1 = router(batch1["router_context"], batch1["observation.state"]).logits
        acc0 = (torch.argmax(logits0, dim=-1) == 0).float().mean()
        acc1 = (torch.argmax(logits1, dim=-1) == 1).float().mean()

    assert float((acc0 + acc1) / 2.0) > 0.70
