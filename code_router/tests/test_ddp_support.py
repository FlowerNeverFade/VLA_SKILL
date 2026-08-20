from __future__ import annotations

import torch
from torch import nn

import train_router_lora as train_entry
from train_router_lora import (
    PolicyLossWrapper,
    _activate_adapter_without_grad_toggle,
    _next_channel,
    _resolve_steps_per_channel,
)
from vla_skill_router.config import ChannelConfig, ExperimentConfig, TrainConfig
from vla_skill_router.distributed import (
    DistributedInfo,
    broadcast_object,
    effective_global_batch_size,
    read_distributed_info,
    resolve_backend,
)


def test_read_distributed_info_single_rank(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    info = read_distributed_info(backend="gloo", device_hint="cpu")

    assert not info.is_distributed
    assert info.is_rank0
    assert info.world_size == 1
    assert info.backend == "none"


def test_read_distributed_info_multi_rank(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    info = read_distributed_info(backend="gloo", device_hint="cpu")

    assert info.is_distributed
    assert not info.is_rank0
    assert info.rank == 2
    assert info.local_rank == 1
    assert info.world_size == 4
    assert info.backend == "gloo"


def test_backend_and_batch_metadata() -> None:
    assert resolve_backend(None, device_hint="cpu") == "gloo"
    assert effective_global_batch_size(per_device_batch_size=8, world_size=4) == 32
    assert broadcast_object("run_a", DistributedInfo()) == "run_a"


def test_steps_per_channel_cli_overrides_config() -> None:
    class Args:
        steps_per_channel = 7

    cfg = ExperimentConfig(
        channels=(ChannelConfig(channel_id="a", skill_id="skill_a"),),
        train=TrainConfig(steps=100, steps_per_channel=3),
    )

    assert _resolve_steps_per_channel(Args(), cfg) == 7


def test_ddp_channel_schedule_is_rank_independent() -> None:
    counts = [1, 0, 1]

    rank0_choice = _next_channel(global_step=2, channel_cursor=1, channel_steps=counts.copy(), steps_per_channel=2)
    rank1_choice = _next_channel(global_step=2, channel_cursor=1, channel_steps=counts.copy(), steps_per_channel=2)

    assert rank0_choice == rank1_choice == (1, 2)


class FakeAdapterLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._active_adapter = ["old"]


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = FakeAdapterLayer()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch):
        return self.weight * float(batch["scale"])


def test_activate_adapter_without_grad_toggle_preserves_requires_grad() -> None:
    policy = FakePolicy()
    policy.weight.requires_grad_(True)

    _activate_adapter_without_grad_toggle(policy, "skill_a")

    assert policy.layer._active_adapter == ["skill_a"]
    assert policy.weight.requires_grad is True


def test_policy_loss_wrapper_activates_adapter_and_returns_loss(monkeypatch) -> None:
    policy = FakePolicy()

    def fake_loss(model, batch):
        assert model.layer._active_adapter == ["skill_b"]
        return model(batch).sum()

    monkeypatch.setattr(train_entry, "pi05_masked_policy_loss", fake_loss)
    wrapper = PolicyLossWrapper(policy)

    loss = wrapper({"scale": 2.0}, "skill_b")

    assert torch.equal(loss, torch.tensor(2.0))
    assert policy.layer._active_adapter == ["skill_b"]
