from __future__ import annotations

from pathlib import Path

import torch

from train_router_lora import _save_checkpoint
from vla_skill_router.config import ChannelConfig, ExperimentConfig, TrainConfig
from vla_skill_router.distributed import DistributedInfo
from vla_skill_router.real_runtime import make_router_shell_skill_spec
from vla_skill_router.router_model import HardTop1Router
from vla_skill_router.schedule import all_channels_complete, next_round_robin_channel


def test_round_robin_step_counter_stops_completed_channels() -> None:
    counts = [2, 0, 1]

    assert next_round_robin_channel(counts, target_steps=2, start_index=0) == 1
    counts[1] = 2
    assert next_round_robin_channel(counts, target_steps=2, start_index=0) == 2
    counts[2] = 2
    assert all_channels_complete(counts, target_steps=2)
    assert next_round_robin_channel(counts, target_steps=2, start_index=0) is None


class FakePolicy:
    def __init__(self) -> None:
        self.selected: list[list[str] | None] = []

    def set_adapter(self, adapter_name: str) -> None:
        self.current_adapter = adapter_name

    def save_pretrained(self, path: str, **kwargs) -> None:
        self.selected.append(kwargs.get("selected_adapters"))
        Path(path).mkdir(parents=True, exist_ok=True)


def test_checkpoint_saves_each_adapter_with_selected_adapters(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        channels=(
            ChannelConfig(channel_id="a", skill_id="skill_a"),
            ChannelConfig(channel_id="b", skill_id="skill_b"),
        )
    )
    router = HardTop1Router(context_dim=2, num_channels=2, max_state_dim=2, state_embed_dim=2, hidden_dim=4)
    router_optimizer = torch.optim.SGD(router.parameters(), lr=0.1)
    adapter_param = torch.nn.Parameter(torch.tensor([1.0]))
    adapter_optimizer = torch.optim.SGD([adapter_param], lr=0.1)
    policy = FakePolicy()

    _save_checkpoint(
        run_dir=tmp_path,
        cfg=cfg,
        policy=policy,
        router=router,
        step=3,
        channel_steps=[2, 1],
        policy_optimizer=adapter_optimizer,
        router_optimizer=router_optimizer,
        dist_info=DistributedInfo(),
    )

    assert policy.selected == [["a"], ["b"]]
    assert (tmp_path / "router.pt").is_file()
    assert (tmp_path / "router_meta.json").is_file()


def test_checkpoint_is_rank0_only(tmp_path: Path) -> None:
    cfg = ExperimentConfig(channels=(ChannelConfig(channel_id="a", skill_id="skill_a"),))
    router = HardTop1Router(context_dim=2, num_channels=1, max_state_dim=2, state_embed_dim=2, hidden_dim=4)
    router_optimizer = torch.optim.SGD(router.parameters(), lr=0.1)
    adapter_param = torch.nn.Parameter(torch.tensor([1.0]))
    adapter_optimizer = torch.optim.SGD([adapter_param], lr=0.1)
    policy = FakePolicy()

    _save_checkpoint(
        run_dir=tmp_path,
        cfg=cfg,
        policy=policy,
        router=router,
        step=1,
        channel_steps=[1],
        policy_optimizer=adapter_optimizer,
        router_optimizer=router_optimizer,
        dist_info=DistributedInfo(rank=1, local_rank=1, world_size=2, backend="gloo"),
    )

    assert policy.selected == []
    assert not (tmp_path / "router.pt").exists()


def test_router_shell_spec_uses_max_state_and_action_dims(tmp_path: Path, monkeypatch) -> None:
    from vla_skill.schema import SkillSpec
    import vla_skill_router.real_runtime as runtime

    specs = {
        "small": SkillSpec(skill_id="small", display_name="small", state_dim=6, action_dim=6),
        "large": SkillSpec(skill_id="large", display_name="large", state_dim=16, action_dim=12),
    }
    monkeypatch.setattr(runtime, "load_skill_spec", lambda _root, skill_id: specs[skill_id])
    cfg = ExperimentConfig(
        channels=(
            ChannelConfig(channel_id="small", skill_id="small"),
            ChannelConfig(channel_id="large", skill_id="large"),
        ),
        train=TrainConfig(steps=1),
    )

    shell = make_router_shell_skill_spec(cfg)

    assert shell.state_dim == 16
    assert shell.action_dim == 12
