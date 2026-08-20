from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vla_skill_router.config import ChannelConfig, ExperimentConfig, TrainConfig
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


def _make_cfg(num_channels: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        channels=tuple(
            ChannelConfig(channel_id=f"channel_{index}", skill_id=f"skill_{index}")
            for index in range(num_channels)
        ),
        train=TrainConfig(steps=1, steps_per_channel=2),
    )


def test_sharded_queue_claims_each_channel_once(tmp_path: Path) -> None:
    cfg = _make_cfg(num_channels=5)
    progress_path = tmp_path / "adapter_progress.json"
    ensure_adapter_progress(progress_path, cfg, steps_per_channel=2)

    claims = [claim_next_channel(progress_path, rank=rank % 3, pid=100 + rank) for rank in range(5)]

    assert [claim.channel_index for claim in claims if claim is not None] == [0, 1, 2, 3, 4]
    assert claim_next_channel(progress_path, rank=0) is None
    assert progress_summary(progress_path)["running"] == 5


def test_sharded_progress_resume_keeps_completed_and_retries_running(tmp_path: Path) -> None:
    cfg = _make_cfg(num_channels=2)
    progress_path = tmp_path / "adapter_progress.json"
    ensure_adapter_progress(progress_path, cfg, steps_per_channel=2)
    first = claim_next_channel(progress_path, rank=0)
    second = claim_next_channel(progress_path, rank=1)
    assert first is not None and second is not None
    mark_channel_completed(
        progress_path,
        channel_id=first.channel_id,
        steps=2,
        adapter_path=tmp_path / "channels" / first.channel_id,
        rank=0,
    )

    ensure_adapter_progress(progress_path, cfg, steps_per_channel=2)

    counts = progress_summary(progress_path)
    assert counts["completed"] == 1
    assert counts["pending"] == 1
    retry = claim_next_channel(progress_path, rank=2)
    assert retry is not None
    assert retry.channel_id == second.channel_id


def test_gradient_accumulation_defaults_to_world_size() -> None:
    assert resolve_gradient_accumulation_steps(None, world_size=6) == 6
    assert resolve_gradient_accumulation_steps(3, world_size=6) == 3
    assert expected_micro_batches(optimizer_steps=1000, gradient_accumulation_steps=6) == 6000


def test_sharded_progress_records_running_steps(tmp_path: Path) -> None:
    cfg = _make_cfg(num_channels=1)
    progress_path = tmp_path / "adapter_progress.json"
    ensure_adapter_progress(progress_path, cfg, steps_per_channel=5)
    claim = claim_next_channel(progress_path, rank=0)
    assert claim is not None

    mark_channel_progress(progress_path, channel_id=claim.channel_id, steps=3, rank=0)

    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    item = progress["channels"][claim.channel_id]
    assert item["status"] == "running"
    assert item["steps"] == 3
    assert item["owner_rank"] == 0


def test_wait_for_adapter_progress_complete_uses_file_state(tmp_path: Path) -> None:
    cfg = _make_cfg(num_channels=2)
    progress_path = tmp_path / "adapter_progress.json"
    ensure_adapter_progress(progress_path, cfg, steps_per_channel=2)
    first = claim_next_channel(progress_path, rank=0)
    second = claim_next_channel(progress_path, rank=1)
    assert first is not None and second is not None
    mark_channel_completed(
        progress_path,
        channel_id=first.channel_id,
        steps=2,
        adapter_path=tmp_path / "channels" / first.channel_id,
        rank=0,
    )
    mark_channel_completed(
        progress_path,
        channel_id=second.channel_id,
        steps=2,
        adapter_path=tmp_path / "channels" / second.channel_id,
        rank=1,
    )

    counts = wait_for_adapter_progress_complete(progress_path, poll_seconds=0)

    assert counts["completed"] == 2


def test_wait_for_adapter_progress_complete_fails_fast(tmp_path: Path) -> None:
    cfg = _make_cfg(num_channels=1)
    progress_path = tmp_path / "adapter_progress.json"
    ensure_adapter_progress(progress_path, cfg, steps_per_channel=2)
    claim = claim_next_channel(progress_path, rank=0)
    assert claim is not None
    mark_channel_failed(progress_path, channel_id=claim.channel_id, rank=0, error="boom")

    with pytest.raises(RuntimeError, match="failed channels"):
        wait_for_adapter_progress_complete(progress_path, poll_seconds=0)


def test_single_channel_loader_deletes_previous_adapters(monkeypatch) -> None:
    import vla_skill_router.real_runtime as runtime

    class FakePeftModel(torch.nn.Module):
        def __init__(self, adapter_name: str):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.peft_config = {adapter_name: object()}
            self.active_adapter = adapter_name
            self.deleted: list[str] = []

        def parameters(self, recurse: bool = True):
            return iter([self.weight])

        def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
            return iter([(f"lora_{self.active_adapter}", self.weight)])

        def delete_adapter(self, adapter_name: str) -> None:
            self.deleted.append(adapter_name)
            self.peft_config.pop(adapter_name, None)

        def add_adapter(self, adapter_name: str, peft_config) -> None:
            self.peft_config[adapter_name] = peft_config

        def set_adapter(self, adapter_name: str) -> None:
            self.active_adapter = adapter_name

    class FakeBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def fake_get_peft_model(_policy, _config, adapter_name):
        return FakePeftModel(adapter_name)

    cfg = _make_cfg(num_channels=2)
    monkeypatch.setattr(runtime, "PeftModel", FakePeftModel)
    monkeypatch.setattr(runtime, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(runtime, "build_lora_config", lambda *_args, **_kwargs: object())

    policy, _ = runtime.load_or_initialize_single_channel(cfg, FakeBase(), 0)
    policy, _ = runtime.load_or_initialize_single_channel(cfg, policy, 1)

    assert isinstance(policy, FakePeftModel)
    assert set(policy.peft_config) == {"channel_1"}
    assert policy.deleted == ["channel_0"]
    assert policy.active_adapter == "channel_1"


def test_router_control_adapter_replaces_skill_adapters_and_is_trainable(monkeypatch) -> None:
    import vla_skill_router.real_runtime as runtime

    class FakePeftModel(torch.nn.Module):
        def __init__(self, adapter_name: str):
            super().__init__()
            self.router_weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.skill_weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.peft_config = {adapter_name: object()}
            self.active_adapter = adapter_name
            self.deleted: list[str] = []

        def parameters(self, recurse: bool = True):
            return iter([self.router_weight, self.skill_weight])

        def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
            return iter(
                [
                    ("lora_router_control", self.router_weight),
                    ("lora_channel_0", self.skill_weight),
                ]
            )

        def delete_adapter(self, adapter_name: str) -> None:
            self.deleted.append(adapter_name)
            self.peft_config.pop(adapter_name, None)

        def add_adapter(self, adapter_name: str, peft_config) -> None:
            self.peft_config[adapter_name] = peft_config

        def set_adapter(self, adapter_name: str) -> None:
            self.active_adapter = adapter_name

    cfg = _make_cfg(num_channels=1)
    policy = FakePeftModel("channel_0")
    monkeypatch.setattr(runtime, "PeftModel", FakePeftModel)
    monkeypatch.setattr(runtime, "build_lora_config", lambda *_args, **_kwargs: object())

    routed = runtime.load_or_initialize_router_control_adapter(cfg, policy, adapter_name="router_control")

    assert routed.active_adapter == "router_control"
    assert set(routed.peft_config) == {"router_control"}
    assert routed.deleted == ["channel_0"]
    assert routed.router_weight.requires_grad is True
    assert routed.skill_weight.requires_grad is False


def test_masked_policy_loss_resolves_peft_wrapped_pi05() -> None:
    import vla_skill_router.real_runtime as runtime
    from lerobot.policies.pi05 import modeling_pi05 as pi05_modeling

    class FakeCore(torch.nn.Module):
        def forward(self, images, img_masks, tokens, masks, actions):
            assert images == ["image"]
            assert img_masks == ["mask"]
            assert tokens.shape[0] == 2
            return torch.ones(2, 3, 4, requires_grad=True)

    class FakePi05Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeCore()

        def _preprocess_images(self, batch):
            return ["image"], ["mask"]

        def prepare_action(self, batch):
            return torch.zeros(2, 3, 4)

    class FakeBaseModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakePi05Policy()

    class FakePeftWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = FakeBaseModel()
            self.model = self.base_model.model

    batch = {
        pi05_modeling.OBS_LANGUAGE_TOKENS: torch.zeros(2, 5, dtype=torch.long),
        pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
        "action_dim": torch.tensor([2, 4]),
    }

    loss = runtime.pi05_masked_policy_loss(FakePeftWrapper(), batch)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_prefetched_batch_iterator_prepares_batches_ahead() -> None:
    from train_router_lora_sharded import PrefetchedBatchIterator

    class FakeLoader:
        def __init__(self) -> None:
            self.index = 0

        def next(self):
            value = self.index
            self.index += 1
            return {
                "x": torch.tensor([value]),
                "channel_index": torch.tensor([0]),
                "action_dim": torch.tensor([4]),
            }

    def fake_preprocessor(raw):
        return {"x_plus_one": raw["x"] + 1}

    iterator = PrefetchedBatchIterator(
        FakeLoader(),
        fake_preprocessor,
        prefetch_batches=2,
        device="cpu",
    )
    try:
        first = iterator.next()
        second = iterator.next()
    finally:
        iterator.close()

    assert first["x_plus_one"].item() == 1
    assert second["x_plus_one"].item() == 2
    assert first["channel_index"].item() == 0
    assert second["action_dim"].item() == 4


def test_cached_uint8_images_are_restored_to_float() -> None:
    from train_router_lora_sharded import _move_batch_to_device

    batch = {
        "observation.images.base_0_rgb": torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8),
        "action": torch.zeros(1, 2, 3),
    }

    moved = _move_batch_to_device(batch, "cpu")

    image = moved["observation.images.base_0_rgb"]
    assert image.dtype == torch.float32
    assert torch.allclose(image.flatten(), torch.tensor([0.0, 128.0 / 255.0, 1.0]))
    assert moved["action"].dtype == torch.float32


def test_router_labels_use_scheduled_channel_not_cached_label() -> None:
    from train_router_lora_sharded import _scheduled_router_labels

    logits = torch.randn(2, 5, requires_grad=True)
    stale_cached_labels = torch.tensor([999, -1])

    labels = _scheduled_router_labels(logits, channel_index=3)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert stale_cached_labels.tolist() == [999, -1]
    assert labels.tolist() == [3, 3]
    assert logits.grad is not None


def test_router_labels_match_single_item_batch() -> None:
    from train_router_lora_sharded import _scheduled_router_labels

    labels = _scheduled_router_labels(torch.zeros(1, 7), channel_index=6)

    assert labels.shape == (1,)
    assert labels.item() == 6


def test_router_control_checkpoint_resolves_nested_selected_adapter_dir(tmp_path: Path) -> None:
    from train_router_lora_sharded import _resolve_router_control_checkpoint_paths

    nested = tmp_path / "router_control" / "router_control"
    nested.mkdir(parents=True)
    (nested / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "router_control_head.pt").write_bytes(b"head")
    (tmp_path / "router_control_trainer_state.pt").write_bytes(b"state")

    adapter_dir, head_path, state_path = _resolve_router_control_checkpoint_paths(
        tmp_path,
        adapter_name="router_control",
    )

    assert adapter_dir == nested
    assert head_path == tmp_path / "router_control_head.pt"
    assert state_path == tmp_path / "router_control_trainer_state.pt"


def test_router_step_cpu_rendezvous_waits_on_files_not_nccl(tmp_path: Path) -> None:
    from train_router_lora_sharded import _router_step_cpu_rendezvous, _write_router_rendezvous_marker
    from vla_skill_router.distributed import DistributedInfo

    rendezvous_dir = tmp_path / "rendezvous"
    _write_router_rendezvous_marker(rendezvous_dir, rank=1, step=7)

    _router_step_cpu_rendezvous(
        rendezvous_dir,
        step=7,
        dist_info=DistributedInfo(rank=0, local_rank=0, world_size=2, backend="nccl"),
        timeout_seconds=1.0,
        poll_seconds=0.01,
    )

    assert (rendezvous_dir / "step-000000000007" / "rank0.txt").read_text(encoding="utf-8").splitlines()[0] == "7"


def test_router_step_cpu_rendezvous_uses_step_scoped_markers(tmp_path: Path) -> None:
    from train_router_lora_sharded import _router_step_cpu_rendezvous, _write_router_rendezvous_marker
    from vla_skill_router.distributed import DistributedInfo

    rendezvous_dir = tmp_path / "rendezvous"
    _write_router_rendezvous_marker(rendezvous_dir, rank=0, step=7)
    _write_router_rendezvous_marker(rendezvous_dir, rank=1, step=7)
    _write_router_rendezvous_marker(rendezvous_dir, rank=0, step=8)

    _router_step_cpu_rendezvous(
        rendezvous_dir,
        step=7,
        dist_info=DistributedInfo(rank=1, local_rank=1, world_size=2, backend="nccl"),
        timeout_seconds=1.0,
        poll_seconds=0.01,
    )


def test_router_step_cpu_rendezvous_times_out_with_missing_rank(tmp_path: Path) -> None:
    from train_router_lora_sharded import _router_step_cpu_rendezvous
    from vla_skill_router.distributed import DistributedInfo

    with pytest.raises(TimeoutError, match="router step 9"):
        _router_step_cpu_rendezvous(
            tmp_path / "rendezvous",
            step=9,
            dist_info=DistributedInfo(rank=0, local_rank=0, world_size=2, backend="nccl"),
            timeout_seconds=0.01,
            poll_seconds=0.01,
        )
