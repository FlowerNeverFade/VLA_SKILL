from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class ToyAdapterLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._active_adapter = ["a"]
        self.adapters = nn.ModuleDict(
            {
                "a": nn.Linear(1, 1, bias=False),
                "b": nn.Linear(1, 1, bias=False),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapters[self._active_adapter[0]](x)


class ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = ToyAdapterLayer()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.layer(batch["x"]).mean()


def _ddp_policy_loss_worker(rank: int, world_size: int, init_file: str) -> None:
    import train_router_lora as train_entry
    from vla_skill_router.distributed import DistributedInfo, broadcast_object

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        info = DistributedInfo(rank=rank, local_rank=rank, world_size=world_size, backend="gloo")
        run_name = broadcast_object("rank0_run" if rank == 0 else None, info)
        assert run_name == "rank0_run"

        torch.manual_seed(0)
        policy = ToyPolicy()

        def fake_loss(model, batch):
            return model(batch)

        train_entry.pi05_masked_policy_loss = fake_loss
        wrapper = train_entry.PolicyLossWrapper(policy)
        ddp_wrapper = DistributedDataParallel(wrapper, find_unused_parameters=True)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

        for channel_id in ("a", "b"):
            optimizer.zero_grad(set_to_none=True)
            loss = ddp_wrapper({"x": torch.ones(4, 1) * float(rank + 1)}, channel_id)
            loss.backward()
            optimizer.step()

        weights = torch.cat(
            [
                policy.layer.adapters["a"].weight.detach().flatten(),
                policy.layer.adapters["b"].weight.detach().flatten(),
            ]
        )
        gathered = [torch.zeros_like(weights) for _ in range(world_size)]
        dist.all_gather(gathered, weights)
        for item in gathered[1:]:
            assert torch.allclose(gathered[0], item)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_cpu_gloo_ddp_policy_loss_smoke(tmp_path: Path) -> None:
    world_size = 2
    init_file = tmp_path / "ddp_init"

    mp.spawn(
        _ddp_policy_loss_worker,
        args=(world_size, str(init_file)),
        nprocs=world_size,
        join=True,
    )
