from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChannelMeta:
    channel_id: str
    skill_id: str
    channel_index: int
    action_dim: int


class SkillChannelDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], meta: ChannelMeta):
        self.dataset = dataset
        self.meta = meta

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["skill_id"] = self.meta.skill_id
        item["channel_id"] = self.meta.channel_id
        item["channel_index"] = torch.tensor(self.meta.channel_index, dtype=torch.long)
        item["action_dim"] = torch.tensor(self.meta.action_dim, dtype=torch.long)
        return item
