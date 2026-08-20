from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .action import pad_state
from .constants import MAX_STATE_DIM


@dataclass(frozen=True)
class RouterForwardOutput:
    logits: torch.Tensor
    router_input: torch.Tensor
    pooled_context: torch.Tensor
    state_embedding: torch.Tensor
    previous_skill_embedding: torch.Tensor | None = None


class StateEncoder(nn.Module):
    def __init__(self, *, max_state_dim: int = MAX_STATE_DIM, state_embed_dim: int = 64):
        super().__init__()
        self.max_state_dim = max_state_dim
        self.net = nn.Sequential(
            nn.LayerNorm(max_state_dim),
            nn.Linear(max_state_dim, state_embed_dim),
            nn.GELU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(pad_state(state, self.max_state_dim))


class HardTop1Router(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        num_channels: int,
        max_state_dim: int = MAX_STATE_DIM,
        state_embed_dim: int = 64,
        hidden_dim: int = 256,
        use_previous_skill: bool = False,
        previous_skill_embed_dim: int | None = None,
    ):
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive.")
        self.context_dim = context_dim
        self.num_channels = num_channels
        self.use_previous_skill = use_previous_skill
        self.state_encoder = StateEncoder(max_state_dim=max_state_dim, state_embed_dim=state_embed_dim)
        prev_dim = 0
        if use_previous_skill:
            prev_dim = previous_skill_embed_dim or state_embed_dim
            self.previous_skill_embedding = nn.Embedding(num_channels, prev_dim)
        else:
            self.previous_skill_embedding = None
        input_dim = context_dim + state_embed_dim + prev_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )

    def build_router_input(
        self,
        pooled_context: torch.Tensor,
        state: torch.Tensor,
        previous_channel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if pooled_context.ndim != 2:
            raise ValueError(f"pooled_context must have shape [B, C], got {tuple(pooled_context.shape)}.")
        if pooled_context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}, got {pooled_context.shape[-1]}.")
        state_embedding = self.state_encoder(state)
        pieces = [pooled_context, state_embedding]
        prev_embedding = None
        if self.use_previous_skill:
            if previous_channel is None:
                previous_channel = torch.zeros(
                    pooled_context.shape[0], dtype=torch.long, device=pooled_context.device
                )
            prev_embedding = self.previous_skill_embedding(previous_channel.to(device=pooled_context.device))
            pieces.append(prev_embedding)
        return torch.cat(pieces, dim=-1), state_embedding, prev_embedding

    def forward(
        self,
        pooled_context: torch.Tensor,
        state: torch.Tensor,
        previous_channel: torch.Tensor | None = None,
    ) -> RouterForwardOutput:
        router_input, state_embedding, prev_embedding = self.build_router_input(
            pooled_context,
            state,
            previous_channel=previous_channel,
        )
        logits = self.classifier(router_input)
        return RouterForwardOutput(
            logits=logits,
            router_input=router_input,
            pooled_context=pooled_context,
            state_embedding=state_embedding,
            previous_skill_embedding=prev_embedding,
        )

    @torch.no_grad()
    def select(
        self,
        pooled_context: torch.Tensor,
        state: torch.Tensor,
        previous_channel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(pooled_context, state, previous_channel=previous_channel)
        probs = torch.softmax(output.logits, dim=-1)
        return torch.argmax(probs, dim=-1), probs


class LoRAControlRouter(nn.Module):
    """Minimal router head used with an active `router_control` LoRA adapter.

    The LoRA adapter produces the pooled context. This module only appends an
    explicit state embedding and linearly reads out channel logits.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        num_channels: int,
        max_state_dim: int = MAX_STATE_DIM,
        state_embed_dim: int = 64,
    ):
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive.")
        self.context_dim = context_dim
        self.num_channels = num_channels
        self.state_encoder = StateEncoder(max_state_dim=max_state_dim, state_embed_dim=state_embed_dim)
        input_dim = context_dim + state_embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_channels),
        )

    def build_router_input(self, pooled_context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pooled_context.ndim != 2:
            raise ValueError(f"pooled_context must have shape [B, C], got {tuple(pooled_context.shape)}.")
        if pooled_context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context_dim={self.context_dim}, got {pooled_context.shape[-1]}.")
        state_embedding = self.state_encoder(state)
        return torch.cat([pooled_context, state_embedding], dim=-1), state_embedding

    def forward(self, pooled_context: torch.Tensor, state: torch.Tensor) -> RouterForwardOutput:
        router_input, state_embedding = self.build_router_input(pooled_context, state)
        logits = self.classifier(router_input)
        return RouterForwardOutput(
            logits=logits,
            router_input=router_input,
            pooled_context=pooled_context,
            state_embedding=state_embedding,
            previous_skill_embedding=None,
        )

    @torch.no_grad()
    def select(self, pooled_context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(pooled_context, state)
        probs = torch.softmax(output.logits, dim=-1)
        return torch.argmax(probs, dim=-1), probs
