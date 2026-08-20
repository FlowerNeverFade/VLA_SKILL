from __future__ import annotations

import torch

from vla_skill_router.router_model import HardTop1Router, LoRAControlRouter


def test_router_input_explicitly_includes_state_embedding() -> None:
    torch.manual_seed(0)
    router = HardTop1Router(context_dim=4, num_channels=2, max_state_dim=6, state_embed_dim=3, hidden_dim=8)
    context = torch.ones(2, 4)
    state_a = torch.zeros(2, 3)
    state_b = torch.ones(2, 3)

    out_a = router(context, state_a)
    out_b = router(context, state_b)

    assert out_a.router_input.shape == (2, 7)
    assert out_a.state_embedding.shape == (2, 3)
    assert not torch.allclose(out_a.state_embedding, out_b.state_embedding)


def test_router_supports_optional_previous_skill_embedding() -> None:
    router = HardTop1Router(
        context_dim=4,
        num_channels=3,
        max_state_dim=6,
        state_embed_dim=3,
        hidden_dim=8,
        use_previous_skill=True,
        previous_skill_embed_dim=5,
    )
    output = router(torch.ones(2, 4), torch.zeros(2, 6), previous_channel=torch.tensor([0, 2]))

    assert output.router_input.shape == (2, 12)
    assert output.previous_skill_embedding is not None
    assert output.logits.shape == (2, 3)


def test_lora_control_router_uses_minimal_linear_head_and_state() -> None:
    router = LoRAControlRouter(context_dim=4, num_channels=3, max_state_dim=6, state_embed_dim=3)
    output = router(torch.ones(2, 4), torch.zeros(2, 6))

    linear_layers = [module for module in router.classifier.modules() if isinstance(module, torch.nn.Linear)]
    assert len(linear_layers) == 1
    assert output.router_input.shape == (2, 7)
    assert output.state_embedding.shape == (2, 3)
    assert output.logits.shape == (2, 3)
