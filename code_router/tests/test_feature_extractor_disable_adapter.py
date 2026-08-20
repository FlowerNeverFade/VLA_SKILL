from __future__ import annotations

from contextlib import contextmanager

import torch
from lerobot.policies.pi05 import modeling_pi05 as pi05_modeling

from vla_skill_router.features import PI05PrefixFeatureExtractor


class FakePaliGemmaWithExpert:
    def __init__(self, policy) -> None:
        self.policy = policy

    def forward(self, **_kwargs):
        value = 2.0 if self.policy.disabled else -2.0
        return [torch.full((2, 3, 4), value)], None


class FakeModel:
    def __init__(self, policy) -> None:
        self.paligemma_with_expert = FakePaliGemmaWithExpert(policy)

    def embed_prefix(self, *_args):
        return (
            torch.ones(2, 3, 4),
            torch.ones(2, 3, dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.bool),
        )

    def _prepare_attention_masks_4d(self, masks):
        return masks[:, None, :, :]


class FakePolicy:
    def __init__(self) -> None:
        self.disabled = False
        self.model = FakeModel(self)

    def eval(self):
        return self

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def _preprocess_images(self, _batch):
        return [], []


def test_pi05_feature_extractor_disables_adapters_for_router_context() -> None:
    policy = FakePolicy()
    extractor = PI05PrefixFeatureExtractor(policy)
    batch = {
        pi05_modeling.OBS_LANGUAGE_TOKENS: torch.ones(2, 5, dtype=torch.long),
        pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
    }

    context = extractor(batch)

    assert tuple(context.shape) == (2, 4)
    assert torch.allclose(context, torch.full((2, 4), 2.0))
    assert policy.disabled is False


def test_pi05_feature_extractor_resolves_peft_wrapped_policy_model() -> None:
    class FakePeftWrapper:
        def __init__(self) -> None:
            self.base_policy = FakePolicy()
            self.model = self.base_policy

        def eval(self):
            self.base_policy.eval()
            return self

        def disable_adapter(self):
            return self.base_policy.disable_adapter()

        def _preprocess_images(self, batch):
            return self.base_policy._preprocess_images(batch)

    policy = FakePeftWrapper()
    extractor = PI05PrefixFeatureExtractor(policy)
    batch = {
        pi05_modeling.OBS_LANGUAGE_TOKENS: torch.ones(2, 5, dtype=torch.long),
        pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
    }

    context = extractor(batch)

    assert tuple(context.shape) == (2, 4)
    assert torch.allclose(context, torch.full((2, 4), 2.0))
    assert policy.base_policy.disabled is False


def test_pi05_feature_extractor_matches_attention_mask_dtype_to_bfloat16_query() -> None:
    class FakeQProj(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)

    class FakeSelfAttention:
        def __init__(self) -> None:
            self.q_proj = FakeQProj()

    class FakeLayer:
        def __init__(self) -> None:
            self.self_attn = FakeSelfAttention()

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = [FakeLayer()]

    class FakeGemmaModel:
        def __init__(self) -> None:
            self.language_model = FakeLanguageModel()

    class FakeGemma:
        def __init__(self) -> None:
            self.model = FakeGemmaModel()

    class DTypeCheckingPaliGemmaWithExpert(FakePaliGemmaWithExpert):
        def __init__(self, policy) -> None:
            super().__init__(policy)
            self.paligemma = FakeGemma()

        def forward(self, **kwargs):
            assert kwargs["attention_mask"].dtype == torch.bfloat16
            assert kwargs["inputs_embeds"][0].dtype == torch.bfloat16
            return [torch.full((2, 3, 4), 1.0, dtype=torch.bfloat16)], None

    class DTypeModel(FakeModel):
        def __init__(self, policy) -> None:
            self.paligemma_with_expert = DTypeCheckingPaliGemmaWithExpert(policy)

    class DTypePolicy(FakePolicy):
        def __init__(self) -> None:
            super().__init__()
            self.model = DTypeModel(self)

    policy = DTypePolicy()
    extractor = PI05PrefixFeatureExtractor(policy, disable_adapters=False, no_grad=True)
    batch = {
        pi05_modeling.OBS_LANGUAGE_TOKENS: torch.ones(2, 5, dtype=torch.long),
        pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
    }

    context = extractor(batch)

    assert context.dtype == torch.bfloat16


def test_pi05_feature_extractor_can_keep_adapter_enabled_with_gradients() -> None:
    class GradPaliGemmaWithExpert:
        def __init__(self, policy) -> None:
            self.policy = policy

        def forward(self, **_kwargs):
            value = self.policy.scale * torch.ones(2, 3, 4)
            return [value], None

    class GradModel(FakeModel):
        def __init__(self, policy) -> None:
            self.paligemma_with_expert = GradPaliGemmaWithExpert(policy)

    class GradPolicy(FakePolicy):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(3.0))
            self.model = GradModel(self)

    policy = GradPolicy()
    extractor = PI05PrefixFeatureExtractor(policy, disable_adapters=False, no_grad=False)
    batch = {
        pi05_modeling.OBS_LANGUAGE_TOKENS: torch.ones(2, 5, dtype=torch.long),
        pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
    }

    context = extractor(batch)
    context.sum().backward()

    assert policy.disabled is False
    assert policy.scale.grad is not None
    assert policy.scale.grad.item() > 0
