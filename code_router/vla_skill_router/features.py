from __future__ import annotations

from typing import Any

import torch
from contextlib import nullcontext


def _pool_tensor(value: torch.Tensor, batch_size: int) -> torch.Tensor | None:
    if value.ndim < 2 or value.shape[0] != batch_size:
        return None
    if value.dtype == torch.bool:
        return None
    if not torch.is_floating_point(value):
        value = value.float()
    if value.ndim == 2:
        return value
    return value.reshape(batch_size, -1, value.shape[-1]).mean(dim=1)


def pool_nested_hidden(value: Any, batch_size: int) -> torch.Tensor:
    pooled: list[torch.Tensor] = []

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = _pool_tensor(item, batch_size)
            if tensor is not None:
                pooled.append(tensor)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not pooled:
        raise ValueError("Could not find any floating hidden tensors to pool for router context.")
    return torch.cat(pooled, dim=-1)


class StaticFeatureExtractor:
    def __init__(self, context_key: str = "router_context"):
        self.context_key = context_key

    def __call__(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.context_key not in batch:
            raise KeyError(f"Batch is missing router context key `{self.context_key}`.")
        return batch[self.context_key]


class PI05PrefixFeatureExtractor:
    """Frozen PI05 prefix feature extractor for router context.

    The prefix path contains image and language tokens. The router still receives
    explicit state separately, even if the tokenizer has already folded state
    into language tokens.
    """

    def __init__(self, policy, *, disable_adapters: bool = True, no_grad: bool = True):
        self.policy = policy
        self.disable_adapters = disable_adapters
        self.no_grad = no_grad

    @staticmethod
    def _resolve_inner_model(policy):
        current = policy
        for _ in range(4):
            if all(
                hasattr(current, name)
                for name in ("embed_prefix", "_prepare_attention_masks_4d", "paligemma_with_expert")
            ):
                return current
            current = getattr(current, "model", None)
            if current is None:
                break
        raise AttributeError("Could not resolve PI05 inner model with embed_prefix from policy.")

    @staticmethod
    def _attention_dtype(inner_model) -> torch.dtype | None:
        try:
            return (
                inner_model.paligemma_with_expert.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype
            )
        except (AttributeError, IndexError, TypeError):
            return None

    def __call__(self, batch: dict[str, Any]) -> torch.Tensor:
        from lerobot.policies.pi05 import modeling_pi05 as pi05_modeling

        self.policy.eval()
        inner_model = self._resolve_inner_model(self.policy)
        attention_dtype = self._attention_dtype(inner_model)
        adapter_context = self.policy.disable_adapter() if self.disable_adapters and hasattr(self.policy, "disable_adapter") else nullcontext()
        grad_context = torch.no_grad() if self.no_grad else nullcontext()
        with grad_context, adapter_context:
            images, img_masks = self.policy._preprocess_images(batch)
            tokens = batch[pi05_modeling.OBS_LANGUAGE_TOKENS]
            masks = batch[pi05_modeling.OBS_LANGUAGE_ATTENTION_MASK]
            prefix_embs, prefix_pad_masks, prefix_att_masks = inner_model.embed_prefix(
                images,
                img_masks,
                tokens,
                masks,
            )
            if attention_dtype in {torch.float16, torch.bfloat16}:
                prefix_embs = prefix_embs.to(dtype=attention_dtype)
            prefix_att_2d_masks = pi05_modeling.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = inner_model._prepare_attention_masks_4d(prefix_att_2d_masks)
            if attention_dtype in {torch.float16, torch.bfloat16}:
                prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=attention_dtype)
            outputs, _ = inner_model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )
        prefix_out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return pool_nested_hidden(prefix_out, batch_size=prefix_embs.shape[0])
