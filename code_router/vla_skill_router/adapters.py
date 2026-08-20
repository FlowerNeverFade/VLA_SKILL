from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vla_skill.io_utils import load_json


@dataclass(frozen=True)
class ResolvedAdapter:
    adapter_dir: Path | None
    source: str
    stale_path: Path | None = None


def resolve_adapter_dir(
    *,
    skill_output_dir: Path,
    explicit_adapter_dir: Path | None = None,
) -> ResolvedAdapter:
    if explicit_adapter_dir is not None:
        adapter_dir = Path(explicit_adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Explicit adapter_dir does not exist: {adapter_dir}")
        return ResolvedAdapter(adapter_dir=adapter_dir, source="explicit")

    best_meta_path = skill_output_dir / "best_adapter.json"
    stale_path = None
    if best_meta_path.is_file():
        payload = load_json(best_meta_path)
        candidate = Path(payload["adapter_dir"])
        if candidate.exists():
            return ResolvedAdapter(adapter_dir=candidate, source="best_adapter_json")
        stale_path = candidate

    best_symlink = skill_output_dir / "best"
    if best_symlink.exists():
        return ResolvedAdapter(adapter_dir=best_symlink, source="best_symlink", stale_path=stale_path)

    return ResolvedAdapter(adapter_dir=None, source="none", stale_path=stale_path)


def set_module_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def set_only_named_adapter_trainable(model, adapter_name: str) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = f".{adapter_name}." in name or f"_{adapter_name}." in name or adapter_name in name


def freeze_all_adapters(model) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name or "adapter" in name:
            param.requires_grad = False
