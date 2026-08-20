from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peft import PeftModel

from .constants import DEFAULT_BASE_MODEL_PATH, DEFAULT_OUTPUT_ROOT, DEFAULT_SKILL_ROOT
from .dataset import load_skill_spec, load_stats
from .io_utils import load_json
from .lora import build_lora_config
from .pi05 import build_processors, load_base_policy, make_generic_inference_config
from .router import RuleBasedSkillRouter
from .schema import SkillSpec


@dataclass(frozen=True)
class AdapterReference:
    skill_id: str
    group: str | None
    run_name: str | None
    adapter_dir: Path
    summary_path: Path | None = None


class SkillAdapterRegistry:
    def __init__(
        self,
        *,
        skill_root: Path = DEFAULT_SKILL_ROOT,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        base_model_path: Path = DEFAULT_BASE_MODEL_PATH,
        device: str = "cuda",
        dtype: str = "bfloat16",
        tokenizer_name_or_path: str | None = None,
    ):
        self.skill_root = Path(skill_root)
        self.output_root = Path(output_root)
        self.base_model_path = Path(base_model_path)
        self.device = device
        self.dtype = dtype
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.router = RuleBasedSkillRouter.from_skill_root(self.skill_root)
        self._processor_cache: dict[str, tuple[Any, Any]] = {}
        self._shell_model: PeftModel | None = None
        self._loaded_adapters: set[str] = set()

    def _build_shell(self) -> PeftModel:
        if self._shell_model is None:
            config = make_generic_inference_config(
                base_model_path=self.base_model_path,
                device=self.device,
                dtype=self.dtype,
            )
            base_policy = load_base_policy(config, base_model_path=self.base_model_path, strict=True)
            self._shell_model = base_policy.wrap_with_peft(
                peft_config=build_lora_config(
                    "C",
                    base_model_name_or_path=str(self.base_model_path),
                    inference_mode=True,
                )
            )
            self._shell_model.eval()
        return self._shell_model

    def resolve_adapter_reference(
        self,
        skill_id: str,
        *,
        group: str | None = None,
        run_name: str | None = None,
        adapter_dir: Path | None = None,
    ) -> AdapterReference:
        if adapter_dir is not None:
            return AdapterReference(skill_id=skill_id, group=group, run_name=run_name, adapter_dir=Path(adapter_dir))

        skill_output_dir = self.output_root / skill_id
        if group and run_name:
            candidate = skill_output_dir / group / run_name / "best"
            return AdapterReference(skill_id=skill_id, group=group, run_name=run_name, adapter_dir=candidate)
        if group and not run_name:
            group_dir = skill_output_dir / group
            run_dirs = sorted(path for path in group_dir.iterdir() if path.is_dir()) if group_dir.is_dir() else []
            if not run_dirs:
                raise FileNotFoundError(f"No runs found under {group_dir}.")
            selected = max(run_dirs, key=lambda path: path.stat().st_mtime)
            return AdapterReference(skill_id=skill_id, group=group, run_name=selected.name, adapter_dir=selected / "best")

        best_meta_path = skill_output_dir / "best_adapter.json"
        if best_meta_path.is_file():
            payload = load_json(best_meta_path)
            return AdapterReference(
                skill_id=skill_id,
                group=payload.get("group"),
                run_name=payload.get("run_name"),
                adapter_dir=Path(payload["adapter_dir"]),
                summary_path=best_meta_path,
            )

        best_symlink = skill_output_dir / "best"
        if best_symlink.exists():
            return AdapterReference(skill_id=skill_id, group=None, run_name=None, adapter_dir=best_symlink)
        raise FileNotFoundError(
            f"Could not resolve a best adapter for skill `{skill_id}` under {skill_output_dir}. "
            "Run training first or pass --adapter-dir."
        )

    def _get_processors(self, skill_spec: SkillSpec):
        cached = self._processor_cache.get(skill_spec.skill_id)
        if cached is None:
            cached = build_processors(
                skill_spec,
                load_stats(skill_spec),
                device=self.device,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )
            self._processor_cache[skill_spec.skill_id] = cached
        return cached

    def activate(
        self,
        *,
        skill_id: str | None = None,
        task: str | None = None,
        explicit_skill_id: str | None = None,
        group: str | None = None,
        run_name: str | None = None,
        adapter_dir: Path | None = None,
    ) -> tuple[SkillSpec, PeftModel, Any, Any, AdapterReference]:
        selected_skill_id = skill_id
        if selected_skill_id is None:
            if task is None:
                raise ValueError("Either skill_id or task must be provided.")
            selected_skill_id = self.router.resolve(task, explicit_skill_id=explicit_skill_id).skill_spec.skill_id

        skill_spec = load_skill_spec(self.skill_root, selected_skill_id)
        preprocessor, postprocessor = self._get_processors(skill_spec)
        adapter_ref = self.resolve_adapter_reference(
            selected_skill_id,
            group=group,
            run_name=run_name,
            adapter_dir=adapter_dir,
        )

        model = self._build_shell()
        adapter_name = f"{selected_skill_id}__{adapter_ref.group or 'best'}__{adapter_ref.run_name or 'resolved'}"
        if adapter_name not in self._loaded_adapters:
            model.load_adapter(str(adapter_ref.adapter_dir), adapter_name=adapter_name, is_trainable=False)
            self._loaded_adapters.add(adapter_name)
        model.set_adapter(adapter_name, inference_mode=True)
        model.config.chunk_size = skill_spec.chunk_size
        model.config.n_action_steps = skill_spec.n_action_steps
        model.config.output_features["action"].shape = (skill_spec.action_dim,)
        return skill_spec, model, preprocessor, postprocessor, adapter_ref
