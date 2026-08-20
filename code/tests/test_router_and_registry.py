from __future__ import annotations

from pathlib import Path

from vla_skill.io_utils import write_json
from vla_skill.registry import SkillAdapterRegistry
from vla_skill.router import RuleBasedSkillRouter


def test_router_prefers_alias_then_regex_then_keywords(toy_skill_root: Path) -> None:
    router = RuleBasedSkillRouter.from_skill_root(toy_skill_root)

    alias_match = router.resolve("pick mug")
    assert alias_match.skill_spec.skill_id == "pick_mug"
    assert alias_match.reason == "alias"

    regex_match = router.resolve("please open the drawer")
    assert regex_match.skill_spec.skill_id == "open_drawer"
    assert regex_match.reason == "regex"

    keyword_match = router.resolve("place the mug on the tray gently")
    assert keyword_match.skill_spec.skill_id == "pick_mug"
    assert keyword_match.reason == "keyword"


def test_registry_resolves_best_adapter_metadata_without_loading_model(tmp_path: Path, toy_skill_root: Path) -> None:
    output_root = tmp_path / "outputs"
    adapter_dir = output_root / "pick_mug" / "A" / "run_001" / "best"
    adapter_dir.mkdir(parents=True)
    write_json(
        output_root / "pick_mug" / "best_adapter.json",
        {
            "skill_id": "pick_mug",
            "group": "A",
            "run_name": "run_001",
            "adapter_dir": str(adapter_dir),
            "val_loss": 0.123,
            "action_mse": 0.456,
        },
    )

    registry = SkillAdapterRegistry(skill_root=toy_skill_root, output_root=output_root, device="cpu", dtype="float32")
    ref = registry.resolve_adapter_reference("pick_mug")

    assert ref.group == "A"
    assert ref.run_name == "run_001"
    assert ref.adapter_dir == adapter_dir
