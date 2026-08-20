from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import discover_skill_dirs
from .schema import SkillSpec


class RouterNoMatchError(ValueError):
    pass


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


@dataclass(frozen=True)
class RouterMatch:
    skill_spec: SkillSpec
    reason: str
    keyword_score: int = 0


class RuleBasedSkillRouter:
    def __init__(self, skill_specs: list[SkillSpec]):
        if not skill_specs:
            raise ValueError("RuleBasedSkillRouter requires at least one skill.")
        self.skill_specs = skill_specs
        self._skill_map = {spec.skill_id: spec for spec in skill_specs}
        self._compiled_regexes = {
            spec.skill_id: [re.compile(pattern, re.IGNORECASE) for pattern in spec.router.regexes]
            for spec in skill_specs
        }

    @classmethod
    def from_skill_root(cls, skill_root: Path) -> "RuleBasedSkillRouter":
        specs = [SkillSpec.load(skill_dir) for skill_dir in discover_skill_dirs(skill_root)]
        return cls(specs)

    def resolve(self, task: str, explicit_skill_id: str | None = None) -> RouterMatch:
        if explicit_skill_id:
            if explicit_skill_id not in self._skill_map:
                raise RouterNoMatchError(f"Unknown explicit skill_id `{explicit_skill_id}`.")
            return RouterMatch(skill_spec=self._skill_map[explicit_skill_id], reason="explicit_skill_id")

        normalized_task = normalize_text(task)
        alias_candidates: list[RouterMatch] = []
        regex_candidates: list[RouterMatch] = []
        keyword_candidates: list[RouterMatch] = []

        for spec in self.skill_specs:
            alias_space = {normalize_text(spec.display_name), normalize_text(spec.skill_id)}
            alias_space.update(normalize_text(alias) for alias in spec.router.aliases)
            if normalized_task in alias_space:
                alias_candidates.append(RouterMatch(spec, "alias"))
                continue

            if any(regex.search(normalized_task) for regex in self._compiled_regexes[spec.skill_id]):
                regex_candidates.append(RouterMatch(spec, "regex"))
                continue

            keyword_score = sum(
                1 for keyword in {normalize_text(keyword) for keyword in spec.router.keywords} if keyword in normalized_task
            )
            if keyword_score > 0:
                keyword_candidates.append(RouterMatch(spec, "keyword", keyword_score=keyword_score))

        if alias_candidates:
            best = max(alias_candidates, key=lambda item: item.skill_spec.router.priority)
            return best
        if regex_candidates:
            best = max(regex_candidates, key=lambda item: item.skill_spec.router.priority)
            return best
        if keyword_candidates:
            best = max(
                keyword_candidates,
                key=lambda item: (item.keyword_score, item.skill_spec.router.priority, item.skill_spec.skill_id),
            )
            return best
        if len(self.skill_specs) == 1:
            return RouterMatch(self.skill_specs[0], "single_skill_fallback")
        raise RouterNoMatchError(f"No router rule matched task: {task}")
