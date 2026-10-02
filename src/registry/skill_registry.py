"""
Skill Registry - the single source of truth for "what skills exist and how to run
them." Per spec.md section 4: Registry != Router, Registry != Discovery. It does not
decide anything; it just answers "what do we have, and how do I run it."

`build_default_registry()` is the ONE place in the whole codebase that imports
concrete skill classes by name. Everything downstream (Discovery, Router, Planner,
Orchestrator, the LangGraph wiring) only ever talks to the registry/registered
SkillSpec objects - never to CalculatorSkill/TranslatorSkill/etc. directly. Adding
skill #5 means adding one line here; nothing else in src/ changes.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.skills.base import BaseSkill
from src.skills.calculator import CalculatorSkill
from src.skills.general_chat import GeneralChatSkill
from src.skills.summarizer import SummarizerSkill
from src.skills.translator import TranslatorSkill


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    implementation: BaseSkill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, skill: BaseSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"A skill named '{skill.name}' is already registered.")
        self._skills[skill.name] = SkillSpec(
            name=skill.name, description=skill.description, implementation=skill
        )

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"No skill named '{name}' is registered.") from exc

    def has(self, name: str) -> bool:
        return name in self._skills

    def all(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())


def build_default_registry() -> SkillRegistry:
    """The only place that wires up today's 4 concrete skills."""
    registry = SkillRegistry()
    registry.register(SummarizerSkill())
    registry.register(TranslatorSkill())
    registry.register(CalculatorSkill())
    registry.register(GeneralChatSkill())
    return registry
