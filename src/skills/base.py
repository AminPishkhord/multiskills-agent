"""
BaseSkill: the plugin contract every skill must implement.

Per spec.md section 10/17: the Orchestrator must be able to run ANY skill via
`skill.run(input_data)` with zero per-skill branching, so that adding skill #35
never requires touching the Orchestrator, the Router, or the LangGraph wiring -
only registering a new BaseSkill subclass (see src/registry/skill_registry.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas.skill import SkillInput, SkillOutput


class BaseSkill(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, input_data: SkillInput) -> SkillOutput:
        """Execute the skill and return its output. Must not raise for "expected"
        failure modes (e.g. no valid math expression found) - return a clear
        SkillOutput message instead, since a raised exception here aborts the whole
        plan. Let genuinely unexpected errors (network, auth) propagate."""
        raise NotImplementedError
