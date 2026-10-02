"""
The common Skill contract, and the Router's structured output.

Design decision (documented in README.md / spec.md too): every skill shares ONE
generic input/output shape - `SkillInput(text=...)` in, `SkillOutput(output=...)` out
- rather than each skill defining a bespoke schema (e.g. Calculator taking
`expression` instead of `text`). This is what lets the Orchestrator call
`skill.run(input_data)` completely generically (src/orchestration/orchestrator.py)
with no per-skill branching, and lets the Planner wire skills together
(`$step_1.output` -> next step's `text`) without knowing each skill's internal field
names. A skill is still free to do whatever it wants internally with that `text`
(e.g. Calculator extracts an expression from it, Translator detects a target
language from it); the external contract just stays uniform.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class SkillInput(BaseModel):
    text: str = Field(..., description="The text this skill should operate on.")


class SkillOutput(BaseModel):
    output: str = Field(..., description="The skill's natural-language result.")


class RoutingDecision(BaseModel):
    """
    Structured output of the Router node.

    Unlike a fixed enum of 4 skill names, `skills` is validated as a plain list of
    strings and checked against whatever candidate set Skill Discovery handed the
    Router for *this* request (see src/routing/router.py::decide_skills) - this is
    what keeps the Router extensible to new skills without a code/schema change.
    """

    skills: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")

    @field_validator("skills")
    @classmethod
    def _dedupe_and_cap(cls, v: List[str]) -> List[str]:
        seen: list[str] = []
        for skill in v:
            if skill not in seen:
                seen.append(skill)
        return seen[:2]
