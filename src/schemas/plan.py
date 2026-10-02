"""
The Planner's structured output: an ExecutionPlan of 1-2 steps.

`PlanStep.input_text` is either a literal string (usually the original user prompt,
or a sub-string of it) or a reference of the form "$<step_id>.output", meaning
"substitute the referenced step's output here" - this is how sequential/chained
skills (e.g. Summarizer -> Translator) are expressed without the Planner or
Orchestrator needing any skill-specific knowledge.

Structural validation (field types, step count, basic shape) is enforced here via
Pydantic. Semantic validation (do the referenced skills/steps actually exist, is
there a dependency cycle, etc.) is NOT a model's job - see
src/routing/planner.py::validate_plan for that, per the brief's distinction between
"Structural Validation" and "Semantic Validation".
"""
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field, field_validator

_STEP_REF_RE = re.compile(r"^\$(?P<step_id>[A-Za-z0-9_]+)\.output$")


class PlanStep(BaseModel):
    id: str = Field(..., description="Unique id for this step, e.g. 'step_1'.")
    skill: str = Field(..., description="Name of the skill to run, must exist in the registry.")
    input_text: str = Field(
        ...,
        description="Literal text for the skill, or '$<step_id>.output' to chain "
        "from a previous step's output.",
    )
    depends_on: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(..., min_length=1, max_length=2)

    @field_validator("steps")
    @classmethod
    def _unique_step_ids(cls, v: List[PlanStep]) -> List[PlanStep]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate step ids in plan: {ids}")
        return v


def is_step_reference(input_text: str) -> bool:
    return bool(_STEP_REF_RE.match(input_text))


def parse_step_reference(input_text: str) -> str:
    """Return the referenced step id from a '$<id>.output' string, or raise ValueError."""
    match = _STEP_REF_RE.match(input_text)
    if not match:
        raise ValueError(f"Not a valid step reference: {input_text!r}")
    return match.group("step_id")
