"""
Planner, per spec.md section 7-9.

The Router decides WHICH skill(s) are needed; the Planner decides HOW they should run
- in particular, whether a 2-skill request is sequential (one skill's output feeds
the next, e.g. "summarize this and translate it") or parallel/independent (two
unrelated asks in one message, e.g. "hi! also what's 12*7?").

Design decisions (documented in README.md too):
- For a single selected skill, planning is trivial (one step, no LLM call needed) -
  `build_trivial_plan` short-circuits straight to that, saving a request and removing
  a failure point for the common case.
- For two selected skills, an LLM call (`PLANNER_SYSTEM_PROMPT`) decides ordering and
  whether step 2's input should be the original text or `$step_1.output`. This is
  deliberately generic (it reasons over whichever 2 skill *descriptions* were
  selected, not a hardcoded pair list), per spec.md section 17's extensibility
  requirement - adding skill #5 doesn't require teaching the Planner a new pair rule.
- If the Planner's LLM call fails or returns an invalid plan even after retries, we
  fail safe into a deterministic PARALLEL plan (both skills run independently on the
  original prompt) rather than crashing - this never changes meaning incorrectly, it
  just loses the "chain through the summary" optimization for that one request.

Semantic validation (`validate_plan`) is intentionally separate from the Pydantic
schema's structural checks (field types, 1-2 step count - see src/schemas/plan.py):
it checks things only the Registry can answer (does this skill actually exist?) and
things that require looking at the whole plan graph at once (dependency cycles,
dangling references) - this mirrors spec.md section 9's Structural vs Semantic split.
"""
from __future__ import annotations

from src.llm.client import call_structured
from src.registry.skill_registry import SkillSpec
from src.schemas.plan import ExecutionPlan, PlanStep, is_step_reference, parse_step_reference


class PlanValidationError(ValueError):
    pass


_PLANNER_SYSTEM_PROMPT = """You are the Planner of a multi-skill assistant. Exactly
two skills have already been selected for the user's message:

{skill_lines}

The user's message may contain MULTIPLE distinct requests, only some of which belong
to these two skills (there may be a third, unrelated part - ignore it entirely, it
is being handled elsewhere). Your job:

1. For EACH selected skill, extract ONLY the specific portion of the user's message
   that is actually relevant to that skill - never the full original message unless
   the full message genuinely is that skill's relevant content. Example: if the
   message is "hi, what's 2*6, and translate 'Ali is eating food' to English, and
   tell me if science or wealth matters more", and the two selected skills are
   calculator and translator, step_1.input_text should be "what's 2*6" (NOT the
   whole message) and step_2.input_text should be "translate 'Ali is eating food' to
   English" (NOT the whole message, and NOT the science/wealth part).
2. Decide execution order:
   - SEQUENTIAL, if skill B should operate on skill A's output rather than its own
     extracted text (e.g. "summarize this, then translate it" -> translate the
     summary).
   - PARALLEL/independent, if the two skills answer two unrelated extracted parts.

Output ONLY a single JSON object, no prose, no markdown fences, matching this shape
exactly (step ids must be exactly "step_1" and "step_2", in execution order):
{{
  "steps": [
    {{"id": "step_1", "skill": "<first skill name>", "input_text": "<ONLY the text relevant to this skill, extracted from the user's message>", "depends_on": []}},
    {{"id": "step_2", "skill": "<second skill name>", "input_text": "<either this skill's own extracted relevant text (parallel), or exactly the string \\"$step_1.output\\" (sequential)>", "depends_on": [<list with \\"step_1\\" if sequential, else []>]}}
  ]
}}
"""


def build_trivial_plan(skill_name: str, prompt: str) -> ExecutionPlan:
    return ExecutionPlan(
        steps=[PlanStep(id="step_1", skill=skill_name, input_text=prompt, depends_on=[])]
    )


def _build_parallel_fallback_plan(skills: list[SkillSpec], prompt: str) -> ExecutionPlan:
    return ExecutionPlan(
        steps=[
            PlanStep(id=f"step_{i + 1}", skill=spec.name, input_text=prompt, depends_on=[])
            for i, spec in enumerate(skills)
        ]
    )


def decide_plan(llm, prompt: str, selected: list[SkillSpec]) -> ExecutionPlan:
    if len(selected) == 1:
        return build_trivial_plan(selected[0].name, prompt)

    skill_lines = "\n".join(f'- "{s.name}": {s.description}' for s in selected)
    system_prompt = _PLANNER_SYSTEM_PROMPT.format(skill_lines=skill_lines)

    try:
        plan = call_structured(llm, system_prompt, prompt, ExecutionPlan)
        validate_plan(plan, allowed_skill_names={s.name for s in selected})
        return plan
    except Exception:
        return _build_parallel_fallback_plan(selected, prompt)


def validate_plan(plan: ExecutionPlan, allowed_skill_names: set[str]) -> None:
    """Raises PlanValidationError on any semantic problem. Generic over plan size,
    not hardcoded to 2 steps, so it keeps working if the step-count ceiling is ever
    raised."""
    step_ids = {s.id for s in plan.steps}

    for step in plan.steps:
        if step.skill not in allowed_skill_names:
            raise PlanValidationError(
                f"Step '{step.id}' references unknown/unselected skill '{step.skill}'."
            )
        for dep in step.depends_on:
            if dep not in step_ids:
                raise PlanValidationError(f"Step '{step.id}' depends on unknown step '{dep}'.")
            if dep == step.id:
                raise PlanValidationError(f"Step '{step.id}' cannot depend on itself.")
        if is_step_reference(step.input_text):
            ref_id = parse_step_reference(step.input_text)
            if ref_id not in step_ids:
                raise PlanValidationError(
                    f"Step '{step.id}' input references unknown step '{ref_id}'."
                )
            if ref_id not in step.depends_on:
                raise PlanValidationError(
                    f"Step '{step.id}' uses '{step.input_text}' but does not list "
                    f"'{ref_id}' in depends_on."
                )

    _check_no_cycles(plan)


def _check_no_cycles(plan: ExecutionPlan) -> None:
    graph = {s.id: s.depends_on for s in plan.steps}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def visit(node: str, stack: list[str]) -> None:
        color[node] = GRAY
        for dep in graph[node]:
            if color.get(dep) == GRAY:
                raise PlanValidationError(f"Dependency cycle detected: {' -> '.join(stack + [dep])}")
            if color.get(dep) == WHITE:
                visit(dep, stack + [dep])
        color[node] = BLACK

    for node in graph:
        if color[node] == WHITE:
            visit(node, [node])
