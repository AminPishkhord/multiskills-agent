"""
Orchestrator, per spec.md section 10.

The Orchestrator must NOT know that "calculator" or "translator" exist as concepts -
it only knows: given a validated ExecutionPlan, resolve each step's input (literal
text, or a `$step_id.output` reference to an already-completed step), fetch that
step's skill from the Registry by name, and call `skill.run(...)`. There is
deliberately no `if skill == "calculator": ...` anywhere in this file.

Execution order respects `depends_on` via a simple repeated "run whatever's ready"
loop (a topological execution), which works for today's max-2-step plans and remains
correct if the step-count ceiling is ever raised. Steps with no unmet dependencies
that become ready in the same pass are executed sequentially here, not threaded/async
- see README.md "Known Limitations" for the note on adding real concurrency for a
production version.
"""
from __future__ import annotations

from src.registry.skill_registry import SkillRegistry
from src.schemas.plan import ExecutionPlan, PlanStep, is_step_reference, parse_step_reference
from src.schemas.skill import SkillInput, SkillOutput


class PlanExecutionError(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute(self, plan: ExecutionPlan, original_prompt: str) -> dict[str, SkillOutput]:
        results: dict[str, SkillOutput] = {}
        remaining: dict[str, PlanStep] = {step.id: step for step in plan.steps}

        while remaining:
            ready = [s for s in remaining.values() if all(dep in results for dep in s.depends_on)]
            if not ready:
                raise PlanExecutionError(
                    f"No runnable step found - unresolved dependencies among {list(remaining)}."
                )
            for step in ready:
                resolved_text = self._resolve_input_text(step, results, original_prompt)
                spec = self.registry.get(step.skill)  # generic lookup, no branching
                output = spec.implementation.run(SkillInput(text=resolved_text))
                results[step.id] = output
                del remaining[step.id]

        return results

    @staticmethod
    def _resolve_input_text(step: PlanStep, results: dict[str, SkillOutput], original_prompt: str) -> str:
        text = step.input_text
        if is_step_reference(text):
            ref_id = parse_step_reference(text)
            if ref_id not in results:
                raise PlanExecutionError(f"Step '{step.id}' references unresolved step '{ref_id}'.")
            return results[ref_id].output
        return text or original_prompt
