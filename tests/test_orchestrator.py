import pytest

from src.orchestration.orchestrator import Orchestrator, PlanExecutionError
from src.registry.skill_registry import SkillRegistry
from src.schemas.plan import ExecutionPlan, PlanStep
from src.schemas.skill import SkillInput, SkillOutput
from src.skills.base import BaseSkill


class _UppercaseSkill(BaseSkill):
    name = "uppercase"
    description = "Uppercases text."

    def run(self, input_data: SkillInput) -> SkillOutput:
        return SkillOutput(output=input_data.text.upper())


class _ExclaimSkill(BaseSkill):
    name = "exclaim"
    description = "Adds an exclamation mark."

    def run(self, input_data: SkillInput) -> SkillOutput:
        return SkillOutput(output=input_data.text + "!")


@pytest.fixture()
def registry():
    reg = SkillRegistry()
    reg.register(_UppercaseSkill())
    reg.register(_ExclaimSkill())
    return reg


def test_single_step_plan(registry):
    plan = ExecutionPlan(steps=[PlanStep(id="step_1", skill="uppercase", input_text="hi")])
    results = Orchestrator(registry).execute(plan, original_prompt="hi")
    assert results["step_1"].output == "HI"


def test_parallel_steps_both_receive_original_prompt(registry):
    plan = ExecutionPlan(
        steps=[
            PlanStep(id="step_1", skill="uppercase", input_text="hello world"),
            PlanStep(id="step_2", skill="exclaim", input_text="hello world"),
        ]
    )
    results = Orchestrator(registry).execute(plan, original_prompt="hello world")
    assert results["step_1"].output == "HELLO WORLD"
    assert results["step_2"].output == "hello world!"


def test_sequential_chaining_via_step_reference(registry):
    plan = ExecutionPlan(
        steps=[
            PlanStep(id="step_1", skill="uppercase", input_text="hi there", depends_on=[]),
            PlanStep(id="step_2", skill="exclaim", input_text="$step_1.output", depends_on=["step_1"]),
        ]
    )
    results = Orchestrator(registry).execute(plan, original_prompt="hi there")
    assert results["step_1"].output == "HI THERE"
    assert results["step_2"].output == "HI THERE!"  # proves it chained, not the original prompt


def test_empty_input_text_falls_back_to_original_prompt(registry):
    plan = ExecutionPlan(steps=[PlanStep(id="step_1", skill="uppercase", input_text="x")])
    # Simulate a step with blank input_text by bypassing validation (min_length not
    # enforced on input_text) - orchestrator should still behave sanely.
    plan.steps[0].input_text = ""
    results = Orchestrator(registry).execute(plan, original_prompt="fallback text")
    assert results["step_1"].output == "FALLBACK TEXT"


def test_unresolvable_reference_raises():
    registry = SkillRegistry()
    registry.register(_UppercaseSkill())
    plan = ExecutionPlan(
        steps=[PlanStep(id="step_1", skill="uppercase", input_text="$step_missing.output", depends_on=[])]
    )
    with pytest.raises(PlanExecutionError):
        Orchestrator(registry).execute(plan, original_prompt="x")

    # Note: this plan is semantically invalid (dangling reference) and would normally
    # be caught by routing.planner.validate_plan before ever reaching the
    # Orchestrator; this test exercises the Orchestrator's own defense-in-depth.
