import pytest

from src.routing.planner import PlanValidationError, build_trivial_plan, validate_plan
from src.schemas.plan import ExecutionPlan, PlanStep


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(steps=list(steps))


def test_valid_sequential_plan_passes():
    plan = _plan(
        PlanStep(id="step_1", skill="summarizer", input_text="hello", depends_on=[]),
        PlanStep(id="step_2", skill="translator", input_text="$step_1.output", depends_on=["step_1"]),
    )
    validate_plan(plan, allowed_skill_names={"summarizer", "translator"})  # should not raise


def test_valid_parallel_plan_passes():
    plan = _plan(
        PlanStep(id="step_1", skill="calculator", input_text="hi, 2+2?", depends_on=[]),
        PlanStep(id="step_2", skill="general_chat", input_text="hi, 2+2?", depends_on=[]),
    )
    validate_plan(plan, allowed_skill_names={"calculator", "general_chat"})  # should not raise


def test_unknown_skill_rejected():
    plan = build_trivial_plan("not_a_real_skill", "hi")
    with pytest.raises(PlanValidationError):
        validate_plan(plan, allowed_skill_names={"general_chat"})


def test_dangling_depends_on_rejected():
    plan = _plan(PlanStep(id="step_1", skill="summarizer", input_text="hi", depends_on=["step_99"]))
    with pytest.raises(PlanValidationError):
        validate_plan(plan, allowed_skill_names={"summarizer"})


def test_self_dependency_rejected():
    plan = _plan(PlanStep(id="step_1", skill="summarizer", input_text="hi", depends_on=["step_1"]))
    with pytest.raises(PlanValidationError):
        validate_plan(plan, allowed_skill_names={"summarizer"})


def test_reference_without_listed_dependency_rejected():
    # Uses $step_1.output but forgets to declare depends_on=["step_1"].
    plan = _plan(
        PlanStep(id="step_1", skill="summarizer", input_text="hi", depends_on=[]),
        PlanStep(id="step_2", skill="translator", input_text="$step_1.output", depends_on=[]),
    )
    with pytest.raises(PlanValidationError):
        validate_plan(plan, allowed_skill_names={"summarizer", "translator"})


def test_reference_to_unknown_step_rejected():
    plan = _plan(
        PlanStep(id="step_1", skill="translator", input_text="$step_7.output", depends_on=["step_7"])
    )
    with pytest.raises(PlanValidationError):
        validate_plan(plan, allowed_skill_names={"translator"})


def test_dependency_cycle_rejected():
    plan = _plan(
        PlanStep(id="step_1", skill="summarizer", input_text="$step_2.output", depends_on=["step_2"]),
        PlanStep(id="step_2", skill="translator", input_text="$step_1.output", depends_on=["step_1"]),
    )
    with pytest.raises(PlanValidationError):
        validate_plan(plan, allowed_skill_names={"summarizer", "translator"})


def test_build_trivial_plan_shape():
    plan = build_trivial_plan("general_chat", "hello there")
    assert len(plan.steps) == 1
    assert plan.steps[0].skill == "general_chat"
    assert plan.steps[0].input_text == "hello there"
    assert plan.steps[0].depends_on == []
