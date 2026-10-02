import pytest
from pydantic import ValidationError

from src.schemas.plan import ExecutionPlan, PlanStep, is_step_reference, parse_step_reference
from src.schemas.skill import RoutingDecision


def test_routing_decision_caps_at_two_skills():
    decision = RoutingDecision.model_validate(
        {"skills": ["summarizer", "translator", "calculator"], "reasoning": "x"}
    )
    assert decision.skills == ["summarizer", "translator"]


def test_routing_decision_dedupes():
    decision = RoutingDecision.model_validate({"skills": ["general_chat", "general_chat"]})
    assert decision.skills == ["general_chat"]


def test_routing_decision_defaults_to_empty_list():
    decision = RoutingDecision.model_validate({})
    assert decision.skills == []


def test_execution_plan_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate({"steps": []})


def test_execution_plan_rejects_more_than_two_steps():
    steps = [{"id": f"step_{i}", "skill": "x", "input_text": "t"} for i in range(3)]
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate({"steps": steps})


def test_execution_plan_rejects_duplicate_step_ids():
    steps = [
        {"id": "step_1", "skill": "a", "input_text": "t"},
        {"id": "step_1", "skill": "b", "input_text": "t"},
    ]
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate({"steps": steps})


def test_step_reference_helpers():
    assert is_step_reference("$step_1.output") is True
    assert is_step_reference("just some text") is False
    assert parse_step_reference("$step_1.output") == "step_1"
    with pytest.raises(ValueError):
        parse_step_reference("not a reference")
