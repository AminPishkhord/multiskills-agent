"""
Shared graph state, mirroring spec.md section 16's conceptual AgentState:
user_prompt -> candidate_skills -> routing_decision -> plan -> execution_results ->
final_response. Each LangGraph node reads what it needs and writes only its own slice.
"""
from __future__ import annotations

from typing import TypedDict

from src.schemas.plan import ExecutionPlan


class StepResult(TypedDict):
    step_id: str
    skill: str
    output: str


class AgentState(TypedDict, total=False):
    user_prompt: str
    candidate_skills: list[str]
    skills: list[str]
    routing_reasoning: str
    plan: ExecutionPlan
    execution_results: list[StepResult]
    final_response: str
