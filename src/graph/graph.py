"""
Assembles the LangGraph graph, per spec.md sections 15 and 19:

  START -> discovery -> router -> planner -> execute -> combine -> END

Note what's NOT here: there is no per-skill node and no conditional fan-out keyed on
skill name. Dispatch to the actual skill implementation happens generically inside
`execute` (via Orchestrator + Registry - see src/orchestration/orchestrator.py), so
this graph's shape never changes when a skill is added or removed. That's the
concrete difference from a naive "router -> [skill_a, skill_b, skill_c, skill_d] ->
combine" graph: here, scaling to 30 skills means registering 30 skills, not adding 30
nodes and edges.
"""
from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from src.discovery.skill_discovery import FALLBACK_SKILL_NAME, SkillDiscovery
from src.graph.state import AgentState
from src.llm.client import get_llm
from src.orchestration.orchestrator import Orchestrator
from src.registry.skill_registry import SkillRegistry
from src.routing.planner import decide_plan
from src.routing.router import decide_skills

_PERSIAN_RE = re.compile(r"[\u0600-\u06FF]")

_LABELS_EN = {
    "summarizer": "Summary",
    "translator": "Translation",
    "calculator": "Calculation",
    "general_chat": "Reply",
}
_LABELS_FA = {
    "summarizer": "خلاصه",
    "translator": "ترجمه",
    "calculator": "محاسبه",
    "general_chat": "پاسخ",
}


def build_graph(registry: SkillRegistry, discovery: SkillDiscovery):
    def discovery_node(state: AgentState) -> dict:
        return {"candidate_skills": discovery.candidates(state["user_prompt"])}

    def router_node(state: AgentState) -> dict:
        candidate_specs = [registry.get(n) for n in state["candidate_skills"]]
        llm = get_llm(temperature=0.0)
        try:
            decision = decide_skills(llm, state["user_prompt"], candidate_specs)
            skills = decision.skills
            reasoning = decision.reasoning
        except Exception as exc:  # noqa: BLE001 - deliberate fallback boundary
            # skills, reasoning = [], f"Fallback due to routing error: {exc}"
            raise RuntimeError(f"Routing failed: {exc}") from exc
        
        
        if not skills:
            # Fallback behavior (spec.md section 12): nothing parseable / nothing
            # matched -> General Chat if it's registered, else just the first
            # candidate we were handed, so we never return zero skills.
            if registry.has(FALLBACK_SKILL_NAME):
                skills = [FALLBACK_SKILL_NAME]
                reasoning = reasoning or "No clear skill matched; falling back to general_chat."
            else:
                skills = [candidate_specs[0].name]
        print("ROUTER CANDIDATES:", state["candidate_skills"])
        print("ROUTER SELECTED:", skills)
        print("ROUTER REASONING:", reasoning)
        return {"skills": skills, "routing_reasoning": reasoning}

    def planner_node(state: AgentState) -> dict:
        selected_specs = [registry.get(n) for n in state["skills"]]
        print("PLANNER SELECTED:", [s.name for s in selected_specs])
        llm = get_llm(temperature=0.0)
        plan = decide_plan(llm, state["user_prompt"], selected_specs)
        print("PLAN:", plan)
        return {"plan": plan}

    def execute_node(state: AgentState) -> dict:
        orchestrator = Orchestrator(registry)
        results = orchestrator.execute(state["plan"], state["user_prompt"])
        step_results = [
            {"step_id": step.id, "skill": step.skill, "output": results[step.id].output}
            for step in state["plan"].steps
        ]
        return {"execution_results": step_results}

    def combine_node(state: AgentState) -> dict:
        results = state["execution_results"]
        if len(results) == 1:
            return {"final_response": results[0]["output"]}

        persian = bool(_PERSIAN_RE.search(state["user_prompt"]))
        labels = _LABELS_FA if persian else _LABELS_EN
        parts = [f"**{labels.get(r['skill'], r['skill'])}:**\n{r['output']}" for r in results]
        return {"final_response": "\n\n".join(parts)}

    graph = StateGraph(AgentState)
    graph.add_node("discovery", discovery_node)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("execute", execute_node)
    graph.add_node("combine", combine_node)

    graph.add_edge(START, "discovery")
    graph.add_edge("discovery", "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "execute")
    graph.add_edge("execute", "combine")
    graph.add_edge("combine", END)

    return graph.compile()
