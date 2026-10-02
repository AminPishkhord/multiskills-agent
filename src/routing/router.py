"""
Router, per spec.md section 6.

The Router only ever sees the candidate skills that Discovery already narrowed the
request down to (see src/discovery) - never the full registry - so its prompt size
stays bounded as the registry grows. It still enforces "1 <= len(skills) <= 2" and
validates every returned name against the candidate set; if the model returns
something outside that set, or returns nothing parseable at all, `decide_skills`
falls back to `["general_chat"]` (see `src/graph/graph.py` for where that fallback is
actually applied).
"""
from __future__ import annotations

from src.llm.client import call_structured
from src.registry.skill_registry import SkillSpec
from src.schemas.skill import RoutingDecision

# A few worked examples for the canonical 4 skills, shown only if that skill is
# actually among this request's candidates. New skills added later simply won't have
# a canonical example here - a reasonable gap for a prototype; a production version
# could auto-generate a 1-line example per skill from its registered description.
_CANONICAL_EXAMPLES = {
    "calculator": '''User: "what's (45 + 12) * 3?"\n{"skills": ["calculator"], "reasoning": "User asks for an arithmetic computation."}''',
    "translator": '''User: "translate 'good morning' to Persian"\n{"skills": ["translator"], "reasoning": "User asks for a translation."}''',
    "summarizer": '''User: "Can you sum up this article for me? <long text>"\n{"skills": ["summarizer"], "reasoning": "User explicitly asks for a summary of text."}''',
    "general_chat": '''User: "سلام، حالت چطوره؟"\n{"skills": ["general_chat"], "reasoning": "This is a Persian greeting / small talk."}''',
}


def _build_system_prompt(candidates: list[SkillSpec]) -> str:
    skill_lines = "\n".join(f'- "{c.name}": {c.description}' for c in candidates)
    examples = "\n\n".join(
        _CANONICAL_EXAMPLES[c.name] for c in candidates if c.name in _CANONICAL_EXAMPLES
    )

    return f"""You are the routing engine of a multi-skill assistant. You do not
answer the user. Your only job is to decide which skill(s), out of ONLY the
following candidates, are needed to handle the user's message:

{skill_lines}

Rules:
1. The user's message can be in ANY language, including Persian (Farsi, written in
   the Arabic script). Route purely based on the *intent* of the message, regardless
   of language.
2. Return AT LEAST 1 and AT MOST 2 skill names, using the exact names listed above.
3. Only return 2 skills if the message clearly asks for two distinct operations.
4. If the message does not clearly match a specific candidate above, OR if you are
   genuinely unsure / it is ambiguous, prefer "general_chat" if it is among the
   candidates. Do not guess a specific skill just because a message contains numbers
   or foreign words - only route to it when the *intent* clearly matches its
   description.
5. Output ONLY a single JSON object, no prose, no markdown fences, in exactly this
   shape: {{"skills": ["<skill>", ...], "reasoning": "<one short sentence, in English>"}}

Examples:

{examples}
"""


def decide_skills(llm, prompt: str, candidates: list[SkillSpec]) -> RoutingDecision:
    """
    Returns a RoutingDecision whose `skills` are guaranteed to be a subset of
    `candidates`' names (anything else the model returns is filtered out). Raises if
    the LLM call/parse fails entirely after retries - callers decide the fallback.
    """
    candidate_names = {c.name for c in candidates}
    system_prompt = _build_system_prompt(candidates)

    decision = call_structured(llm, system_prompt, prompt, RoutingDecision)
    filtered_skills = [s for s in decision.skills if s in candidate_names][:2]
    return RoutingDecision(skills=filtered_skills, reasoning=decision.reasoning)
