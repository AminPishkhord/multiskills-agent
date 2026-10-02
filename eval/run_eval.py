"""
Runs Discovery + Router (not the full Planner/Orchestrator pipeline, to keep scoring
fast and cheap) against eval/eval_set.json and writes a Markdown accuracy report to
eval/eval_report.md.

Needs a valid OPENROUTER_API_KEY in your environment / .env file. Run from the
project root:

    python -m eval.run_eval

Scored two ways:
  - Exact match: predicted skill set equals expected skill set exactly.
  - Set-overlap (Jaccard): intersection / union size, for partial credit on
    multi-skill prompts where only one of the two skills was caught.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.discovery.skill_discovery import FALLBACK_SKILL_NAME, SkillDiscovery
from src.llm.client import get_llm
from src.registry.skill_registry import build_default_registry
from src.routing.router import decide_skills

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
REPORT_PATH = Path(__file__).parent / "eval_report.md"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def route_prompt(registry, discovery, llm, prompt: str) -> tuple[list[str], str]:
    candidate_names = discovery.candidates(prompt)
    candidate_specs = [registry.get(n) for n in candidate_names]
    try:
        decision = decide_skills(llm, prompt, candidate_specs)
        skills = decision.skills or [FALLBACK_SKILL_NAME]
        return skills, decision.reasoning
    except Exception as exc:  # noqa: BLE001
        return [FALLBACK_SKILL_NAME], f"fallback: {exc}"


def main() -> None:
    settings = get_settings()
    registry = build_default_registry()
    discovery = SkillDiscovery(registry, top_k=settings.discovery_top_k)
    llm = get_llm(temperature=0.0)

    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    rows = []
    exact_hits = 0
    jaccard_sum = 0.0

    for case in cases:
        expected = set(case["expected_skills"])
        predicted_list, reasoning = route_prompt(registry, discovery, llm, case["prompt"])
        predicted = set(predicted_list)

        is_exact = predicted == expected
        overlap = jaccard(predicted, expected)
        exact_hits += int(is_exact)
        jaccard_sum += overlap

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "reasoning": reasoning,
                "exact": is_exact,
                "jaccard": round(overlap, 2),
            }
        )

    n = len(cases)
    exact_accuracy = exact_hits / n if n else 0.0
    mean_jaccard = jaccard_sum / n if n else 0.0

    lines = [
        "# Skill-Detection Evaluation Report",
        "",
        f"Total cases: {n}",
        f"Exact-match accuracy: {exact_hits}/{n} = {exact_accuracy:.1%}",
        f"Mean Jaccard (partial credit) score: {mean_jaccard:.2f}",
        "",
        "| ID | Category | Prompt | Expected | Predicted | Exact | Jaccard | Reasoning |",
        "|----|----------|--------|----------|-----------|-------|---------|-----------|",
    ]
    for r in rows:
        prompt_preview = r["prompt"].replace("|", "\\|")
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:57] + "..."
        lines.append(
            f"| {r['id']} | {r['category']} | {prompt_preview} | "
            f"{', '.join(r['expected'])} | {', '.join(r['predicted'])} | "
            f"{'✅' if r['exact'] else '❌'} | {r['jaccard']} | {r['reasoning']} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({exact_hits}/{n} exact matches, mean Jaccard {mean_jaccard:.2f})")


if __name__ == "__main__":
    main()
