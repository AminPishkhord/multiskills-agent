"""
CLI entry point.

Usage:
    python -m src.main
    python -m src.main --once "summarize this: ..."
    python -m src.main --verbose   # also prints candidates/skills/plan
"""
from __future__ import annotations

import argparse
import sys

from src.config import get_settings
from src.discovery.skill_discovery import SkillDiscovery
from src.graph.graph import build_graph
from src.registry.skill_registry import build_default_registry


def _build_app():
    settings = get_settings()
    registry = build_default_registry()
    discovery = SkillDiscovery(registry, top_k=settings.discovery_top_k)
    return build_graph(registry, discovery)


def run_once(app, prompt: str, verbose: bool) -> str:
    result = app.invoke({"user_prompt": prompt})
    if verbose:
        plan_steps = [(s.id, s.skill, s.input_text) for s in result["plan"].steps]
        print(
            f"[candidates: {result.get('candidate_skills')}] "
            f"[skills: {result.get('skills')}] "
            f"[reasoning: {result.get('routing_reasoning')}] "
            f"[plan: {plan_steps}]",
            file=sys.stderr,
        )
    return result.get("final_response", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="MAP multi-skill agent CLI")
    parser.add_argument("--once", type=str, default=None, help="Run a single prompt and exit.")
    parser.add_argument("--verbose", action="store_true", help="Print discovery/routing/plan details.")
    args = parser.parse_args()

    app = _build_app()

    if args.once is not None:
        print(run_once(app, args.once, args.verbose))
        return

    print("MAP multi-skill agent. Type a message (English or Persian). Ctrl+C or 'exit' to quit.\n")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        try:
            print(run_once(app, prompt, args.verbose))
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
