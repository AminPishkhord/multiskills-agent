"""
Centralized configuration, loaded from environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    site_url: str
    app_name: str
    temperature: float = 0.1
    # How many candidate skills Skill Discovery hands to the Router. With only 4
    # skills registered today this effectively returns all of them; the point is
    # that this number does NOT grow as skills are added (see src/discovery).
    discovery_top_k: int = 4


def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in "
            "a free API key from https://openrouter.ai/keys"
        )

    return Settings(
        openrouter_api_key=api_key,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        site_url=os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        app_name=os.getenv("OPENROUTER_APP_NAME", "MAP-Multiskill-agent"),
        discovery_top_k=int(os.getenv("DISCOVERY_TOP_K", "4")),
    )
