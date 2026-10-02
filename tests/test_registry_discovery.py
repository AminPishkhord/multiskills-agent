import pytest

from src.discovery.skill_discovery import FALLBACK_SKILL_NAME, SkillDiscovery
from src.registry.skill_registry import SkillRegistry, build_default_registry
from src.schemas.skill import SkillInput, SkillOutput
from src.skills.base import BaseSkill

class _FakeIndex:
    """Deterministic stand-in for a real embedding backend, so this test checks
    SkillDiscovery's own top_k/fallback logic, not any particular model's quality."""

    def __init__(self, ranking_order: list[str]):
        self._ranking_order = ranking_order

    def fit(self, names, corpus):
        pass  # no-op: ranking is fixed up front

    def rank(self, query):
        return list(self._ranking_order)


def test_discovery_narrows_when_registry_exceeds_top_k():
    registry = SkillRegistry()
    registry.register(_EchoSkill("cooking", "recipes, ingredients, cooking instructions, baking"))
    registry.register(_EchoSkill("weather", "weather forecast, temperature, rain, climate"))
    registry.register(_EchoSkill("sports", "football, basketball, sports scores and teams"))
    registry.register(_EchoSkill("music", "songs, albums, musicians, concerts, playlists"))
    registry.register(_EchoSkill(FALLBACK_SKILL_NAME, "general conversation and small talk"))

    fake_index = _FakeIndex(ranking_order=["weather", "cooking", "sports", "music", FALLBACK_SKILL_NAME])
    discovery = SkillDiscovery(registry, top_k=2, index=fake_index)
    candidates = discovery.candidates("what's the weather forecast for tomorrow, will it rain?")

    assert len(candidates) == 2
    assert "weather" in candidates
    assert FALLBACK_SKILL_NAME in candidates


class _EchoSkill(BaseSkill):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, input_data: SkillInput) -> SkillOutput:
        return SkillOutput(output=input_data.text)


def test_registry_register_get_and_names():
    registry = SkillRegistry()
    registry.register(_EchoSkill("alpha", "does alpha things"))
    registry.register(_EchoSkill("beta", "does beta things"))

    assert set(registry.names()) == {"alpha", "beta"}
    assert registry.get("alpha").description == "does alpha things"
    assert registry.has("beta") is True
    assert registry.has("gamma") is False


def test_registry_rejects_duplicate_names():
    registry = SkillRegistry()
    registry.register(_EchoSkill("alpha", "first"))
    with pytest.raises(ValueError):
        registry.register(_EchoSkill("alpha", "second"))


def test_registry_get_unknown_raises_keyerror():
    registry = SkillRegistry()
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_default_registry_has_the_four_required_skills():
    registry = build_default_registry()
    assert set(registry.names()) == {"summarizer", "translator", "calculator", "general_chat"}


def test_discovery_small_registry_returns_everything():
    # With top_k >= number of registered skills, Discovery is a pass-through -
    # exactly the "4 skills today" behavior the architecture is meant to preserve.
    registry = build_default_registry()
    discovery = SkillDiscovery(registry, top_k=4)
    candidates = discovery.candidates("completely irrelevant text about nothing in particular")
    assert set(candidates) == set(registry.names())


# def test_discovery_narrows_when_registry_exceeds_top_k():
#     registry = SkillRegistry()
#     registry.register(_EchoSkill("cooking", "recipes, ingredients, cooking instructions, baking"))
#     registry.register(_EchoSkill("weather", "weather forecast, temperature, rain, climate"))
#     registry.register(_EchoSkill("sports", "football, basketball, sports scores and teams"))
#     registry.register(_EchoSkill("music", "songs, albums, musicians, concerts, playlists"))
#     registry.register(_EchoSkill(FALLBACK_SKILL_NAME, "general conversation and small talk"))

#     discovery = SkillDiscovery(registry, top_k=2)
#     candidates = discovery.candidates("what's the weather forecast for tomorrow, will it rain?")

#     assert len(candidates) == 2
#     assert "weather" in candidates
#     # Fallback skill must always be reachable even though it's not topically similar.
#     assert FALLBACK_SKILL_NAME in candidates
