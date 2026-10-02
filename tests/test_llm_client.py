import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.llm.client import call_structured


class _DummySchema(BaseModel):
    value: int
    label: str = ""


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._responses.pop(0))


def test_parses_clean_json():
    llm = _FakeLLM(['{"value": 7, "label": "seven"}'])
    result = call_structured(llm, "system", "user", _DummySchema)
    assert result.value == 7
    assert result.label == "seven"


def test_strips_surrounding_prose_and_fences():
    llm = _FakeLLM(['Sure, here:\n```json\n{"value": 3}\n```'])
    result = call_structured(llm, "system", "user", _DummySchema)
    assert result.value == 3


def test_retries_then_succeeds():
    llm = _FakeLLM(["not json at all", '{"value": 1}'])
    result = call_structured(llm, "system", "user", _DummySchema, max_retries=2)
    assert result.value == 1
    assert llm.calls == 2


def test_raises_after_exhausting_retries():
    llm = _FakeLLM(["nope", "still nope", "nope again"])
    with pytest.raises(Exception):
        call_structured(llm, "system", "user", _DummySchema, max_retries=2)
