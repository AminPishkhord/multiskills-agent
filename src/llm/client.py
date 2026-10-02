"""
LLM client setup (OpenRouter, via the OpenAI-compatible Chat Completions API) plus a
small helper for getting Pydantic-validated structured output out of a free model.

We deliberately do NOT rely on OpenAI-style tool/function calling for structured
output: free OpenRouter models have inconsistent tool-calling support. Instead we
prompt the model to emit raw JSON, extract the JSON object from the response, and
validate it with Pydantic - retrying with the validation error fed back to the model
if the first attempt fails. This is more robust across the free-model landscape, and
is used by every LLM-structured-output consumer in this project: the Router, the
Planner, and the Calculator skill's expression-extraction step.
"""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from src.config import get_settings

T = TypeVar("T", bound=BaseModel)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature if temperature is None else temperature,
        default_headers={
            "HTTP-Referer": settings.site_url,
            "X-Title": settings.app_name,
        },
    )


def _extract_json(text: str) -> str:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return match.group(0)


def call_structured(
    llm,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    max_retries: int = 2,
) -> T:
    """
    Call the LLM, asking for JSON matching `schema`, and return a validated instance.
    Retries with the validation error appended to the prompt if parsing/validation
    fails. Raises the last error if all retries are exhausted (callers should catch
    this and fall back to a safe default, e.g. General Chat).
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        response = llm.invoke(messages)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        try:
            json_str = _extract_json(raw_text)
            data = json.loads(json_str)
            return schema.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous reply was not valid JSON matching the required "
                        f"schema. Error: {exc}. Reply again with ONLY a single valid "
                        "JSON object, no prose, no markdown code fences."
                    )
                )
            )

    assert last_error is not None
    raise last_error
