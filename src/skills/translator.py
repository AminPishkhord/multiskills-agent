from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import get_llm
from src.schemas.skill import SkillInput, SkillOutput
from src.skills.base import BaseSkill

_SYSTEM_PROMPT = """You are the Translator skill of a multi-skill assistant.
The user's message contains a request to translate some text, and usually (but not
always) names a target language. The message may ALSO contain other unrelated
requests (math questions, opinions, small talk) - you must completely ignore those
and focus ONLY on the translation part.

Instructions:
1. Identify ONLY the specific text/phrase that needs to be translated, ignoring any
   other sentences, questions, or requests in the same message.
2. If the target language is NOT explicitly stated: if the source text is Persian
   (Farsi), translate it to English; otherwise translate it to Persian (Farsi). This
   is a deliberate default given this assistant's bilingual Persian/English focus.
3. Produce a fluent, natural, accurate translation - not a word-for-word gloss.
   Preserve tone, register, and any names/numbers exactly.
4. Output ONLY the translated text itself. Do NOT answer any other question in the
   message, do NOT give opinions, do NOT add commentary, do NOT repeat the original
   text, do NOT use markdown or bullet points - just the plain translated text.
"""


class TranslatorSkill(BaseSkill):
    name = "translator"
    description = (
        "Translates text from one language to another, including Persian <-> "
        "English. Use for requests that explicitly ask for a translation."
    )

    def run(self, input_data: SkillInput) -> SkillOutput:
        llm = get_llm(temperature=0.2)
        response = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=input_data.text)]
        )
        return SkillOutput(output=response.content)
