from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import get_llm
from src.schemas.skill import SkillInput, SkillOutput
from src.skills.base import BaseSkill

_SYSTEM_PROMPT = """You are the General Chat skill of a friendly, helpful
multi-skill assistant (the other skills are Summarizer, Translator, and Calculator).
Handle greetings, small talk, opinions, and general questions.

Instructions:
1. Reply in the SAME language the user wrote in (if they wrote in Persian, reply in
   natural, colloquial Persian; if English, reply in English; etc.).
2. Keep replies concise and conversational.
3. If the message seems like it might have needed Summarizer, Translator, or
   Calculator but was too ambiguous to route confidently, you may briefly mention
   what you can help with (summaries, translations, calculations) as well as
   answering generally - but do not be pushy about it.
"""


class GeneralChatSkill(BaseSkill):
    name = "general_chat"
    description = (
        "Handles greetings, small talk, opinions, and general knowledge questions "
        "that are not a summarization, translation, or calculation request. Also "
        "the fallback skill when no other skill clearly applies."
    )

    def run(self, input_data: SkillInput) -> SkillOutput:
        llm = get_llm(temperature=0.5)
        response = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=input_data.text)]
        )
        return SkillOutput(output=response.content)
