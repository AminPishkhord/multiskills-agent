from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import get_llm
from src.schemas.skill import SkillInput, SkillOutput
from src.skills.base import BaseSkill

_SYSTEM_PROMPT = """You are the Summarizer skill of a multi-skill assistant.
The user will give you a free-form message that contains, somewhere in it, a request
to summarize a piece of text (the text to summarize may be embedded in the same
message, e.g. after a colon, or may be the message minus the instruction words).

Instructions:
1. Identify the actual text to summarize (ignore instruction phrases like "summarize
   this:", "tl;dr", "خلاصه کن", etc. - those are not part of the content).
2. Detect the language of that text.
3. Write the summary in the SAME language as the source text (e.g. Persian text gets
   a Persian summary; English text gets an English summary), unless the user
   explicitly asked for the summary in a different language.
4. The summary must be clearly shorter than the original (roughly 20-30% of the
   original length for longer texts, or 1-2 sentences for short texts), must preserve
   the key facts/claims, and must not add opinions or information not present in the
   source.
5. Output ONLY the summary text itself - no preamble like "Here is the summary:", no
   labels, no markdown.
"""


class SummarizerSkill(BaseSkill):
    name = "summarizer"
    description = (
        "Summarizes or condenses a piece of text into a shorter version that "
        "preserves the key facts. Use for requests that ask for a summary, tl;dr, "
        "or to shorten/condense text."
    )

    def run(self, input_data: SkillInput) -> SkillOutput:
        llm = get_llm(temperature=0.2)
        response = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=input_data.text)]
        )
        return SkillOutput(output=response.content)
