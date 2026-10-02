"""
Calculator skill.

Per the brief: the LLM must NEVER guess the numeric result. Here the LLM's only job
is to extract/normalize a free-form request (possibly Persian, possibly a word
problem) into a plain arithmetic expression string; `_safe_eval` then evaluates it
deterministically via a whitelisted AST walker - no `eval()`/`exec()` on raw input.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Callable

from pydantic import BaseModel, Field

from src.llm.client import call_structured, get_llm
from src.schemas.skill import SkillInput, SkillOutput
from src.skills.base import BaseSkill

_PERSIAN_RE = re.compile(r"[\u0600-\u06FF]")


# ---------------------------------------------------------------------------
# Safe AST-based expression evaluator (the "real computational tool call")
# ---------------------------------------------------------------------------

class CalculatorError(ValueError):
    """Raised when an expression is invalid or contains disallowed constructs."""


_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": math.pow,
    "min": min,
    "max": max,
}

_ALLOWED_NAMES: dict[str, float] = {"pi": math.pi, "e": math.e}

_MAX_EXPRESSION_LENGTH = 500


def safe_eval(expression: str) -> float:
    if not expression or not expression.strip():
        raise CalculatorError("Empty expression.")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Could not parse expression: {exc}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Disallowed constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise CalculatorError(f"Disallowed operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        try:
            return op_func(left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero.") from exc

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise CalculatorError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise CalculatorError("Disallowed function call.")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not allowed.")
        args = [_eval_node(arg) for arg in node.args]
        try:
            return _ALLOWED_FUNCS[node.func.id](*args)
        except (TypeError, ValueError) as exc:
            raise CalculatorError(f"Error calling {node.func.id}(): {exc}") from exc

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise CalculatorError(f"Disallowed name: {node.id}")

    raise CalculatorError(f"Disallowed expression element: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

class _ExpressionExtraction(BaseModel):
    expression: str = Field(default="")
    found: bool = Field(default=False)


_EXTRACTION_SYSTEM_PROMPT = """You extract mathematical expressions from free-form
user requests (which may be in English, Persian, or mixed). You do NOT compute
anything yourself - you only normalize the request into a single Python-evaluable
arithmetic expression.

Allowed in the output expression: digits, + - * / // % ** ( ), and these function
names only: sqrt, abs, round, floor, ceil, log, log10, log2, exp, sin, cos, tan, pow,
min, max - plus the names pi and e. Convert words/word-problems into this form, e.g.:
- "23 percent of 480" -> "(23/100)*480"
- "بیست و سه درصد از ۴۸۰" -> "(23/100)*480"
- "جذر ۱۶ به علاوه ۲ به توان ۳" -> "sqrt(16) + 2**3"
- "half of 18 minus 4" -> "(18/2) - 4"

Convert any Persian/Arabic-indic digits to normal digits in the expression.

Output ONLY a single JSON object, no prose, no markdown fences, in exactly this shape:
{"expression": "<python-evaluable expression, or empty string>", "found": true|false}

Set "found" to false and "expression" to "" if the message does not actually contain
a clear, unambiguous mathematical computation.
"""


class CalculatorSkill(BaseSkill):
    name = "calculator"
    description = (
        "Performs mathematical calculations: arithmetic, percentages, roots, "
        "exponents, and simple numeric word problems. Use for requests that ask to "
        "compute, calculate, or work out a numeric result."
    )

    def run(self, input_data: SkillInput) -> SkillOutput:
        text = input_data.text
        persian = bool(_PERSIAN_RE.search(text))
        llm = get_llm(temperature=0.0)

        try:
            extraction = call_structured(llm, _EXTRACTION_SYSTEM_PROMPT, text, _ExpressionExtraction)
            if not extraction.found or not extraction.expression.strip():
                raise CalculatorError("No clear mathematical expression was found.")

            result = safe_eval(extraction.expression)

            if persian:
                output = f"نتیجه محاسبه‌ی «{extraction.expression}» برابر است با: {result}"
            else:
                output = f"The result of `{extraction.expression}` is: {result}"

        except CalculatorError as exc:
            if persian:
                output = f"متأسفم، نتونستم یک عبارت ریاضی واضح در پیام شما پیدا کنم یا محاسبه کنم. ({exc})"
            else:
                output = f"Sorry, I couldn't find or compute a clear mathematical expression. ({exc})"

        return SkillOutput(output=output)
