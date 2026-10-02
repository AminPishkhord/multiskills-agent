import math

import pytest

from src.skills.calculator import CalculatorError, safe_eval


def test_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20
    assert safe_eval("10 / 4") == 2.5
    assert safe_eval("10 // 4") == 2
    assert safe_eval("10 % 3") == 1
    assert safe_eval("2 ** 10") == 1024


def test_functions_and_constants():
    assert safe_eval("sqrt(16)") == 4.0
    assert math.isclose(safe_eval("pi"), math.pi)
    assert math.isclose(safe_eval("sin(0)"), 0.0)
    assert safe_eval("max(3, 7, 2)") == 7
    assert safe_eval("round(3.14159, 2)") == 3.14


def test_percentage_style_expression():
    assert math.isclose(safe_eval("(23/100)*480"), 110.4)


def test_unary_minus():
    assert safe_eval("-5 + 3") == -2


def test_division_by_zero_raises():
    with pytest.raises(CalculatorError):
        safe_eval("1/0")


def test_empty_expression_raises():
    with pytest.raises(CalculatorError):
        safe_eval("")
    with pytest.raises(CalculatorError):
        safe_eval("   ")


def test_disallowed_function_call_raises():
    with pytest.raises(CalculatorError):
        safe_eval('__import__("os").system("echo hi")')


def test_disallowed_name_raises():
    with pytest.raises(CalculatorError):
        safe_eval("some_undefined_name + 1")


def test_string_literal_raises():
    with pytest.raises(CalculatorError):
        safe_eval("'a' + 'b'")


def test_syntax_error_raises():
    with pytest.raises(CalculatorError):
        safe_eval("2 + * 3")


def test_expression_too_long_raises():
    with pytest.raises(CalculatorError):
        safe_eval("1+" * 1000 + "1")
