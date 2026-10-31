"""Parsing: notation, natural-language goals, and loud failures."""

import pytest

from aria.core.ast import Congruence, Derivative, Equation, Matrix
from aria.core.parse import ParseError, parse, parse_problem
from aria.core.state import GoalKind


@pytest.mark.parametrize(
    "text",
    [
        "2x + 4 = 10",
        "3 x^2 - 2 x + 1",
        "(x + 1)(x - 1)",
        "1/2 x + 3/4",
        "sqrt(x) + log(y)",
        "exp(2 x)",
        "sin x + cos(x)",
        "2^10",
        "[[1, 2], [3, 4]]",
        "x ≡ 3 mod 7",
        "1.5e-3 y",
    ],
)
def test_notation_parses(text):
    assert parse(text) is not None


def test_equation_structure():
    expr = parse("2x + 4 = 10")
    assert isinstance(expr, Equation)
    assert str(expr.lhs) == "4 + 2 x"
    assert str(expr.rhs) == "10"


def test_congruence_structure():
    expr = parse("x ≡ 3 mod 7")
    assert isinstance(expr, Congruence)
    assert str(expr.modulus) == "7"


def test_matrix_shape():
    matrix = parse("[[1, 2, 3], [4, 5, 6]]")
    assert isinstance(matrix, Matrix)
    assert matrix.shape == (2, 3)


def test_derivative_notation():
    expr = parse("d/dx[x^2 + 3 x]")
    assert isinstance(expr, Derivative)
    assert expr.var.name == "x"


def test_implicit_multiplication_precedence():
    assert parse("2x^2") == parse("2 * x^2")


@pytest.mark.parametrize("text", ["2 +", "(x + 1", "x @ y", ""])
def test_invalid_input_raises(text):
    with pytest.raises(ParseError):
        parse(text)


def test_goal_extraction():
    state = parse_problem("solve for x in 2x + 4 = 10")
    assert state.goal.kind is GoalKind.SOLVE
    assert state.goal.var.name == "x"


def test_goal_extraction_without_for_clause():
    state = parse_problem("solve x^2 - 9 = 0")
    assert state.goal.kind is GoalKind.SOLVE


def test_instruction_words_are_not_variables():
    state = parse_problem("simplify (x + 1)^2")
    assert [s.name for s in state.symbols] == ["x"]


def test_verify_goal():
    state = parse_problem("verify log(a*b) = log(a) + log(b)")
    assert state.goal.kind is GoalKind.VERIFY


def test_differentiate_goal_infers_variable():
    state = parse_problem("differentiate x^3 with respect to x")
    assert state.goal.kind is GoalKind.DIFFERENTIATE
    assert state.goal.var.name == "x"
