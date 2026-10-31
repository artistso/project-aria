"""Rewrite rules and calculus."""

import pytest

from aria.core.parse import parse
from aria.reason import rules as R
from aria.reason.engine import SymbolicEngine


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("x^2 + 5x + 6", "(x + 2) (x + 3)"),
        ("x^2 - 9", "(x - 3) (x + 3)"),
        ("2x^2 + 4x", "2 x (2 + x)"),
        ("x^2 - 5x + 6", "(x - 2) (x - 3)"),
    ],
)
def test_factoring(expr, expected):
    engine = SymbolicEngine()
    derivation = engine.factor(parse(expr))
    assert derivation.status == "solved"
    assert derivation.result == parse(expected)
    assert derivation.valid


def test_log_expansion():
    engine = SymbolicEngine()
    derivation = engine.simplify(parse("log(a*b)"))
    assert derivation.result == parse("log(a) + log(b)")
    assert derivation.valid


def test_difference_of_squares_rule_is_justified():
    rule = R.RULES["difference_of_squares"]
    result = rule.apply(parse("x^2 - 16"))
    assert result is not None
    assert "square" in result[1]


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("x^3", "3 x^2"),
        ("sin(x)", "cos(x)"),
        ("cos(x)", "-sin(x)"),
        ("log(x)", "1/x"),
        ("exp(2x)", "2 exp(2x)"),
        ("x^2 + 3x", "2 x + 3"),
    ],
)
def test_derivatives(expr, expected):
    from aria.core.ast import Symbol

    engine = SymbolicEngine()
    derivation = engine.differentiate(parse(expr), Symbol("x"))
    assert derivation.status == "solved", derivation.notes
    assert derivation.result == parse(expected)


def test_product_rule():
    from aria.core.ast import Symbol

    engine = SymbolicEngine()
    derivation = engine.differentiate(parse("x^2 sin(x)"), Symbol("x"))
    assert derivation.status == "solved"
    assert engine.verifier.equivalent(derivation.result, parse("2x sin(x) + x^2 cos(x)")).ok


def test_rule_application_is_bottom_up():
    expr = parse("(x + x) * (1 + 1)")
    rewritten, _ = R.apply_rule_everywhere(expr, R.RULES["collect_like_terms"])
    assert rewritten == parse("2 x * 2")
