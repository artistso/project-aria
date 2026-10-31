"""Symbolic object model: exactness, canonicalisation, rendering."""

from fractions import Fraction

import pytest

from aria.core.ast import Number, Symbol, add, mul, num, power, sym


def test_exact_arithmetic_never_uses_floats():
    value = add(Fraction(1, 3), Fraction(2, 3))
    assert isinstance(value, Number)
    assert value.value == 1
    assert value.is_exact


def test_decimal_literals_are_exact():
    assert num(0.5).value == Fraction(1, 2)
    assert Number(Fraction("0.125")).value == Fraction(1, 8)


def test_like_terms_are_not_folded_by_the_constructor():
    expr = add(mul(2, sym("x")), mul(3, sym("x")))
    assert str(expr).count("x") == 2  # 2x + 3x, collapsing is a *rule*, not parsing


def test_identical_factors_become_powers():
    expr = mul(sym("x"), sym("x"))
    assert expr == power(sym("x"), 2)


def test_zero_and_one_identities():
    assert add() == num(0)
    assert mul() == num(1)
    assert add(num(0), sym("x")) == sym("x")
    assert mul(num(1), sym("x")) == sym("x")


def test_canonical_forms_hash_equal():
    left = add(sym("y"), sym("x"))
    right = add(sym("x"), sym("y"))
    assert left == right
    assert hash(left) == hash(right)


def test_render_roundtrip():
    from aria.core.parse import parse

    for text in ("x^2 + 2 x + 1", "log(a b)", "(1 + x) (2 + x)", "x^(1/2)"):
        expr = parse(text)
        assert parse(str(expr)) == expr


def test_substitution_is_structural():
    expr = parse_expr("x^2 + y")
    replaced = expr.subs({sym("x"): num(3)})
    assert str(replaced) == "9 + y"


def parse_expr(text: str):
    from aria.core.parse import parse

    return parse(text)
