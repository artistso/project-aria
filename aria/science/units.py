"""Dimensional analysis.

Physical claims carry an invariant that pure text models have no access to: the
dimensions on both sides of a law must agree.  ARIA represents a dimension as an
exponent vector over the seven SI base dimensions, which makes multiplication,
division and exponentiation exact linear algebra over the rationals -- and makes
"is this equation physically possible?" a decidable question.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.ast import (
    Add,
    Expr,
    Func,
    Matrix,
    Mul,
    Number,
    Pow,
    Symbol,
)
from ..core.parse import ParseError, parse

__all__ = [
    "Dimension",
    "DimensionError",
    "DIMENSIONLESS",
    "BASE_DIMENSIONS",
    "parse_unit",
    "infer_dimension",
    "check_equation",
    "DimensionalReport",
    "format_dimension",
]


BASE_DIMENSIONS: Tuple[str, ...] = ("M", "L", "T", "I", "Th", "N", "J")
BASE_LABELS = {
    "M": "mass",
    "L": "length",
    "T": "time",
    "I": "electric current",
    "Th": "temperature",
    "N": "amount of substance",
    "J": "luminous intensity",
}


class DimensionError(ValueError):
    """Raised when an expression cannot be given a consistent dimension."""


@dataclass(frozen=True)
class Dimension:
    """Exponent vector over the SI base dimensions."""

    exponents: Tuple[Fraction, ...] = (Fraction(0),) * 7
    label: str = ""

    def __post_init__(self) -> None:
        exponents = tuple(Fraction(e) for e in self.exponents)
        if len(exponents) != len(BASE_DIMENSIONS):
            raise DimensionError("dimension vector must have 7 components")
        object.__setattr__(self, "exponents", exponents)

    # -- algebra -----------------------------------------------------------

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, exponent: Fraction | int) -> "Dimension":
        power = Fraction(exponent)
        return Dimension(tuple(a * power for a in self.exponents))

    # -- predicates --------------------------------------------------------

    @property
    def is_dimensionless(self) -> bool:
        return all(exponent == 0 for exponent in self.exponents)

    def compatible_with(self, other: "Dimension") -> bool:
        return self.exponents == other.exponents

    def as_dict(self) -> Dict[str, Fraction]:
        return {name: exp for name, exp in zip(BASE_DIMENSIONS, self.exponents) if exp != 0}

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        return format_dimension(self)


DIMENSIONLESS = Dimension()


def format_dimension(dimension: Dimension) -> str:
    """Render a dimension in base units (``kg m^2 / s^2`` style)."""
    numerator: List[str] = []
    denominator: List[str] = []
    for name, exponent in zip(BASE_DIMENSIONS, dimension.exponents):
        if exponent == 0:
            continue
        symbol = BASE_UNIT_SYMBOLS[name]
        magnitude = abs(exponent)
        chunk = symbol if magnitude == 1 else f"{symbol}^{_render_exponent(magnitude)}"
        (numerator if exponent > 0 else denominator).append(chunk)
    if not numerator:
        numerator = ["1"]
    text = " ".join(numerator)
    if denominator:
        text += " / " + " ".join(denominator)
    return text


def _render_exponent(exponent: Fraction) -> str:
    if exponent.denominator == 1:
        return str(exponent.numerator)
    return f"({exponent.numerator}/{exponent.denominator})"


BASE_UNIT_SYMBOLS = {
    "M": "kg",
    "L": "m",
    "T": "s",
    "I": "A",
    "Th": "K",
    "N": "mol",
    "J": "cd",
}


# ---------------------------------------------------------------------------
# Unit vocabulary
# ---------------------------------------------------------------------------


def _dim(**kwargs: Fraction | int) -> Dimension:
    exponents = []
    for name in BASE_DIMENSIONS:
        exponents.append(Fraction(kwargs.get(name, 0)))
    return Dimension(tuple(exponents))


UNITS: Dict[str, Dimension] = {
    # SI base
    "kg": _dim(M=1),
    "g": _dim(M=1),
    "m": _dim(L=1),
    "s": _dim(T=1),
    "A": _dim(I=1),
    "K": _dim(Th=1),
    "mol": _dim(N=1),
    "cd": _dim(J=1),
    # time and length
    "min": _dim(T=1),
    "h": _dim(T=1),
    "km": _dim(L=1),
    "cm": _dim(L=1),
    "mm": _dim(L=1),
    # mechanical
    "N": _dim(M=1, L=1, T=-2),
    "J": _dim(M=1, L=2, T=-2),
    "W": _dim(M=1, L=2, T=-3),
    "Pa": _dim(M=1, L=-1, T=-2),
    "Hz": _dim(T=-1),
    # electromagnetic
    "C": _dim(I=1, T=1),
    "V": _dim(M=1, L=2, T=-3, I=-1),
    "ohm": _dim(M=1, L=2, T=-3, I=-2),
    "F": _dim(M=-1, L=-2, T=4, I=2),
    "T": _dim(M=1, T=-2, I=-1),
    "Wb": _dim(M=1, L=2, T=-2, I=-1),
    # dimensionless
    "rad": _dim(),
    "sr": _dim(),
}


def parse_unit(text: str) -> Dimension:
    """Parse a unit expression such as ``kg*m^2/(s^3*A)``."""
    tokens = _tokenize_unit(text)
    position = 0

    def parse_expression() -> Dimension:
        nonlocal position
        value = parse_term()
        while position < len(tokens) and tokens[position] in ("*", "/"):
            operator = tokens[position]
            position += 1
            right = parse_term()
            value = value * right if operator == "*" else value / right
        return value

    def parse_term() -> Dimension:
        nonlocal position
        if position >= len(tokens):
            raise DimensionError(f"unexpected end of unit expression {text!r}")
        token = tokens[position]
        position += 1
        if token == "(":
            value = parse_expression()
            if position >= len(tokens) or tokens[position] != ")":
                raise DimensionError(f"unbalanced parentheses in unit expression {text!r}")
            position += 1
        elif token.isdigit():
            value = Dimension()  # numeric factors are dimensionless
        else:
            if token not in UNITS:
                raise DimensionError(f"unknown unit {token!r}")
            value = UNITS[token]
        if position < len(tokens) and tokens[position] == "^":
            position += 1
            if position >= len(tokens):
                raise DimensionError(f"missing exponent in {text!r}")
            sign = 1
            if tokens[position] == "-":
                sign = -1
                position += 1
            if position >= len(tokens):
                raise DimensionError(f"missing exponent in {text!r}")
            exponent_token = tokens[position]
            position += 1
            value = value ** (sign * Fraction(exponent_token))
        return value

    result = parse_expression()
    if position != len(tokens):
        raise DimensionError(f"trailing tokens in unit expression {text!r}")
    return result


def _tokenize_unit(text: str) -> List[str]:
    tokens: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "*/^()-":
            tokens.append(char)
            index += 1
            continue
        start = index
        while index < len(text) and (text[index].isalnum() or text[index] in "_.'"):
            index += 1
        if start == index:
            raise DimensionError(f"unexpected character {char!r} in unit expression")
        tokens.append(text[start:index])
    return tokens


# ---------------------------------------------------------------------------
# Dimensional inference over expressions
# ---------------------------------------------------------------------------


def infer_dimension(expr: Expr, symbols: Mapping[str, Dimension]) -> Dimension:
    """Infer the dimension of an expression from symbol dimensions."""
    if isinstance(expr, Number):
        if not expr.is_exact:
            raise DimensionError("inexact numeric literals have no exact dimension")
        return DIMENSIONLESS
    if isinstance(expr, Symbol):
        if expr.name in symbols:
            return symbols[expr.name]
        if expr.name in UNITS and expr.name not in {"m", "s", "F", "T", "N", "J", "C", "V", "W", "K", "A"}:
            return UNITS[expr.name]
        if expr.name in {"pi", "e", "tau", "i"}:
            return DIMENSIONLESS
        raise DimensionError(
            f"no dimension declared for {expr.name!r}; declare it (e.g. {expr.name}=kg*m/s^2)"
        )
    if isinstance(expr, Add):
        first: Optional[Dimension] = None
        for term in expr.terms:
            current = infer_dimension(term, symbols)
            if first is None:
                first = current
            elif not first.compatible_with(current):
                raise DimensionError(
                    f"cannot add {first} and {current}: inhomogeneous sum"
                )
        return first or DIMENSIONLESS
    if isinstance(expr, Mul):
        result = DIMENSIONLESS
        for factor in expr.factors:
            result = result * infer_dimension(factor, symbols)
        return result
    if isinstance(expr, Pow):
        if not isinstance(expr.exp, Number) or not expr.exp.is_exact:
            raise DimensionError("only numeric exponents are dimensionally meaningful")
        exponent = expr.exp.value
        if exponent.denominator != 1:
            raise DimensionError("fractional exponents require care; use integer powers")
        return infer_dimension(expr.base, symbols) ** int(exponent)
    if isinstance(expr, Func):
        for argument in expr.args:
            if not infer_dimension(argument, symbols).is_dimensionless:
                raise DimensionError(
                    f"{expr.name}() requires a dimensionless argument, got {infer_dimension(argument, symbols)}"
                )
        return DIMENSIONLESS
    if isinstance(expr, Matrix):
        shapes = {infer_dimension(cell, symbols) for row in expr.rows for cell in row}
        if len(shapes) > 1:
            raise DimensionError("matrix entries have inconsistent dimensions")
        return shapes.pop()
    raise DimensionError(f"cannot assign a dimension to {type(expr).__name__}")


@dataclass
class DimensionalReport:
    """Outcome of checking a physical equation for dimensional consistency."""

    equation: str
    left: Optional[Dimension] = None
    right: Optional[Dimension] = None
    consistent: bool = False
    detail: str = ""
    symbol_dimensions: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.symbol_dimensions is None:
            self.symbol_dimensions = {}

    def to_text(self) -> str:
        lines = [f"Equation: {self.equation}"]
        if self.left is not None:
            lines.append(f"  left  dimension: {self.left}")
        if self.right is not None:
            lines.append(f"  right dimension: {self.right}")
        for name, dim in sorted(self.symbol_dimensions.items()):
            lines.append(f"  [{name}] = {dim}")
        lines.append(f"  verdict: {'CONSISTENT' if self.consistent else 'INCONSISTENT'} — {self.detail}")
        return "\n".join(lines)


def check_equation(
    equation_text: str, symbol_dimensions: Mapping[str, str]
) -> DimensionalReport:
    """Check that both sides of ``equation_text`` carry the same dimension."""
    from ..core.ast import Equation

    parsed = parse(equation_text)
    if not isinstance(parsed, Equation):
        raise DimensionError("expected an equation such as 'F = m a'")
    symbols: Dict[str, Dimension] = {}
    for name, unit_text in symbol_dimensions.items():
        symbols[name] = parse_unit(unit_text)
    report = DimensionalReport(equation_text)
    report.symbol_dimensions = {
        name: str(dimension) for name, dimension in sorted(symbols.items())
    }
    try:
        left = infer_dimension(parsed.lhs, symbols)
        right = infer_dimension(parsed.rhs, symbols)
    except DimensionError as error:
        report.detail = str(error)
        return report
    report.left = left
    report.right = right
    report.consistent = left.compatible_with(right)
    report.detail = (
        "both sides have the same dimension"
        if report.consistent
        else f"{left} != {right}"
    )
    return report
