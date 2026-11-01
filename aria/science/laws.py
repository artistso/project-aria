"""A small library of physical laws, checked and inverted dimensionally.

Two capabilities matter here:

* :func:`check_law` validates a law's dimensional consistency, which is the
  invariant that distinguishes a physically meaningful equation from one that
  merely looks like physics.
* :func:`derive_dimension` treats a law as a linear constraint on the exponents
  of its quantities, so ARIA can answer "what must the dimension of ``a`` be for
  ``F = m a`` to hold?" without being told the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Tuple

from ..core.ast import Add, Equation, Expr, Func, Mul, Number, Pow, Symbol
from ..core.parse import parse
from .units import (
    DIMENSIONLESS,
    Dimension,
    DimensionError,
    DimensionalReport,
    check_equation,
    format_dimension,
    infer_dimension,
    parse_unit,
)

__all__ = ["Law", "LAW_LIBRARY", "check_law", "derive_dimension", "list_laws"]


@dataclass(frozen=True)
class Law:
    name: str
    equation: str
    dimensions: Dict[str, str]
    domain: str = "classical mechanics"

    def check(self) -> DimensionalReport:
        return check_equation(self.equation, self.dimensions)

    def __str__(self) -> str:
        return f"{self.name}: {self.equation}"


LAW_LIBRARY: Dict[str, Law] = {
    "newton_second": Law(
        "newton_second",
        "F = m a",
        {"F": "kg*m/s^2", "m": "kg", "a": "m/s^2"},
        "classical mechanics",
    ),
    "mass_energy": Law(
        "mass_energy",
        "E = m c^2",
        {"E": "kg*m^2/s^2", "m": "kg", "c": "m/s"},
        "relativity",
    ),
    "kinetic_energy": Law(
        "kinetic_energy",
        "E_k = 1/2 m v^2",
        {"E_k": "kg*m^2/s^2", "m": "kg", "v": "m/s"},
        "classical mechanics",
    ),
    "gravitation": Law(
        "gravitation",
        "F = G m1 m2 / r^2",
        {"F": "kg*m/s^2", "G": "m^3/(kg*s^2)", "m1": "kg", "m2": "kg", "r": "m"},
        "classical mechanics",
    ),
    "ohms_law": Law(
        "ohms_law",
        "V = I R",
        {"V": "kg*m^2/(s^3*A)", "I": "A", "R": "kg*m^2/(s^3*A^2)"},
        "electromagnetism",
    ),
    "coulomb": Law(
        "coulomb",
        "F = k q1 q2 / r^2",
        {"F": "kg*m/s^2", "k": "kg*m^3/(s^4*A^2)", "q1": "A*s", "q2": "A*s", "r": "m"},
        "electromagnetism",
    ),
    "wave_speed": Law(
        "wave_speed",
        "v = f λ",
        {"v": "m/s", "f": "1/s", "λ": "m"},
        "waves",
    ),
    "ideal_gas": Law(
        "ideal_gas",
        "P V = n R T",
        {
            "P": "kg/(m*s^2)",
            "V": "m^3",
            "n": "mol",
            "R": "kg*m^2/(s^2*K*mol)",
            "T": "K",
        },
        "thermodynamics",
    ),
    "planck_energy": Law(
        "planck_energy",
        "E = h f",
        {"E": "kg*m^2/s^2", "h": "kg*m^2/s", "f": "1/s"},
        "quantum mechanics",
    ),
}


def list_laws() -> List[Law]:
    return [LAW_LIBRARY[name] for name in sorted(LAW_LIBRARY)]


def check_law(name: str) -> DimensionalReport:
    if name not in LAW_LIBRARY:
        raise KeyError(f"unknown law {name!r}; known laws: {', '.join(sorted(LAW_LIBRARY))}")
    return LAW_LIBRARY[name].check()


# ---------------------------------------------------------------------------
# Dimensional inversion
# ---------------------------------------------------------------------------


def _linear(
    expr: Expr, symbols: Mapping[str, Dimension], unknown: str
) -> Tuple[Dimension, Fraction]:
    """Dimension contribution plus the exponent of ``unknown``'s dimension."""
    if isinstance(expr, Number):
        return DIMENSIONLESS, Fraction(0)
    if isinstance(expr, Symbol):
        if expr.name == unknown:
            return DIMENSIONLESS, Fraction(1)
        if expr.name in {"pi", "e", "tau", "i"}:
            return DIMENSIONLESS, Fraction(0)
        if expr.name in symbols:
            return symbols[expr.name], Fraction(0)
        raise DimensionError(f"no dimension declared for {expr.name!r}")
    if isinstance(expr, Add):
        left_dim, left_coeff = _linear(expr.terms[0], symbols, unknown)
        for term in expr.terms[1:]:
            right_dim, right_coeff = _linear(term, symbols, unknown)
            if left_coeff != right_coeff:
                raise DimensionError(
                    "cannot invert an equation whose unknown appears in an inhomogeneous sum"
                )
            if not left_dim.compatible_with(right_dim):
                raise DimensionError(f"inhomogeneous sum: {left_dim} vs {right_dim}")
        return left_dim, left_coeff
    if isinstance(expr, Mul):
        dimension = DIMENSIONLESS
        coefficient = Fraction(0)
        for factor in expr.factors:
            factor_dim, factor_coeff = _linear(factor, symbols, unknown)
            dimension = dimension * factor_dim
            coefficient += factor_coeff
        return dimension, coefficient
    if isinstance(expr, Pow):
        if not isinstance(expr.exp, Number) or not expr.exp.is_exact:
            raise DimensionError("only numeric exponents can be inverted")
        exponent = Fraction(expr.exp.value)
        base_dim, base_coeff = _linear(expr.base, symbols, unknown)
        return base_dim**exponent, base_coeff * exponent
    if isinstance(expr, Func):
        for argument in expr.args:
            dimension, coefficient = _linear(argument, symbols, unknown)
            if not dimension.is_dimensionless or coefficient != 0:
                raise DimensionError(f"{expr.name}() requires a dimensionless argument")
        return DIMENSIONLESS, Fraction(0)
    raise DimensionError(f"cannot analyse {type(expr).__name__}")


def derive_dimension(
    equation_text: str, symbol_dimensions: Mapping[str, str], unknown: str
) -> Dimension:
    """Solve a law for the dimension of ``unknown``."""
    parsed = parse(equation_text)
    if not isinstance(parsed, Equation):
        raise DimensionError("expected an equation")
    symbols = {name: parse_unit(unit) for name, unit in symbol_dimensions.items()}
    left_dim, left_coeff = _linear(parsed.lhs, symbols, unknown)
    right_dim, right_coeff = _linear(parsed.rhs, symbols, unknown)
    coefficient = left_coeff - right_coeff
    if coefficient == 0:
        raise DimensionError(
            f"{unknown!r} does not appear with a solvable power in {equation_text!r}"
        )
    combined = left_dim / right_dim
    return Dimension(tuple(-exponent / coefficient for exponent in combined.exponents))


def explain_derivation(
    equation_text: str, symbol_dimensions: Mapping[str, str], unknown: str
) -> str:
    dimension = derive_dimension(equation_text, symbol_dimensions, unknown)
    return (
        f"Solving {equation_text} for [{unknown}] gives [{unknown}] = {dimension}\n"
        f"  base units: {format_dimension(dimension)}"
    )
