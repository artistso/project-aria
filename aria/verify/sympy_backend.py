"""Bridge between ARIA expression trees and SymPy.

SymPy is an *optional* dependency: it is the strongest available verification
backend (symbolic simplification, transcendental identities, calculus), but the
core representation, parser and rewriting engine are pure Python.  When SymPy is
missing, :class:`~aria.verify.verifier.Verifier` degrades to an internal
polynomial normaliser plus numeric probing and reports lower confidence instead
of pretending to prove things.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Any

from ..core.ast import (
    Add,
    Congruence,
    Derivative,
    Equation,
    Expr,
    Func,
    Matrix,
    Mul,
    Number,
    Pow,
    Relation,
    Symbol,
)

__all__ = ["available", "to_sympy", "from_sympy", "backend_name"]

try:  # pragma: no cover - exercised implicitly by the test-suite
    import sympy as sp

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    sp = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def available() -> bool:
    """True when SymPy could be imported."""
    return sp is not None


def backend_name() -> str:
    if sp is None:
        return "internal"
    return f"sympy-{sp.__version__}"


_FUNCTION_MAP = {
    "log": "log",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "sec": "sec",
    "csc": "csc",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "abs": "Abs",
    "sign": "sign",
    "floor": "floor",
    "ceil": "ceiling",
    "max": "Max",
    "min": "Min",
    "gcd": "gcd",
    "lcm": "lcm",
}

_CONSTANT_MAP = {"pi": "pi", "e": "E", "tau": "tau", "inf": "oo", "i": "I"}


@lru_cache(maxsize=256)
def _symbol(name: str, assumptions: tuple[str, ...]) -> Any:
    assert sp is not None
    kwargs = {}
    for flag in assumptions:
        if flag in {"positive", "negative", "nonzero", "integer", "real", "rational"}:
            kwargs[flag] = True
    if "positive" in kwargs:
        kwargs.setdefault("real", True)
    return sp.Symbol(name, **kwargs)


def to_sympy(expr: Expr) -> Any:
    """Convert an ARIA expression into its SymPy counterpart."""
    if sp is None:  # pragma: no cover
        raise RuntimeError("SymPy backend unavailable") from _IMPORT_ERROR

    if isinstance(expr, Number):
        if expr.is_exact:
            return sp.Rational(expr.value.numerator, expr.value.denominator)
        return sp.Float(expr.value)
    if isinstance(expr, Symbol):
        if expr.name in _CONSTANT_MAP:
            return getattr(sp, _CONSTANT_MAP[expr.name])
        return _symbol(expr.name, tuple(sorted(expr.assumptions)))
    if isinstance(expr, Add):
        return sp.Add(*(to_sympy(t) for t in expr.terms))
    if isinstance(expr, Mul):
        return sp.Mul(*(to_sympy(f) for f in expr.factors))
    if isinstance(expr, Pow):
        return sp.Pow(to_sympy(expr.base), to_sympy(expr.exp))
    if isinstance(expr, Func):
        name = _FUNCTION_MAP.get(expr.name)
        if name is None:
            return sp.Function(expr.name)(*(to_sympy(a) for a in expr.args))
        return getattr(sp, name)(*(to_sympy(a) for a in expr.args))
    if isinstance(expr, Derivative):
        return sp.diff(to_sympy(expr.expr), to_sympy(expr.var))
    if isinstance(expr, Equation):
        return sp.Eq(to_sympy(expr.lhs), to_sympy(expr.rhs))
    if isinstance(expr, Relation):
        op = {
            "<": "StrictLessThan",
            "<=": "LessThan",
            ">": "StrictGreaterThan",
            ">=": "GreaterThan",
            "!=": "Unequality",
        }[expr.op]
        return getattr(sp, op)(to_sympy(expr.lhs), to_sympy(expr.rhs))
    if isinstance(expr, Congruence):
        return sp.Equivalent(
            to_sympy(expr.lhs) % to_sympy(expr.modulus) == to_sympy(expr.rhs) % to_sympy(expr.modulus),
            True,
        )
    if isinstance(expr, Matrix):
        return sp.Matrix([[to_sympy(c) for c in row] for row in expr.rows])
    raise TypeError(f"cannot convert {type(expr).__name__} to SymPy")


def from_sympy(obj: Any) -> Expr:
    """Convert a SymPy object back into an ARIA expression."""
    if sp is None:  # pragma: no cover
        raise RuntimeError("SymPy backend unavailable") from _IMPORT_ERROR

    if isinstance(obj, sp.Rational):
        return Number(Fraction(int(obj.p), int(obj.q)))
    if isinstance(obj, sp.Integer):
        return Number(Fraction(int(obj)))
    if isinstance(obj, sp.Float):
        return Number(float(obj))
    if isinstance(obj, sp.Symbol):
        return Symbol(str(obj))
    if isinstance(obj, sp.Add):
        parts = [from_sympy(a) for a in obj.args]
        return Add(tuple(parts)) if len(parts) > 1 else parts[0]
    if isinstance(obj, sp.Mul):
        return Mul(tuple(from_sympy(a) for a in obj.args))
    if isinstance(obj, sp.Pow):
        return Pow(from_sympy(obj.base), from_sympy(obj.exp))
    if isinstance(obj, sp.Equality):
        return Equation(from_sympy(obj.lhs), from_sympy(obj.rhs))
    if isinstance(obj, sp.MatrixBase):
        return Matrix(tuple(tuple(from_sympy(c) for c in row) for row in obj.tolist()))
    if isinstance(obj, sp.Function):
        return Func(str(obj.func), tuple(from_sympy(a) for a in obj.args))
    if obj is sp.pi:
        return Symbol("pi")
    if obj is sp.E:
        return Symbol("e")
    raise TypeError(f"cannot convert {obj!r} from SymPy")
