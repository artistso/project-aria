"""Core symbolic object model for ARIA.

ARIA represents mathematical content as typed expression trees rather than as
token strings.  Everything the reasoning and verification layers manipulate is
an :class:`Expr` node, which gives us exactness (rational arithmetic, no
floating point drift), structural equality (canonical forms hash equal), and a
natural substrate for graph/tree neural encoders.

The node set is deliberately small:

====================  ==========================================================
Node                  Meaning
====================  ==========================================================
``Number``            exact rational or inexact float literal
``Symbol``            a variable with optional assumptions
``Add``               n-ary sum
``Mul``               n-ary product
``Pow``               base ** exponent
``Func``              named function application (``log``, ``sin``, ...)
``Derivative``        d/d(var) of an expression
``Equation``          lhs = rhs
``Relation``          lhs <op> rhs for ``<``, ``<=``, ``>``, ``>=``, ``!=``
``Congruence``        lhs ≡ rhs (mod n)
``Matrix``            row-major matrix of expressions
====================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

__all__ = [
    "Expr",
    "Number",
    "Symbol",
    "Add",
    "Mul",
    "Pow",
    "Func",
    "Derivative",
    "Equation",
    "Relation",
    "Congruence",
    "Matrix",
    "num",
    "sym",
    "add",
    "mul",
    "power",
    "canonical_key",
    "ZERO",
    "ONE",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Expr:
    """Base class for ARIA symbolic objects."""

    # -- tree structure ----------------------------------------------------

    def children(self) -> Tuple["Expr", ...]:
        return ()

    def walk(self) -> Iterator["Expr"]:
        yield self
        for child in self.children():
            yield from child.walk()

    def _rebuild(self, children: Sequence["Expr"]) -> "Expr":
        if tuple(children) == self.children():
            return self
        raise NotImplementedError(f"{type(self).__name__} cannot be rebuilt")

    @property
    def depth(self) -> int:
        return 1 + max((c.depth for c in self.children()), default=0)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children())

    @property
    def free_symbols(self) -> Tuple["Symbol", ...]:
        found: Dict[str, Symbol] = {}
        for node in self.walk():
            if isinstance(node, Symbol):
                found.setdefault(node.name, node)
        return tuple(found[name] for name in sorted(found))

    @property
    def operators(self) -> Tuple[str, ...]:
        """Multiset of node heads, useful as a structural feature."""
        return tuple(sorted(n.head for n in self.walk()))

    @property
    def head(self) -> str:
        return type(self).__name__

    # -- rewriting ---------------------------------------------------------

    def subs(self, mapping: "Mapping[Any, Expr]") -> "Expr":
        """Capture-avoiding substitution.

        ``mapping`` may be keyed by :class:`Symbol`, by symbol name, or by an
        arbitrary expression node (structural replacement).
        """
        key: Any = self
        if key in mapping:  # type: ignore[operator]
            return mapping[key]
        if isinstance(self, Symbol) and self.name in mapping:
            return mapping[self.name]
        children = tuple(c.subs(mapping) for c in self.children())
        return self._rebuild(children)

    def replace(self, predicate: Callable[["Expr"], bool], func: Callable[["Expr"], "Expr"]) -> "Expr":
        """Bottom-up rewriting: apply ``func`` wherever ``predicate`` holds."""
        node: Expr = self
        if predicate(node):
            return func(node)
        children = tuple(c.replace(predicate, func) for c in self.children())
        return self._rebuild(children)

    # -- protocol ----------------------------------------------------------

    def canonical(self) -> "Expr":
        return self

    def __str__(self) -> str:
        return render(self)

    __repr__ = __str__

    # -- ergonomic arithmetic (kept exact; never silently coerced to float) --

    def __add__(self, other: Any) -> "Expr":
        return add(self, other)

    def __radd__(self, other: Any) -> "Expr":
        return add(other, self)

    def __sub__(self, other: Any) -> "Expr":
        return add(self, mul(-1, check_expr(other)))

    def __rsub__(self, other: Any) -> "Expr":
        return add(other, mul(-1, self))

    def __mul__(self, other: Any) -> "Expr":
        return mul(self, other)

    def __rmul__(self, other: Any) -> "Expr":
        return mul(other, self)

    def __truediv__(self, other: Any) -> "Expr":
        return mul(self, power(other, -1))

    def __rtruediv__(self, other: Any) -> "Expr":
        return mul(other, power(self, -1))

    def __pow__(self, other: Any) -> "Expr":
        return power(self, other)

    def __neg__(self) -> "Expr":
        return mul(-1, self)

    def __pos__(self) -> "Expr":
        return self


def canonical_key(expr: Expr) -> Tuple:
    """Total order over expressions, used for deterministic canonical forms."""
    return expr.canonical_key()


def _sort_key(expr: Expr) -> Tuple:
    return expr.canonical_key()


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Number(Expr):
    """An exact rational (``Fraction``) or inexact (``float``) numeric literal."""

    value: Any

    def __post_init__(self) -> None:
        value = self.value
        if isinstance(value, bool):  # guard: bools are ints
            raise TypeError("boolean is not a number")
        if isinstance(value, int):
            object.__setattr__(self, "value", Fraction(value))
        elif isinstance(value, float):
            object.__setattr__(self, "value", value)
        elif isinstance(value, Fraction):
            pass
        else:
            raise TypeError(f"unsupported numeric literal: {value!r}")

    # -- predicates --------------------------------------------------------

    @property
    def is_exact(self) -> bool:
        return isinstance(self.value, Fraction)

    @property
    def is_integer(self) -> bool:
        return self.is_exact and self.value.denominator == 1

    @property
    def is_rational(self) -> bool:
        return self.is_exact

    @property
    def is_zero(self) -> bool:
        return self.value == 0

    @property
    def is_one(self) -> bool:
        return self.value == 1

    @property
    def is_negative(self) -> bool:
        return self.value < 0

    def as_fraction(self) -> Fraction:
        if not self.is_exact:
            return Fraction(self.value).limit_denominator(10**9)
        return self.value

    def head_key(self) -> Tuple:
        if self.is_exact:
            return ("num", self.value.numerator, self.value.denominator)
        return ("float", float(self.value))

    def canonical_key(self) -> Tuple:
        return (0, self.head_key())

    def __str__(self) -> str:
        return render(self)


SymbolAssumptions = frozenset


@dataclass(frozen=True)
class Symbol(Expr):
    """A variable, optionally carrying assumptions (``positive``, ``integer``...)."""

    name: str
    assumptions: SymbolAssumptions = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", frozenset(self.assumptions))

    @property
    def is_integer(self) -> bool:
        return "integer" in self.assumptions

    @property
    def is_positive(self) -> bool:
        return "positive" in self.assumptions

    @property
    def is_nonzero(self) -> bool:
        return "nonzero" in self.assumptions

    def canonical_key(self) -> Tuple:
        return (1, self.name, tuple(sorted(self.assumptions)))

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Compound nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Add(Expr):
    terms: Tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))

    def children(self) -> Tuple[Expr, ...]:
        return self.terms

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return add(*children)

    def canonical(self) -> Expr:
        return add(*(t.canonical() for t in self.terms))

    def canonical_key(self) -> Tuple:
        return (2, tuple(t.canonical_key() for t in self.terms))

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Mul(Expr):
    factors: Tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))

    def children(self) -> Tuple[Expr, ...]:
        return self.factors

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return mul(*children)

    def canonical(self) -> Expr:
        return mul(*(f.canonical() for f in self.factors))

    def canonical_key(self) -> Tuple:
        return (3, tuple(f.canonical_key() for f in self.factors))

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Pow(Expr):
    base: Expr
    exp: Expr

    def children(self) -> Tuple[Expr, ...]:
        return (self.base, self.exp)

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return power(children[0], children[1])

    def canonical(self) -> Expr:
        return power(self.base.canonical(), self.exp.canonical())

    def canonical_key(self) -> Tuple:
        return (4, self.base.canonical_key(), self.exp.canonical_key())

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Func(Expr):
    """A named function application, e.g. ``log(x)`` or ``sin(2 x)``."""

    name: str
    args: Tuple[Expr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))

    def children(self) -> Tuple[Expr, ...]:
        return self.args

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return func(self.name, *children)

    def canonical(self) -> Expr:
        return func(self.name, *(a.canonical() for a in self.args))

    def canonical_key(self) -> Tuple:
        return (5, self.name, tuple(a.canonical_key() for a in self.args))

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Derivative(Expr):
    expr: Expr
    var: Symbol

    def children(self) -> Tuple[Expr, ...]:
        return (self.expr, self.var)

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return Derivative(children[0], children[1])  # type: ignore[arg-type]

    def canonical_key(self) -> Tuple:
        return (6, self.expr.canonical_key(), self.var.canonical_key())

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Equation(Expr):
    lhs: Expr
    rhs: Expr

    def children(self) -> Tuple[Expr, ...]:
        return (self.lhs, self.rhs)

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return Equation(children[0], children[1])

    def canonical_key(self) -> Tuple:
        return (7, self.lhs.canonical_key(), self.rhs.canonical_key())

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Relation(Expr):
    op: str
    lhs: Expr
    rhs: Expr

    def children(self) -> Tuple[Expr, ...]:
        return (self.lhs, self.rhs)

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return Relation(self.op, children[0], children[1])

    def canonical_key(self) -> Tuple:
        return (8, self.op, self.lhs.canonical_key(), self.rhs.canonical_key())

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Congruence(Expr):
    """``lhs ≡ rhs (mod modulus)`` -- a first-class modular arithmetic object."""

    lhs: Expr
    rhs: Expr
    modulus: Expr

    def children(self) -> Tuple[Expr, ...]:
        return (self.lhs, self.rhs, self.modulus)

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        return Congruence(children[0], children[1], children[2])

    def canonical_key(self) -> Tuple:
        return (9, self.lhs.canonical_key(), self.rhs.canonical_key(), self.modulus.canonical_key())

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Matrix(Expr):
    rows: Tuple[Tuple[Expr, ...], ...]

    def __post_init__(self) -> None:
        rows = tuple(tuple(r) for r in self.rows)
        object.__setattr__(self, "rows", rows)
        if rows and len({len(r) for r in rows}) != 1:
            raise ValueError("ragged matrix rows")

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def children(self) -> Tuple[Expr, ...]:
        return tuple(c for row in self.rows for c in row)

    def _rebuild(self, children: Sequence[Expr]) -> Expr:
        cols = self.shape[1]
        rows = tuple(tuple(children[i * cols : (i + 1) * cols]) for i in range(self.shape[0]))
        return Matrix(rows)

    def canonical_key(self) -> Tuple:
        return (10, tuple(tuple(c.canonical_key() for c in row) for row in self.rows))

    def __str__(self) -> str:
        return render(self)


# ---------------------------------------------------------------------------
# Smart constructors
# ---------------------------------------------------------------------------

ZERO = Number(Fraction(0))
ONE = Number(Fraction(1))


def num(value: Any) -> Number:
    return Number(value)


def sym(name: str, *assumptions: str) -> Symbol:
    return Symbol(name, frozenset(assumptions))


def check_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a symbolic value")
    if isinstance(value, (int, float, Fraction)):
        return Number(value)
    raise TypeError(f"cannot coerce {value!r} into a symbolic expression")


def add(*terms: Any) -> Expr:
    """Flattened, constant-folded, canonically ordered sum."""
    flat: list[Expr] = []
    constant = Fraction(0)
    inexact = 0.0
    has_inexact = False
    for term in terms:
        expr = check_expr(term)
        if isinstance(expr, Add):
            flat.extend(expr.terms)
        elif isinstance(expr, Number):
            if expr.is_exact:
                constant += expr.value
            else:
                inexact += float(expr.value)
                has_inexact = True
        else:
            flat.append(expr)
    if has_inexact:
        value: Any = float(constant) + inexact if constant else inexact
        if value != 0:
            flat.append(Number(value))
    elif constant != 0:
        flat.append(Number(constant))
    flat.sort(key=_sort_key)
    if not flat:
        return ZERO
    if len(flat) == 1:
        return flat[0]
    return Add(tuple(flat))


def mul(*factors: Any) -> Expr:
    """Flattened, constant-folded, canonically ordered product.

    Identical factors are collected into powers, and numeric coefficients are
    folded.  ``mul()`` is ``1`` and ``mul(x)`` is ``x``.
    """
    coefficient = Fraction(1)
    has_inexact = False
    inexact = 1.0
    exponents: Dict[Tuple, Tuple[Expr, Expr]] = {}
    order: list[Tuple] = []

    def fold_number(value: Any) -> None:
        nonlocal coefficient, has_inexact, inexact
        if isinstance(value, Fraction):
            coefficient *= value
        else:
            has_inexact = True
            inexact *= float(value)

    def push(expr: Expr) -> None:
        if isinstance(expr, Mul):
            for factor in expr.factors:
                push(factor)
            return
        if isinstance(expr, Number):
            fold_number(expr.value)
            return
        base = expr.base if isinstance(expr, Pow) else expr
        exponent = expr.exp if isinstance(expr, Pow) else ONE
        key = base.canonical_key()
        if key in exponents:
            existing_base, existing_exponent = exponents[key]
            exponents[key] = (existing_base, add(existing_exponent, exponent))
        else:
            exponents[key] = (base, exponent)
            order.append(key)

    for factor in factors:
        push(check_expr(factor))

    flat: list[Expr] = []
    for key in order:
        base, exponent = exponents[key]
        if isinstance(exponent, Number) and exponent.value == 0:
            continue
        if isinstance(exponent, Number) and exponent.value == 1:
            flat.append(base)
        else:
            flat.append(Pow(base, exponent))

    flat.sort(key=_sort_key)
    coefficient_value: Any = float(coefficient) * inexact if has_inexact else coefficient

    if coefficient_value == 0:
        return ZERO
    if coefficient_value != 1 or not flat:
        flat.insert(0, Number(coefficient_value))
    if not flat:
        return ONE
    if len(flat) == 1:
        return flat[0]
    return Mul(tuple(flat))


def power(base: Any, exp: Any) -> Expr:
    b = check_expr(base)
    e = check_expr(exp)
    if isinstance(b, Number) and b.is_zero and isinstance(e, Number) and e.value > 0:
        return ZERO
    if isinstance(e, Number):
        if e.value == 0:
            return ONE
        if e.value == 1:
            return b
        if e.value == Fraction(-1):
            return _reciprocal(b)
    if isinstance(b, Number) and isinstance(e, Number) and b.is_exact and e.is_exact:
        if e.value.denominator == 1:
            exponent = int(e.value)
            if 0 <= exponent <= 64:
                return Number(b.value**exponent)
    return Pow(b, e)


def _reciprocal(expr: Expr) -> Expr:
    if isinstance(expr, Number):
        if expr.value == 0:
            raise ZeroDivisionError("division by zero")
        return Number(Fraction(1) / expr.as_fraction())
    return Pow(expr, Number(Fraction(-1)))


def func(name: str, *args: Any) -> Expr:
    return Func(name, tuple(check_expr(a) for a in args))


def eq(lhs: Any, rhs: Any) -> Equation:
    return Equation(check_expr(lhs), check_expr(rhs))


def neg(expr: Any) -> Expr:
    return mul(-1, expr)


def div(lhs: Any, rhs: Any) -> Expr:
    return mul(lhs, power(rhs, -1))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PRECEDENCE = {
    "Equation": 0,
    "Relation": 0,
    "Congruence": 0,
    "Add": 1,
    "Mul": 2,
    "Pow": 4,
    "Derivative": 5,
    "Func": 6,
    "Matrix": 6,
    "Number": 7,
    "Symbol": 7,
}


def _precedence(expr: Expr) -> int:
    return _PRECEDENCE.get(expr.head, 7)


def _render_operand(expr: Expr, parent_prec: int, side: str = "either") -> str:
    text = render(expr)
    if _precedence(expr) < parent_prec:
        return f"({text})"
    if side == "right" and _precedence(expr) == parent_prec:
        return f"({text})"
    return text


def render(expr: Expr) -> str:
    """Human-readable infix rendering (round-trips through the parser)."""
    if isinstance(expr, Number):
        value = expr.value
        if isinstance(value, Fraction):
            if value.denominator == 1:
                return str(value.numerator)
            return f"{value.numerator}/{value.denominator}"
        return repr(value) if value != int(value) else f"{value:.1f}"
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, Add):
        pieces: list[str] = []
        for i, term in enumerate(expr.terms):
            text = _render_operand(term, 1)
            negative = (
                isinstance(term, Number)
                and term.is_negative
                or isinstance(term, Mul)
                and isinstance(term.factors[0], Number)
                and term.factors[0].is_negative
            )
            if i == 0:
                pieces.append(text)
            elif negative:
                pieces.append(f"- {text.lstrip('-')}")
            else:
                pieces.append(f"+ {text}")
        return " ".join(pieces)
    if isinstance(expr, Mul):
        pieces: list[str] = []
        for i, factor in enumerate(expr.factors):
            if i == 0 and isinstance(factor, Number) and factor.is_negative:
                magnitude = _render_operand(Number(-factor.value), 2)
                pieces.append("-" if magnitude == "1" else f"-{magnitude}")
            else:
                pieces.append(_render_operand(factor, 2))
        if pieces and pieces[0] == "-":
            return "-" + " ".join(pieces[1:])
        return " ".join(pieces)
    if isinstance(expr, Pow):
        base = _render_operand(expr.base, 3)
        exponent = expr.exp
        if isinstance(exponent, Number) and exponent.is_exact and exponent.value.denominator != 1:
            return f"{base}^({render(exponent)})"
        if isinstance(exponent, Number) and exponent.is_negative:
            return f"{base}^({render(exponent)})"
        return f"{base}^{_render_operand(expr.exp, 4)}"
    if isinstance(expr, Func):
        return f"{expr.name}({', '.join(render(a) for a in expr.args)})"
    if isinstance(expr, Derivative):
        return f"d/d{expr.var.name}[{render(expr.expr)}]"
    if isinstance(expr, Equation):
        return f"{render(expr.lhs)} = {render(expr.rhs)}"
    if isinstance(expr, Relation):
        return f"{render(expr.lhs)} {expr.op} {render(expr.rhs)}"
    if isinstance(expr, Congruence):
        return f"{render(expr.lhs)} ≡ {render(expr.rhs)} (mod {render(expr.modulus)})"
    if isinstance(expr, Matrix):
        rows = "; ".join(", ".join(render(c) for c in row) for row in expr.rows)
        return f"[[{rows}]]"
    return object.__repr__(expr)


# ---------------------------------------------------------------------------
# Structural helpers used across layers
# ---------------------------------------------------------------------------


def is_zero(expr: Expr) -> bool:
    return isinstance(expr, Number) and expr.is_zero


def is_one(expr: Expr) -> bool:
    return isinstance(expr, Number) and expr.is_one


def is_atom(expr: Expr) -> bool:
    return isinstance(expr, (Number, Symbol))


def is_num(expr: Expr) -> bool:
    return isinstance(expr, Number)


def coeff_and_rest(expr: Expr) -> Tuple[Fraction, Expr]:
    """Split a term into a rational coefficient and its non-numeric part."""
    if isinstance(expr, Number):
        return (expr.value if expr.is_exact else Fraction(expr.value), ONE)
    if isinstance(expr, Mul) and expr.factors and isinstance(expr.factors[0], Number):
        head = expr.factors[0]
        rest = mul(*expr.factors[1:])
        coeff = head.value if head.is_exact else Fraction(head.value)
        return (coeff, rest)
    return (Fraction(1), expr)


def negate(expr: Expr) -> Expr:
    return mul(-1, expr)
