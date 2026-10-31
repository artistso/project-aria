"""Exact polynomial and rational-function utilities.

These routines give ARIA a verification path that needs no third-party CAS:
every expression built from ``+ - * /`` and integer powers is converted to a
rational normal form, and equality is decided exactly by comparing canonical
polynomials.  That covers the overwhelming majority of algebra benchmarks and
is the fallback when SymPy is not installed.

Representation: a polynomial is a ``dict`` from a *monomial key* to a rational
coefficient.  The monomial key is a sorted tuple of ``(symbol_name, exponent)``
pairs, so ``x**2 y`` and ``y x**2`` map to the same key.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ast import (
    Add,
    Expr,
    Func,
    Mul,
    Number,
    Pow,
    Symbol,
    add,
    mul,
    power,
    sym,
)

__all__ = [
    "poly_divmod",
    "poly_divide",
    "poly_gcd",
    "MonomialKey",
    "Polynomial",
    "normal_form",
    "check_equivalent",
    "univariate_coefficients",
    "from_univariate",
    "degree",
    "rational_roots",
    "factor_quadratic",
    "extended_gcd",
    "mod_inverse",
    "MAX_TERMS",
]

MonomialKey = Tuple[Tuple[str, int], ...]
Polynomial = Dict[MonomialKey, Fraction]

#: Guard against combinatorial blow-up during expansion.
MAX_TERMS = 20_000

_ONE_MONOMIAL: MonomialKey = ()


class _TooComplex(Exception):
    pass


def _mono_mul(a: MonomialKey, b: MonomialKey) -> MonomialKey:
    powers: Dict[str, int] = {}
    for name, exp in a + b:
        powers[name] = powers.get(name, 0) + exp
    return tuple(sorted((name, exp) for name, exp in powers.items() if exp))


def _mono_pow(a: MonomialKey, exponent: int) -> MonomialKey:
    if exponent == 0:
        return _ONE_MONOMIAL
    return tuple(sorted((name, exp * exponent) for name, exp in a if exp * exponent))


def _poly_add(a: Polynomial, b: Polynomial) -> Polynomial:
    result = dict(a)
    for key, value in b.items():
        total = result.get(key, Fraction(0)) + value
        if total == 0:
            result.pop(key, None)
        else:
            result[key] = total
    return result


def _poly_mul(a: Polynomial, b: Polynomial) -> Polynomial:
    if len(a) * len(b) > MAX_TERMS:
        raise _TooComplex("polynomial product too large")
    result: Polynomial = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            key = _mono_mul(ka, kb)
            result[key] = result.get(key, Fraction(0)) + va * vb
    return {k: v for k, v in result.items() if v != 0}


def _poly_scale(a: Polynomial, factor: Fraction) -> Polynomial:
    return {k: v * factor for k, v in a.items() if v * factor != 0}


def _poly_pow(a: Polynomial, exponent: int) -> Polynomial:
    if exponent < 0:
        raise _TooComplex("negative polynomial exponent")
    result: Polynomial = {_ONE_MONOMIAL: Fraction(1)}
    for _ in range(exponent):
        result = _poly_mul(result, a)
    return result


def _normal_form(expr: Expr, depth: int = 0) -> Tuple[Polynomial, Polynomial]:
    """Return ``(numerator, denominator)`` polynomials for ``expr``."""
    if depth > 60:
        raise _TooComplex("expression nesting too deep")
    if isinstance(expr, Number):
        if not expr.is_exact:
            raise _TooComplex("inexact coefficient")
        return ({_ONE_MONOMIAL: expr.value}, {_ONE_MONOMIAL: Fraction(1)})
    if isinstance(expr, Symbol):
        return ({(("$" + expr.name, 1),): Fraction(1)}, {_ONE_MONOMIAL: Fraction(1)})
    if isinstance(expr, Add):
        numerator: Polynomial = {}
        denominator: Polynomial = {_ONE_MONOMIAL: Fraction(1)}
        for term in expr.terms:
            tn, td = _normal_form(term, depth + 1)
            # numerator/denominator += tn/td  -> (numerator*td + tn*denominator)/(denominator*td)
            numerator = _poly_add(_poly_mul(numerator, td), _poly_mul(tn, denominator))
            denominator = _poly_mul(denominator, td)
        return (numerator, denominator)
    if isinstance(expr, Mul):
        numerator = {_ONE_MONOMIAL: Fraction(1)}
        denominator = {_ONE_MONOMIAL: Fraction(1)}
        for factor in expr.factors:
            fn, fd = _normal_form(factor, depth + 1)
            numerator = _poly_mul(numerator, fn)
            denominator = _poly_mul(denominator, fd)
        return (numerator, denominator)
    if isinstance(expr, Pow):
        if not isinstance(expr.exp, Number) or not expr.exp.is_exact:
            raise _TooComplex("non-rational exponent")
        exponent = expr.exp.value
        if exponent.denominator != 1:
            raise _TooComplex("fractional exponent")
        base_num, base_den = _normal_form(expr.base, depth + 1)
        exp_int = int(exponent)
        if exp_int >= 0:
            return (_poly_pow(base_num, exp_int), _poly_pow(base_den, exp_int))
        if not base_num or not base_den:
            raise _TooComplex("division by zero")
        return (_poly_pow(base_den, -exp_int), _poly_pow(base_num, -exp_int))
    raise _TooComplex(f"non-polynomial node {type(expr).__name__}")


def normal_form(expr: Expr) -> Optional[Tuple[Polynomial, Polynomial]]:
    """Exact rational normal form, or ``None`` if the expression is transcendental."""
    try:
        numerator, denominator = _normal_form(expr)
    except (_TooComplex, ZeroDivisionError):
        return None
    if not denominator:
        return None
    # Normalise the scaling so equal rational functions produce identical pairs.
    lead = max(numerator) if numerator else _ONE_MONOMIAL
    scale = numerator.get(lead, Fraction(1))
    if scale != 0:
        numerator = _poly_scale(numerator, Fraction(1) / scale)
        denominator = _poly_scale(denominator, Fraction(1) / scale)
    return numerator, denominator


def check_equivalent(lhs: Expr, rhs: Expr) -> Optional[bool]:
    """Decide exact equivalence over the rationals.

    Returns ``True``/``False`` when both sides are rational functions, and
    ``None`` when the expressions involve transcendentals.
    """
    left = normal_form(lhs)
    right = normal_form(rhs)
    if left is None or right is None:
        return None
    ln, ld = left
    rn, rd = right
    difference = _poly_add(_poly_mul(ln, rd), _poly_scale(_poly_mul(rn, ld), Fraction(-1)))
    return not difference


def _symbol_of(monomial: MonomialKey) -> Optional[str]:
    if len(monomial) != 1:
        return None
    name, exp = monomial[0]
    if exp != 1:
        return None
    return name[1:] if name.startswith("$") else None


def univariate_coefficients(expr: Expr, var: Symbol) -> Optional[Dict[int, Fraction]]:
    """Coefficients ``{degree: coefficient}`` of ``expr`` as a polynomial in ``var``.

    Returns ``None`` when the expression is not polynomial in ``var``.
    """
    form = normal_form(expr)
    if form is None:
        return None
    numerator, denominator = form
    if len(denominator) != 1 or _ONE_MONOMIAL not in denominator:
        # constant denominators are fine after scaling; anything else is rational
        if len(denominator) != 1:
            return None
    scale = denominator.get(_ONE_MONOMIAL, Fraction(1))
    coefficients: Dict[int, Fraction] = {}
    for monomial, value in numerator.items():
        degree = 0
        for name, exp in monomial:
            if name == "$" + var.name:
                degree += exp
            else:
                return None
        coefficients[degree] = coefficients.get(degree, Fraction(0)) + value / scale
    return {degree: coeff for degree, coeff in coefficients.items() if coeff != 0}


def from_univariate(coefficients: Dict[int, Fraction], var: Symbol) -> Expr:
    """Rebuild an expression from ``{degree: coefficient}``."""
    terms: List[Expr] = []
    for degree in sorted(coefficients, reverse=True):
        coefficient = coefficients[degree]
        if coefficient == 0:
            continue
        if degree == 0:
            terms.append(Number(coefficient))
        elif degree == 1:
            terms.append(mul(coefficient, var))
        else:
            terms.append(mul(coefficient, power(var, degree)))
    return add(*terms)


def degree(expr: Expr, var: Symbol) -> Optional[int]:
    coefficients = univariate_coefficients(expr, var)
    if coefficients is None:
        return None
    return max(coefficients, default=0)


def rational_roots(coefficients: Dict[int, Fraction]) -> List[Fraction]:
    """All rational roots of a univariate polynomial (rational root theorem)."""
    if not coefficients:
        return []
    degree_max = max(coefficients)
    if degree_max == 0:
        return []
    leading = coefficients.get(degree_max, Fraction(1))
    constant = coefficients.get(0, Fraction(0))
    if constant == 0:
        return [Fraction(0)]

    positives: List[Fraction] = []
    for p in _positive_divisors(abs(constant.numerator)):
        for q in _positive_divisors(abs(leading.numerator)):
            for sign in (1, -1):
                candidate = Fraction(sign * p, q)
                if candidate not in positives and _evaluate_poly(coefficients, candidate) == 0:
                    positives.append(candidate)
    return sorted(positives)


def _positive_divisors(n: int) -> List[int]:
    if n == 0:
        return [1]
    divisors = set()
    step = 1
    while step * step <= n:
        if n % step == 0:
            divisors.add(step)
            divisors.add(n // step)
        step += 1
    return sorted(divisors)


def _evaluate_poly(coefficients: Dict[int, Fraction], value: Fraction) -> Fraction:
    total = Fraction(0)
    for degree, coefficient in coefficients.items():
        total += coefficient * value**degree
    return total


def factor_quadratic(coefficients: Dict[int, Fraction], var: Optional[Symbol] = None) -> Optional[Tuple[Expr, Expr]]:
    """Factor ``a x^2 + b x + c`` into two rational linear factors when possible."""
    if set(coefficients) - {0, 1, 2} or 2 not in coefficients:
        return None
    a = coefficients.get(2, Fraction(0))
    b = coefficients.get(1, Fraction(0))
    c = coefficients.get(0, Fraction(0))
    if a == 0:
        return None
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    root = _rational_sqrt(discriminant)
    if root is None:
        return None
    x = var or sym("x")
    r1 = (-b + root) / (2 * a)
    r2 = (-b - root) / (2 * a)
    return (mul(a, _linear_factor(r1, x)), _linear_factor(r2, x))


def _rational_sqrt(value: Fraction) -> Optional[Fraction]:
    if value < 0:
        return None
    num = _exact_sqrt(value.numerator)
    den = _exact_sqrt(value.denominator)
    if num is None or den is None:
        return None
    return Fraction(num, den)


def _exact_sqrt(n: int) -> Optional[int]:
    if n < 0:
        return None
    r = int(n**0.5)
    for candidate in (r - 1, r, r + 1):
        if candidate >= 0 and candidate * candidate == n:
            return candidate
    return None


def _linear_factor(root: Fraction, var: Symbol) -> Expr:
    return add(var, -root)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``a*x + b*y = g = gcd(a, b)``."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s, old_t


def mod_inverse(a: int, modulus: int) -> Optional[int]:
    """Modular inverse of ``a`` modulo ``modulus``, or ``None`` if it does not exist."""
    a %= modulus
    if a == 0:
        return None
    g, x, _ = extended_gcd(a, modulus)
    if g != 1:
        return None
    return x % modulus


# ---------------------------------------------------------------------------
# Univariate polynomial arithmetic (for rational simplification)
# ---------------------------------------------------------------------------


def poly_trim(coefficients: Dict[int, Fraction]) -> Dict[int, Fraction]:
    """Drop zero coefficients and coerce every coefficient to an exact Fraction."""
    trimmed: Dict[int, Fraction] = {}
    for degree, coeff in coefficients.items():
        value = Fraction(coeff)
        if value != 0:
            trimmed[degree] = value
    return trimmed


def poly_divmod(
    numerator: Dict[int, Fraction], denominator: Dict[int, Fraction]
) -> Optional[Tuple[Dict[int, Fraction], Dict[int, Fraction]]]:
    """Euclidean division of univariate polynomials over the rationals."""
    denominator = poly_trim(denominator)
    if not denominator:
        return None
    remainder = poly_trim(dict(numerator))
    quotient: Dict[int, Fraction] = {}
    denominator_degree = max(denominator)
    leading = denominator[denominator_degree]
    guard = 0
    while remainder and max(remainder) >= denominator_degree:
        guard += 1
        if guard > 500:
            return None
        shift = max(remainder) - denominator_degree
        factor = remainder[max(remainder)] / leading
        quotient[shift] = quotient.get(shift, Fraction(0)) + factor
        for degree, coeff in denominator.items():
            target = degree + shift
            remainder[target] = remainder.get(target, Fraction(0)) - factor * coeff
        remainder = poly_trim(remainder)
    return poly_trim(quotient), remainder


def poly_divide(
    numerator: Dict[int, Fraction], denominator: Dict[int, Fraction]
) -> Optional[Dict[int, Fraction]]:
    """Exact quotient, or ``None`` when the division is not exact."""
    result = poly_divmod(numerator, denominator)
    if result is None:
        return None
    quotient, remainder = result
    if remainder:
        return None
    return quotient


def poly_gcd(
    left: Dict[int, Fraction], right: Dict[int, Fraction]
) -> Optional[Dict[int, Fraction]]:
    """Monic greatest common divisor of two univariate polynomials."""
    a = poly_trim(left)
    b = poly_trim(right)
    guard = 0
    while b:
        guard += 1
        if guard > 500:
            return None
        result = poly_divmod(a, b)
        if result is None:
            return None
        a, b = b, result[1]
    a = poly_trim(a)
    if not a:
        return {0: Fraction(0)}
    leading = a[max(a)]
    return {degree: coeff / leading for degree, coeff in a.items()}


def poly_content(coefficients: Dict[int, Fraction]) -> Fraction:
    """Rational content of a univariate polynomial."""
    coefficients = poly_trim(coefficients)
    if not coefficients:
        return Fraction(0)
    return max(coefficients.values(), key=abs)
