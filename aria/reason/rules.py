"""Rewrite-rule library.

A rule is a *local, justified* transformation: given an expression it either
returns a new expression together with the mathematical justification for the
rewrite, or returns ``None``.  Nothing here is heuristic in the statistical
sense -- every rule is an identity or an admissible algebraic operation, which
is what makes step-level verification meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core import poly
from ..core.ast import (
    Add,
    Derivative,
    Expr,
    Func,
    Mul,
    Number,
    Pow,
    Symbol,
    add,
    coeff_and_rest,
    div,
    mul,
    num,
    power,
    render,
)

__all__ = [
    "Rule",
    "Candidate",
    "RULES",
    "rule",
    "simplify_rules",
    "factor_rules",
    "expand_rules",
    "combine_like_terms",
    "expand",
    "differentiate",
    "apply_rule_everywhere",
]

Candidate = Tuple[Expr, str]


@dataclass(frozen=True)
class Rule:
    """A named, justified rewrite."""

    name: str
    kind: str
    description: str
    fn: Callable[[Expr], Optional[Candidate]]

    def apply(self, expr: Expr) -> Optional[Candidate]:
        try:
            return self.fn(expr)
        except (ZeroDivisionError, ValueError, TypeError, RecursionError):
            return None


RULES: Dict[str, Rule] = {}


def rule(name: str, kind: str, description: str):
    def decorator(fn: Callable[[Expr], Optional[Candidate]]) -> Callable[[Expr], Optional[Candidate]]:
        RULES[name] = Rule(name, kind, description, fn)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Algebraic rules
# ---------------------------------------------------------------------------


def combine_like_terms(expr: Expr) -> Optional[Candidate]:
    """``2x + 3x -> 5x`` (and the general like-term collection)."""
    if not isinstance(expr, Add):
        return None
    groups: Dict[Tuple, List] = {}
    for term in expr.terms:
        coefficient, rest = coeff_and_rest(term)
        key = rest.canonical_key()
        if key in groups:
            groups[key][1] += coefficient
        else:
            groups[key] = [rest, coefficient]
    terms: List[Expr] = []
    for rest, coefficient in groups.values():
        if coefficient == 0:
            continue
        terms.append(rest if coefficient == 1 else mul(coefficient, rest))
    rebuilt = add(*terms)
    if rebuilt == expr:
        return None
    return rebuilt, "collect like terms"


@rule("collect_like_terms", "algebra", "combine terms with the same symbolic part")
def _rule_collect(expr: Expr) -> Optional[Candidate]:
    return combine_like_terms(expr)


@rule("distribute", "algebra", "multiply out a product over a sum")
def _rule_distribute(expr: Expr) -> Optional[Candidate]:
    if isinstance(expr, Mul):
        # Never distribute over a denominator: a/(b + c) is already simpler
        # than a/b + a/c, and the rewrite would destroy the rational form.
        if any(_is_negative_power(f) for f in expr.factors):
            return None
        if not any(isinstance(f, Add) for f in expr.factors):
            return None
        expanded = expand(Mul(expr.factors))
        if expanded == expr:
            return None
        return expanded, "distribute multiplication over addition"
    if isinstance(expr, Pow) and isinstance(expr.base, Add):
        if isinstance(expr.exp, Number) and expr.exp.is_exact and 2 <= int(expr.exp.value) <= 8:
            expanded = expand(expr)
            if expanded != expr:
                return expanded, f"expand the power {render(expr)}"
    return None


def _is_negative_power(expr: Expr) -> bool:
    return isinstance(expr, Pow) and isinstance(expr.exp, Number) and expr.exp.value < 0


@rule("simplify_radical", "algebra", "sqrt(k^2 n) -> k sqrt(n)")
def _rule_simplify_radical(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Pow):
        return None
    exponent = expr.exp
    if not (isinstance(exponent, Number) and exponent.is_exact and exponent.value == Fraction(1, 2)):
        return None
    base = expr.base
    if not (isinstance(base, Number) and base.is_exact and base.value.denominator == 1 and base.value > 1):
        return None
    value = int(base.value)
    multiplier = 1
    remainder = value
    candidate = 2
    while candidate * candidate <= remainder:
        while remainder % (candidate * candidate) == 0:
            remainder //= candidate * candidate
            multiplier *= candidate
        candidate += 1
    if multiplier == 1:
        return None
    rewritten = mul(multiplier, power(Number(remainder), Number(Fraction(1, 2)))) if remainder > 1 else num(multiplier)
    return rewritten, f"extract the square factor: sqrt({value}) = {multiplier} sqrt({remainder})"
    return None


def expand(expr: Expr, depth: int = 0) -> Expr:
    """Fully distribute products and non-negative integer powers."""
    if depth > 40:
        return expr
    if isinstance(expr, Add):
        return add(*(expand(t, depth + 1) for t in expr.terms))
    if isinstance(expr, Mul):
        factors = [expand(f, depth + 1) for f in expr.factors]
        result: Expr = num(1)
        for factor in factors:
            result = _distribute_product(result, factor, depth + 1)
        return result
    if isinstance(expr, Pow):
        base = expand(expr.base, depth + 1)
        if isinstance(expr.exp, Number) and expr.exp.is_exact and expr.exp.value.denominator == 1:
            exponent = int(expr.exp.value)
            if 1 <= exponent <= 12 and isinstance(base, Add):
                result: Expr = num(1)
                for _ in range(exponent):
                    result = _distribute_product(result, base, depth + 1)
                return result
            if exponent == 0:
                return num(1)
        return power(base, expand(expr.exp, depth + 1))
    return expr


def _distribute_product(left: Expr, right: Expr, depth: int) -> Expr:
    if isinstance(left, Add) and isinstance(right, Add):
        return add(*(_distribute_product(t, right, depth) for t in left.terms))
    if isinstance(left, Add):
        return add(*(mul(t, right) for t in left.terms))
    if isinstance(right, Add):
        return add(*(mul(left, t) for t in right.terms))
    return mul(left, right)


def _is_square(expr: Expr) -> Optional[Expr]:
    if isinstance(expr, Pow) and isinstance(expr.exp, Number) and expr.exp.value == 2:
        return expr.base
    if isinstance(expr, Number) and expr.is_exact and expr.value >= 0:
        root = poly._rational_sqrt(expr.value)
        if root is not None:
            return Number(root)
    if isinstance(expr, Mul):
        coeff, rest = coeff_and_rest(expr)
        if coeff > 0:
            root = poly._rational_sqrt(coeff)
            if root is not None:
                inner = _is_square(rest)
                if inner is not None:
                    return mul(root, inner)
    return None


@rule("difference_of_squares", "factor", "a^2 - b^2 -> (a - b)(a + b)")
def _rule_difference_of_squares(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Add) or len(expr.terms) != 2:
        return None
    positive: Optional[Expr] = None
    negative: Optional[Expr] = None
    for term in expr.terms:
        coefficient, rest = coeff_and_rest(term)
        if coefficient == 0:
            return None
        sign = 1 if coefficient > 0 else -1
        magnitude = mul(abs(coefficient), rest)
        root = _is_square(magnitude)
        if root is None:
            return None
        if sign > 0 and positive is None:
            positive = root
        elif sign < 0 and negative is None:
            negative = root
        else:
            return None
    if positive is None or negative is None:
        return None
    factored = mul(add(positive, mul(-1, negative)), add(positive, negative))
    return factored, "difference of two squares"


@rule("common_factor", "factor", "ab + ac -> a(b + c)")
def _rule_common_factor(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Add) or len(expr.terms) < 2:
        return None
    decomposed = [_decompose_term(term) for term in expr.terms]
    # numeric content: factor out the gcd of integer coefficients when there is one
    coefficients = [Fraction(coeff) for coeff, _ in decomposed]
    scale = Fraction(1)
    if all(c.denominator == 1 for c in coefficients) and all(c != 0 for c in coefficients):
        from math import gcd

        value = 0
        for c in coefficients:
            value = gcd(value, int(abs(c)))
        scale = Fraction(value)

    # symbolic content: factors common to *every* term, counted with multiplicity
    common_items: List[Expr] = []
    reference = decomposed[0][1]
    for index, item in enumerate(reference):
        key = item.canonical_key()
        if all(len(other) > index and other[index].canonical_key() == key for _, other in decomposed[1:]):
            common_items.append(item)
    # keep only the longest shared prefix in canonical order
    shared: List[Expr] = []
    for item in common_items:
        key = item.canonical_key()
        if all(any(f.canonical_key() == key for f in other[1]) for other in decomposed):
            shared.append(item)

    if scale == 1 and not shared:
        return None

    common_expr = mul(scale, *shared)
    remainders: List[Expr] = []
    for coefficient, factors in decomposed:
        remaining = list(factors)
        for item in shared:
            for position, factor in enumerate(remaining):
                if factor.canonical_key() == item.canonical_key():
                    remaining.pop(position)
                    break
        remainders.append(mul(coefficient / scale, *remaining))
    factored = mul(common_expr, add(*remainders))
    if factored == expr:
        return None
    return factored, f"factor out {render(common_expr)}"


def _decompose_term(term: Expr) -> Tuple[Fraction, List[Expr]]:
    """Coefficient and *expanded* factor list of a term (x^2 -> [x, x])."""
    coefficient, rest = coeff_and_rest(term)
    factors: List[Expr] = []
    if not (isinstance(rest, Number) and rest.is_one):
        items = rest.factors if isinstance(rest, Mul) else (rest,)
        for factor in items:
            if (
                isinstance(factor, Pow)
                and isinstance(factor.exp, Number)
                and factor.exp.is_exact
                and factor.exp.value.denominator == 1
                and 0 < int(factor.exp.value) <= 12
            ):
                factors.extend([factor.base] * int(factor.exp.value))
            else:
                factors.append(factor)
    factors.sort(key=lambda e: e.canonical_key())
    return Fraction(coefficient), factors


@rule("quadratic_factor", "factor", "a x^2 + b x + c -> a(x - r1)(x - r2)")
def _rule_quadratic_factor(expr: Expr) -> Optional[Candidate]:
    variables = expr.free_symbols
    if len(variables) != 1:
        return None
    var = variables[0]
    coefficients = poly.univariate_coefficients(expr, var)
    if coefficients is None or max(coefficients, default=0) != 2:
        return None
    roots = poly.rational_roots(coefficients)
    if len(roots) != 2:
        return None
    leading = coefficients[2]
    factored = mul(
        leading,
        add(var, -roots[0]),
        add(var, -roots[1]),
    )
    if factored == expr:
        return None
    return factored, f"factor with rational roots {roots[0]} and {roots[1]}"


@rule("reduce_rational", "algebra", "cancel a common polynomial factor in a rational expression")
def _rule_reduce_rational(expr: Expr) -> Optional[Candidate]:
    numerator, denominator = _split_quotient(expr)
    if denominator is None:
        return None
    variables = sorted({s.name for s in (numerator.free_symbols + denominator.free_symbols)})
    if len(variables) != 1:
        return None
    var = Symbol(variables[0])
    num_coeffs = poly.univariate_coefficients(numerator, var)
    den_coeffs = poly.univariate_coefficients(denominator, var)
    if num_coeffs is None or den_coeffs is None:
        return None
    common = poly.poly_gcd(num_coeffs, den_coeffs)
    if common is None or _degree(common) == 0:
        return None
    reduced_num = poly.poly_divide(num_coeffs, common)
    reduced_den = poly.poly_divide(den_coeffs, common)
    if reduced_num is None or reduced_den is None:
        return None
    rebuilt = div(poly.from_univariate(reduced_num, var), poly.from_univariate(reduced_den, var))
    if rebuilt == expr:
        return None
    return rebuilt, "cancel common polynomial factor"


def _degree(coefficients: Dict[int, Fraction]) -> int:
    return max(coefficients, default=0)


def _split_quotient(expr: Expr) -> Tuple[Expr, Optional[Expr]]:
    """Split ``expr`` into (numerator, denominator) for a single quotient."""
    if isinstance(expr, Mul):
        numerators: List[Expr] = []
        denominators: List[Expr] = []
        for factor in expr.factors:
            if isinstance(factor, Pow) and isinstance(factor.exp, Number) and factor.exp.value < 0:
                exponent = -factor.exp.value
                denominators.append(power(factor.base, exponent) if exponent != 1 else factor.base)
            else:
                numerators.append(factor)
        if len(denominators) == 1:
            return mul(*numerators), denominators[0]
        if denominators:
            return mul(*numerators), mul(*denominators)
        return expr, None
    if isinstance(expr, Pow) and isinstance(expr.exp, Number) and expr.exp.value == -1:
        return num(1), expr.base
    return expr, None


# ---------------------------------------------------------------------------
# Logarithmic / exponential rules
# ---------------------------------------------------------------------------


@rule("log_product", "log", "log(a b) -> log(a) + log(b)")
def _rule_log_product(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Func) or expr.name != "log" or len(expr.args) != 1:
        return None
    argument = expr.args[0]
    if not isinstance(argument, Mul) or len(argument.factors) < 2:
        return None
    rewritten = add(*(Func("log", (f,)) for f in argument.factors))
    return rewritten, "logarithm of a product"


@rule("log_quotient", "log", "log(a/b) -> log(a) - log(b)")
def _rule_log_quotient(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Func) or expr.name != "log" or len(expr.args) != 1:
        return None
    numerator, denominator = _split_quotient(expr.args[0])
    if denominator is None:
        return None
    rewritten = add(Func("log", (numerator,)), mul(-1, Func("log", (denominator,))))
    return rewritten, "logarithm of a quotient"


@rule("log_power", "log", "log(a^n) -> n log(a)")
def _rule_log_power(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Func) or expr.name != "log" or len(expr.args) != 1:
        return None
    argument = expr.args[0]
    if not isinstance(argument, Pow):
        return None
    rewritten = mul(argument.exp, Func("log", (argument.base,)))
    return rewritten, "logarithm of a power"


@rule("exp_log_inverse", "log", "exp(log(a)) -> a and log(exp(a)) -> a")
def _rule_exp_log_inverse(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Func) or len(expr.args) != 1:
        return None
    inner = expr.args[0]
    if expr.name == "exp" and isinstance(inner, Func) and inner.name == "log":
        return inner.args[0], "exp and log are inverse"
    if expr.name == "log" and isinstance(inner, Func) and inner.name == "exp":
        return inner.args[0], "log and exp are inverse"
    return None


@rule("trig_pythagorean", "trig", "sin^2(x) + cos^2(x) -> 1")
def _rule_pythagorean(expr: Expr) -> Optional[Candidate]:
    if not isinstance(expr, Add):
        return None
    for i, first in enumerate(expr.terms):
        for j, second in enumerate(expr.terms):
            if i >= j:
                continue
            for a, b in ((first, second), (second, first)):
                if _is_trig_square(a, "sin") and _is_trig_square(b, "cos") and _trig_argument(a) == _trig_argument(b):
                    rest = [t for k, t in enumerate(expr.terms) if k not in (i, j)]
                    return add(num(1), *rest), "Pythagorean identity"
    return None


def _is_trig_square(expr: Expr, name: str) -> bool:
    return (
        isinstance(expr, Pow)
        and isinstance(expr.exp, Number)
        and expr.exp.value == 2
        and isinstance(expr.base, Func)
        and expr.base.name == name
        and len(expr.base.args) == 1
    )


def _trig_argument(expr: Expr) -> Optional[Expr]:
    if isinstance(expr, Pow) and isinstance(expr.base, Func) and len(expr.base.args) == 1:
        return expr.base.args[0]
    return None


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------


def differentiate(expr: Expr, var: Symbol) -> Tuple[Expr, List[str]]:
    """Symbolic differentiation with a record of the rules used."""
    used: List[str] = []

    def d(node: Expr) -> Expr:
        if isinstance(node, Derivative):
            return d(node.expr)
        if isinstance(node, Number):
            used.append("constant rule")
            return num(0)
        if isinstance(node, Symbol):
            used.append("power rule")
            return num(1) if node.name == var.name else num(0)
        if isinstance(node, Add):
            used.append("sum rule")
            return add(*(d(t) for t in node.terms))
        if isinstance(node, Mul):
            used.append("product rule")
            terms: List[Expr] = []
            for index, factor in enumerate(node.factors):
                piece = d(factor)
                if isinstance(piece, Number) and piece.is_zero:
                    continue
                terms.append(mul(*(node.factors[:index] + (piece,) + node.factors[index + 1 :])))
            return add(*terms)
        if isinstance(node, Pow):
            base, exponent = node.base, node.exp
            if isinstance(exponent, Number) and exponent.is_exact:
                used.append("power rule")
                dbase = d(base)
                if isinstance(dbase, Number) and dbase.is_zero:
                    return num(0)
                return mul(exponent, power(base, exponent.value - 1), dbase)
            used.append("general power rule")
            # f^g = exp(g log f)
            return mul(
                power(base, exponent),
                add(mul(d(exponent), Func("log", (base,))), div(mul(exponent, d(base)), base)),
            )
        if isinstance(node, Func):
            inner = node.args[0]
            dinner = d(inner)
            if node.name == "log":
                used.append("chain rule: d/dx log(u) = u'/u")
                return div(dinner, inner)
            if node.name == "exp":
                used.append("chain rule: d/dx exp(u) = exp(u) u'")
                return mul(node, dinner)
            if node.name == "sin":
                used.append("chain rule: d/dx sin(u) = cos(u) u'")
                return mul(Func("cos", (inner,)), dinner)
            if node.name == "cos":
                used.append("chain rule: d/dx cos(u) = -sin(u) u'")
                return mul(-1, Func("sin", (inner,)), dinner)
            if node.name == "tan":
                used.append("chain rule: d/dx tan(u) = (1 + tan^2(u)) u'")
                return mul(add(num(1), power(node, 2)), dinner)
            if node.name == "abs":
                used.append("chain rule: d/dx |u| = sign(u) u'")
                return mul(Func("sign", (inner,)), dinner)
            used.append(f"unsupported derivative of {node.name}")
            return Derivative(mul(node, dinner), var)
        raise TypeError(f"cannot differentiate {node}")

    return d(expr), used


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------


def apply_rule_everywhere(expr: Expr, the_rule: Rule) -> Tuple[Expr, Optional[str]]:
    """Apply ``the_rule`` bottom-up at every node; return the rewritten tree."""
    justification: Optional[str] = None

    def visit(node: Expr) -> Expr:
        nonlocal justification
        rebuilt = node._rebuild(tuple(visit(c) for c in node.children()))
        candidate = the_rule.apply(rebuilt)
        if candidate is None:
            return rebuilt
        after, why = candidate
        justification = why if justification is None else justification
        return after

    return visit(expr), justification


SIMPLIFY_RULES = (
    "distribute",
    "simplify_radical",
    "collect_like_terms",
    "reduce_rational",
    "log_product",
    "log_quotient",
    "log_power",
    "exp_log_inverse",
    "trig_pythagorean",
)

FACTOR_RULES = ("common_factor", "difference_of_squares", "quadratic_factor")

EXPAND_RULES = ("distribute",)


def simplify_rules() -> List[Rule]:
    return [RULES[name] for name in SIMPLIFY_RULES]


def factor_rules() -> List[Rule]:
    return [RULES[name] for name in FACTOR_RULES]


def expand_rules() -> List[Rule]:
    return [RULES[name] for name in EXPAND_RULES]
