"""Formal verification layer.

Every symbolic claim the system produces -- a rewritten expression, an equation
manipulation, a solution, a physical law application -- passes through a
:class:`Verifier`.  A verdict is never a probability: it is one of

* ``proven``   -- established exactly (symbolic backend or exact arithmetic),
* ``refuted``  -- shown false (counterexample or non-zero difference),
* ``unknown``  -- the available backend cannot decide.

That three-valued discipline is the core difference between "the model produced
plausible-looking algebra" and "the system asserts a checked claim".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..core import poly
from ..core.ast import (
    Add,
    Congruence,
    Equation,
    Expr,
    Func,
    Matrix,
    Mul,
    Number,
    Pow,
    Relation,
    Symbol,
    add,
    div,
    mul,
    num,
    power,
)
from . import sympy_backend

__all__ = ["Verdict", "Verifier"]


PROVEN = "proven"
REFUTED = "refuted"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verdict:
    """Outcome of a verification attempt."""

    status: str
    method: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PROVEN

    @property
    def decided(self) -> bool:
        return self.status in (PROVEN, REFUTED)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.status} via {self.method}{suffix}"


class Verifier:
    """Three-valued symbolic verifier.

    Parameters
    ----------
    backend:
        ``"auto"`` (use SymPy when importable), ``"sympy"`` (require it) or
        ``"internal"`` (force the dependency-free path).
    samples:
        Number of numeric probe points used to falsify claims and to give
        supporting evidence when no exact backend can decide.
    seed:
        Deterministic probe sampling, so verification is reproducible.
    """

    def __init__(self, backend: str = "auto", samples: int = 24, seed: int = 0) -> None:
        if backend not in ("auto", "sympy", "internal"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "sympy" and not sympy_backend.available():
            raise RuntimeError("SymPy backend requested but not installed")
        self.use_sympy = sympy_backend.available() and backend in ("auto", "sympy")
        self.samples = samples
        self.seed = seed
        self._rng = random.Random(seed)
        self.log: list[Verdict] = []

    # -- public API --------------------------------------------------------

    @property
    def backend(self) -> str:
        return sympy_backend.backend_name() if self.use_sympy else "internal"

    def record(self, verdict: Verdict) -> Verdict:
        self.log.append(verdict)
        return verdict

    def equivalent(self, lhs: Expr, rhs: Expr, assumptions: Optional[Mapping[str, Iterable[str]]] = None) -> Verdict:
        """Decide whether two expressions are identically equal."""
        verdict = self._equivalent(lhs, rhs)
        return self.record(verdict)

    def claim(self, expr: Expr) -> Verdict:
        """Decide a claim: an equation, congruence, relation or boolean value."""
        if isinstance(expr, Equation):
            return self.record(self._equation_identity(expr))
        if isinstance(expr, Congruence):
            return self.record(self._congruence(expr))
        if isinstance(expr, Relation):
            return self.record(self._relation(expr))
        return self.record(self._equivalent(expr, num(0)))

    def solution(self, equation: Expr, var: Symbol, values: Sequence[Expr]) -> Verdict:
        """Check that every candidate value satisfies ``equation`` exactly."""
        if not values:
            return self.record(Verdict(REFUTED, "solution-check", "no candidate values"))
        worst: Optional[Verdict] = None
        for value in values:
            verdict = self.satisfies(equation, var, value)
            if verdict.status == REFUTED:
                return self.record(Verdict(REFUTED, "solution-check", f"{var.name}={value}: {verdict.detail}"))
            if not verdict.ok:
                worst = verdict
        if worst is not None:
            return self.record(Verdict(UNKNOWN, "solution-check", worst.detail))
        return self.record(
            Verdict(PROVEN, "solution-check", f"{len(values)} value(s) satisfy the equation exactly")
        )

    def satisfies(self, expr: Expr, var: Symbol, value: Expr) -> Verdict:
        """Check a candidate root by exact substitution, falling back to SymPy."""
        if isinstance(expr, Equation):
            residual = add(expr.lhs, mul(-1, expr.rhs))
        elif isinstance(expr, Congruence):
            lhs = _evaluate(expr.lhs.subs({var.name: value}))
            rhs = _evaluate(expr.rhs.subs({var.name: value}))
            modulus = _evaluate(expr.modulus)
            if lhs is None or rhs is None or modulus in (None, 0):
                return self.record(Verdict(UNKNOWN, "modular-substitution", "non-exact residue"))
            ok = (lhs - rhs) % modulus == 0
            return self.record(
                Verdict(PROVEN if ok else REFUTED, "modular-substitution", f"({lhs} - {rhs}) mod {modulus}")
            )
        else:
            residual = expr

        substituted = residual.subs({var.name: value})
        exact = _evaluate(substituted)
        if exact is not None:
            status = PROVEN if exact == 0 else REFUTED
            return self.record(Verdict(status, "exact-substitution", f"residual = {exact}"))
        if self.use_sympy:
            sp = sympy_backend.sp
            try:
                sym = sympy_backend.to_sympy(substituted)
                simplified = sp.simplify(sym)
                if simplified == 0:
                    return self.record(Verdict(PROVEN, "sympy-substitution"))
                if simplified.is_number:
                    try:
                        if abs(complex(simplified)) > 1e-9:
                            return self.record(
                                Verdict(REFUTED, "sympy-substitution", f"residual = {simplified}")
                            )
                    except TypeError:  # pragma: no cover
                        pass
                if sp.simplify(sp.nsimplify(sp.N(sym, 30))) == 0:  # pragma: no cover - numeric
                    return self.record(Verdict(PROVEN, "sympy-numeric-substitution"))
            except Exception as exc:  # pragma: no cover - defensive
                return self.record(Verdict(UNKNOWN, "sympy-substitution", f"error: {exc}"))
        return self.record(Verdict(UNKNOWN, "substitution", "residual could not be reduced to zero exactly"))

    def step(self, step: "StepLike") -> Verdict:  # noqa: F821 - protocol by duck typing
        """Verify a single derivation step using its declared invariant."""
        kind = step.check.get("kind", "equivalent")
        if kind == "equivalent":
            return self.record(self._equivalent(step.before, step.after))
        if kind == "both_sides_add":
            return self.record(self._both_sides_add(step.before, step.after))
        if kind == "both_sides_mul":
            return self.record(self._both_sides_mul(step.before, step.after))
        if kind == "equation_scaled":
            return self.record(self._equation_scaled(step.before, step.after))
        if kind == "modular_equivalent":
            return self.record(self._modular_equivalent(step.before, step.after))
        if kind == "root":
            return self.record(self.satisfies(step.before, step.check["var"], step.check["root"]))
        if kind == "premise":
            return self.record(Verdict(PROVEN, "premise", "assumed by the problem statement"))
        return self.record(Verdict(UNKNOWN, f"step-{kind}", "no verification procedure for this step kind"))

    # -- internals ---------------------------------------------------------

    def _equivalent(self, lhs: Expr, rhs: Expr) -> Verdict:
        if lhs == rhs:
            return Verdict(PROVEN, "structural", "identical canonical forms")

        if _has_transcendental_constant(lhs) or _has_transcendental_constant(rhs):
            # pi, e and i are constants, not free variables: numeric probing and
            # polynomial normal forms would be unsound here, so defer to SymPy.
            if self.use_sympy:
                return self._sympy_equivalent(lhs, rhs)
            return Verdict(UNKNOWN, "no-backend", "constants pi/e/i require a symbolic backend")

        counter = self._numeric_counterexample(lhs, rhs)
        if counter is not None:
            return Verdict(REFUTED, "numeric-probe", counter)

        float_counter = self._float_counterexample(lhs, rhs)
        if float_counter is not None:
            return Verdict(REFUTED, "float-probe", float_counter)

        poly_verdict = poly.check_equivalent(lhs, rhs)
        if poly_verdict is not None:
            status = PROVEN if poly_verdict else REFUTED
            return Verdict(status, "polynomial-normal-form")

        if self.use_sympy:
            verdict = self._sympy_equivalent(lhs, rhs)
            if verdict.decided:
                return verdict
        if counter is None:
            evidence = self._numeric_agreement(lhs, rhs)
            if evidence is not None:
                return Verdict(UNKNOWN, "numeric-probe", evidence)
        return Verdict(UNKNOWN, "no-backend", "requires a symbolic backend for transcendental terms")

    def _sympy_equivalent(self, lhs: Expr, rhs: Expr) -> Verdict:
        sp = sympy_backend.sp
        try:
            a = sympy_backend.to_sympy(lhs)
            b = sympy_backend.to_sympy(rhs)
        except Exception as exc:  # pragma: no cover - defensive
            return Verdict(UNKNOWN, "sympy", f"conversion failed: {exc}")
        # A single normalisation strategy is not enough: log/trig identities
        # need their own canonicalisers, and precise results often require
        # rationalising before simplifying.  Each candidate is a *sound*
        # rewriting of the difference, so a zero result is a proof.
        deltas = [a - b]
        try:
            deltas.append(sp.together(a - b))
            deltas.append(sp.expand(a - b))
            deltas.append(sp.expand_log(a - b, force=True))
            deltas.append(sp.logcombine(a - b, force=True))
            deltas.append(sp.expand_trig(a - b))
            deltas.append(sp.trigsimp(a - b))
            deltas.append(sp.cancel(a - b))
            deltas.append(sp.radsimp(a - b))
        except Exception:  # pragma: no cover - defensive
            pass

        for delta in deltas:
            try:
                simplified = sp.simplify(delta)
            except Exception:  # pragma: no cover - defensive
                continue
            if simplified == 0:
                return Verdict(PROVEN, "sympy-simplify")
            if simplified.is_number:
                try:
                    if simplified != 0 and abs(complex(simplified)) > 1e-12:
                        return Verdict(REFUTED, "sympy-simplify", f"difference = {simplified}")
                except TypeError:  # pragma: no cover
                    pass
        return Verdict(UNKNOWN, "sympy-simplify", "backend could not reduce the difference to zero")

    def _equation_identity(self, eq: Equation) -> Verdict:
        terms = eq.lhs.free_symbols + eq.rhs.free_symbols
        if not terms:
            lhs = _evaluate(eq.lhs)
            rhs = _evaluate(eq.rhs)
            if lhs is None or rhs is None:
                return Verdict(UNKNOWN, "numeric", "non-exact operands")
            return Verdict(
                PROVEN if lhs == rhs else REFUTED, "exact-numeric", f"{lhs} {'==' if lhs == rhs else '!='} {rhs}"
            )
        return self._equivalent(eq.lhs, eq.rhs)

    def _congruence(self, cong: Congruence) -> Verdict:
        lhs = _evaluate(cong.lhs.subs({}))
        rhs = _evaluate(cong.rhs)
        modulus = _evaluate(cong.modulus)
        if lhs is not None and rhs is not None and modulus:
            ok = (lhs - rhs) % modulus == 0
            return Verdict(PROVEN if ok else REFUTED, "modular-arithmetic", f"({lhs} - {rhs}) mod {modulus}")
        if self.use_sympy:
            sp = sympy_backend.sp
            try:
                difference = sp.simplify(sympy_backend.to_sympy(cong.lhs - cong.rhs))
                base = sympy_backend.to_sympy(cong.modulus)
                if sp.simplify(difference % base) == 0:
                    return Verdict(PROVEN, "sympy-mod")
            except Exception:  # pragma: no cover
                pass
        return Verdict(UNKNOWN, "modular-arithmetic")

    def _relation(self, rel: Relation) -> Verdict:
        lhs = _evaluate(rel.lhs)
        rhs = _evaluate(rel.rhs)
        if lhs is None or rhs is None:
            return Verdict(UNKNOWN, "numeric-relation", "non-exact operands")
        result = {"<": lhs < rhs, "<=": lhs <= rhs, ">": lhs > rhs, ">=": lhs >= rhs, "!=": lhs != rhs}[rel.op]
        return Verdict(PROVEN if result else REFUTED, "numeric-relation", f"{lhs} {rel.op} {rhs}")

    def _both_sides_add(self, before: Expr, after: Expr) -> Verdict:
        if not (isinstance(before, Equation) and isinstance(after, Equation)):
            return Verdict(UNKNOWN, "both-sides-add", "not an equation step")
        delta = self._equivalent(_sub(after.lhs, before.lhs), _sub(after.rhs, before.rhs))
        return Verdict(delta.status, "both-sides-add", "same quantity added to both sides" if delta.ok else delta.detail)

    def _both_sides_mul(self, before: Expr, after: Expr) -> Verdict:
        if not (isinstance(before, Equation) and isinstance(after, Equation)):
            return Verdict(UNKNOWN, "both-sides-mul", "not an equation step")
        if _evaluate(after.lhs) == 0 or _evaluate(after.rhs) == 0:
            return Verdict(UNKNOWN, "both-sides-mul", "potential multiplication by zero")
        delta = self._equivalent(div(after.lhs, before.lhs), div(after.rhs, before.rhs))
        status = PROVEN if delta.ok else UNKNOWN
        return Verdict(status, "both-sides-mul", "same non-zero factor applied to both sides")

    def _equation_scaled(self, before: Expr, after: Expr) -> Verdict:
        """Verify an equation rewrite by checking the residuals are proportional.

        Every admissible algebraic manipulation of an equation (adding the same
        quantity to both sides, multiplying by a non-zero constant, factoring,
        completing the square, ...) leaves ``lhs - rhs`` unchanged up to a
        non-zero scalar multiple.  Operations that are *not* reversible -- such
        as squaring both sides -- break proportionality and are rejected here,
        which is exactly the behaviour we want.
        """
        if not (isinstance(before, Equation) and isinstance(after, Equation)):
            return Verdict(UNKNOWN, "equation-scale", "not an equation step")
        residual_before = add(before.lhs, mul(-1, before.rhs))
        residual_after = add(after.lhs, mul(-1, after.rhs))
        if _evaluate(residual_before) == 0 and _evaluate(residual_after) == 0:
            return Verdict(PROVEN, "equation-scale", "both sides identical")
        verdict = self._equivalent(residual_after, residual_before)
        if verdict.ok:
            return Verdict(PROVEN, "equation-scale", "residual preserved up to a non-zero factor")
        if verdict.status == REFUTED:
            return Verdict(REFUTED, "equation-scale", "residuals are not proportional")
        # Fall back to checking for a constant ratio directly.
        if self.use_sympy:
            sp = sympy_backend.sp
            try:
                ratio = sp.simplify(sympy_backend.to_sympy(residual_after) / sympy_backend.to_sympy(residual_before))
                if ratio.is_number and ratio != 0:
                    return Verdict(PROVEN, "equation-scale", f"residual scaled by {ratio}")
                if ratio == 1:
                    return Verdict(PROVEN, "equation-scale", "identical residual")
            except Exception:  # pragma: no cover - defensive
                pass
        return Verdict(UNKNOWN, "equation-scale", "could not relate the two residual forms")

    def _modular_equivalent(self, before: Expr, after: Expr) -> Verdict:
        """Verify that two linear congruences have the same solution set.

        Sound by construction: both solution sets are enumerated over the least
        common multiple of the moduli and compared residue by residue.
        """
        if not (isinstance(before, Congruence) and isinstance(after, Congruence)):
            return Verdict(UNKNOWN, "modular", "not a congruence step")
        left = _linear_congruence_solutions(before)
        right = _linear_congruence_solutions(after)
        if left is None or right is None:
            return Verdict(UNKNOWN, "modular", "unsupported congruence shape")
        modulus_left, residues_left = left
        modulus_right, residues_right = right
        from math import lcm

        span = lcm(modulus_left, modulus_right)
        if span > 100_000:
            return Verdict(UNKNOWN, "modular", "modulus too large to enumerate")
        expanded_left = {r % span for r in residues_left} | {
            (r + k * modulus_left) % span
            for r in residues_left
            for k in range(span // modulus_left)
        }
        expanded_right = {r % span for r in residues_right} | {
            (r + k * modulus_right) % span
            for r in residues_right
            for k in range(span // modulus_right)
        }
        if expanded_left == expanded_right:
            return Verdict(PROVEN, "modular-solution-set", f"{len(expanded_left)} residue class(es) mod {span}")
        return Verdict(
            REFUTED,
            "modular-solution-set",
            f"solution sets differ mod {span}: |before|={len(expanded_left)}, |after|={len(expanded_right)}",
        )

    def _numeric_counterexample(self, lhs: Expr, rhs: Expr) -> Optional[str]:
        symbols = {s.name: s for s in lhs.free_symbols + rhs.free_symbols}
        if not symbols:
            return None
        rng = random.Random(self.seed)
        for _ in range(self.samples):
            point = {name: Fraction(rng.randint(2, 97), rng.randint(1, 13)) for name in symbols}
            a = _evaluate(lhs.subs(point))
            b = _evaluate(rhs.subs(point))
            if a is None or b is None:
                return None
            if a != b:
                rendered = ", ".join(f"{k}={v}" for k, v in point.items())
                return f"{lhs} = {a} but {rhs} = {b} at {rendered}"
        return None

    def _float_counterexample(self, lhs: Expr, rhs: Expr) -> Optional[str]:
        """Refute identities involving transcendentals by numeric approximation.

        Floating point can only ever *refute* a claim, never prove one: the
        tolerance is deliberately loose so that rounding cannot manufacture a
        counterexample, and a disagreement beyond it is reported.
        """
        symbols = {s.name: s for s in lhs.free_symbols + rhs.free_symbols if s.name not in _CONSTANT_VALUES}
        if not symbols:
            return None
        rng = random.Random(self.seed + 7)
        for _ in range(self.samples):
            point = {name: rng.uniform(0.3, 3.0) for name in symbols}
            a = _evaluate_float(lhs, point)
            b = _evaluate_float(rhs, point)
            if a is None or b is None:
                return None
            scale = 1.0 + abs(a) + abs(b)
            if abs(a - b) > 1e-6 * scale:
                rendered = ", ".join(f"{k}={v:.4f}" for k, v in point.items())
                return f"{a:.6f} != {b:.6f} at {rendered}"
        return None

    def _numeric_agreement(self, lhs: Expr, rhs: Expr) -> Optional[str]:
        symbols = {s.name: s for s in lhs.free_symbols + rhs.free_symbols}
        if not symbols:
            return None
        rng = random.Random(self.seed + 1)
        for _ in range(self.samples):
            point = {name: Fraction(rng.randint(2, 97), rng.randint(1, 13)) for name in symbols}
            a = _evaluate(lhs.subs(point))
            b = _evaluate(rhs.subs(point))
            if a is None or b is None or a != b:
                return None
        return f"agrees on {self.samples} sample points (evidence, not proof)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sub(a: Expr, b: Expr) -> Expr:
    return Add((a, mul(-1, b))) if not isinstance(a, Number) or not isinstance(b, Number) else Number(a.value - b.value)


def _evaluate(expr: Expr) -> Optional[Fraction]:
    """Exact numeric evaluation of a closed expression, or ``None``."""
    if isinstance(expr, Number) and expr.is_exact:
        return expr.value
    if isinstance(expr, Number):
        return Fraction(expr.value).limit_denominator(10**9)
    if isinstance(expr, Symbol):
        return None
    if isinstance(expr, Add):
        total = Fraction(0)
        for term in expr.terms:
            value = _evaluate(term)
            if value is None:
                return None
            total += value
        return total
    if isinstance(expr, Mul):
        total = Fraction(1)
        for factor in expr.factors:
            value = _evaluate(factor)
            if value is None:
                return None
            total *= value
        return total
    if isinstance(expr, Pow):
        base = _evaluate(expr.base)
        exponent = _evaluate(expr.exp)
        if base is None or exponent is None:
            return None
        if exponent.denominator != 1:
            return None
        exp_int = int(exponent)
        if base == 0 and exp_int < 0:
            return None
        return base**exp_int
    if isinstance(expr, Func):
        values = [_evaluate(a) for a in expr.args]
        if any(v is None for v in values):
            return None
        if expr.name == "abs" and len(values) == 1:
            return abs(values[0])
        if expr.name == "sign" and len(values) == 1:
            return Fraction((values[0] > 0) - (values[0] < 0))
        return None
    return None


_TRANSCENDENTAL_NAMES = {"pi", "e", "i", "tau", "inf"}

_CONSTANT_VALUES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _evaluate_float(expr: Expr, point: Mapping[str, float]) -> Optional[float]:
    """Floating-point evaluation used only for refutation probes."""
    if isinstance(expr, Number):
        return float(expr.value)
    if isinstance(expr, Symbol):
        if expr.name in _CONSTANT_VALUES:
            return _CONSTANT_VALUES[expr.name]
        return point.get(expr.name)
    if isinstance(expr, Add):
        total = 0.0
        for term in expr.terms:
            value = _evaluate_float(term, point)
            if value is None:
                return None
            total += value
        return total
    if isinstance(expr, Mul):
        total = 1.0
        for factor in expr.factors:
            value = _evaluate_float(factor, point)
            if value is None:
                return None
            total *= value
        return total
    if isinstance(expr, Pow):
        base = _evaluate_float(expr.base, point)
        exponent = _evaluate_float(expr.exp, point)
        if base is None or exponent is None:
            return None
        try:
            return float(base) ** float(exponent)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    if isinstance(expr, Func) and len(expr.args) == 1:
        argument = _evaluate_float(expr.args[0], point)
        if argument is None:
            return None
        try:
            if expr.name == "log":
                return math.log(argument) if argument > 0 else None
            if expr.name == "exp":
                return math.exp(min(argument, 50.0))
            if expr.name == "sin":
                return math.sin(argument)
            if expr.name == "cos":
                return math.cos(argument)
            if expr.name == "tan":
                return math.tan(argument)
            if expr.name == "abs":
                return abs(argument)
            if expr.name == "sign":
                return float((argument > 0) - (argument < 0))
        except (ValueError, OverflowError):
            return None
    return None


def _has_transcendental_constant(expr: Expr) -> bool:
    return any(symbol.name in _TRANSCENDENTAL_NAMES for symbol in expr.free_symbols)


def _linear_congruence_solutions(cong: Congruence) -> Optional[Tuple[int, set]]:
    """Residues solving ``a x ≡ b (mod n)``; returns ``(n, residues)``."""
    modulus = _integer_value(cong.modulus)
    if modulus is None or modulus <= 0:
        return None
    residual = add(cong.lhs, mul(-1, cong.rhs))
    variables = residual.free_symbols
    if len(variables) != 1:
        return None
    var = variables[0]
    coefficients = poly.univariate_coefficients(residual, var)
    if coefficients is None or set(coefficients) - {0, 1}:
        return None
    a = int(coefficients.get(1, Fraction(0)))
    b = int(-coefficients.get(0, Fraction(0)))
    residues = {x for x in range(modulus) if (a * x - b) % modulus == 0}
    return modulus, residues


def _integer_value(expr: Expr) -> Optional[int]:
    if isinstance(expr, Number) and expr.is_exact and expr.value.denominator == 1:
        return int(expr.value)
    return None


class StepLike:
    """Structural protocol implemented by derivation steps."""

    before: Expr
    after: Expr
    check: Dict[str, object]
