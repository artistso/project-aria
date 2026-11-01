"""The symbolic reasoning engine.

The engine is a *planner over verified rewrites*: it selects rules, applies
them, records each transformation with its justification and invariant, and
asks the verifier to check every step.  A derivation that contains a refuted
step is never returned as solved.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import poly
from ..core.ast import (
    Add,
    Congruence,
    Derivative,
    Equation,
    Expr,
    Func,
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
    render,
)
from ..core.state import Goal, GoalKind, SymbolicState
from ..verify.verifier import Verdict, Verifier
from . import rules as R
from .derivation import OPEN, PARTIAL, REFUTED, SOLVED, Derivation

__all__ = ["SymbolicEngine"]


class SymbolicEngine:
    """Plans and verifies symbolic derivations.

    Parameters
    ----------
    verifier:
        Verification backend; a default :class:`Verifier` is created if omitted.
    max_iterations:
        Upper bound on simplification/rewrite rounds, which keeps pathological
        inputs from looping.
    """

    def __init__(self, verifier: Optional[Verifier] = None, max_iterations: int = 12) -> None:
        self.verifier = verifier or Verifier()
        self.max_iterations = max_iterations

    # -- dispatch ----------------------------------------------------------

    def run(self, state: SymbolicState) -> Derivation:
        """Execute the goal recorded in ``state``."""
        goal = state.goal
        kind = goal.kind
        if kind is GoalKind.SOLVE:
            target = state.primary
            if isinstance(target, Congruence):
                return self.solve_congruence(target, goal.var or target.lhs.free_symbols[0])
            if isinstance(target, Equation):
                if goal.var is None:
                    raise ValueError("solve goal requires a variable")
                return self.solve(target, goal.var)
            # A bare expression: solve 'expr = 0' implicitly.
            if goal.var is None:
                symbols = target.free_symbols
                if len(symbols) != 1:
                    raise ValueError("cannot infer the variable to solve for")
                return self.solve(Equation(target, num(0)), symbols[0])
            return self.solve(Equation(target, num(0)), goal.var)
        if kind is GoalKind.SIMPLIFY or kind is GoalKind.COMPUTE:
            return self.simplify(state.primary, goal)
        if kind is GoalKind.EXPAND:
            return self.expand(state.primary, goal)
        if kind is GoalKind.FACTOR:
            return self.factor(state.primary, goal)
        if kind is GoalKind.DIFFERENTIATE:
            if goal.var is None:
                raise ValueError("differentiation requires a variable")
            return self.differentiate(goal.target or state.primary, goal.var, goal)
        if kind is GoalKind.VERIFY:
            return self.prove(state.primary, goal)
        if kind is GoalKind.DERIVE:
            return self.derive(state)
        return self.analyze(state)

    def analyze(self, state: SymbolicState) -> Derivation:
        derivation = Derivation(state.goal)
        derivation.result = state.primary
        derivation.status = OPEN
        derivation.notes = "no goal specified; showing the parsed symbolic form only"
        return derivation

    # -- simplification ----------------------------------------------------

    def simplify(self, expr: Expr, goal: Optional[Goal] = None) -> Derivation:
        derivation = Derivation(goal or Goal(GoalKind.SIMPLIFY, expr))
        current = expr
        for _ in range(self.max_iterations):
            progressed = False
            for rule in R.simplify_rules():
                candidate, why = R.apply_rule_everywhere(current, rule)
                if candidate != current:
                    if why is None:
                        # The tree changed only through canonical reconstruction
                        # (constant folding, cancellation of inverses).  Label it
                        # honestly instead of attributing it to the attempted rule.
                        derivation.add(
                            "normalise",
                            current,
                            candidate,
                            "canonicalise by folding constants and identities",
                            self.verifier,
                        )
                    else:
                        derivation.add(rule.name, current, candidate, why, self.verifier)
                    current = candidate
                    progressed = True
            if not progressed:
                break
        derivation.result = current
        final = self.verifier.equivalent(expr, current)
        if final.ok:
            derivation.status = SOLVED
        elif final.status == "refuted":
            derivation.status = REFUTED
            derivation.notes = f"final form failed verification: {final}"
        else:
            derivation.status = PARTIAL
            derivation.notes = f"chain unverified by backend: {final}"
        return derivation

    # -- expansion ---------------------------------------------------------

    def expand(self, expr: Expr, goal: Optional[Goal] = None) -> Derivation:
        derivation = Derivation(goal or Goal(GoalKind.EXPAND, expr))
        expanded = R.expand(expr)
        if expanded != expr:
            derivation.add("distribute", expr, expanded, "distribute multiplication over addition", self.verifier)
        current = expanded
        for rule in R.simplify_rules():
            candidate, why = R.apply_rule_everywhere(current, rule)
            if candidate != current:
                derivation.add(rule.name, current, candidate, why or rule.description, self.verifier)
                current = candidate
        derivation.result = current
        verdict = self.verifier.equivalent(expr, current)
        derivation.status = SOLVED if verdict.ok else PARTIAL
        if not verdict.ok:
            derivation.notes = str(verdict)
        return derivation

    # -- factoring ---------------------------------------------------------

    def factor(self, expr: Expr, goal: Optional[Goal] = None) -> Derivation:
        derivation = Derivation(goal or Goal(GoalKind.FACTOR, expr))
        current = expr
        for _ in range(self.max_iterations):
            progressed = False
            for rule in R.factor_rules():
                candidate = rule.apply(current)
                if candidate is not None and candidate[0] != current:
                    after, why = candidate
                    derivation.add(rule.name, current, after, why, self.verifier)
                    current = after
                    progressed = True
            if not progressed:
                break
        derivation.result = current
        verdict = self.verifier.equivalent(expr, current)
        if verdict.ok:
            derivation.status = SOLVED
        elif current == expr:
            derivation.status = PARTIAL
            derivation.notes = "no factorisation rule applies to this expression"
        else:
            derivation.status = REFUTED
            derivation.notes = str(verdict)
        return derivation

    # -- calculus ----------------------------------------------------------

    def differentiate(self, expr: Expr, var: Symbol, goal: Optional[Goal] = None) -> Derivation:
        derivation = Derivation(goal or Goal(GoalKind.DIFFERENTIATE, expr, var))
        source = expr.expr if isinstance(expr, Derivative) else expr
        raw, used = R.differentiate(source, var)
        derivative_node = Derivative(source, var)
        derivation.add(
            "differentiate",
            derivative_node,
            raw,
            "apply " + ", ".join(_unique(used)),
            self.verifier,
        )
        simplified = self.simplify(raw)
        for step in simplified.steps:
            derivation.add(step.rule, step.before, step.after, step.justification, self.verifier)
        derivation.result = simplified.result
        expected = self._expected_derivative(source, var)
        if expected is None:
            derivation.status = PARTIAL
            derivation.notes = "no independent derivative backend available"
        else:
            verdict = self.verifier.equivalent(simplified.result, expected)
            if verdict.ok:
                derivation.status = SOLVED
            else:
                derivation.status = REFUTED
                derivation.notes = f"derivative check failed: {verdict}"
            derivation.result = simplified.result
        return derivation

    def _expected_derivative(self, expr: Expr, var: Symbol) -> Optional[Expr]:
        from ..verify import sympy_backend

        if not sympy_backend.available():
            return None
        try:
            return sympy_backend.from_sympy(
                sympy_backend.sp.diff(sympy_backend.to_sympy(expr), sympy_backend.to_sympy(var))
            )
        except Exception:  # pragma: no cover - defensive
            return None

    # -- equation solving --------------------------------------------------

    def solve(self, equation: Equation, var: Symbol) -> Derivation:
        derivation = Derivation(Goal(GoalKind.SOLVE, equation, var))
        residual = add(equation.lhs, mul(-1, equation.rhs))
        if residual != equation.lhs or not (isinstance(equation.rhs, Number) and equation.rhs.is_zero):
            derivation.add(
                "move_all_to_one_side",
                equation,
                Equation(residual, num(0)),
                "subtract the right-hand side from both sides",
                self.verifier,
                {"kind": "equation_scaled"},
            )
        current = residual
        for _ in range(self.max_iterations):
            progressed = False
            for rule in R.simplify_rules():
                candidate, why = R.apply_rule_everywhere(current, rule)
                if candidate != current:
                    derivation.add(
                        rule.name if why else "normalise",
                        Equation(current, num(0)),
                        Equation(candidate, num(0)),
                        why or "canonicalise by folding constants and identities",
                        self.verifier,
                        {"kind": "equation_scaled"},
                    )
                    current = candidate
                    progressed = True
            if not progressed:
                break

        coefficients = poly.univariate_coefficients(current, var)
        if coefficients is None:
            derivation.status = PARTIAL
            derivation.notes = f"expression is not polynomial in {var.name}; a proof-assistant backend is required"
            return derivation

        degree = max(coefficients, default=0)
        if degree == 0:
            if not coefficients:
                derivation.status = SOLVED
                derivation.notes = "the equation holds for every value (both sides identically equal)"
            else:
                derivation.status = REFUTED
                derivation.notes = "the equation has no solution: the residual is a non-zero constant"
            return derivation

        if degree == 1:
            self._linear_steps(derivation, current, coefficients, var)
        elif degree == 2:
            self._quadratic_steps(derivation, current, coefficients, var)
        else:
            self._higher_degree_steps(derivation, current, coefficients, var)

        self._finalise_solutions(derivation, current, var)
        return derivation

    def _linear_steps(
        self, derivation: Derivation, current: Expr, coefficients: Dict[int, Fraction], var: Symbol
    ) -> None:
        a = coefficients.get(1, Fraction(0))
        b = coefficients.get(0, Fraction(0))
        if a == 0:
            return
        isolated = Equation(mul(a, var), Number(-b))
        derivation.add(
            "isolate_variable_term",
            Equation(current, num(0)),
            isolated,
            f"add {render(Number(-b))} to both sides",
            self.verifier,
            {"kind": "equation_scaled"},
        )
        solution = Number(-b / a)
        if a != 1:
            derivation.add(
                "divide_by_coefficient",
                isolated,
                Equation(var, solution),
                f"divide both sides by {render(Number(a))}",
                self.verifier,
                {"kind": "both_sides_mul"},
            )
        derivation.solutions = (solution,)
        derivation.result = solution

    def _quadratic_steps(
        self, derivation: Derivation, current: Expr, coefficients: Dict[int, Fraction], var: Symbol
    ) -> None:
        a = coefficients.get(2, Fraction(0))
        b = coefficients.get(1, Fraction(0))
        c = coefficients.get(0, Fraction(0))
        roots = poly.rational_roots(coefficients)
        if len(roots) == 2:
            factored = mul(a, add(var, -roots[0]), add(var, -roots[1]))
            derivation.add(
                "factor_by_roots",
                Equation(current, num(0)),
                Equation(factored, num(0)),
                f"factor using the rational roots {roots[0]} and {roots[1]}",
                self.verifier,
                {"kind": "equation_scaled"},
            )
            self._root_steps(derivation, factored, sorted(roots), var)
            return

        discriminant = b * b - 4 * a * c
        monic = Equation(add(power(var, 2), mul(b / a, var), Number(c / a)), num(0))
        if monic != Equation(current, num(0)):
            derivation.add(
                "normalise_monic",
                Equation(current, num(0)),
                monic,
                f"divide both sides by the leading coefficient {render(Number(a))}",
                self.verifier,
                {"kind": "equation_scaled"},
            )
        if discriminant < 0:
            real = power(Number(-discriminant), Number(Fraction(1, 2)))
            imaginary = Symbol("i")
            base = div(Number(-b), Number(2 * a))
            scale = div(real, Number(2 * a))
            roots_expr = [
                _radical_simplify(add(base, mul(scale, imaginary))),
                _radical_simplify(add(base, mul(-scale, imaginary))),
            ]
            derivation.notes = "the discriminant is negative, so the solutions are complex"
        else:
            root = power(Number(discriminant), Number(Fraction(1, 2)))
            base = div(Number(-b), Number(2 * a))
            scale = div(root, Number(2 * a))
            roots_expr = [_radical_simplify(add(base, scale)), _radical_simplify(add(base, mul(-1, scale)))]
        unique_roots: List[Expr] = []
        for value in roots_expr:
            if not any(value.canonical_key() == existing.canonical_key() for existing in unique_roots):
                unique_roots.append(value)
        roots_expr = unique_roots
        for value in roots_expr:
            derivation.add(
                "quadratic_formula",
                monic,
                Equation(var, value),
                "apply the quadratic formula",
                self.verifier,
                {"kind": "root", "var": var, "root": value},
            )
        derivation.solutions = tuple(roots_expr)
        derivation.result = roots_expr[0]

    def _higher_degree_steps(
        self, derivation: Derivation, current: Expr, coefficients: Dict[int, Fraction], var: Symbol
    ) -> None:
        degree = max(coefficients)
        roots = poly.rational_roots(coefficients)
        if roots and len(roots) == degree:
            factored = mul(coefficients[degree], *(add(var, -r) for r in roots))
            derivation.add(
                "factor_by_roots",
                Equation(current, num(0)),
                Equation(factored, num(0)),
                f"factor completely using the rational roots {sorted(roots)}",
                self.verifier,
                {"kind": "equation_scaled"},
            )
            self._root_steps(derivation, factored, sorted(roots), var)
        else:
            derivation.status = PARTIAL
            derivation.notes = (
                f"polynomial of degree {degree} found, but only {len(roots)} rational root(s); "
                "irrational or complex roots need a proof-assistant backend"
            )
            derivation.result = current

    def _root_steps(self, derivation: Derivation, factored: Expr, roots: Sequence[Fraction], var: Symbol) -> None:
        solution_values: List[Expr] = []
        for root in roots:
            value = Number(root)
            derivation.add(
                "zero_product_rule",
                Equation(factored, num(0)),
                Equation(var, value),
                f"the factor {render(add(var, -root))} vanishes when {var.name} = {render(value)}",
                self.verifier,
                {"kind": "root", "var": var, "root": value},
            )
            solution_values.append(value)
        derivation.solutions = tuple(solution_values)
        derivation.result = solution_values[0] if solution_values else None

    def _finalise_solutions(self, derivation: Derivation, residual: Expr, var: Symbol) -> None:
        if not derivation.solutions:
            if derivation.status == OPEN:
                derivation.status = PARTIAL
            return
        verdict = self.verifier.solution(Equation(residual, num(0)), var, derivation.solutions)
        if verdict.ok:
            derivation.status = SOLVED
        elif verdict.status == "refuted":
            derivation.status = REFUTED
            derivation.notes = f"candidate solutions rejected: {verdict}"
        else:
            derivation.status = PARTIAL
            derivation.notes = f"solutions produced but not fully verified: {verdict}"

    # -- modular arithmetic ------------------------------------------------

    def solve_congruence(self, congruence: Congruence, var: Symbol) -> Derivation:
        derivation = Derivation(Goal(GoalKind.SOLVE, congruence, var))
        modulus_value = _integer_value(congruence.modulus)
        if modulus_value is None or modulus_value == 0:
            derivation.status = PARTIAL
            derivation.notes = "modulus must be a non-zero integer"
            return derivation
        residual = add(congruence.lhs, mul(-1, congruence.rhs))
        coefficients = poly.univariate_coefficients(residual, var)
        if coefficients is None or set(coefficients) - {0, 1}:
            derivation.status = PARTIAL
            derivation.notes = "only linear congruences a*x ≡ b (mod n) are supported"
            return derivation
        a = int(coefficients.get(1, Fraction(0)))
        b = int(-coefficients.get(0, Fraction(0)))
        a_reduced = a % modulus_value
        b_reduced = b % modulus_value
        normalised = Congruence(mul(a_reduced, var), Number(b_reduced), Number(modulus_value))
        if normalised != congruence:
            derivation.add(
            "reduce_coefficients",
            congruence,
            normalised,
                "reduce both sides modulo the modulus",
                self.verifier,
                {"kind": "modular_equivalent"},
            )
        common = gcd(a_reduced, modulus_value)
        if b_reduced % common != 0:
            derivation.status = SOLVED
            derivation.solutions = ()
            derivation.result = None
            derivation.notes = (
                f"no solution: gcd({a_reduced}, {modulus_value}) = {common} does not divide {b_reduced}"
            )
            return derivation
        if common > 1:
            step_modulus = modulus_value // common
            step_a = a_reduced // common
            step_b = b_reduced // common
            derivation.add(
                "divide_congruence",
                normalised,
                Congruence(mul(step_a, var), Number(step_b), Number(step_modulus)),
                f"divide the congruence by {common}; the modulus becomes {step_modulus}",
                self.verifier,
                {"kind": "premise"},
            )
            a_reduced, b_reduced, modulus_value = step_a, step_b, step_modulus
        inverse = poly.mod_inverse(a_reduced, modulus_value)
        if inverse is None:
            derivation.status = PARTIAL
            derivation.notes = "coefficient is not invertible modulo the modulus"
            return derivation
        residue = (b_reduced * inverse) % modulus_value
        derivation.add(
            "multiply_by_inverse",
            Congruence(mul(a_reduced, var), Number(b_reduced), Number(modulus_value)),
            Congruence(var, Number(residue), Number(modulus_value)),
            f"multiply both sides by the modular inverse {inverse}",
            self.verifier,
            {"kind": "modular_equivalent"},
        )
        solutions = [Number(residue + k * modulus_value) for k in range(common)]
        derivation.solutions = tuple(solutions)
        derivation.result = solutions[0]
        verdict = self.verifier.solution(congruence, var, solutions)
        if verdict.ok:
            derivation.status = SOLVED
        elif verdict.status == "refuted":
            derivation.status = REFUTED
            derivation.notes = str(verdict)
        else:
            derivation.status = PARTIAL
            derivation.notes = str(verdict)
        return derivation

    # -- verification and proof -------------------------------------------

    def prove(self, claim: Expr, goal: Optional[Goal] = None) -> Derivation:
        derivation = Derivation(goal or Goal(GoalKind.VERIFY, claim))
        verdict = self.verifier.claim(claim)
        derivation.result = claim
        if verdict.ok:
            derivation.status = SOLVED
            derivation.notes = f"verified: {verdict}"
        elif verdict.status == "refuted":
            derivation.status = REFUTED
            derivation.notes = f"refuted: {verdict}"
        else:
            derivation.status = PARTIAL
            derivation.notes = f"undecided: {verdict}"
        return derivation

    def derive(self, state: SymbolicState) -> Derivation:
        derivation = Derivation(state.goal)
        target = state.goal.target or state.expressions[-1]
        derivation.result = target
        verdict = self.verifier.claim(target) if len(state.expressions) == 1 else Verdict("unknown", "no-backend")
        if len(state.expressions) > 1:
            derivation.status = PARTIAL
            derivation.notes = (
                "multi-premise derivation requires the proof-assistant bridge (see docs/ROADMAP.md); "
                "premises were parsed and type-checked but not discharged"
            )
        elif verdict.ok:
            derivation.status = SOLVED
            derivation.notes = str(verdict)
        else:
            derivation.status = REFUTED if verdict.status == "refuted" else PARTIAL
            derivation.notes = str(verdict)
        return derivation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _integer_value(expr: Expr) -> Optional[int]:
    if isinstance(expr, Number) and expr.is_exact and expr.value.denominator == 1:
        return int(expr.value)
    return None


def _radical_simplify(expr: Expr) -> Expr:
    """Apply the radical rule until fixpoint (keeps solution forms readable)."""
    current = expr
    for _ in range(4):
        candidate, _why = R.apply_rule_everywhere(current, R.RULES["simplify_radical"])
        if candidate == current:
            break
        current = candidate
    return current


def _unique(items: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
