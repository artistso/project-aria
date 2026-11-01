"""Benchmark task suite.

Each task pairs a natural-language / semi-formal prompt with an *independently
stated* expected answer, so scoring never depends on the engine's own trace.
Expected answers are compared symbolically (``Verifier.equivalent``), not as
strings, which is what makes "exact correctness" meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

__all__ = ["Task", "TASKS", "tasks_for", "SUITES"]


@dataclass(frozen=True)
class Task:
    """A single benchmark item."""

    task_id: str
    suite: str
    prompt: str
    expected_solutions: Tuple[str, ...] = ()
    expected_result: Optional[str] = None
    expected_status: Optional[str] = None
    dimensions: Dict[str, str] = field(default_factory=dict)
    note: str = ""


SUITES = (
    "solve",
    "simplify",
    "factor",
    "log-exp",
    "modular",
    "calculus",
    "verify",
    "units",
)


def _t(*args, **kwargs) -> Task:
    return Task(*args, **kwargs)


TASKS: Tuple[Task, ...] = (
    # -- linear and quadratic solving -------------------------------------
    _t("solve-01", "solve", "solve for x in 2x + 4 = 10", expected_solutions=("3",)),
    _t("solve-02", "solve", "solve for x in 3x - 7 = 2x + 5", expected_solutions=("12",)),
    _t("solve-03", "solve", "solve x^2 - 9 = 0", expected_solutions=("-3", "3")),
    _t("solve-04", "solve", "solve x^2 + 5x + 6 = 0", expected_solutions=("-3", "-2")),
    _t("solve-05", "solve", "solve x^2 - 2x - 3 = 0", expected_solutions=("-1", "3")),
    _t("solve-06", "solve", "solve x^2 - 2 = 0", expected_solutions=("2^(1/2)", "-2^(1/2)")),
    _t("solve-07", "solve", "solve 5x = 3", expected_solutions=("3/5",)),
    _t("solve-08", "solve", "solve x/2 + 1/3 = 5/6", expected_solutions=("1",)),
    _t("solve-09", "solve", "solve 4x - 8 = 4x - 8", expected_status="solved", note="identity: every x solves it"),
    _t("solve-10", "solve", "solve 2x + 1 = 2x + 5", expected_status="refuted", note="contradiction"),
    _t("solve-11", "solve", "solve x^2 - 6x + 9 = 0", expected_solutions=("3",)),
    _t("solve-12", "solve", "solve 3x + 2 = 11", expected_solutions=("3",)),
    # -- simplification ---------------------------------------------------
    _t("simp-01", "simplify", "simplify 2x + 3x - x", expected_result="4x"),
    _t("simp-02", "simplify", "simplify (x+1)^2 - x^2", expected_result="2x + 1"),
    _t("simp-03", "simplify", "simplify (x+1)^3", expected_result="x^3 + 3x^2 + 3x + 1"),
    _t("simp-04", "simplify", "simplify (x^2 - 1)/(x - 1)", expected_result="x + 1"),
    _t("simp-05", "simplify", "simplify (x^2 + 2x + 1)/(x + 1)", expected_result="x + 1"),
    _t("simp-06", "simplify", "simplify 3(x + 2) - 2(x + 1)", expected_result="x + 4"),
    _t("simp-07", "simplify", "simplify (a+b)^2 - (a-b)^2", expected_result="4 a b"),
    # -- factoring --------------------------------------------------------
    _t("fact-01", "factor", "factor x^2 - 4", expected_result="(x - 2)(x + 2)"),
    _t("fact-02", "factor", "factor x^2 + 5x + 6", expected_result="(x + 2)(x + 3)"),
    _t("fact-03", "factor", "factor 2x^2 + 4x", expected_result="2x(x + 2)"),
    _t("fact-04", "factor", "factor 9x^2 - 16", expected_result="(3x - 4)(3x + 4)"),
    _t("fact-05", "factor", "factor x^2 - 5x + 6", expected_result="(x - 2)(x - 3)"),
    # -- logs and exponentials -------------------------------------------
    _t("log-01", "log-exp", "simplify log(a*b) - log(a)", expected_result="log(b)"),
    _t("log-02", "log-exp", "simplify log(a^3)", expected_result="3 log(a)"),
    _t("log-03", "log-exp", "simplify log(a/b) + log(b)", expected_result="log(a)"),
    _t("log-04", "log-exp", "verify exp(log(x)) = x", expected_status="solved"),
    _t("log-05", "log-exp", "verify log(a*b) = log(a) + log(b)", expected_status="solved"),
    _t("log-06", "log-exp", "verify log(a+b) = log(a) + log(b)", expected_status="refuted"),
    # -- modular arithmetic ----------------------------------------------
    _t("mod-01", "modular", "solve 5x ≡ 3 mod 7", expected_solutions=("2",)),
    _t("mod-02", "modular", "solve 3x ≡ 1 mod 7", expected_solutions=("5",)),
    _t("mod-03", "modular", "solve 4x ≡ 6 mod 8", expected_solutions=(), note="no solution"),
    _t("mod-04", "modular", "solve x ≡ 3 mod 7", expected_solutions=("3",)),
    _t("mod-05", "modular", "verify 17 ≡ 2 mod 5", expected_status="solved"),
    # -- calculus ---------------------------------------------------------
    _t("calc-01", "calculus", "differentiate x^3 + 2x^2 with respect to x", expected_result="3x^2 + 4x"),
    _t("calc-02", "calculus", "differentiate sin(x) with respect to x", expected_result="cos(x)"),
    _t("calc-03", "calculus", "differentiate log(x) with respect to x", expected_result="1/x"),
    _t("calc-04", "calculus", "differentiate x^2 sin(x) with respect to x", expected_result="2x sin(x) + x^2 cos(x)"),
    _t("calc-05", "calculus", "differentiate exp(2x) with respect to x", expected_result="2 exp(2x)"),
    # -- identity verification -------------------------------------------
    _t("ver-01", "verify", "verify (a+b)^2 = a^2 + 2ab + b^2", expected_status="solved"),
    _t("ver-02", "verify", "verify (a+b)^2 = a^2 + b^2", expected_status="refuted"),
    _t("ver-03", "verify", "verify sin(x)^2 + cos(x)^2 = 1", expected_status="solved"),
    _t("ver-04", "verify", "verify 2 + 2 = 4", expected_status="solved"),
    _t("ver-05", "verify", "verify 2 + 2 = 5", expected_status="refuted"),
    _t("ver-06", "verify", "verify ((x^2 - 1)/(x - 1)) = x + 1", expected_status="solved"),
)

#: Physical laws used by the dimensional-analysis suite: ``law -> symbol dims``.
PHYSICS_LAWS: Dict[str, Dict[str, str]] = {
    "newton_second": {
        "equation": "F = m a",
        "dims": {"F": "kg*m/s^2", "m": "kg", "a": "m/s^2"},
    },
    "mass_energy": {
        "equation": "E = m c^2",
        "dims": {"E": "kg*m^2/s^2", "m": "kg", "c": "m/s"},
    },
    "kinetic_energy": {
        "equation": "E_k = 1/2 m v^2",
        "dims": {"E_k": "kg*m^2/s^2", "m": "kg", "v": "m/s"},
    },
    "gravitational": {
        "equation": "F = G m1 m2 / r^2",
        "dims": {"F": "kg*m/s^2", "G": "m^3/(kg*s^2)", "m1": "kg", "m2": "kg", "r": "m"},
    },
    "ohms_law": {
        "equation": "V = I R",
        "dims": {"V": "kg*m^2/(s^3*A)", "I": "A", "R": "kg*m^2/(s^3*A^2)"},
    },
    "coulomb": {
        "equation": "F = k q1 q2 / r^2",
        "dims": {"F": "kg*m/s^2", "k": "kg*m^3/(s^4*A^2)", "q1": "A*s", "q2": "A*s", "r": "m"},
    },
    "wave_speed": {
        "equation": "v = f λ",
        "dims": {"v": "m/s", "f": "1/s", "λ": "m"},
    },
}


def tasks_for(suite: Optional[str] = None) -> Tuple[Task, ...]:
    """All tasks, or only those in ``suite``."""
    if suite is None:
        return TASKS
    return tuple(task for task in TASKS if task.suite == suite)
