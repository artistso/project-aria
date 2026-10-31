"""Derivations: auditable chains of verified symbolic transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.ast import Expr
from ..core.state import Goal, GoalKind
from ..verify.verifier import Verdict

__all__ = ["DerivationStep", "Derivation", "SOLVED", "PARTIAL", "OPEN", "REFUTED"]

SOLVED = "solved"
PARTIAL = "partial"
OPEN = "open"
REFUTED = "refuted"


@dataclass
class DerivationStep:
    """One justified transformation from ``before`` to ``after``."""

    rule: str
    before: Expr
    after: Expr
    justification: str
    check: Dict[str, Any] = field(default_factory=dict)
    verdict: Optional[Verdict] = None

    @property
    def verified(self) -> bool:
        return bool(self.verdict and self.verdict.ok)

    @property
    def invalid(self) -> bool:
        return bool(self.verdict and self.verdict.status == "refuted")

    def __str__(self) -> str:
        status = str(self.verdict) if self.verdict else "unverified"
        return f"{self.before}  →  {self.after}   [{self.rule}: {self.justification}; {status}]"


@dataclass
class Derivation:
    """A goal, the ordered steps taken toward it, and the verified result."""

    goal: Goal
    steps: List[DerivationStep] = field(default_factory=list)
    result: Optional[Expr] = None
    solutions: Tuple[Expr, ...] = ()
    status: str = OPEN
    notes: str = ""

    # -- construction ------------------------------------------------------

    def add(
        self,
        rule: str,
        before: Expr,
        after: Expr,
        justification: str,
        verifier=None,
        check: Optional[Dict[str, Any]] = None,
    ) -> DerivationStep:
        step = DerivationStep(rule, before, after, justification, check or {"kind": "equivalent"})
        if verifier is not None:
            step.verdict = verifier.step(step)
        self.steps.append(step)
        return step

    # -- inspection --------------------------------------------------------

    @property
    def valid(self) -> bool:
        """True when every step was verified and none was refuted."""
        return all(step.verified for step in self.steps) and not self.invalid_steps

    @property
    def invalid_steps(self) -> List[DerivationStep]:
        return [s for s in self.steps if s.invalid]

    @property
    def unverified_steps(self) -> List[DerivationStep]:
        return [s for s in self.steps if s.verdict is None]

    @property
    def trace_accuracy(self) -> float:
        """Fraction of steps whose invariant checked out."""
        if not self.steps:
            return 1.0
        return sum(1 for s in self.steps if s.verified) / len(self.steps)

    def summary(self) -> Dict[str, Any]:
        return {
            "goal": str(self.goal),
            "status": self.status,
            "steps": len(self.steps),
            "verified_steps": sum(1 for s in self.steps if s.verified),
            "invalid_steps": len(self.invalid_steps),
            "unverified_steps": len(self.unverified_steps),
            "result": str(self.result) if self.result is not None else None,
            "solutions": [str(s) for s in self.solutions],
            "trace_accuracy": round(self.trace_accuracy, 4),
        }

    # -- rendering ---------------------------------------------------------

    def to_text(self, width: int = 0) -> str:
        lines = [f"Goal: {self.goal}", ""]
        current: Optional[Expr] = self.steps[0].before if self.steps else self.result
        if current is not None:
            lines.append(f"  0.  {current}")
        for index, step in enumerate(self.steps, start=1):
            flag = "✓" if step.verified else ("✗" if step.invalid else "?")
            lines.append(f"  {index}.  {step.after}      {flag} {step.rule} — {step.justification}")
        if self.solutions:
            lines.append("")
            lines.append("Solutions: " + ", ".join(f"{self.goal.var.name if self.goal.var else 'x'} = {s}" for s in self.solutions))
        if self.result is not None and not self.solutions:
            lines.append("")
            lines.append(f"Result: {self.result}")
        lines.append(f"Status: {self.status}  (trace accuracy {self.trace_accuracy:.0%})")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [f"### {self.goal}", "", "| # | Step | Rule | Justification | Verified |", "|---|---|---|---|---|"]
        current: Optional[Expr] = self.steps[0].before if self.steps else None
        if current is not None:
            lines.append(f"| 0 | `{current}` | — | initial symbolic state | ✓ |")
        for index, step in enumerate(self.steps, start=1):
            flag = "✓" if step.verified else ("✗" if step.invalid else "?")
            lines.append(
                f"| {index} | `{step.after}` | {step.rule} | {step.justification} | {flag} |"
            )
        lines.append("")
        if self.solutions:
            var = self.goal.var.name if self.goal.var else "x"
            lines.append("**Solutions:** " + ", ".join(f"`{var} = {s}`" for s in self.solutions))
        elif self.result is not None:
            lines.append(f"**Result:** `{self.result}`")
        lines.append(f"**Status:** {self.status} · trace accuracy {self.trace_accuracy:.0%}")
        if self.notes:
            lines.append(f"_{self.notes}_")
        return "\n".join(lines)
