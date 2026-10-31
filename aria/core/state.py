"""Symbolic state: the structured object the reasoner operates on."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from .ast import Expr, Symbol

__all__ = ["GoalKind", "Goal", "SymbolicState"]


class GoalKind(str, Enum):
    """What the system is being asked to do with a symbolic state."""

    SOLVE = "solve"
    SIMPLIFY = "simplify"
    FACTOR = "factor"
    EXPAND = "expand"
    DIFFERENTIATE = "differentiate"
    VERIFY = "verify"
    PROVE = "prove"
    COMPUTE = "compute"
    DERIVE = "derive"
    ANALYZE = "analyze"


@dataclass(frozen=True)
class Goal:
    """A typed objective, e.g. ``Goal(Solve(x))``."""

    kind: GoalKind
    target: Optional[Expr] = None
    var: Optional[Symbol] = None

    def __str__(self) -> str:
        if self.var is not None:
            return f"Goal({self.kind.value}({self.var.name}))"
        if self.target is not None:
            return f"Goal({self.kind.value}({self.target}))"
        return f"Goal({self.kind.value})"


@dataclass
class SymbolicState:
    """A typed symbolic problem statement.

    ``expressions`` holds the formal statements extracted from the input (an
    equation, an expression to simplify, ...), ``goal`` records the objective,
    and ``assumptions`` carries declared properties of symbols (``x`` positive,
    ``n`` integer, ...).  Everything downstream -- embeddings, rewriting,
    verification -- consumes this object, never raw text.
    """

    expressions: Tuple[Expr, ...] = ()
    goal: Goal = field(default_factory=lambda: Goal(GoalKind.ANALYZE))
    assumptions: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    source_text: str = ""

    def __post_init__(self) -> None:
        self.expressions = tuple(self.expressions)
        self.assumptions = {k: frozenset(v) for k, v in self.assumptions.items()}

    @property
    def symbols(self) -> Tuple[Symbol, ...]:
        found: Dict[str, Symbol] = {}
        for expr in self.expressions:
            for symbol in expr.free_symbols:
                found.setdefault(symbol.name, symbol)
        return tuple(found[name] for name in sorted(found))

    @property
    def primary(self) -> Expr:
        if not self.expressions:
            raise ValueError("symbolic state has no expressions")
        return self.expressions[0]

    def with_assumption(self, name: str, *assumptions: str) -> "SymbolicState":
        merged = set(self.assumptions.get(name, frozenset()))
        merged.update(assumptions)
        updated = dict(self.assumptions)
        updated[name] = frozenset(merged)
        return SymbolicState(self.expressions, self.goal, updated, self.source_text)

    def __str__(self) -> str:
        body = ", ".join(str(e) for e in self.expressions) or "<empty>"
        return f"SymbolicState({body}; {self.goal})"
