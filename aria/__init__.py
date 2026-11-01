"""ARIA -- Algebraic Reasoning & Inference Architecture.

A neuro-symbolic reasoning stack in which symbolic objects, not text tokens, are
the substrate for mathematical and physical reasoning:

    from aria import parse_problem, SymbolicEngine

    state = parse_problem("solve for x in 2x + 4 = 10")
    derivation = SymbolicEngine().run(state)
    print(derivation.to_text())      # every step verified
    print(derivation.solutions)      # (3,)
"""

from __future__ import annotations

from .core.ast import Equation, Expr, Number, Symbol, add, mul, num, power, sym
from .core.parse import ParseError, parse, parse_problem
from .core.state import Goal, GoalKind, SymbolicState
from .embed.tree_embed import TreeEmbedding, embed
from .reason.derivation import Derivation, DerivationStep
from .reason.engine import SymbolicEngine
from .science.laws import LAW_LIBRARY, check_law, derive_dimension
from .science.units import Dimension, check_equation, parse_unit
from .verify.sympy_backend import available as sympy_available
from .verify.verifier import Verdict, Verifier

__version__ = "0.1.0"

__all__ = [
    "Derivation",
    "DerivationStep",
    "Dimension",
    "Equation",
    "Expr",
    "Goal",
    "GoalKind",
    "LAW_LIBRARY",
    "Number",
    "ParseError",
    "Symbol",
    "SymbolicEngine",
    "SymbolicState",
    "TreeEmbedding",
    "Verdict",
    "Verifier",
    "__version__",
    "add",
    "check_equation",
    "check_law",
    "derive_dimension",
    "embed",
    "mul",
    "num",
    "parse",
    "parse_problem",
    "parse_unit",
    "power",
    "sym",
    "sympy_available",
]
