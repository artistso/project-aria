"""The ARIA math channel protocol.

The model is allowed to say exactly one thing: *which* mathematical operation the
user wants and *what* the expression is.  It never states an answer.  That single
restriction is what turns a language model into a translator at the boundary of a
symbolic runtime, and it is enforced three ways:

1. a grammar (``aria/llm/grammar.py``) makes malformed output unrepresentable;
2. :func:`parse_contract` validates whatever arrived anyway;
3. ARIA's own parser and verifier decide the mathematics.

Wire format (three lines, order fixed)::

    goal: solve
    var: x
    expr: 2x + 4 = 10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..core.ast import Expr, Symbol
from ..core.parse import ParseError, parse
from ..core.state import Goal, GoalKind, SymbolicState
from .client import Message

__all__ = [
    "MathRequest",
    "ContractError",
    "parse_contract",
    "to_state",
    "SYSTEM_PROMPT",
    "translation_messages",
    "narration_messages",
    "DETERMINISTIC_TRANSLATION",
]

#: Contract goal tags mapped onto ARIA goal kinds.
GOAL_TAGS = {
    "solve": GoalKind.SOLVE,
    "simplify": GoalKind.SIMPLIFY,
    "factor": GoalKind.FACTOR,
    "expand": GoalKind.EXPAND,
    "verify": GoalKind.VERIFY,
    "differentiate": GoalKind.DIFFERENTIATE,
    "compute": GoalKind.COMPUTE,
    "analyze": GoalKind.ANALYZE,
}

_GOALS_REQUIRING_VAR = {"solve", "differentiate"}


class ContractError(ValueError):
    """Raised when the model's output does not satisfy the contract."""


@dataclass(frozen=True)
class MathRequest:
    """What the model is permitted to produce: a goal and an expression."""

    goal: str
    expression: str
    variable: Optional[str] = None

    def as_text(self) -> str:
        lines = [f"goal: {self.goal}"]
        if self.variable:
            lines.append(f"var: {self.variable}")
        lines.append(f"expr: {self.expression}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def parse_contract(text: str) -> MathRequest:
    """Parse the three-line contract, tolerating stray whitespace and fences."""
    if text is None:
        raise ContractError("no output from the model")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rstrip("`").strip()

    fields: dict[str, str] = {}
    for line in cleaned.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key in ("goal", "var", "expr") and value:
            fields.setdefault(key, value)

    if "goal" not in fields:
        raise ContractError(f"missing 'goal:' line in model output: {text[:200]!r}")
    goal = fields["goal"].lower()
    if goal not in GOAL_TAGS:
        raise ContractError(
            f"unknown goal {goal!r}; expected one of {', '.join(sorted(GOAL_TAGS))}"
        )
    if "expr" not in fields:
        raise ContractError(f"missing 'expr:' line in model output: {text[:200]!r}")

    expression = fields["expr"]
    try:
        parse(expression)
    except ParseError as error:
        raise ContractError(f"expression does not parse ({error}); received {expression!r}") from error

    variable = fields.get("var")
    if goal in _GOALS_REQUIRING_VAR and not variable:
        raise ContractError(f"goal {goal!r} requires a 'var:' line")

    return MathRequest(goal=goal, expression=expression, variable=variable)


def to_state(request: MathRequest) -> SymbolicState:
    """Turn a validated request into a fully typed symbolic state."""
    expression: Expr = parse(request.expression)
    kind = GOAL_TAGS[request.goal]
    var: Optional[Symbol] = None

    if request.variable:
        var = Symbol(request.variable)
    elif kind in (GoalKind.SOLVE, GoalKind.DIFFERENTIATE):
        candidates = expression.free_symbols
        if len(candidates) != 1:
            raise ContractError(
                f"goal {request.goal!r} needs an unambiguous variable; "
                f"found {[s.name for s in candidates]}"
            )
        var = candidates[0]

    return SymbolicState((expression,), Goal(kind, expression, var), source_text=request.expression)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the natural-language front end of a symbolic mathematics engine.

You do NOT solve anything. You do NOT compute. You do NOT explain.
You translate the user's request into exactly three lines and stop.

Format (no other text, no markdown, no comments):

goal: <solve|simplify|factor|expand|verify|differentiate|compute|analyze>
var: <single variable name, only for solve and differentiate>
expr: <the expression or equation in plain mathematical notation>

Parsing rules for `expr`:
- use * for multiplication between numbers and parentheses, and plain
  juxtaposition for products of variables (2x, x y)
- use ^ for powers, / for division, parentheses for grouping
- write equations with = and congruences as: a ≡ b mod n
- functions: log(x), exp(x), sin(x), cos(x), tan(x), sqrt(x)
- differentiate with: d/dx[expr]
- no units, no prose, no trailing punctuation

Examples:

user: solve 2x + 4 = 10
goal: solve
var: x
expr: 2x + 4 = 10

user: what is (x+1)^2 - x^2?
goal: simplify
expr: (x+1)^2 - x^2

user: is log(a*b) the same as log(a)+log(b)?
goal: verify
expr: log(a*b) = log(a) + log(b)

user: differentiate x^3 + 2x with respect to x
goal: differentiate
var: x
expr: x^3 + 2x
"""

#: Stand-in "model" used by ``--dry-run`` and tests: ARIA's own parser plays the
#: translator, so the whole loop can be exercised without any weights.
DETERMINISTIC_TRANSLATION = "__deterministic__"


def translation_messages(user_text: str) -> List[Message]:
    return [
        Message("system", SYSTEM_PROMPT),
        Message("user", user_text),
    ]


def narration_messages(user_text: str, verified_trace: str) -> List[Message]:
    """Prompt for the text back-end: explain only what the verifier approved."""
    system = (
        "You explain results from a symbolic mathematics engine to a user.\n"
        "Every fact you need is in the VERIFIED TRACE. Do not add claims, do not "
        "recompute, do not mention steps that are not listed. If the trace reports "
        "status 'partial' or 'refuted', say so plainly and explain what was and was "
        "not established. Be concise: at most four sentences.\n"
    )
    user = (
        f"User question: {user_text}\n\n"
        f"VERIFIED TRACE\n{verified_trace}\n"
        "Explain the result to the user."
    )
    return [
        Message("system", system),
        Message("user", user),
    ]
