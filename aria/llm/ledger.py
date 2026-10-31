"""The ledger: a record the system keeps of its own correct and incorrect work.

A reward is a number somebody hands you. A ledger is a record you keep. The
difference is not cosmetic: a number can be maximised without being understood,
while a record of *process* -- which rule was applied, whether the engine could
check it, and where it first broke -- cannot. This module replaces the scalar
with that record, and replaces the score with a school.

Four rules make the record trustworthy:

1. **The verifier writes the verdicts, the model writes the reflection.**
   ``outcome``, ``steps`` and ``first_error`` come from ARIA's symbolic engine.
   Only ``note`` is authored by the model. A self-report that nobody checks is
   not a record, it is a story.

2. **Nothing is scored.** An attempt is ``demonstrated`` when its contract
   parsed, every step the engine took was verified, and the run reached a
   decision. There is no partial credit to farm, so there is nothing to game.

3. **Advancement is by demonstrated process, held over distinct problems.**
   Three clean, distinct proofs at a rung -- not three attempts, not a high
   average -- hold that rung. One failure does not remove a grade; it puts the
   rung into *repair*, and advancement waits until the repair is clean twice.

4. **A rung the engine cannot decide is marked ``blocked``, not failed.**
   Congruences currently come back ``partial``: ARIA cannot close them. The
   ledger says so on the card instead of blaming the learner for a missing
   rule. Blocked rungs are excluded from graduation until the engine grows.

The ladder runs from grade 1 to grade 12, then graduation. College is not
"more grade"; it is the point where every gradable rung is held and the system
has nothing left to be taught by the current verifier -- only claims to attack.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..bench.tasks import TASKS, Task
from ..core.ast import Congruence, Equation, Expr, Func, Symbol
from ..core.parse import ParseError, parse, parse_problem
from ..core.poly import degree
from .protocol import SYSTEM_PROMPT, ContractError, MathRequest, parse_contract, to_state

__all__ = [
    "DEFAULT_PATH",
    "CURRICULUM",
    "COLLEGE",
    "Judgement",
    "Entry",
    "SkillRecord",
    "Ledger",
    "classify",
    "judge",
    "report",
    "grade_of",
    "contract_of",
]


#: Where the ledger lives. ``ARIA_LEDGER`` overrides it (tests, CI, second runs).
DEFAULT_PATH = Path(
    os.environ.get("ARIA_LEDGER", str(Path.home() / ".aria" / "ledger.jsonl"))
)

#: Graduation: every gradable rung held.
COLLEGE = 13

#: How much clean, distinct work holds a rung.
CLEAN_TO_HOLD = 3
DISTINCT_TO_HOLD = 3

#: Outcomes that mean "the engine stopped honestly" rather than "the run broke".
_UNDECIDED = ("partial", "open", "rejected")

#: (level, skill, what it means, seed problems)
CURRICULUM: Tuple[Tuple[int, str, str, Tuple[str, ...]], ...] = (
    (1, "arithmetic", "evaluate a number, or check an arithmetic claim",
     ("verify 17 * 23 = 391", "verify 2^10 = 1024", "verify 12 * 12 = 144",
      "verify 99 - 47 = 52")),
    (2, "simplify", "rewrite an expression into canonical form",
     ("simplify 3x + 4x - 2x", "simplify 2(x + 3) - x", "simplify x + x + x")),
    (3, "expand", "distribute products",
     ("expand (x + 2)(x - 3)", "expand (x + 1)^3", "expand (x - 4)^2",
      "expand (2x + 1)(x + 5)")),
    (4, "linear_equation", "solve a degree-1 equation",
     ("solve 9x - 3 = 24", "solve 2x + 7 = 15", "solve 6x + 1 = 19")),
    (5, "factor", "factor a polynomial",
     ("factor x^2 - 9", "factor x^2 + 7x + 12", "factor x^2 - 3x")),
    (6, "quadratic_equation", "solve a degree-2 equation",
     ("solve x^2 - 4 = 0", "solve x^2 + x - 6 = 0", "solve x^2 - 8x + 16 = 0")),
    (7, "derivative_polynomial", "differentiate a polynomial",
     ("differentiate 4x^3 - 2x with respect to x", "differentiate x^2 with respect to x",
      "differentiate 5x^4 + 3x with respect to x")),
    (8, "derivative_transcendental", "differentiate through a function",
     ("differentiate cos(x) with respect to x", "differentiate exp(3x) with respect to x",
      "differentiate log(x) with respect to x")),
    (9, "log_exp", "work with logarithms and exponentials",
     ("verify log(a) + log(b) = log(a*b)", "verify log(a^2) = 2 log(a)",
      "verify log(a/b) = log(a) - log(b)")),
    (10, "trig", "work with trigonometric expressions",
     ("verify 1 - sin(x)^2 = cos(x)^2", "verify tan(x) = sin(x)/cos(x)",
      "verify sin(2x) = 2 sin(x) cos(x)")),
    (11, "identity", "decide whether a statement holds for all values",
     ("verify (a - b)^2 = a^2 - 2ab + b^2", "verify (a + b)^3 = a^3 + 3a^2 b + 3a b^2 + b^3",
      "verify (a + b)(a - b) = a^2 - b^2")),
    (12, "modular", "work with congruences and divisibility",
     ("solve 5x ≡ 3 mod 7", "verify 17 ≡ 2 mod 5")),
)

LEVEL_OF: Dict[str, int] = {name: level for level, name, _, _ in CURRICULUM}
LABEL_OF: Dict[str, str] = {name: label for _, name, label, _ in CURRICULUM}
SEEDS: Dict[str, Tuple[str, ...]] = {name: seeds for _, name, _, seeds in CURRICULUM}
SCHOOL_SKILLS = tuple(name for _, name, _, _ in CURRICULUM)

_TRIG = {"sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan"}
_LOG = {"log", "ln", "exp"}


# ---------------------------------------------------------------------------
# Which rung does a request exercise?
# ---------------------------------------------------------------------------


def _equation_sides(expr: Expr) -> Tuple[Expr, ...]:
    if isinstance(expr, Equation):
        return (expr.lhs, expr.rhs)  # type: ignore[attr-defined]
    return (expr,)


def classify(request: MathRequest) -> str:
    """Name the skill a contract exercises. Never raises; falls back to simplify."""
    goal = request.goal
    try:
        expr = parse(request.expression)
    except (ParseError, ValueError, TypeError):
        return "simplify"

    functions = {node.name for node in expr.walk() if isinstance(node, Func)}
    variable = Symbol(request.variable) if request.variable else None
    sides = _equation_sides(expr)
    symbolic = any(side.free_symbols for side in sides)

    if isinstance(expr, Congruence) or "mod" in request.expression or "≡" in request.expression:
        return "modular"
    if goal == "differentiate":
        return "derivative_transcendental" if functions & (_TRIG | _LOG) else "derivative_polynomial"
    if goal == "verify":
        if functions & _TRIG:
            return "trig"
        if functions & _LOG:
            return "log_exp"
        return "identity" if symbolic else "arithmetic"
    if goal == "factor":
        return "factor"
    if goal == "expand":
        return "expand"
    if goal == "compute":
        return "arithmetic" if not symbolic else "simplify"
    if goal == "analyze":
        return "identity" if symbolic else "arithmetic"
    if goal == "solve":
        if functions & _LOG:
            return "log_exp"
        if functions & _TRIG:
            return "trig"
        highest = 0
        for side in sides:
            try:
                found = degree(side, variable) if variable is not None else None
            except Exception:
                found = None
            highest = max(highest, found or 0)
        if highest == 1:
            return "linear_equation"
        if highest == 2:
            return "quadratic_equation"
        if highest >= 3:
            return "higher_degree_equation"
        return "simplify"
    return "simplify"


# ---------------------------------------------------------------------------
# One judgement: what happened, written by the engine
# ---------------------------------------------------------------------------


@dataclass
class Judgement:
    """The verifier's account of a single attempt. Contains no score."""

    contract_ok: bool
    outcome: str
    skill: str
    goal: str = ""
    expression: str = ""
    variable: Optional[str] = None
    steps: List[dict] = field(default_factory=list)
    first_error: str = ""
    detail: str = ""

    @property
    def verified(self) -> bool:
        """Every step checked out and the run reached a decision."""
        return (
            self.contract_ok
            and self.outcome in ("solved", "refuted")
            and not any(not step["ok"] for step in self.steps)
        )

    @property
    def demonstrated(self) -> bool:
        """A verified decision: nothing the engine produced went unchecked.

        Step count is recorded but never required -- a claim decided with no
        rewrites still had every check the engine made pass.
        """
        return self.verified

    def as_dict(self) -> dict:
        return {
            "contract_ok": self.contract_ok,
            "outcome": self.outcome,
            "skill": self.skill,
            "goal": self.goal,
            "expression": self.expression,
            "variable": self.variable,
            "steps": self.steps,
            "first_error": self.first_error,
            "verified": self.verified,
            "demonstrated": self.demonstrated,
            "detail": self.detail,
        }


def judge(completion: str, engine=None) -> Judgement:
    """Check a completion end to end. This is the only writer of verdicts."""
    from ..reason.engine import SymbolicEngine

    engine = engine or SymbolicEngine()
    try:
        request = parse_contract(completion)
    except ContractError as error:
        return Judgement(contract_ok=False, outcome="invalid", skill="unknown", detail=str(error))

    base = dict(
        contract_ok=True,
        skill=classify(request),
        goal=request.goal,
        expression=request.expression,
        variable=request.variable,
    )

    try:
        state = to_state(request)
        derivation = engine.run(state)
    except (ContractError, ValueError, TypeError) as error:
        return Judgement(outcome="rejected", detail=f"rejected: {error}", **base)

    steps = [
        {
            "rule": step.rule,
            "before": str(step.before),
            "after": str(step.after),
            "justification": step.justification,
            "ok": bool(step.verified),
            "refuted": bool(step.invalid),
        }
        for step in derivation.steps
    ]
    first_error = ""
    for step in derivation.steps:
        if not step.verified:
            first_error = f"{step.rule}: {step.justification} ({step.before} -> {step.after})"
            break

    return Judgement(
        outcome=derivation.status,
        steps=steps,
        first_error=first_error,
        detail=derivation.notes,
        **base,
    )


def contract_of(prompt: str) -> Optional[str]:
    """Turn a natural-language prompt into the contract ARIA's parser would emit."""
    request = request_from_prompt(prompt)
    if request is None:
        return None
    return (
        f"goal: {request.goal}\n"
        + (f"var: {request.variable}\n" if request.variable else "")
        + f"expr: {request.expression}"
    )


def request_from_prompt(prompt: str) -> Optional[MathRequest]:
    """Parse a natural-language problem statement into a math-channel request."""
    try:
        state = parse_problem(prompt)
    except Exception:
        return None
    return MathRequest(
        goal=state.goal.kind.value,
        expression=str(state.primary),
        variable=state.goal.var.name if state.goal.var is not None else None,
    )


# ---------------------------------------------------------------------------
# Entries and the ledger file
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Entry:
    """One line of the ledger."""

    seq: int
    when: str
    question: str
    skill: str
    contract: str
    outcome: str
    verified: bool
    demonstrated: bool
    steps: List[dict] = field(default_factory=list)
    first_error: str = ""
    note: str = ""
    author: str = "verifier"  # who wrote the arithmetic fields

    @property
    def expression(self) -> str:
        for line in self.contract.splitlines():
            if line.startswith("expr:"):
                return line.split(":", 1)[1].strip()
        return ""

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "when": self.when,
            "question": self.question,
            "skill": self.skill,
            "contract": self.contract,
            "outcome": self.outcome,
            "verified": self.verified,
            "demonstrated": self.demonstrated,
            "steps": self.steps,
            "first_error": self.first_error,
            "note": self.note,
            "author": {"verdict": "verifier", "note": "model" if self.note else "none"},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entry":
        return cls(
            seq=int(data.get("seq", 0)),
            when=str(data.get("when", "")),
            question=str(data.get("question", "")),
            skill=str(data.get("skill", "unknown")),
            contract=str(data.get("contract", "")),
            outcome=str(data.get("outcome", "")),
            verified=bool(data.get("verified", False)),
            demonstrated=bool(data.get("demonstrated", False)),
            steps=list(data.get("steps", [])),
            first_error=str(data.get("first_error", "")),
            note=str(data.get("note", "")),
        )


@dataclass
class SkillRecord:
    skill: str
    state: str = "learning"  # learning | held | repair | blocked
    attempts: int = 0
    demonstrated: int = 0
    streak: int = 0
    distinct: int = 0
    last_seq: int = 0
    last_error: str = ""
    outcomes: Tuple[str, ...] = ()

    @property
    def level(self) -> int:
        return LEVEL_OF.get(self.skill, 0)

    @property
    def label(self) -> str:
        return LABEL_OF.get(self.skill, self.skill)


class Ledger:
    """An append-only file of attempts, plus the school's rules over it."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH

    # -- reading -----------------------------------------------------------

    def entries(self) -> List[Entry]:
        if not self.path.exists():
            return []
        out: List[Entry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Entry.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                continue
        return out

    def next_seq(self) -> int:
        entries = self.entries()
        return (entries[-1].seq + 1) if entries else 1

    # -- writing -----------------------------------------------------------

    def record(
        self,
        question: str,
        completion: str,
        note: str = "",
        engine=None,
    ) -> Entry:
        """Judge an attempt and append it. Verdicts come from the engine."""
        verdict = judge(completion, engine=engine)
        entry = Entry(
            seq=self.next_seq(),
            when=_now(),
            question=question,
            skill=verdict.skill,
            contract=completion.strip(),
            outcome=verdict.outcome,
            verified=verdict.verified,
            demonstrated=verdict.demonstrated,
            steps=verdict.steps,
            first_error=verdict.first_error,
            note=note.strip(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
        return entry

    # -- mastery -----------------------------------------------------------

    def skills(self) -> Dict[str, SkillRecord]:
        records: Dict[str, SkillRecord] = {
            name: SkillRecord(skill=name) for name in LEVEL_OF
        }
        seen: Dict[str, set] = {name: set() for name in LEVEL_OF}
        outcomes: Dict[str, List[str]] = {name: [] for name in LEVEL_OF}

        for entry in self.entries():
            record = records.get(entry.skill)
            if record is None:
                continue
            record.attempts += 1
            record.last_seq = entry.seq
            outcomes[entry.skill].append(entry.outcome)
            if entry.demonstrated:
                record.demonstrated += 1
                record.streak += 1
                seen[entry.skill].add(entry.expression)
            else:
                record.streak = 0
                record.last_error = entry.first_error or entry.outcome

        for name, record in records.items():
            record.distinct = len(seen[name])
            record.outcomes = tuple(outcomes[name])
            record.state = _state_of(record)
        return records

    def grade(self) -> int:
        """The rung the system is working at; ``COLLEGE`` once nothing is left."""
        return grade_of(self.skills())

    def card(self) -> dict:
        records = self.skills()
        grade = grade_of(records)
        return {
            "grade": grade,
            "college": grade >= COLLEGE,
            "ledger": str(self.path),
            "entries": len(self.entries()),
            "held": [name for name in SCHOOL_SKILLS if records[name].state == "held"],
            "blocked": [name for name in SCHOOL_SKILLS if records[name].state == "blocked"],
            "skills": {
                name: {
                    "level": record.level,
                    "state": record.state,
                    "attempts": record.attempts,
                    "demonstrated": record.demonstrated,
                    "distinct": record.distinct,
                    "streak": record.streak,
                    "last_error": record.last_error,
                }
                for name, record in records.items()
            },
            "open_repairs": [
                name for name in SCHOOL_SKILLS if records[name].state == "repair"
            ],
        }

    # -- curriculum --------------------------------------------------------

    def frontier(self) -> Optional[str]:
        """The rung to work on: repairs first, then the lowest unheld gradable one."""
        records = self.skills()
        repairs = [
            record
            for record in records.values()
            if record.state == "repair" and record.attempts
        ]
        if repairs:
            return min(repairs, key=lambda r: (r.level, r.last_seq)).skill
        for level, name, _, _ in CURRICULUM:
            if records[name].state not in ("held", "blocked"):
                return name
        return None

    def next_task(self) -> Optional[dict]:
        """An unattempted problem at the frontier: benchmark first, then seeds."""
        skill = self.frontier()
        if skill is None:
            return None
        # A problem is identified by (rung, expression): the same expression
        # asked as `simplify` and as `expand` is two different exercises.
        seen = {(entry.skill, entry.expression) for entry in self.entries()}
        for task in TASKS:
            if _task_skill(task) != skill:
                continue
            if (skill, _task_expression(task)) in seen:
                continue
            return {
                "skill": skill,
                "level": LEVEL_OF[skill],
                "task_id": task.task_id,
                "prompt": task.prompt,
                "source": "bench",
            }
        for prompt in SEEDS.get(skill, ()):
            contract = contract_of(prompt)
            if contract is None:
                continue
            expression = ""
            for line in contract.splitlines():
                if line.startswith("expr:"):
                    expression = line.split(":", 1)[1].strip()
            if (skill, expression) in seen:
                continue
            return {
                "skill": skill,
                "level": LEVEL_OF[skill],
                "task_id": None,
                "prompt": prompt,
                "source": "seed",
            }
        return {"skill": skill, "level": LEVEL_OF[skill], "task_id": None, "prompt": None, "source": "none"}

    # -- training data -----------------------------------------------------

    def to_sft_records(self) -> List[dict]:
        """Q&A pairs where the *process* was proven -- nothing else.

        Failed attempts are never targets. A failure is only included when a
        later attempt on the same expression succeeded: then the corrected pair
        is what gets learned, and the earlier error stays in the ledger as
        history rather than as a lesson.
        """
        good: Dict[str, dict] = {}
        for entry in sorted(self.entries(), key=lambda e: e.seq):
            if entry.verified and entry.contract:
                good[entry.expression] = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": entry.question},
                        {"role": "assistant", "content": entry.contract},
                    ],
                    "skill": entry.skill,
                    "level": LEVEL_OF.get(entry.skill, 0),
                }
        return sorted(good.values(), key=lambda r: (r["level"], r["skill"]))


def _state_of(record: SkillRecord) -> str:
    if record.streak >= CLEAN_TO_HOLD and record.distinct >= DISTINCT_TO_HOLD:
        return "held"
    if record.attempts and record.demonstrated and record.streak == 0:
        return "repair"
    if record.attempts and not record.demonstrated and set(record.outcomes) <= set(_UNDECIDED):
        # Never wrong, never finished: the engine cannot decide this rung yet.
        return "blocked"
    return "learning"


def grade_of(records: Dict[str, SkillRecord]) -> int:
    """Level of the first rung that is neither held nor blocked; 0 if below grade 1."""
    for level, name, _, _ in CURRICULUM:
        record = records.get(name)
        if record is None or record.state not in ("held", "blocked"):
            return level
    return COLLEGE


def _task_skill(task: Task) -> Optional[str]:
    request = request_from_prompt(task.prompt)
    return classify(request) if request is not None else None


def _task_expression(task: Task) -> str:
    request = request_from_prompt(task.prompt)
    return request.expression if request is not None else ""


# ---------------------------------------------------------------------------
# The report card
# ---------------------------------------------------------------------------


def report(ledger: Ledger) -> str:
    """Render the grade card as text."""
    card = ledger.card()
    grade = card["grade"]
    held = len(card["held"])
    dark = len(card["blocked"])
    if grade >= COLLEGE:
        headline = (
            f"graduated (college) -- {held} of {len(SCHOOL_SKILLS)} rungs held"
            + (f", {dark} dark" if dark else "")
        )
    else:
        headline = f"studying grade {grade} -- {held} of {len(SCHOOL_SKILLS)} rungs held"
    lines = [
        f"ARIA report card -- {card['entries']} record(s), {headline}",
        "",
    ]
    for level, skill, label, _ in CURRICULUM:
        data = card["skills"][skill]
        mark = {
            "held": "HELD   ",
            "repair": "REPAIR ",
            "blocked": "DARK   ",
            "learning": "learning",
        }[data["state"]]
        tally = f"{data['demonstrated']}/{data['attempts']} proven"
        if data["distinct"]:
            tally += f", {data['distinct']} distinct"
        if data["streak"]:
            tally += f", streak {data['streak']}"
        line = f"  {level:>2}  {skill:<24} {mark} {tally}"
        if data["state"] == "repair" and data["last_error"]:
            line += f"\n      last break: {data['last_error'][:110]}"
        if data["state"] == "blocked":
            line += "\n      the engine returns `partial` here: this rung cannot be graded yet"
        lines.append(line)

    if card["open_repairs"]:
        lines += ["", f"open repairs: {', '.join(card['open_repairs'])}"]
    if card["blocked"]:
        lines += ["", f"dark rungs (verifier cannot decide): {', '.join(card['blocked'])}"]

    following = ledger.next_task()
    if following is None:
        lines += ["", "next: nothing left to be taught. Write a general claim and try to break it."]
    elif following["prompt"]:
        lines += ["", f"next: [{following['skill']}] {following['prompt']}"]
    else:
        lines += ["", f"next: [{following['skill']}] no unattempted problem left -- author one."]
    return "\n".join(lines)
