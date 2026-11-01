"""Training-data helpers and a retained legacy scalar diagnostic.

The accepted weight-update path is staged SFT from the ledger's demonstrated
processes (``aria ledger sft``), after a school rung is held.  The benchmark
SFT builder remains useful for format smoke tests.  RL prompt and scalar score
helpers are retained only for historical comparison; they are not the school,
not the accepted training objective, and not a correctness certificate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from ..bench.tasks import TASKS, Task
from ..core.parse import parse_problem
from ..core.state import GoalKind
from ..reason.engine import SymbolicEngine
from .protocol import SYSTEM_PROMPT, ContractError, MathRequest, parse_contract, to_state

__all__ = [
    "SftExample",
    "RewardBreakdown",
    "build_sft_examples",
    "to_chat_jsonl",
    "build_rl_prompts",
    "score_completion",
    "reward",
    "GOAL_TEMPLATES",
    "reward_source",
]


# ---------------------------------------------------------------------------
# Natural-language templates per goal
# ---------------------------------------------------------------------------

GOAL_TEMPLATES = {
    "solve": (
        "solve for {var} in {expr}",
        "find {var} such that {expr}",
        "what is {var} if {expr}?",
        "{expr} — solve for {var}",
    ),
    "simplify": (
        "simplify {expr}",
        "reduce {expr}",
        "what is {expr}?",
        "write {expr} in simplest form",
    ),
    "factor": (
        "factor {expr}",
        "factorise {expr}",
        "factor {expr} completely",
    ),
    "expand": (
        "expand {expr}",
        "multiply out {expr}",
    ),
    "verify": (
        "verify {expr}",
        "is it true that {expr}?",
        "check whether {expr}",
    ),
    "differentiate": (
        "differentiate {expr} with respect to {var}",
        "find the derivative of {expr} with respect to {var}",
        "d/d{var} of {expr}",
    ),
    "compute": ("compute {expr}", "evaluate {expr}"),
    "analyze": ("look at {expr}",),
}

_GOAL_FROM_KIND = {
    GoalKind.SOLVE: "solve",
    GoalKind.SIMPLIFY: "simplify",
    GoalKind.FACTOR: "factor",
    GoalKind.EXPAND: "expand",
    GoalKind.VERIFY: "verify",
    GoalKind.DIFFERENTIATE: "differentiate",
    GoalKind.COMPUTE: "compute",
    GoalKind.ANALYZE: "analyze",
}


@dataclass(frozen=True)
class SftExample:
    """One supervised translation example."""

    user: str
    contract: str
    source: str = "benchmark"

    def as_messages(self, system: str = SYSTEM_PROMPT) -> List[dict]:
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.contract},
        ]


# ---------------------------------------------------------------------------
# Building data
# ---------------------------------------------------------------------------


def _request_from_prompt(prompt: str) -> Optional[MathRequest]:
    try:
        state = parse_problem(prompt)
    except Exception:
        return None
    goal = _GOAL_FROM_KIND.get(state.goal.kind)
    if goal is None:
        return None
    return MathRequest(
        goal=goal,
        expression=str(state.primary),
        variable=state.goal.var.name if state.goal.var is not None else None,
    )


def _paraphrases(request: MathRequest, limit: int) -> List[str]:
    templates = GOAL_TEMPLATES.get(request.goal, ())
    rendered: List[str] = []
    for template in templates:
        if "{var}" in template and not request.variable:
            continue
        text = template.format(expr=request.expression, var=request.variable or "x")
        if text not in rendered:
            rendered.append(text)
        if len(rendered) >= limit:
            break
    return rendered


def build_sft_examples(
    tasks: Sequence[Task] = TASKS,
    extra_prompts: Sequence[str] = (),
    paraphrases_per_task: int = 2,
    include_original: bool = True,
) -> List[SftExample]:
    """Build ``(question → contract)`` pairs from the benchmark and extras.

    The contract is produced by ARIA's own parser, so every training target is
    guaranteed to parse and to be solvable by the engine. The model learns the
    *translation*, never an answer.
    """
    examples: List[SftExample] = []
    seen: set[tuple[str, str]] = set()

    for task in tasks:
        if task.suite == "units":  # dimensional checks are not part of this contract
            continue
        request = _request_from_prompt(task.prompt)
        if request is None:
            continue
        contract = request.as_text()
        candidates = ([task.prompt] if include_original else []) + _paraphrases(
            request, paraphrases_per_task
        )
        for user_text in candidates:
            key = (user_text.strip().lower(), contract)
            if key in seen:
                continue
            seen.add(key)
            examples.append(SftExample(user=user_text, contract=contract, source="benchmark"))

    for prompt in extra_prompts:
        request = _request_from_prompt(prompt)
        if request is None:
            continue
        contract = request.as_text()
        key = (prompt.strip().lower(), contract)
        if key in seen:
            continue
        seen.add(key)
        examples.append(SftExample(user=prompt, contract=contract, source="synthetic"))

    return examples


def to_chat_jsonl(examples: Iterable[SftExample], system: str = SYSTEM_PROMPT) -> str:
    """Serialise examples as JSONL of ``{"messages": [...]}`` (TRL/unsloth format)."""
    return "\n".join(
        json.dumps({"messages": example.as_messages(system)}, ensure_ascii=False)
        for example in examples
    ) + "\n"


def build_rl_prompts(tasks: Sequence[Task] = TASKS) -> List[dict]:
    """Legacy prompt-only records for scalar-reward experiments."""
    records: List[dict] = []
    for task in tasks:
        if task.suite == "units":
            continue
        request = _request_from_prompt(task.prompt)
        if request is None:
            continue
        records.append(
            {
                "prompt": task.prompt,
                "expected_goal": request.goal,
                "expected_expression": request.expression,
                "reward": "aria.llm.dataset:reward",
            }
        )
    return records


# ---------------------------------------------------------------------------
# Legacy scalar diagnostic (not used by the staged school)
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Component view of a completion's score (0.0–1.0)."""

    total: float
    contract_ok: bool
    status: str = "unparsed"
    trace_accuracy: float = 0.0
    steps: int = 0
    invalid_steps: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "contract_ok": self.contract_ok,
            "status": self.status,
            "trace_accuracy": round(self.trace_accuracy, 4),
            "steps": self.steps,
            "invalid_steps": self.invalid_steps,
            "detail": self.detail,
        }


def score_completion(completion: str, engine: Optional[SymbolicEngine] = None) -> RewardBreakdown:
    """Return a legacy scalar for reproducing old reward experiments.

    This diagnostic is deliberately not the ledger, not a training target for
    ARIA's accepted curriculum, and not proof of correctness.  It is retained so
    the historical failure can be measured: a broken rational solve can score
    0.90 because the scalar gives partial credit to two passing steps.
    Use :class:`aria.llm.ledger.Ledger` and its staged school instead.
    """
    engine = engine or SymbolicEngine()
    try:
        request = parse_contract(completion)
    except ContractError as error:
        return RewardBreakdown(total=0.0, contract_ok=False, detail=str(error))

    try:
        state = to_state(request)
        derivation = engine.run(state)
    except (ContractError, ValueError, TypeError) as error:
        return RewardBreakdown(total=0.05, contract_ok=True, detail=f"rejected: {error}")

    accuracy = derivation.trace_accuracy
    status = derivation.status
    steps = len(derivation.steps)
    invalid = len(derivation.invalid_steps)

    if status == "solved":
        total = 0.70 + 0.30 * accuracy
        if steps and derivation.valid:
            total = 1.0
    elif status == "refuted":
        total = 0.50
    elif status == "partial":
        total = 0.30 + 0.40 * accuracy
    elif status == "open":
        total = 0.20 + 0.40 * accuracy
    else:
        total = 0.10

    if invalid:
        total = min(total, 0.25)  # a refuted step is worse than an honest unknown

    return RewardBreakdown(
        total=max(0.0, min(1.0, total)),
        contract_ok=True,
        status=status,
        trace_accuracy=accuracy,
        steps=steps,
        invalid_steps=invalid,
        detail=derivation.notes,
    )


def reward(completion: str, engine: Optional[SymbolicEngine] = None) -> float:
    """Legacy scalar diagnostic; do not use for ledger advancement or default training."""
    return score_completion(completion, engine).total


def reward_source() -> str:
    """Legacy import path for explicitly opted-in scalar-reward experiments."""
    return "aria.llm.dataset:reward"
