"""The math channel: text in, verified mathematics, text out.

    user text ──▶ model (grammar-constrained) ──▶ contract ──▶ ARIA parser
              ──▶ rule engine + verifier ──▶ verified trace ──▶ model ──▶ prose

The model appears twice, and in neither place is it allowed to state a
mathematical fact.  On the way in it is a translator; on the way out it is a
narrator that may only describe a trace the verifier already approved.  All
mathematics happens in between, in exact symbolic form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..core.parse import ParseError, parse_problem
from ..core.state import Goal, GoalKind, SymbolicState
from ..reason.derivation import Derivation
from ..reason.engine import SymbolicEngine
from .client import ChatClient, ClientError, Message, StubClient
from .grammar import build_grammar
from .protocol import (
    DETERMINISTIC_TRANSLATION,
    ContractError,
    MathRequest,
    narration_messages,
    parse_contract,
    to_state,
    translation_messages,
)

__all__ = ["ChannelResult", "MathChannel", "deterministic_translator", "dry_run_client"]


# ---------------------------------------------------------------------------
# A translator that needs no weights (used by --dry-run and tests)
# ---------------------------------------------------------------------------


def deterministic_translator(user_text: str) -> str:
    """Translate using ARIA's own instruction grammar.

    This is *not* a model: it shows the exact contract the real model must
    produce, so the pipeline can be demonstrated and tested offline.
    """
    state = parse_problem(user_text)
    kind = state.goal.kind
    goal = {
        GoalKind.SOLVE: "solve",
        GoalKind.SIMPLIFY: "simplify",
        GoalKind.COMPUTE: "compute",
        GoalKind.FACTOR: "factor",
        GoalKind.EXPAND: "expand",
        GoalKind.VERIFY: "verify",
        GoalKind.DIFFERENTIATE: "differentiate",
    }.get(kind, "analyze")
    lines = [f"goal: {goal}"]
    if state.goal.var is not None:
        lines.append(f"var: {state.goal.var.name}")
    lines.append(f"expr: {state.primary}")
    return "\n".join(lines)


def dry_run_client() -> StubClient:
    """Build a stub client that behaves like a perfectly obedient translator."""

    def respond(messages: Sequence[Message]) -> str:
        user = messages[-1].content
        if "VERIFIED TRACE" in user:
            return ""  # narration is skipped in dry-run mode
        return deterministic_translator(user)

    return StubClient(respond)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class ChannelResult:
    user_text: str
    request: Optional[MathRequest] = None
    derivation: Optional[Derivation] = None
    explanation: str = ""
    contract_text: str = ""
    repairs: int = 0
    error: Optional[str] = None
    trace: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.derivation is not None

    def to_text(self) -> str:
        lines = [f"Problem: {self.user_text}", ""]
        if self.request is not None:
            lines.append("Model-produced contract (the only text the model may emit):")
            lines.extend(f"  {line}" for line in self.contract_text.splitlines())
            lines.append("")
        if self.derivation is not None:
            lines.append("Verified derivation:")
            lines.append(self.derivation.to_text())
            lines.append("")
        if self.explanation:
            lines.append("Explanation (narrating the verified trace only):")
            lines.append(self.explanation)
        if self.repairs:
            lines.append(f"\nContract repairs: {self.repairs}")
        if self.error:
            lines.append(f"\nError: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The channel
# ---------------------------------------------------------------------------


class MathChannel:
    """Couples a local language model to ARIA's verified symbolic runtime.

    Parameters
    ----------
    client:
        Any :class:`~aria.llm.client.ChatClient` (llama.cpp, LM Studio, Ollama,
        a stub).
    engine:
        Symbolic engine used for the actual mathematics.
    grammar_mode:
        ``"loose"``, ``"strict"`` or ``None`` to disable constrained decoding.
    max_repairs:
        How many times to re-prompt the model with its own contract error.
    narrate:
        Whether to ask the model for a prose explanation of the verified trace.
    """

    def __init__(
        self,
        client: ChatClient,
        engine: Optional[SymbolicEngine] = None,
        grammar_mode: Optional[str] = "loose",
        max_repairs: int = 2,
        narrate: bool = True,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.engine = engine or SymbolicEngine()
        self.grammar = build_grammar(grammar_mode) if grammar_mode else None
        self.max_repairs = max_repairs
        self.narrate = narrate
        self.max_tokens = max_tokens

    # -- main entry point --------------------------------------------------

    def ask(self, user_text: str) -> ChannelResult:
        result = ChannelResult(user_text=user_text)

        messages = translation_messages(user_text)
        request: Optional[MathRequest] = None

        for attempt in range(self.max_repairs + 1):
            try:
                raw = self.client.complete(
                    messages, grammar=self.grammar, temperature=0.0, max_tokens=self.max_tokens
                )
            except ClientError as error:
                result.error = f"inference failed: {error}"
                return result

            result.trace.append(f"attempt {attempt + 1}: {raw.strip()[:160]}")
            try:
                request = parse_contract(raw)
                result.contract_text = request.as_text()
                result.repairs = attempt
                break
            except ContractError as error:
                if attempt >= self.max_repairs:
                    result.error = f"contract rejected after {attempt + 1} attempts: {error}"
                    result.contract_text = raw.strip()
                    return result
                messages = messages + [
                    Message("assistant", raw),
                    Message(
                        "user",
                        f"That output violated the contract: {error}. "
                        "Reply with the three-line contract only, no other text.",
                    ),
                ]

        if request is None:  # pragma: no cover - defensive
            result.error = "no contract produced"
            return result

        result.request = request
        try:
            state = to_state(request)
            derivation = self.engine.run(state)
        except (ContractError, ValueError, TypeError, ParseError) as error:
            result.error = f"symbolic layer rejected the request: {error}"
            return result

        result.derivation = derivation

        if self.narrate:
            result.explanation = self._narrate(user_text, derivation)

        return result

    # -- narration ---------------------------------------------------------

    def _narrate(self, user_text: str, derivation: Derivation) -> str:
        trace = derivation.to_text()
        try:
            text = self.client.complete(
                narration_messages(user_text, trace), grammar=None, temperature=0.2, max_tokens=512
            )
        except ClientError as error:
            return f"(narration unavailable: {error})"
        text = text.strip()
        return text or "(no explanation produced)"
