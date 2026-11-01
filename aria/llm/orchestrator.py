"""The math channel: natural language at the edges, symbolic work in the middle.

The translator proposes a structured contract. The symbolic runtime checks the
mathematics and keeps the task outcome separate from the process audit. A
narrator may describe that frame, but its prose is not a new verification step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ..core.parse import ParseError, parse_problem
from ..core.state import Goal, GoalKind, SymbolicState
from ..reason.derivation import Derivation
from ..reason.engine import SymbolicEngine
from .client import ChatClient, ClientError, Message, StubClient
from .context import (
    ContextBlock,
    ContextBudgetError,
    ContextBuilder,
    ContextUse,
    ReasoningFrame,
    Scale,
)
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
        if "ARIA FRAME" in user:
            return ""  # narration is skipped in dry-run mode
        marker = "--- USER ---\n"
        if user.startswith("ARIA CROSS-SCALE CONTEXT") and marker in user:
            user = user.split(marker, 1)[1].split("\n\n--- ", 1)[0]
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
    frame: Optional[ReasoningFrame] = None
    explanation: str = ""
    contract_text: str = ""
    repairs: int = 0
    error: Optional[str] = None
    trace: List[str] = field(default_factory=list)
    context_uses: List[ContextUse] = field(default_factory=list)
    context_failures: List[ContextBudgetError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.derivation is not None

    def as_dict(self) -> dict:
        """A machine-readable result with the full context manifest."""
        return {
            "user_text": self.user_text,
            "contract": self.contract_text,
            "status": self.derivation.status if self.derivation else None,
            "solutions": [str(value) for value in self.derivation.solutions]
            if self.derivation
            else [],
            "explanation": self.explanation,
            "repairs": self.repairs,
            "error": self.error,
            "reasoning_frame": self.frame.as_dict() if self.frame else None,
            "contexts": [use.as_dict() for use in self.context_uses],
            "context_failures": [failure.as_dict() for failure in self.context_failures],
            "trace": list(self.trace),
        }

    def to_text(self) -> str:
        lines = [f"Problem: {self.user_text}", ""]
        if self.request is not None:
            lines.append("Accepted translator contract (validated before symbolic execution):")
            lines.extend(f"  {line}" for line in self.contract_text.splitlines())
            lines.append("")
        if self.derivation is not None:
            lines.append("Symbolic engine trace:")
            lines.append(self.derivation.to_text())
            lines.append("")
        if self.frame is not None:
            assert self.frame.audit_state is not None
            lines.append(
                "Cross-scale state: "
                f"outcome={self.frame.task_outcome}; "
                f"audit={self.frame.audit_state.value} ({self.frame.audit_state.ket_label}); "
                f"open obligations={len(self.frame.open_obligations)}"
            )
        if self.explanation:
            lines.append("Explanation (narrating the ARIA frame):")
            lines.append(self.explanation)
        if self.context_uses or self.context_failures:
            lines.append("Context manifest:")
            for use in self.context_uses:
                summary = use.pack.summary()
                omitted = summary["omitted_ids"]
                line = (
                    f"  {use.phase}: {summary['completeness']} "
                    f"({summary['used_units']} {summary['unit']}"
                )
                if summary["budget_units"] is not None:
                    line += f" / {summary['budget_units']} {summary['unit']}"
                line += ")"
                if omitted:
                    line += f"; omitted whole blocks: {', '.join(omitted)} (recoverable)"
                lines.append(line)
            for failure in self.context_failures:
                lines.append(f"  {failure}")
        if self.repairs:
            lines.append(f"\nContract repairs: {self.repairs}")
        if self.error:
            lines.append(f"\nError: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The channel
# ---------------------------------------------------------------------------


class MathChannel:
    """Couples a supplied language model endpoint to ARIA's symbolic runtime.

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
        Whether to ask the model for prose about the ARIA frame; narration does not
        change the symbolic audit state.
    context_budget:
        Optional input-context limit measured by ``context_measure``. The
        default measure is character count; inject the model's tokenizer for a
        token limit. Required context is never clipped.
    context_measure:
        Callable returning a non-negative integer size for a text string.
    """

    def __init__(
        self,
        client: ChatClient,
        engine: Optional[SymbolicEngine] = None,
        grammar_mode: Optional[str] = "loose",
        max_repairs: int = 2,
        narrate: bool = True,
        max_tokens: int = 256,
        context_budget: Optional[int] = None,
        context_measure: Optional[Callable[[str], int]] = None,
        context_unit: Optional[str] = None,
    ) -> None:
        self.client = client
        self.engine = engine or SymbolicEngine()
        self.grammar = build_grammar(grammar_mode) if grammar_mode else None
        self.max_repairs = max_repairs
        self.narrate = narrate
        self.max_tokens = max_tokens
        measure = context_measure or len
        unit = context_unit or ("chars" if context_measure is None else "units")
        self.context_builder = ContextBuilder(
            budget_units=context_budget,
            measure=measure,
            unit_label=unit,
        )

    # -- main entry point --------------------------------------------------

    def ask(self, user_text: str) -> ChannelResult:
        result = ChannelResult(user_text=user_text)

        messages = translation_messages(user_text)
        request: Optional[MathRequest] = None

        for attempt in range(self.max_repairs + 1):
            system_messages = [message for message in messages if message.role == "system"]
            conversation = [message for message in messages if message.role != "system"]
            transcript = "\n\n".join(
                f"--- {message.role.upper()} ---\n{message.content}" for message in conversation
            )
            reserved_system = "\n\n".join(message.content for message in system_messages)
            block = ContextBlock(
                id=f"translation-attempt-{attempt + 1}",
                scale=Scale.MACRO,
                text=transcript,
                source="aria.llm.protocol",
                required=True,
                priority=1000,
            )
            try:
                pack = self.context_builder.build(
                    (block,),
                    reserved_text=reserved_system,
                    phase=f"translation attempt {attempt + 1}",
                )
            except ContextBudgetError as error:
                result.context_failures.append(error)
                result.error = f"translation held before model call: {error}"
                return result
            result.context_uses.append(ContextUse(pack.phase, pack))

            # Send exactly the context that was measured, not the unbounded source
            # messages that happened to be used to build it. Keep trusted system
            # instructions in their actual role; the packed transcript is input data.
            packed_messages = [*system_messages, Message("user", pack.text)]
            try:
                raw = self.client.complete(
                    packed_messages,
                    grammar=self.grammar,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            except ClientError as error:
                result.error = f"inference failed: {error}"
                return result

            # Keep the entire exchange. If it must be shortened for display, the
            # caller must request that explicitly; the stored trace is lossless.
            result.trace.append(f"attempt {attempt + 1}: {raw.strip()}")
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
        result.frame = ReasoningFrame.from_derivation(derivation)

        if self.narrate:
            result.explanation = self._narrate(user_text, result.frame, result)

        return result

    # -- narration ---------------------------------------------------------

    def _narrate(
        self,
        user_text: str,
        frame: ReasoningFrame,
        result: ChannelResult,
    ) -> str:
        template = narration_messages(user_text, "")
        reserved = "\n\n".join(
            f"--- {message.role.upper()} ---\n{message.content}" for message in template
        )
        try:
            pack = self.context_builder.build(
                frame.context_blocks(),
                reserved_text=reserved,
                phase="narration",
            )
        except ContextBudgetError as error:
            result.context_failures.append(error)
            return f"(narration held: {error}; the symbolic derivation is retained)"
        result.context_uses.append(ContextUse(pack.phase, pack))

        try:
            text = self.client.complete(
                narration_messages(user_text, pack.text),
                grammar=None,
                temperature=0.2,
                max_tokens=512,
            )
        except ClientError as error:
            return f"(narration unavailable: {error})"
        text = text.strip()
        return text or "(no explanation produced)"
