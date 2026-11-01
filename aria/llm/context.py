"""Cross-scale reasoning frames and fail-closed context packing.

ARIA keeps three views of one checked result:

* macro: the task, outcome, result, audit state, and open obligations;
* meso: a branch/checkpoint and its compiled audit state;
* micro: individual claims, rewrite steps, and verifier evidence.

Context packing removes only whole optional blocks.  The original blocks remain
attached to :class:`ContextPack`, so a larger budget can rebuild the context.
Mandatory context is never clipped: if it cannot fit, the caller must hold the
request rather than send an incomplete prompt.

The bra-ket labels are display aliases only; they do not denote amplitudes,
probabilities, superposition, or a physical state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..reason.derivation import Derivation, REFUTED, SOLVED

__all__ = [
    "AuditState",
    "BranchMode",
    "CandidateRoute",
    "ContextBlock",
    "ContextBudgetError",
    "ContextBuilder",
    "ContextCompleteness",
    "ContextPack",
    "ContextUse",
    "HeldState",
    "MicroEvidence",
    "MesoBranch",
    "ReasoningFrame",
    "Scale",
    "StateEvent",
    "compile_branch_states",
    "least_resistance_route",
]


class Scale(str, Enum):
    """A view of reasoning, ordered from task-level to atomic detail."""

    MACRO = "macro"
    MESO = "meso"
    MICRO = "micro"


class AuditState(str, Enum):
    """Three-valued state for evidence or the process audit.

    ``PASS`` means the scoped check passed, ``FAIL`` means it was refuted, and
    ``HELD`` means the available evidence did not decide it.  This is distinct
    from ``ReasoningFrame.task_outcome``: a mathematically false claim can have
    a ``refuted`` task outcome and a ``PASS`` audit when the refutation itself
    was correctly checked.
    """

    PASS = "pass"
    FAIL = "fail"
    HELD = "held"

    @property
    def ket_label(self) -> str:
        """Compact display alias; not a quantum-mechanical quantity."""
        return {
            AuditState.PASS: "|1⟩",
            AuditState.FAIL: "|0⟩",
            AuditState.HELD: "|?⟩",
        }[self]

    @classmethod
    def from_verifier_status(cls, status: str) -> "AuditState":
        """Map a process-audit status, not a mathematical task outcome.

        ``refuted`` is intentionally rejected: it says a claim is false, not
        whether the reasoning process that refuted it was checked correctly.
        The caller must provide that audit result separately.
        """
        normalized = status.strip().lower()
        mapping = {
            "proven": cls.PASS,
            "pass": cls.PASS,
            "verified": cls.PASS,
            "fail": cls.FAIL,
            "invalid": cls.FAIL,
            "unknown": cls.HELD,
            "partial": cls.HELD,
            "open": cls.HELD,
            "held": cls.HELD,
        }
        try:
            return mapping[normalized]
        except KeyError as error:
            if normalized == "refuted":
                raise ValueError(
                    "'refuted' is a task outcome; supply the process audit separately"
                ) from error
            raise ValueError(f"unknown verifier audit status {status!r}") from error


class BranchMode(str, Enum):
    """How independent meso branches compile into one decision state."""

    ALL = "all"
    ANY = "any"


def compile_branch_states(
    states: Iterable[AuditState], mode: BranchMode | str = BranchMode.ALL
) -> AuditState:
    """Compile branch states without converting uncertainty into success.

    ``ALL`` is conjunctive: any failed required branch fails the compilation;
    it passes only when every included branch passes; otherwise it is held.
    ``ANY`` is alternative: one verified path is enough to pass; it fails only
    when every alternative fails; unresolved alternatives keep it held.
    An empty set is held in both modes.
    """
    selected_mode = BranchMode(mode)
    values = tuple(states)
    if not values:
        return AuditState.HELD

    if selected_mode is BranchMode.ANY:
        if AuditState.PASS in values:
            return AuditState.PASS
        if AuditState.HELD in values:
            return AuditState.HELD
        return AuditState.FAIL

    if AuditState.FAIL in values:
        return AuditState.FAIL
    if all(value is AuditState.PASS for value in values):
        return AuditState.PASS
    return AuditState.HELD


class StateEvent(str, Enum):
    """Events that may or may not change a held audit register."""

    VERIFIER_PASSED = "verifier_passed"
    VERIFIER_FAILED = "verifier_failed"
    VERIFIER_HELD = "verifier_held"
    CONTEXT_REDUCED = "context_reduced"
    CONTEXT_RECOVERED = "context_recovered"
    NARRATION = "narration"


@dataclass(frozen=True)
class HeldState:
    """Immutable decision register; only a verifier event can resolve it.

    Context truncation/recovery and prose narration are observational events:
    they return the same object and cannot collapse a held decision into
    pass/fail.  A new task should receive a new register rather than mutate the
    old claim's state.
    """

    state: AuditState = AuditState.HELD
    reason: str = "awaiting a verifier decision"
    revision: int = 0
    last_event: str = "initial"
    evidence_ref: str = ""

    def on_event(
        self,
        event: StateEvent | str,
        *,
        evidence_ref: str = "",
        reason: str = "",
    ) -> "HeldState":
        selected_event = StateEvent(event)
        if selected_event in {
            StateEvent.CONTEXT_REDUCED,
            StateEvent.CONTEXT_RECOVERED,
            StateEvent.NARRATION,
        }:
            return self

        if not evidence_ref.strip():
            raise ValueError("a verifier transition requires an evidence reference")
        target = {
            StateEvent.VERIFIER_PASSED: AuditState.PASS,
            StateEvent.VERIFIER_FAILED: AuditState.FAIL,
            StateEvent.VERIFIER_HELD: AuditState.HELD,
        }[selected_event]
        return HeldState(
            state=target,
            reason=reason.strip() or f"state set by {selected_event.value}",
            revision=self.revision + 1,
            last_event=selected_event.value,
            evidence_ref=evidence_ref.strip(),
        )

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "revision": self.revision,
            "last_event": self.last_event,
            "evidence_ref": self.evidence_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HeldState":
        return cls(
            state=AuditState(data.get("state", AuditState.HELD.value)),
            reason=str(data.get("reason", "awaiting a verifier decision")),
            revision=int(data.get("revision", 0)),
            last_event=str(data.get("last_event", "initial")),
            evidence_ref=str(data.get("evidence_ref", "")),
        )


@dataclass(frozen=True)
class MicroEvidence:
    """An atomic claim, rewrite, or verifier record."""

    id: str
    statement: str
    audit_state: AuditState
    source: str
    detail: str = ""
    required: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("micro evidence needs a stable id")
        if not self.statement.strip():
            raise ValueError("micro evidence needs a statement")
        if not self.source.strip():
            raise ValueError("micro evidence needs provenance")
        object.__setattr__(self, "audit_state", AuditState(self.audit_state))

    def context_text(self) -> str:
        lines = [f"Statement: {self.statement}"]
        if self.detail:
            lines.append(f"Detail: {self.detail}")
        lines.append(f"Provenance: {self.source}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "audit_state": self.audit_state.value,
            "source": self.source,
            "detail": self.detail,
            "required": self.required,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class MesoBranch:
    """A branch/checkpoint containing irreducible micro-evidence."""

    id: str
    title: str
    evidence: Tuple[MicroEvidence, ...] = ()
    required: bool = True
    priority: int = 0
    source: str = "aria.reason.engine"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a meso branch needs a stable id")
        if not self.title.strip():
            raise ValueError("a meso branch needs a title")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.source.strip():
            raise ValueError("a meso branch needs provenance")

    @property
    def audit_state(self) -> AuditState:
        return compile_branch_states(item.audit_state for item in self.evidence)

    def summary(self) -> str:
        counts = {state: 0 for state in AuditState}
        for item in self.evidence:
            counts[item.audit_state] += 1
        lines = [
            f"Branch: {self.title}",
            f"Branch audit state: {self.audit_state.value} ({self.audit_state.ket_label})",
            (
                f"Micro records: {len(self.evidence)}; "
                f"pass={counts[AuditState.PASS]}, fail={counts[AuditState.FAIL]}, "
                f"held={counts[AuditState.HELD]}"
            ),
        ]
        unresolved = [item.id for item in self.evidence if item.audit_state is AuditState.HELD]
        invalid = [item.id for item in self.evidence if item.audit_state is AuditState.FAIL]
        if unresolved:
            lines.append("Open obligations: " + ", ".join(unresolved))
        if invalid:
            lines.append("Rejected evidence: " + ", ".join(invalid))
        if not self.evidence:
            lines.append("No atomic evidence was recorded for this branch.")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "audit_state": self.audit_state.value,
            "required": self.required,
            "priority": self.priority,
            "source": self.source,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class ReasoningFrame:
    """A macro result compiled from one or more meso branches."""

    goal: str
    task_outcome: str
    result: str
    branches: Tuple[MesoBranch, ...] = ()
    branch_mode: BranchMode = BranchMode.ALL
    audit_state: Optional[AuditState] = None
    assumptions: Tuple[str, ...] = ()
    source: str = "aria.reason.engine"
    held_state: Optional[HeldState] = None

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("a reasoning frame needs a goal")
        if not self.task_outcome.strip():
            raise ValueError("a reasoning frame needs a task outcome")
        object.__setattr__(self, "branches", tuple(self.branches))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "branch_mode", BranchMode(self.branch_mode))
        if not self.source.strip():
            raise ValueError("a reasoning frame needs provenance")

        compiled = self.audit_state
        if compiled is None:
            branch_states = [branch.audit_state for branch in self.branches]
            if self.branch_mode is BranchMode.ALL:
                branch_states = [
                    branch.audit_state for branch in self.branches if branch.required
                ]
            compiled = compile_branch_states(branch_states, self.branch_mode)
            object.__setattr__(self, "audit_state", compiled)
        else:
            compiled = AuditState(compiled)
            object.__setattr__(self, "audit_state", compiled)

        if self.held_state is None:
            event = {
                AuditState.PASS: StateEvent.VERIFIER_PASSED,
                AuditState.FAIL: StateEvent.VERIFIER_FAILED,
                AuditState.HELD: StateEvent.VERIFIER_HELD,
            }[compiled]
            register = HeldState().on_event(
                event,
                evidence_ref=self.source,
                reason=f"macro audit is {compiled.value}",
            )
            object.__setattr__(self, "held_state", register)

    @classmethod
    def from_derivation(cls, derivation: Derivation) -> "ReasoningFrame":
        """Project ARIA's verified derivation into macro/meso/micro views."""
        if derivation.invalid_steps:
            audit_state = AuditState.FAIL
        elif derivation.unverified_steps:
            audit_state = AuditState.HELD
        elif derivation.status in (SOLVED, REFUTED) and derivation.valid:
            # A correct refutation is a successful audit of a false claim.
            audit_state = AuditState.PASS
        else:
            audit_state = AuditState.HELD

        if derivation.solutions:
            variable = derivation.goal.var.name if derivation.goal.var else "x"
            result_text = "; ".join(f"{variable} = {solution}" for solution in derivation.solutions)
        elif derivation.result is not None:
            result_text = str(derivation.result)
        else:
            result_text = "not established"

        evidence: List[MicroEvidence] = [
            MicroEvidence(
                id="outcome",
                statement=f"Engine outcome: {derivation.status}; result: {result_text}",
                audit_state=audit_state,
                source="aria.reason.engine + aria.verify.verifier",
                detail=derivation.notes,
                priority=100,
            )
        ]
        for index, step in enumerate(derivation.steps, start=1):
            if step.invalid:
                state = AuditState.FAIL
            elif step.verified:
                state = AuditState.PASS
            else:
                state = AuditState.HELD
            verdict_source = (
                f"aria.verify.verifier:{step.verdict.method}"
                if step.verdict is not None
                else "aria.verify.verifier:missing-verdict"
            )
            evidence.append(
                MicroEvidence(
                    id=f"step-{index:03d}",
                    statement=f"{step.before} → {step.after} ({step.rule})",
                    audit_state=state,
                    source=verdict_source,
                    detail=step.justification,
                    required=False,
                    priority=100 if state is not AuditState.PASS else 0,
                )
            )

        branch = MesoBranch(
            id="selected",
            title="Selected derivation",
            evidence=tuple(evidence),
            required=True,
            source="aria.reason.engine",
        )
        return cls(
            goal=str(derivation.goal),
            task_outcome=derivation.status,
            result=result_text,
            branches=(branch,),
            branch_mode=BranchMode.ALL,
            audit_state=audit_state,
            assumptions=(),
            source="aria.reason.engine",
        )

    @property
    def open_obligations(self) -> Tuple[str, ...]:
        return tuple(
            f"{branch.id}/{item.id}"
            for branch in self.branches
            for item in branch.evidence
            if item.audit_state is AuditState.HELD
        )

    @property
    def rejected_evidence(self) -> Tuple[str, ...]:
        return tuple(
            f"{branch.id}/{item.id}"
            for branch in self.branches
            for item in branch.evidence
            if item.audit_state is AuditState.FAIL
        )

    def context_blocks(self) -> Tuple["ContextBlock", ...]:
        """Build required macro/meso summaries plus optional atomic detail."""
        assert self.audit_state is not None
        assert self.held_state is not None
        interpretation = (
            "Task outcome records the engine's mathematical conclusion; audit state records "
            "whether the reasoning process was checked. Do not conflate them."
        )
        lines = [
            f"Goal: {self.goal}",
            f"Task outcome: {self.task_outcome}",
            f"Result: {self.result or 'not established'}",
            f"Audit state: {self.audit_state.value} ({self.audit_state.ket_label})",
            f"Branch mode: {self.branch_mode.value}; branch count: {len(self.branches)}",
            "Open obligations: " + (", ".join(self.open_obligations) or "none recorded"),
            "Rejected evidence: " + (", ".join(self.rejected_evidence) or "none recorded"),
            "Assumptions: " + ("; ".join(self.assumptions) or "none recorded"),
            (
                f"Held register: {self.held_state.state.value}, revision "
                f"{self.held_state.revision}, event {self.held_state.last_event}."
            ),
            interpretation,
        ]
        blocks: List[ContextBlock] = [
            ContextBlock(
                id="task",
                scale=Scale.MACRO,
                text="\n".join(lines),
                source=self.source,
                audit_state=self.audit_state,
                required=True,
                priority=1000,
            )
        ]
        for branch in self.branches:
            blocks.append(
                ContextBlock(
                    id=branch.id,
                    scale=Scale.MESO,
                    text=branch.summary(),
                    source=branch.source,
                    audit_state=branch.audit_state,
                    required=branch.required,
                    priority=branch.priority,
                )
            )
            for item in branch.evidence:
                blocks.append(
                    ContextBlock(
                        id=f"{branch.id}/{item.id}",
                        scale=Scale.MICRO,
                        text=item.context_text(),
                        source=item.source,
                        audit_state=item.audit_state,
                        required=item.required,
                        priority=item.priority,
                    )
                )
        return tuple(blocks)

    def as_dict(self) -> dict:
        assert self.audit_state is not None
        assert self.held_state is not None
        return {
            "goal": self.goal,
            "task_outcome": self.task_outcome,
            "result": self.result,
            "audit_state": self.audit_state.value,
            "branch_mode": self.branch_mode.value,
            "assumptions": list(self.assumptions),
            "open_obligations": list(self.open_obligations),
            "rejected_evidence": list(self.rejected_evidence),
            "source": self.source,
            "held_state": self.held_state.as_dict(),
            "branches": [branch.as_dict() for branch in self.branches],
        }


@dataclass(frozen=True)
class CandidateRoute:
    """A candidate path and the measured resistance to prefer among valid paths."""

    id: str
    resistance: float
    audit_state: AuditState
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a candidate route needs a stable id")
        if not isfinite(self.resistance) or self.resistance < 0:
            raise ValueError("route resistance must be a finite, non-negative value")
        object.__setattr__(self, "audit_state", AuditState(self.audit_state))
        if self.audit_state is AuditState.PASS and not self.evidence_ref.strip():
            raise ValueError("a truth-preserving route needs a verifier evidence reference")


def least_resistance_route(routes: Iterable[CandidateRoute]) -> Optional[CandidateRoute]:
    """Choose the least-cost route *only* from verifier-passing candidates.

    A cheap ``HELD`` or ``FAIL`` route is never promoted by its cost.  If no
    route has a passing audit and evidence reference, return ``None`` so the
    caller can keep the task held.
    """
    eligible = [route for route in routes if route.audit_state is AuditState.PASS]
    if not eligible:
        return None
    return min(eligible, key=lambda route: (route.resistance, route.id))


@dataclass(frozen=True)
class ContextBlock:
    """One indivisible context unit at macro, meso, or micro scale."""

    id: str
    scale: Scale
    text: str
    source: str
    audit_state: Optional[AuditState] = None
    required: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a context block needs a stable id")
        if not self.text.strip():
            raise ValueError("a context block cannot be empty")
        if not self.source.strip():
            raise ValueError("a context block needs provenance")
        object.__setattr__(self, "scale", Scale(self.scale))
        if self.audit_state is not None:
            object.__setattr__(self, "audit_state", AuditState(self.audit_state))
        if self.scale is Scale.MACRO and not self.required:
            object.__setattr__(self, "required", True)

    def header(self) -> str:
        state = (
            f" audit={self.audit_state.value}:{self.audit_state.ket_label}"
            if self.audit_state is not None
            else ""
        )
        return f"[{self.scale.value.upper()} id={self.id}{state} source={self.source}]"

    def manifest(self) -> dict:
        return {
            "id": self.id,
            "scale": self.scale.value,
            "source": self.source,
            "audit_state": self.audit_state.value if self.audit_state else None,
            "required": self.required or self.scale is Scale.MACRO,
            "priority": self.priority,
        }

    def as_dict(self) -> dict:
        """Full source record, including text needed to recover an omission."""
        return {**self.manifest(), "text": self.text}


class ContextCompleteness(str, Enum):
    COMPLETE = "complete"
    REDUCED = "reduced"


class ContextBudgetError(ValueError):
    """Mandatory context did not fit; no model request should be made."""

    def __init__(
        self,
        *,
        phase: str,
        budget_units: int,
        required_units: int,
        required_ids: Sequence[str],
        source_blocks: Sequence[ContextBlock],
        reserved_text: str,
        measure: Callable[[str], int],
        unit_label: str,
    ) -> None:
        self.phase = phase
        self.budget_units = budget_units
        self.required_units = required_units
        self.required_ids = tuple(required_ids)
        self.source_blocks = tuple(source_blocks)
        self.reserved_text = reserved_text
        self.measure = measure
        self.unit_label = unit_label
        super().__init__(
            f"HELD before {phase}: mandatory context requires at least {required_units} "
            f"{unit_label}, budget is {budget_units}; nothing was clipped or sent. "
            "Increase the context budget and rebuild, or split the request."
        )

    def rebuild(self, budget_units: int) -> "ContextPack":
        return ContextBuilder(
            budget_units=budget_units,
            measure=self.measure,
            unit_label=self.unit_label,
        ).build(self.source_blocks, reserved_text=self.reserved_text, phase=self.phase)

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "state": AuditState.HELD.value,
            "budget_units": self.budget_units,
            "required_units": self.required_units,
            "unit": self.unit_label,
            "required_ids": list(self.required_ids),
            "recoverable": True,
            "source_blocks": [block.as_dict() for block in self.source_blocks],
            "reserved_text": self.reserved_text,
            "message": str(self),
        }


@dataclass(frozen=True)
class ContextPack:
    """Packed prompt context plus an intact manifest for recovery."""

    phase: str
    text: str
    budget_units: Optional[int]
    used_units: int
    unit_label: str
    completeness: ContextCompleteness
    included_ids: Tuple[str, ...]
    omitted_ids: Tuple[str, ...]
    source_blocks: Tuple[ContextBlock, ...] = field(repr=False)
    reserved_text: str = field(default="", repr=False)
    measure: Callable[[str], int] = field(default=len, repr=False, compare=False)

    @property
    def reduced(self) -> bool:
        return self.completeness is ContextCompleteness.REDUCED

    @property
    def recoverable(self) -> bool:
        return True

    def omitted_manifest(self) -> List[dict]:
        omitted = set(self.omitted_ids)
        return [block.manifest() for block in self.source_blocks if block.id in omitted]

    def summary(self) -> dict:
        return {
            "phase": self.phase,
            "completeness": self.completeness.value,
            "budget_units": self.budget_units,
            "used_units": self.used_units,
            "unit": self.unit_label,
            "included_ids": list(self.included_ids),
            "omitted_ids": list(self.omitted_ids),
            "omitted": self.omitted_manifest(),
            "recoverable": True,
        }

    def as_dict(self) -> dict:
        """Serialize both the current view and all source text for recovery."""
        return {
            **self.summary(),
            "source_blocks": [block.as_dict() for block in self.source_blocks],
            "reserved_text": self.reserved_text,
        }

    def rebuild(self, budget_units: Optional[int] = None) -> "ContextPack":
        """Rebuild from every original block; no omitted text was discarded."""
        if budget_units is None:
            builder = ContextBuilder(None, measure=self.measure, unit_label=self.unit_label)
        else:
            builder = ContextBuilder(budget_units, measure=self.measure, unit_label=self.unit_label)
        return builder.build(
            self.source_blocks,
            reserved_text=self.reserved_text,
            phase=self.phase,
        )


@dataclass(frozen=True)
class ContextUse:
    """One context pack used or measured by a MathChannel phase."""

    phase: str
    pack: ContextPack

    def as_dict(self) -> dict:
        return self.pack.as_dict()


class ContextBuilder:
    """Pack required macro/meso context first, then optional whole micro blocks."""

    _SCALE_ORDER = {Scale.MACRO: 0, Scale.MESO: 1, Scale.MICRO: 2}

    def __init__(
        self,
        budget_units: Optional[int],
        measure: Callable[[str], int] = len,
        unit_label: str = "chars",
    ) -> None:
        if budget_units is not None and budget_units < 0:
            raise ValueError("context budget cannot be negative")
        if not unit_label.strip():
            raise ValueError("context unit label cannot be empty")
        self.budget_units = budget_units
        self.measure = measure
        self.unit_label = unit_label

    def _ordered(self, blocks: Sequence[ContextBlock]) -> Tuple[ContextBlock, ...]:
        indexed = list(enumerate(blocks))
        indexed.sort(key=lambda pair: (self._SCALE_ORDER[pair[1].scale], pair[0]))
        return tuple(block for _, block in indexed)

    def _render(self, blocks: Sequence[ContextBlock], reduced: bool) -> str:
        lines = [
            "ARIA CROSS-SCALE CONTEXT",
            "CONTEXT_COMPLETENESS: REDUCED" if reduced else "CONTEXT_COMPLETENESS: COMPLETE",
        ]
        for block in self._ordered(blocks):
            lines.extend((block.header(), block.text))
        if reduced:
            lines.append(
                "TRUNCATION NOTICE: whole context blocks were omitted. Their source IDs are "
                "available to ARIA for recovery; do not infer omitted details. If an omitted "
                "detail is material, keep the result held and request the full context."
            )
        return "\n".join(lines)

    def _cost(self, text: str, reserved_text: str) -> int:
        joined = f"{reserved_text}\n{text}" if reserved_text and text else reserved_text or text
        cost = self.measure(joined)
        if not isinstance(cost, int) or cost < 0:
            raise ValueError("context measure must return a non-negative integer")
        return cost

    def build(
        self,
        blocks: Sequence[ContextBlock],
        *,
        reserved_text: str = "",
        phase: str = "context",
    ) -> ContextPack:
        source_blocks = tuple(blocks)
        ids = [block.id for block in source_blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("context block ids must be unique within a pack")

        required = tuple(
            block for block in source_blocks if block.required or block.scale is Scale.MACRO
        )
        optional = tuple(block for block in source_blocks if block not in required)
        all_ordered = self._ordered(source_blocks)
        complete_text = self._render(all_ordered, reduced=False)
        complete_cost = self._cost(complete_text, reserved_text)
        if self.budget_units is None or complete_cost <= self.budget_units:
            return ContextPack(
                phase=phase,
                text=complete_text,
                budget_units=self.budget_units,
                used_units=complete_cost,
                unit_label=self.unit_label,
                completeness=ContextCompleteness.COMPLETE,
                included_ids=tuple(block.id for block in all_ordered),
                omitted_ids=(),
                source_blocks=source_blocks,
                reserved_text=reserved_text,
                measure=self.measure,
            )

        selected = list(required)
        required_reduced_text = self._render(self._ordered(selected), reduced=True)
        required_cost = self._cost(required_reduced_text, reserved_text)
        required_ids = [block.id for block in self._ordered(required)]
        if required_cost > self.budget_units:
            raise ContextBudgetError(
                phase=phase,
                budget_units=self.budget_units,
                required_units=required_cost,
                required_ids=required_ids,
                source_blocks=source_blocks,
                reserved_text=reserved_text,
                measure=self.measure,
                unit_label=self.unit_label,
            )

        scale_rank = {Scale.MACRO: 0, Scale.MESO: 1, Scale.MICRO: 2}
        optional_index = {block.id: index for index, block in enumerate(source_blocks)}
        candidates = sorted(
            optional,
            key=lambda block: (
                -block.priority,
                scale_rank[block.scale],
                optional_index[block.id],
            ),
        )
        selected_ids = {block.id for block in selected}
        omitted: List[ContextBlock] = []
        for candidate in candidates:
            trial = [*selected, candidate]
            trial_text = self._render(self._ordered(trial), reduced=True)
            if self._cost(trial_text, reserved_text) <= self.budget_units:
                selected.append(candidate)
                selected_ids.add(candidate.id)
            else:
                omitted.append(candidate)

        selected_ordered = self._ordered(selected)
        omitted_ordered = tuple(block for block in source_blocks if block.id not in selected_ids)
        reduced_text = self._render(selected_ordered, reduced=True)
        used = self._cost(reduced_text, reserved_text)
        if used > self.budget_units:
            # Required context was checked above; this is a defensive guard.
            raise ContextBudgetError(
                phase=phase,
                budget_units=self.budget_units,
                required_units=used,
                required_ids=required_ids,
                source_blocks=source_blocks,
                reserved_text=reserved_text,
                measure=self.measure,
                unit_label=self.unit_label,
            )

        return ContextPack(
            phase=phase,
            text=reduced_text,
            budget_units=self.budget_units,
            used_units=used,
            unit_label=self.unit_label,
            completeness=ContextCompleteness.REDUCED,
            included_ids=tuple(block.id for block in selected_ordered),
            omitted_ids=tuple(block.id for block in omitted_ordered),
            source_blocks=source_blocks,
            reserved_text=reserved_text,
            measure=self.measure,
        )
